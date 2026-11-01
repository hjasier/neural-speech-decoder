"""Errores silenciosos de CTC y estado compartido entre flujos."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.data.preprocessing import TrainingNormalization, build_smoother
from src.inference.ctc import IncrementalCTC, decode_ctc
from src.inference.speech import OnlineSpeechDecoder
from src.models.factory import construir_modelo
from src.models.speech_gru import SpeechGRU
from src.training.checkpoints import save_checkpoint
from src.utils.seed import fijar_seed


@pytest.fixture
def online_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_habla_online_pequena: dict,
) -> tuple[Path, SpeechGRU, dict[str, TrainingNormalization]]:
    """Guarda el formato real con pesos sintéticos y estadísticas distintas por sesión."""
    cfg = cfg_habla_online_pequena
    fijar_seed(0)
    model = construir_modelo(cfg).eval()
    with torch.no_grad():
        model.input_layers[1].weight.mul_(0.5)
        model.input_layers[1].bias.fill_(0.2)
    normalization = {
        name: TrainingNormalization(
            session=name, mean=np.linspace(-3, 3, model.input_features) + index,
            std=np.linspace(0.5, 2, model.input_features) + index, frames=100,
            epsilon=cfg["preprocesamiento"]["normalizacion_epsilon"],
        )
        for index, name in enumerate(cfg["datos"]["sesiones"])
    }
    dataset = SimpleNamespace(
        sessions=cfg["datos"]["sesiones"],
        normalization_states={name: values.to_dict() for name, values in normalization.items()},
    )
    monkeypatch.setattr("src.training.checkpoints.preparation_signature",
                        lambda config: config["protocolo"]["sha256"])
    metrics = {"per": 1.0, "completo": 1}
    path = tmp_path / "synthetic_online.pt"
    save_checkpoint(path, model, cfg, dataset, 1, metrics, split="val",
                    exit_metrics={depth: metrics.copy() for depth in model.available_exits})
    return path, model, normalization


# Repeticiones en la frontera y blanks entre fonemas iguales no deben alterar el resultado.
def test_incremental_ctc_preserves_runs_and_blank_boundaries() -> None:
    """Partir el camino CTC en paquetes conserva también el fonema de índice cero."""
    blank = 3
    path = torch.tensor([0, 0, blank, 0, 1, 1, blank, 1, 1, 2, blank, 2, 2])
    logits = F.one_hot(path, num_classes=blank + 1).float()
    decoder = IncrementalCTC(blank)
    actual = []
    for packet in torch.tensor_split(logits, [1, 2, 3, 5, 7, 9]):
        actual.extend(decoder.push(packet))
    expected = [0, 0, 1, 1, 2, 2]
    assert actual == expected
    assert decode_ctc(logits.unsqueeze(0), torch.tensor([len(path)]), blank) == [expected]
    decoder.reset()
    assert decoder.push(logits[-2:]) == [2]


# Cambiar el retardo conserva todas las dimensiones de pesos, pero cambia el modelo temporal.
def test_checkpoint_prevents_changing_delay_with_same_weights(
    online_checkpoint: tuple[Path, SpeechGRU, dict[str, TrainingNormalization]],
) -> None:
    """Un checkpoint válido no permite aplicar sus pesos con otro retardo causal."""
    path, _, _ = online_checkpoint
    OnlineSpeechDecoder.from_checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["config"]["preprocesamiento"]["suavizado"]["retardo_bins"] += 1
    torch.save(state, path)
    with pytest.raises(ValueError):
        OnlineSpeechDecoder.from_checkpoint(path)


# Compara logits reales: solo mirar argmax podría ocultar estados GRU mal continuados.
def test_interleaved_streams_and_reset_match_complete_sequences(
    online_checkpoint: tuple[Path, SpeechGRU, dict[str, TrainingNormalization]],
) -> None:
    """Dos sesiones y prefijos intercalados conservan logits, emisiones y reinicio."""
    path, original, normalization = online_checkpoint
    decoder = OnlineSpeechDecoder.from_checkpoint(path)
    checkpoint = decoder.checkpoint
    streams = {"a": decoder.start("sesion_a"), "b": decoder.start("sesion_b", layers=(2,))}
    raw = {"a": torch.randn(97, original.input_features).numpy(),
           "b": torch.randn(77, original.input_features).numpy()}
    expected: dict[str, dict[int, torch.Tensor]] = {}
    expected_ids: dict[str, dict[int, list[int]]] = {}
    smoother = build_smoother(checkpoint.config)
    with torch.inference_mode():
        for index, name in enumerate(("a", "b")):
            session = checkpoint.sessions[index]
            values = normalization[session].transform_features(raw[name], session)
            expected[name] = original.forward_exits(
                smoother(torch.from_numpy(values).unsqueeze(0)), torch.tensor([index]),
                layers=streams[name].layers,
            )
            expected_ids[name] = {
                depth: decode_ctc(logits, torch.tensor([logits.shape[1]]), checkpoint.blank_id)[0]
                for depth, logits in expected[name].items()
            }
    observed: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {depth: [] for depth in original.available_exits} for name in ("a", "b", "again")
    }
    current = {"flow": "a"}
    heads = {int(depth): head for depth, head in checkpoint.model.intermediate_heads.items()}
    heads[original.num_layers] = checkpoint.model.output_layer
    hooks = [
        head.register_forward_hook(
            lambda _module, _inputs, output, depth=depth:
            observed[current["flow"]][depth].append(output.detach().clone())
        )
        for depth, head in heads.items()
    ]
    updates = {name: {depth: [] for depth in stream.layers} for name, stream in streams.items()}
    packets = {"a": np.split(raw["a"], [1, 3, 8, 16, 29]),
               "b": np.split(raw["b"], [2, 17, 18, 31, 57])}
    try:
        for index in range(len(packets["a"])):
            for name in streams:
                current["flow"] = name
                for depth, phones in streams[name].push(packets[name][index]).items():
                    updates[name][depth].extend(phones)
        for name, stream in streams.items():
            for depth, logits in expected[name].items():
                torch.testing.assert_close(torch.cat(observed[name][depth], dim=1), logits,
                                           atol=1e-6, rtol=1e-5)
            assert updates[name] == stream.hypotheses == expected_ids[name]
        assert observed["b"][3] == observed["b"][5] == []
        assert streams["b"].finish() == streams["b"].finish() == {2: []}
        assert streams["b"].hypotheses == expected_ids["b"]

        # Reiniciar una frase abierta debe borrar filtro, solapamiento, GRU y CTC.
        streams["a"].reset()
        current["flow"] = "again"
        assert streams["a"].push(raw["a"]) == expected_ids["a"]
        assert streams["a"].finish() == {depth: [] for depth in streams["a"].layers}
        assert streams["a"].hypotheses == expected_ids["a"]
        for depth, logits in expected["a"].items():
            torch.testing.assert_close(torch.cat(observed["again"][depth], dim=1), logits,
                                       atol=1e-6, rtol=1e-5)
    finally:
        for hook in hooks:
            hook.remove()
