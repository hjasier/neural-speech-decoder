"""Causalidad y continuidad del preprocesamiento entre paquetes."""

import numpy as np
import torch

from src.data.preprocessing import TrainingNormalization, build_smoother
from src.inference.preprocessing import OnlinePreprocessor
from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed


# Comprueba que los bordes de paquetes y el reinicio no cambian ventanas, sesiones ni salidas.
def test_packets_and_reset_preserve_full_sequence(cfg_habla_online_pequena: dict) -> None:
    """Paquetes irregulares, incluidos bins sueltos, coinciden con dos frases completas."""
    cfg = cfg_habla_online_pequena
    fijar_seed(0)
    model = construir_modelo(cfg).eval()
    smoother = build_smoother(cfg)
    count = model.input_features
    normalization = TrainingNormalization(
        session=cfg["datos"]["sesiones"][1], mean=np.linspace(-3, 3, count),
        std=np.linspace(0.5, 2, count), frames=100,
        epsilon=cfg["preprocesamiento"]["normalizacion_epsilon"],
    )
    processor = OnlinePreprocessor(model, smoother, normalization, cfg["datos"]["sesiones"])
    with torch.inference_mode():
        model.input_layers[1].weight.mul_(0.5)
        model.input_layers[1].bias.fill_(0.2)
        for length in (97, 41):
            processor.reset()
            raw = torch.randn(length, count).numpy()
            normalized = ((raw - normalization.mean) / (normalization.std + normalization.epsilon)).astype(np.float32)
            full = smoother(torch.from_numpy(normalized).unsqueeze(0))
            expected_windows = model.prepare_windows(full, torch.tensor([1]))
            expected_outputs = model.forward_exits(full, torch.tensor([1]))
            windows, pieces = [], {depth: [] for depth in model.available_exits}
            states = None
            for packet in np.split(raw, [1, 3, 8, 16, 29]):
                block = processor.push(packet)
                windows.append(block)
                if block.shape[1]:
                    outputs, states = model.forward_windows(block, states)
                    for depth in pieces:
                        pieces[depth].append(outputs[depth])
            windows.append(processor.finish())
            assert processor.finish().shape[1] == 0
            torch.testing.assert_close(torch.cat(windows, dim=1), expected_windows, atol=1e-6, rtol=1e-5)
            for depth in pieces:
                torch.testing.assert_close(torch.cat(pieces[depth], dim=1), expected_outputs[depth],
                                           atol=1e-6, rtol=1e-5)


# Comprueba el retardo del filtro y que modificar el futuro no altera la parte ya calculada.
def test_smoothing_has_declared_delay_without_future_input(cfg_habla_online_pequena: dict) -> None:
    """Un impulso alcanza su máximo tras el retardo configurado y no afecta al pasado."""
    cfg = cfg_habla_online_pequena
    smoother = build_smoother(cfg)
    delay = cfg["preprocesamiento"]["suavizado"]["retardo_bins"]
    signal = torch.zeros(1, 4 * delay + 1, cfg["modelo"]["entrada"])
    signal[0, delay, 0] = 1
    changed = signal.clone()
    boundary = 2 * delay + 1
    changed[:, boundary:] = 100
    with torch.inference_mode():
        original = smoother(signal)
        alternative = smoother(changed)
    assert original[0, :, 0].argmax().item() == 2 * delay
    torch.testing.assert_close(original[:, :boundary], alternative[:, :boundary], atol=0, rtol=0)
