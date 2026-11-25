"""Mejoras de la Fase 2: activables por configuración sin alterar los runs anteriores."""

from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_queue import queue_jobs
from src.models.factory import construir_modelo
from src.models.speech_gru import SpeechGRU
from src.training.speech import _time_mask, time_mask_spec

ROOT = Path(__file__).resolve().parents[1]
CAUSAL = yaml.safe_load((ROOT / "configs/final/t12_causal.yaml").read_text())


def _cfg(spec: dict | None) -> dict:
    training = {} if spec is None else {"enmascarado_temporal": spec}
    return {"entrenamiento": training}


# Un tramo sobre el relleno no aumentaría nada, y uno fuera de rango taparía otra frase.
def test_time_mask_zeroes_bounded_spans_inside_each_sentence() -> None:
    torch.manual_seed(0)
    features = torch.ones(64, 50, 3)
    lengths = torch.randint(5, 51, (64,))
    masked = _time_mask(features, lengths, _cfg({"mascaras": 2, "ancho_max_bins": 6}))
    zero = masked.eq(0).all(dim=-1)
    assert torch.equal(masked.eq(0).any(dim=-1), zero)  # se tapa el paso entero, no canales sueltos
    steps = torch.arange(50)[None, :]
    assert not (zero & (steps >= lengths[:, None])).any()
    assert int(zero.sum(dim=1).max()) <= 12 and int(zero.sum()) > 0
    assert torch.equal(features, torch.ones(64, 50, 3))  # no modifica el tensor de entrada


# Sin la clave, los perfiles de la referencia deben entrenar exactamente igual que antes.
def test_time_mask_is_inert_without_its_key_and_rejects_bad_specs() -> None:
    features = torch.randn(4, 20, 3)
    state = torch.get_rng_state()
    assert _time_mask(features, torch.full((4,), 20), _cfg(None)) is features
    assert torch.equal(torch.get_rng_state(), state)
    assert time_mask_spec(CAUSAL) is None
    for invalid in ({"mascaras": 0, "ancho_max_bins": 5}, {"mascaras": 2}, {"mascaras": 2, "ancho_max_bins": 5.0},
                    {"mascaras": 2, "ancho_max_bins": 5, "extra": 1}, [2, 5]):
        with pytest.raises(ValueError, match="enmascarado_temporal"):
            time_mask_spec(_cfg(invalid))


def _model(layer_norm: bool) -> SpeechGRU:
    torch.manual_seed(0)
    return SpeechGRU(input_features=4, input_size=4, hidden_size=8, num_layers=3, num_sessions=2, num_classes=5,
                     kernel_size=3, stride=2, input_dropout=0.0, gru_dropout=0.0, intermediate_layers=(2,),
                     layer_norm=layer_norm)


# Los checkpoints de la referencia se cargan con strict=True: no pueden aparecer claves nuevas.
def test_layer_norm_off_keeps_the_original_weights_and_keys() -> None:
    plain, normalized = _model(False), _model(True)
    assert plain.layer_norms is None
    extra = set(normalized.state_dict()) - set(plain.state_dict())
    assert extra and all(key.startswith("layer_norms.") for key in extra)
    for key, value in plain.state_dict().items():
        assert torch.equal(value, normalized.state_dict()[key])
    assert construir_modelo(CAUSAL).layer_norms is None
    with pytest.raises(ValueError, match="layer_norm"):
        SpeechGRU(input_features=4, input_size=4, hidden_size=8, num_layers=3, num_sessions=2, num_classes=5,
                  kernel_size=3, stride=2, input_dropout=0.0, gru_dropout=0.0, layer_norm="sí")


# La demo y la evaluación por bloques continúan estados entre paquetes: la norma no puede romperlo.
def test_layer_norm_keeps_block_streaming_equal_to_the_full_sentence() -> None:
    model = _model(True).eval()
    with torch.no_grad():
        for norm in model.layer_norms:
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
        windows = model.prepare_windows(torch.randn(2, 41, 4), torch.tensor([0, 1]))
        full, _ = model.forward_windows(windows)
        first, states = model.forward_windows(windows[:, :7])
        second, _ = model.forward_windows(windows[:, 7:], states)
    for depth in model.available_exits:
        torch.testing.assert_close(torch.cat([first[depth], second[depth]], dim=1), full[depth])
    cfg = {**CAUSAL, "modelo": {**CAUSAL["modelo"], "layer_norm": True}}
    assert len(construir_modelo(cfg).layer_norms) == CAUSAL["modelo"]["capas"]


# Cada variante debe diferir de la referencia solo en lo que declara; si no, la comparación no vale.
def test_phase2_variants_change_only_their_declared_keys() -> None:
    jobs = queue_jobs(ROOT / "configs/final/cola_fase2.yaml")
    assert len(jobs) == 15 and len({job.config["salida"]["directorio"] for job in jobs}) == 15
    expected = {
        "t12_causal_mascara": {("entrenamiento", "enmascarado_temporal")},
        "t12_causal_layernorm": {("modelo", "layer_norm")},
        "t12_causal_20k": {("entrenamiento", "pasos"), ("entrenamiento", "learning_rate")},
    }
    for job in jobs:
        changed = {(section, key) for section in ("modelo", "entrenamiento", "preprocesamiento", "datos")
                   for key in set(job.config[section]) | set(CAUSAL[section])
                   if job.config[section].get(key) != CAUSAL[section].get(key)}
        assert changed == expected[job.variant], job.name
        assert job.config["protocolo"] == CAUSAL["protocolo"] and job.config["evaluacion"] == CAUSAL["evaluacion"]
        time_mask_spec(job.config)
    longer = next(job for job in jobs if job.variant == "t12_causal_20k").config["entrenamiento"]
    assert longer["learning_rate"]["decay_pasos"] == longer["pasos"] - 1


# El paso de entrenamiento real debe aplicar las dos mejoras sin romper CTC ni gradientes.
def test_training_step_runs_with_mask_and_layer_norm() -> None:
    from torch import nn

    from src.data.speech_dataset import SpeechSample, collate_speech
    from src.training.speech import _backward_microbatch, _criterion

    cfg = yaml.safe_load((ROOT / "configs/speech_salidas_t4.yaml").read_text())
    cfg["modelo"].update(entrada=4, capa_dia=4, unidades=8, kernel=2, stride=2, dropout_entrada=0.0,
                         dropout_gru=0.0, layer_norm=True)
    cfg["datos"]["sesiones"] = ["day_a", "day_b"]
    cfg["entrenamiento"]["ruido"] = {"blanco_sd": 0.0, "offset_sd": 0.0}
    cfg["entrenamiento"]["enmascarado_temporal"] = {"mascaras": 2, "ancho_max_bins": 3}
    rows = [SpeechSample(torch.linspace(-1, 1, length * 4).reshape(length, 4), torch.tensor(ids), index % 2,
                         "reference", index, 1)
            for index, (length, ids) in enumerate(((14, [0, 1]), (12, [1, 1]), (10, [2])))]
    torch.manual_seed(0)
    model = construir_modelo(cfg).train()
    loss = _backward_microbatch(model, collate_speech(rows), cfg, nn.Identity(), _criterion(cfg),
                                torch.device("cpu"), 3)
    assert torch.isfinite(torch.tensor(loss))
    assert all(norm.weight.grad is not None for norm in model.layer_norms)
