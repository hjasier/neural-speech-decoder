"""Recuentos de PER y reloj de llegada: cálculos que pueden fallar sin excepción."""

from itertools import count
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.preprocessing import TrainingNormalization, build_smoother
from src.inference.checkpoints import SpeechCheckpoint
from src.inference.speech import OnlineSpeechDecoder
from src.models.factory import construir_modelo
from src.training.speech_online_evaluation import OnlineTrial, evaluate_online
from src.utils.seed import fijar_seed


@pytest.fixture
def online_evaluation_case(cfg_habla_online_pequena: dict) -> tuple[OnlineSpeechDecoder, list[OnlineTrial], dict]:
    """Dos frases de longitudes distintas; todas las salidas predicen el fonema cero."""
    cfg = cfg_habla_online_pequena
    fijar_seed(0)
    model = construir_modelo(cfg).eval()
    with torch.no_grad():
        for head in [model.output_layer, *model.intermediate_heads.values()]:
            head.weight.zero_()
            head.bias.zero_()
            head.bias[0] = 1
    sessions = tuple(cfg["datos"]["sesiones"])
    normalization = {
        name: TrainingNormalization(name, np.zeros(model.input_features), np.ones(model.input_features),
                                    100, cfg["preprocesamiento"]["normalizacion_epsilon"])
        for name in sessions
    }
    checkpoint = SpeechCheckpoint(model, build_smoother(cfg), cfg, sessions,
                                  tuple(cfg["datos"]["fonemas"]), len(cfg["datos"]["fonemas"]),
                                  normalization, 0, "synthetic")
    trials = [OnlineTrial(np.zeros((length, model.input_features), dtype=np.float32), ids,
                          sessions[index], index, 1, "referencia sintética")
              for index, (length, ids) in enumerate(((19, (0,)), (25, (1, 2, 3, 4))))]
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/speech_online_evaluacion.yaml").read_text())
    spec["calentamiento"] = 0
    return OnlineSpeechDecoder(checkpoint), trials, spec


# Promediar PER por frase o por sesión daría 0,5 en lugar de los cuatro errores entre cinco fonemas.
def test_online_per_uses_total_phone_counts(online_evaluation_case: tuple) -> None:
    """Incluye todas las frases y agrega errores, sin promediar sus porcentajes."""
    decoder, trials, spec = online_evaluation_case
    result = evaluate_online(decoder, iter(trials), spec, expected_trials=len(trials))
    for depth in decoder.checkpoint.model.available_exits:
        assert result["global"][depth] == {"errores": 4, "fonemas": 5, "frases": 2,
                                           "per": 4 / 5, "completo": 1}
        assert result["by_session"][depth]["sesion_a"]["errores"] == 0
        assert result["by_session"][depth]["sesion_b"]["errores"] == 4


# Un bloque tarda 120 ms pero llega cada 80 ms: ignorar la cola falsearía el primer fonema.
def test_online_timing_accounts_for_arrival_backlog_and_total_duration(
    online_evaluation_case: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El RTF usa tiempo total de señal y el primer fonema incluye espera y cálculo."""
    decoder, trials, spec = online_evaluation_case
    ticks = count()
    monkeypatch.setattr("src.training.speech_online_evaluation.perf_counter", lambda: next(ticks) * 0.12)
    rows = []
    result = evaluate_online(decoder, iter(trials), spec, expected_trials=len(trials), prediction_sink=rows.append)
    for depth in decoder.checkpoint.model.available_exits:
        measured = result["timing"][depth]
        # Apertura + 5 o 7 paquetes + cierre; duraciones de señal de 380 y 500 ms.
        assert measured["processing_ms"] == pytest.approx(1920)
        assert measured["signal_ms"] == pytest.approx(880)
        assert measured["rtf"] == pytest.approx(1920 / 880)
        assert measured["packet_processing"]["observations"] == 12
        first = rows[0]["profiles"][depth]
        assert first["first_phone_input_ms"] == pytest.approx(320)
        assert first["first_phone_ready_ms"] == pytest.approx(600)
        assert first["packets"][-1]["input_ms"] == pytest.approx(380)
