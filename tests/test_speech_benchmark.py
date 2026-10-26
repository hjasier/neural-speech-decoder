"""Contrato de etapas, prefijos reales, muestras y normalización del benchmark."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.data.preprocessing import TrainingNormalization, build_smoother
from src.data.speech import SpeechSession, SpeechTrial
from src.data.speech_dataset import SpeechSample
from src.models.factory import construir_modelo
from src.training.speech_benchmark import (
    BenchmarkCase, _timed, benchmark_profiles, prepare_benchmark_cases,
)
from src.utils.seed import fijar_seed


def _fixture():
    cfg = yaml.safe_load(Path("configs/speech_salidas_t4.yaml").read_text())
    cfg["datos"]["sesiones"] = ["fixture"]
    cfg["modelo"].update(capa_dia=4, unidades=8, kernel=2, stride=2,
                         dropout_entrada=0.0, dropout_gru=0.0)
    spec = yaml.safe_load(Path("configs/speech_benchmark.yaml").read_text())
    spec.update(dispositivo="cpu", calentamiento=1, repeticiones=2)
    stats = TrainingNormalization("fixture", np.zeros(256), np.ones(256), 36, 1e-8)
    cases = []
    for i, length in enumerate((14, 10)):
        values = (np.arange(length * 256).reshape(length, 256) % 17).astype(np.float32)
        raw = SpeechTrial(i, 1, "fixture", values, values + 1)
        sample = SpeechSample(torch.from_numpy(stats.transform(raw, "fixture")), torch.tensor([0, 1]), 0, "fixture", i, 1)
        cases.append(BenchmarkCase(i, "fixture", raw, sample))
    return cfg, spec, {"fixture": stats}, cases


def test_benchmark_uses_real_prefixes_and_keeps_raw_and_forward_routes_equal():
    cfg, spec, stats, cases = _fixture()
    fijar_seed(0)
    model = construir_modelo(cfg)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    result = benchmark_profiles(model, build_smoother(cfg), cases, stats, spec, torch.device("cpu"))
    assert result["optimizer_steps"] == 0 and result["decoded_profiles_match"] is True
    assert len(result["measurements"]) == 3 * 2 * 2 * 2
    for depth in (2, 3, 5):
        assert result["profiles"][depth]["executed_layers"] == list(range(1, depth + 1))
        for stage in ("forward", "raw_to_phones"):
            summary = result["profiles"][depth][stage]
            assert summary["observations"] == 4
            assert 0 < summary["p50_ms"] <= summary["p95_ms"]
            assert summary["gpu_peak_allocated_bytes"] is None
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert result["resident_model_parameters"] == sum(parameter.numel() for parameter in model.parameters())


def test_timer_waits_for_device_before_and_after_operation(monkeypatch):
    order = []
    monkeypatch.setattr("src.training.speech_benchmark._sync", lambda _device: order.append("sync"))
    milliseconds, value = _timed(lambda: order.append("call") or 17, torch.device("cuda"))
    assert order == ["sync", "call", "sync"] and value == 17 and milliseconds > 0


def test_input_selection_is_reproducible_and_rejects_changed_raw_recording(tmp_path, monkeypatch):
    cfg, spec, stats, cases = _fixture()
    cfg["datos"]["raiz"] = str(tmp_path / "raw")
    manifest = {"files": [{"split": "train", "session": "fixture", "source_sha256": "fixed"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    class Dataset:
        split = "val"
        root = tmp_path
        sessions = ("fixture",)
        session_indices = [range(2)]
        normalization_states = {"fixture": stats["fixture"].to_dict()}

        def __getitem__(self, index):
            return cases[index].sample

    monkeypatch.setattr("src.training.speech_benchmark.file_sha256", lambda _path: "fixed")
    monkeypatch.setattr("src.training.speech_benchmark.preparation_signature", lambda _cfg: "fixture_signature")
    monkeypatch.setattr("src.training.speech_benchmark.load_session", lambda _path: SpeechSession("fixture", tuple(case.raw for case in cases)))
    _, _, first = prepare_benchmark_cases(Dataset(), cfg, spec)
    _, _, second = prepare_benchmark_cases(Dataset(), cfg, spec)
    assert first["input_signature"] == second["input_signature"]
    assert first["samples"] == second["samples"]
    monkeypatch.setattr("src.training.speech_benchmark.file_sha256", lambda _path: "changed")
    with pytest.raises(ValueError, match="cambiado"):
        prepare_benchmark_cases(Dataset(), cfg, spec)
