"""Regresiones del protocolo T12: separación, fuga de datos y selección."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.data.preprocessing import fit_train_normalization
from src.data.speech import SpeechSession, SpeechTrial
from src.data.speech_dataset import SpeechBatch
from src.data.speech_protocol import read_speech_protocol, split_training_blocks
from src.training.speech import entrenar_habla
from src.training.checkpoints import load_checkpoint


def test_protocol_covers_official_trials_without_overlapping_blocks():
    cfg = yaml.safe_load(Path("configs/speech_comparacion.yaml").read_text())
    protocol = read_speech_protocol(cfg)
    assert protocol["totals"] == {"train": 5180, "val": 1460, "test": 680}
    for session in protocol["sessions"]:
        assert not set(session["train"]) & set(session["val"])
        assert not set(session["train_blocks"]) & set(session["val_blocks"])
        assert sorted(session["train"] + session["val"]) == list(range(session["sources"]["train"]["trials"]))
        assert session["test"] == list(range(session["sources"]["test"]["trials"]))


def test_changed_partition_is_rejected_before_using_data(tmp_path):
    cfg = yaml.safe_load(Path("configs/speech_comparacion.yaml").read_text())
    path = tmp_path / "partition.json"
    path.write_bytes(Path(cfg["protocolo"]["manifiesto"]).read_bytes() + b"\n")
    cfg["protocolo"]["manifiesto"] = str(path)
    with pytest.raises(ValueError, match="SHA-256"):
        read_speech_protocol(cfg)


def test_block_split_keeps_whole_groups_even_with_unequal_sizes():
    blocks = [1] * 20 + [2] * 40 + [3] * 20 + [4] * 20
    train, val = split_training_blocks(blocks, 0.2)
    assert {blocks[i] for i in val} == {4}
    assert {blocks[i] for i in train}.isdisjoint({blocks[i] for i in val})


def test_validation_signal_cannot_change_training_statistics(monkeypatch):
    def session(value: float) -> SpeechSession:
        trials = []
        for index, data in enumerate(([1.0, 3.0], [value, value])):
            array = np.tile(np.asarray(data, dtype=np.float32)[:, None], (1, 256))
            trials.append(SpeechTrial(index, index + 1, "a", array, array))
        return SpeechSession("day", tuple(trials))

    monkeypatch.setattr("src.data.preprocessing.load_session", lambda _: session(10.0))
    first = fit_train_normalization(Path("unused"), "day", 1e-8, trial_indices=[0])
    monkeypatch.setattr("src.data.preprocessing.load_session", lambda _: session(1e9))
    second = fit_train_normalization(Path("unused"), "day", 1e-8, trial_indices=[0])
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.std, second.std)
    np.testing.assert_array_equal(second.mean, np.full(256, 2.0))
    assert second.frames == 2


def test_best_validation_checkpoint_survives_regression_and_tie(tmp_path, monkeypatch):
    cfg = yaml.safe_load(Path("configs/speech_comparacion.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["entrenamiento"]["pasos"] = 3
    cfg["evaluacion"]["cada_pasos"] = 1
    cfg["salida"]["directorio"] = str(tmp_path)
    model = torch.nn.Linear(1, 1)
    batch = SpeechBatch(torch.zeros(1, 1, 1), torch.tensor([0]), torch.tensor([1]),
                        torch.tensor([1]), torch.tensor([0]), ("a",), (0,), (1,))

    class Loader:
        dataset = SimpleNamespace(normalization_states={}, sessions=cfg["datos"]["sesiones"])

        def __iter__(self):
            return iter([batch, batch, batch])

    run = SimpleNamespace(summary={}, log=lambda *_args, **_kwargs: None)

    def backward(*_args):
        loss = model.weight.square().sum()
        loss.backward()
        return loss.item()

    # El último punto empata en PER, aunque tenga menor CTC; debe conservarse el primero.
    values = iter([(0.3, 10.0), (0.5, 9.0), (0.3, 8.0)])

    def evaluate(_model, loader, *_args):
        assert loader is validation_loader
        per, ctc = next(values)
        return {"per": per, "ctc_loss": ctc, "frases": 1, "errores": round(per * 10),
                "fonemas": 10, "completo": 1, "segundos": 0.0}

    validation_loader = object()
    monkeypatch.setattr("src.training.speech._backward_microbatch", backward)
    monkeypatch.setattr("src.training.speech._regularization", lambda *_: model.weight.sum() * 0)
    monkeypatch.setattr("src.training.speech.evaluar_habla", evaluate)
    monkeypatch.setattr("src.training.checkpoints.preparation_signature", lambda config: config["protocolo"]["sha256"])
    result = entrenar_habla(model, Loader(), validation_loader, cfg, run, torch.nn.Identity(),
                           torch.device("cpu"), tmp_path)
    best = torch.load(tmp_path / "mejor_val.pt", weights_only=True)
    last = torch.load(tmp_path / "ultimo.pt", weights_only=True)
    assert best["step"] == 1 and last["step"] == 3
    assert best["format_version"] == 2 and "test_metrics" not in best
    assert best["val_metrics"]["ctc_loss"] == 10.0
    assert result == best["val_metrics"]
    assert json.loads((tmp_path / "resultado.json").read_text())["test_evaluated"] is False
    restored = torch.nn.Linear(1, 1)
    loaded = load_checkpoint(tmp_path / "mejor_val.pt", restored, cfg, Loader.dataset)
    assert loaded["step"] == 1
    assert all(torch.equal(restored.state_dict()[name], value) for name, value in best["model_state"].items())
