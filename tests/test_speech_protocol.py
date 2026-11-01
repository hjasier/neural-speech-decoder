"""Separación de los datos T12 y normalización sin fuga de validación."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.data.preprocessing import fit_train_normalization
from src.data.speech import SpeechSession, SpeechTrial
from src.data.speech_protocol import read_speech_protocol


# Comprueba que la partición incluye todos los ensayos previstos sin compartir frases ni bloques entre entrenamiento y validación.
def test_protocol_covers_official_trials_without_overlapping_blocks():
    cfg = yaml.safe_load(Path("configs/speech_comparacion_t4.yaml").read_text())
    protocol = read_speech_protocol(cfg)
    assert protocol["totals"] == {"train": 5180, "val": 1460, "test": 680}
    for session in protocol["sessions"]:
        assert not set(session["train"]) & set(session["val"])
        assert not set(session["train_blocks"]) & set(session["val_blocks"])
        assert sorted(session["train"] + session["val"]) == list(range(session["sources"]["train"]["trials"]))
        assert session["test"] == list(range(session["sources"]["test"]["trials"]))


# Comprueba que cambiar los datos de validación no altera la media ni la desviación usadas para normalizar el entrenamiento.
#Si usáramos toda la sesión para normalizar, cambiar la validación alteraría esas estadísticas y el test fallaría.
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
