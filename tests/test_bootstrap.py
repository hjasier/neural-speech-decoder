"""Intervalos de confianza: tasa agregada, reproducibilidad y separación por tipo de habla."""

import pytest
import yaml
from pathlib import Path

from scripts.summarize_speech_predictions import summarize_predictions
from src.training.bootstrap import bootstrap_rate_interval

OPTIONS = {"resamples": 2000, "confidence": 0.95, "seed": 0}


# Promediar tasas por frase daría (1/1 + 0/9)/2 = 0,5 en lugar de 1 error entre 10 símbolos.
def test_rate_aggregates_counts_instead_of_averaging_sentence_rates() -> None:
    result = bootstrap_rate_interval([1, 0], [1, 9], **OPTIONS)
    assert result["rate"] == pytest.approx(0.1)
    assert result["low"] <= result["rate"] <= result["high"]
    assert (result["errors"], result["total"], result["trials"]) == (1, 10, 2)


# Con la misma semilla el intervalo debe repetirse; si todas las frases son iguales, no hay anchura.
def test_interval_is_reproducible_and_degenerate_for_identical_sentences() -> None:
    errors, totals = [0, 1, 2, 3, 4] * 20, [10] * 100
    assert bootstrap_rate_interval(errors, totals, **OPTIONS) == bootstrap_rate_interval(errors, totals, **OPTIONS)
    assert bootstrap_rate_interval(errors, totals, **{**OPTIONS, "seed": 1}) != bootstrap_rate_interval(errors, totals, **OPTIONS)
    constant = bootstrap_rate_interval([2] * 50, [10] * 50, **OPTIONS)
    assert constant["low"] == constant["high"] == constant["rate"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bootstrap_rate_interval([], [], **OPTIONS)


# La Tabla 1 del paper separa habla vocalizada y silenciosa: mezclar sesiones cambiaría la comparación.
def test_summary_separates_silent_sessions_listed_in_the_config() -> None:
    spec = yaml.safe_load(Path("configs/final/resumen_metricas.yaml").read_text())
    spec["bootstrap"]["remuestreos"] = 200

    def row(session: str, word_errors: int) -> dict:
        return {"split": "val", "session": session, "profiles": {"5": {
            "errores_fonemas": 1, "fonemas": 10, "errores_palabras": word_errors, "palabras": 4}}}

    rows = [row("t12.2022.08.13", 0)] * 3 + [row("t12.2022.08.18", 2)] * 2
    summary = summarize_predictions(rows, spec)["by_exit"]["5"]
    assert summary["vocalizada"]["wer"]["rate"] == 0 and summary["vocalizada"]["wer"]["trials"] == 3
    assert summary["silenciosa"]["wer"]["rate"] == pytest.approx(0.5)
    assert summary["global"]["wer"]["rate"] == pytest.approx(4 / 20)
    assert summary["global"]["per"]["rate"] == pytest.approx(0.1)
