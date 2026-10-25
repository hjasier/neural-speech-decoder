"""Condiciones comunes, selección de semilla y paquete Colab de la comparación."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest
import yaml

from scripts.configure_speech_comparison import comparison_config
from scripts.export_colab import export_notebook
from src.data.speech_protocol import read_speech_protocol


PLAN = Path("configs/speech_comparativa.yaml")


def test_all_combinations_preserve_protocol_and_choose_seed():
    plan = yaml.safe_load(PLAN.read_text())
    depths = {"baseline_5": 5, "conjunta_235": 5, "control_2": 2, "control_3": 3}
    for variant, expected_depth in depths.items():
        for index, seed in enumerate(plan["semillas"]):
            cfg = comparison_config(PLAN, variant, index)
            assert cfg["experimento"]["seed"] == seed
            assert cfg["experimento"]["nombre"].endswith(f"_seed{seed}")
            assert cfg["modelo"]["capas"] == expected_depth
            assert cfg["modelo"].get("salidas_intermedias", []) == ([2, 3] if variant == "conjunta_235" else [])
            assert ("ctc_pesos" in cfg["entrenamiento"]) == (variant == "conjunta_235")
            assert cfg["entrenamiento"]["pasos"] == 10001
            assert cfg["comparacion"]["variante"] == variant
            assert read_speech_protocol(cfg)["totals"] == {"train": 5180, "val": 1460, "test": 680}


@pytest.mark.parametrize("field", ["budget", "normalization", "time_limit"])
def test_changed_comparison_conditions_are_rejected(tmp_path, monkeypatch, field):
    plan = yaml.safe_load(PLAN.read_text())
    for variant, path in plan["variantes"].items():
        cfg = yaml.safe_load(Path(path).read_text())
        if variant == "control_2":
            if field == "budget":
                cfg["entrenamiento"]["pasos"] -= 1
            elif field == "normalization":
                cfg["preprocesamiento"]["normalizacion_epsilon"] *= 10
            else:
                cfg["entrenamiento"]["max_segundos"] = 1000
        destination = tmp_path / f"{variant}.yaml"
        destination.write_text(yaml.safe_dump(cfg))
        plan["variantes"][variant] = destination.name
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="comunes"):
        comparison_config(Path("plan.yaml"), "baseline_5", 0)


@pytest.mark.parametrize("variant,index", [("missing", 0), ("baseline_5", -1), ("baseline_5", 3)])
def test_unknown_variant_or_seed_index_is_rejected(variant, index):
    with pytest.raises(ValueError):
        comparison_config(PLAN, variant, index)
