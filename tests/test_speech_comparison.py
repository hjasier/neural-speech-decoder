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


def test_comparison_notebook_extracts_all_configs_and_preserves_old_ones(tmp_path, monkeypatch):
    notebook = nbformat.read("notebooks/09_comparativa_t12_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None
        module = ast.parse(cell.source)
        for node in ast.walk(module):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and any(
                        isinstance(target, ast.Name) and target.id in {"config_code", "probe", "verification", "extract", "verify"}
                        for target in node.targets)):
                compile(node.value.value, "notebook:embedded", "exec")
    bundle = next(cell for cell in notebook.cells if "source_bundle" in cell.metadata.get("tags", []))
    namespace = {"PROJECT": tmp_path / "project"}
    exec(compile(bundle.source, "source_bundle", "exec"), namespace)
    assert len(namespace["SOURCE_MANIFEST"]["files"]) == 48
    monkeypatch.chdir(namespace["PROJECT"])
    for variant in ("baseline_5", "conjunta_235", "control_2", "control_3"):
        cfg = comparison_config(PLAN, variant, 1)
        assert cfg["experimento"]["seed"] == 1
    subprocess.run([sys.executable, "scripts/configure_speech_comparison.py", "--variant", "control_3",
                    "--seed-index", "1", "--output", "configs/active.yaml"],
                   check=True, capture_output=True, text=True)
    config_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "config_code =" in cell.source)
    node = next(node for node in ast.parse(config_cell.source).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "config_code" for target in node.targets))
    subprocess.run([sys.executable, "-c", ast.literal_eval(node.value), "configs/active.yaml"],
                   check=True, capture_output=True, text=True)
    effective = yaml.safe_load(Path("configs/speech_colab.yaml").read_text())
    assert effective["modelo"]["capas"] == 3
    assert effective["experimento"]["seed"] == 1
    assert effective["comparacion"]["variante"] == "control_3"
    assert effective["wandb"]["modo"] == "online"
    for name in ("04_entrenamiento_colab.ipynb", "05_alice_colab.ipynb", "06_demo_ahpsn.ipynb",
                 "07_baseline_t12_colab.ipynb", "08_salidas_t12_colab.ipynb"):
        with pytest.raises(ValueError, match="históricos|conservar"):
            export_notebook(tmp_path / name, perfil="speech_comparativa")
        assert not (tmp_path / name).exists()
