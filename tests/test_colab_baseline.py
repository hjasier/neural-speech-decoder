"""Contrato del lanzador remoto: fuentes verificadas y protocolo sin test."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest
import yaml

from scripts.export_colab import export_notebook
from src.data.speech_protocol import read_speech_protocol


def test_baseline_snapshot_extracts_and_contains_fixed_protocol(tmp_path, monkeypatch):
    notebook = nbformat.read("notebooks/07_baseline_t12_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        module = ast.parse(cell.source)
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"config_code", "probe", "verification", "extract", "verify"}:
                        compile(ast.literal_eval(node.value), f"notebook:{target.id}", "exec")
    bundle = next(cell for cell in notebook.cells if "source_bundle" in cell.metadata.get("tags", []))
    project = tmp_path / "project"
    namespace = {"PROJECT": project}
    exec(compile(bundle.source, "source_bundle", "exec"), namespace)
    assert namespace["SOURCE_MANIFEST"]["files"] == notebook.metadata.source_snapshot["files"]
    monkeypatch.chdir(project)
    cfg = yaml.safe_load(Path("configs/speech_comparacion_t4.yaml").read_text())
    assert read_speech_protocol(cfg)["totals"] == {"train": 5180, "val": 1460, "test": 680}
    assert cfg["modelo"]["capas"] == 5 and cfg["modelo"]["unidades"] == 1024
    assert cfg["entrenamiento"]["pasos"] == 10001
    assert cfg["entrenamiento"].get("max_segundos") is None
    assert cfg["entrenamiento"]["micro_batch_size"] == 16
    config_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "config_code =" in cell.source)
    node = next(node for node in ast.parse(config_cell.source).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "config_code" for target in node.targets))
    subprocess.run([sys.executable, "-c", ast.literal_eval(node.value), "configs/speech_comparacion_t4.yaml"],
                   check=True, capture_output=True, text=True)
    effective = yaml.safe_load(Path("configs/speech_colab.yaml").read_text())
    for key in ("protocolo", "modelo", "entrenamiento", "evaluacion"):
        assert effective[key] == cfg[key]
    assert effective["wandb"]["modo"] == "online"


def test_comparison_export_cannot_overwrite_historical_notebooks(tmp_path):
    for name in ("04_entrenamiento_colab.ipynb", "05_alice_colab.ipynb", "06_demo_ahpsn.ipynb"):
        with pytest.raises(ValueError, match="históricos"):
            export_notebook(tmp_path / name, perfil="speech_comparacion")
        assert not (tmp_path / name).exists()


def test_exit_snapshot_contains_short_real_run_and_fresh_reload(tmp_path, monkeypatch):
    notebook = nbformat.read("notebooks/08_salidas_t12_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        module = ast.parse(cell.source)
        assert cell.execution_count is None and not cell.outputs
        for node in ast.walk(module):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)) and any(
                isinstance(target, ast.Name) and target.id in {"config_code", "probe", "verification", "extract", "verify"}
                for target in node.targets
            ):
                compile(ast.literal_eval(node.value), "notebook:embedded", "exec")
    bundle = next(cell for cell in notebook.cells if "source_bundle" in cell.metadata.get("tags", []))
    namespace = {"PROJECT": tmp_path / "project"}
    exec(compile(bundle.source, "source_bundle", "exec"), namespace)
    assert len(namespace["SOURCE_MANIFEST"]["files"]) == 44
    monkeypatch.chdir(namespace["PROJECT"])
    short = yaml.safe_load(Path("configs/speech_salidas_e1.yaml").read_text())
    full = yaml.safe_load(Path("configs/speech_salidas_t4.yaml").read_text())
    assert short["entrenamiento"]["pasos"] == 200
    assert full["entrenamiento"]["pasos"] == 10001
    assert short["modelo"] == full["modelo"]
    assert short["entrenamiento"]["learning_rate"] == full["entrenamiento"]["learning_rate"]
    assert short["entrenamiento"]["ctc_pesos"] == {"2": 0.25, "3": 0.25, "5": 0.5}
    assert read_speech_protocol(short)["totals"] == {"train": 5180, "val": 1460, "test": 680}
    assert Path("scripts/verify_speech_checkpoint.py").is_file()
    assert Path("src/training/verification.py").is_file()
    reload_cell = next(cell for cell in notebook.cells if "#@title 11." in cell.source)
    assert 'ejecutar(["scripts/verify_speech_checkpoint.py"' in reload_cell.source


def test_exit_export_preserves_previous_four_notebooks(tmp_path):
    for name in ("04_entrenamiento_colab.ipynb", "05_alice_colab.ipynb", "06_demo_ahpsn.ipynb",
                 "07_baseline_t12_colab.ipynb"):
        with pytest.raises(ValueError, match="históricos|conservar"):
            export_notebook(tmp_path / name, perfil="speech_salidas")
        assert not (tmp_path / name).exists()
