"""Integración en procesos separados con archivos diminutos; no mide calidad T12."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from src.data.speech_dataset import file_sha256, preparation_signature


ROOT = Path(__file__).absolute().parents[1]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _prepared_fixture(tmp_path, monkeypatch, width=4):
    cfg = yaml.safe_load((ROOT / "configs/speech_salidas_e1.yaml").read_text())
    cfg["experimento"]["nombre"] = "unit_fixture_not_t12"
    cfg["datos"].update(raiz="raw", preparados="prepared", sesiones=["fixture"], batch_size=3)
    cfg["etiquetas"]["salida"] = "labels"
    cfg["modelo"].update(entrada=width, capa_dia=4, unidades=8, kernel=2, stride=2,
                         dropout_entrada=0.0, dropout_gru=0.0)
    cfg["entrenamiento"].update(pasos=3, micro_batch_size=2, imprimir_cada=1)
    cfg["entrenamiento"]["ruido"] = {"blanco_sd": 0.0, "offset_sd": 0.0}
    cfg["evaluacion"].update(cada_pasos=1, micro_batch_size=2)
    cfg["ejecucion"]["dispositivo"] = "cpu"
    cfg["salida"]["directorio"] = "checkpoints/fixture"
    cfg["wandb"].update(modo="offline", directorio="tracking")
    cfg["protocolo"].update(nombre="fixture", fraccion_validacion=0.5, manifiesto="split.json")
    monkeypatch.chdir(tmp_path)
    protocol = {
        "format_version": 1, "name": "fixture", "method": "ultimos_bloques_por_sesion",
        "validation_fraction_target": 0.5,
        "checkpoint_selection": "minimum validation PER; first checkpoint wins ties",
        "totals": {"train": 3, "val": 3, "test": 1},
        "sessions": [{"session": "fixture", "train": [0, 1, 2], "val": [3, 4, 5], "test": [0],
                      "train_blocks": [1], "val_blocks": [2], "sources": {
                          "train": {"sha256": "a" * 64, "trials": 6},
                          "test": {"sha256": "b" * 64, "trials": 1}}}],
    }
    _write_json(Path("split.json"), protocol)
    cfg["protocolo"]["sha256"] = file_sha256(Path("split.json"))
    ids = [[0, 1], [1, 1], [2]] * 2
    labels = {"session": "fixture", "split": "train", "transcriptions_sha256": "c" * 64,
              "targets": [{"trial_index": i, "reference": "synthetic fixture", "ids": target,
                           "minimum_ctc_steps": len(target) + sum(a == b for a, b in zip(target, target[1:]))}
                          for i, target in enumerate(ids)]}
    _write_json(Path("labels/train/fixture.json"), labels)
    _write_json(Path("labels/manifest.json"), {
        "format_version": 1, "sessions": ["fixture"], "phones": cfg["datos"]["fonemas"], "blank_id": 40,
        "label_config": cfg["etiquetas"], "totals": {"train": 6, "test": 1},
        "files": [{"path": "train/fixture.json", "sha256": file_sha256(Path("labels/train/fixture.json"))}],
    })
    lengths = [14, 12, 10] * 2
    values = np.concatenate([np.linspace(-1 + i, 1 + i, length * width).reshape(length, width)
                             for i, length in enumerate(lengths)])
    mean, std = values[:36].mean(0), values[:36].std(0)
    epsilon = cfg["preprocesamiento"]["normalizacion_epsilon"]
    Path("prepared/train").mkdir(parents=True)
    np.save("prepared/train/fixture.npy", ((values - mean) / (std + epsilon)).astype(np.float32))
    _write_json(Path("prepared/manifest.json"), {
        "format_version": 2, "signature": preparation_signature(cfg),
        "protocol_sha256": cfg["protocolo"]["sha256"], "totals": protocol["totals"],
        "normalizations": {"fixture": {"session": "fixture", "fitted_split": "train", "frames": 36,
                                       "epsilon": epsilon, "mean": mean.tolist(), "std": std.tolist(),
                                       "fit_trial_indices": [0, 1, 2],
                                       "fit_protocol_sha256": cfg["protocolo"]["sha256"]}},
        "files": [{"split": "train", "session": "fixture", "path": "train/fixture.npy",
                   "sha256": file_sha256(Path("prepared/train/fixture.npy")),
                   "source_sha256": "a" * 64, "transcriptions_sha256": "c" * 64,
                   "offsets": [0, 14, 26, 36, 50, 62, 72], "blocks": [1, 1, 1, 2, 2, 2]}],
    })
    Path("config.yaml").write_text(yaml.safe_dump(cfg))
    return cfg


@pytest.mark.parametrize("depth", [None, 2, 3])
def test_training_reload_and_val_cli_in_separate_processes(tmp_path, monkeypatch, depth):
    cfg = _prepared_fixture(tmp_path, monkeypatch)
    if depth is not None:
        cfg["modelo"]["capas"] = depth
        cfg["modelo"].pop("salidas_intermedias")
        cfg["entrenamiento"].pop("ctc_pesos")
        Path("config.yaml").write_text(yaml.safe_dump(cfg))
    expected_exits = (2, 3, 5) if depth is None else (depth,)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}

    def execute(script, *args):
        result = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    execute("run.py", "--config", "config.yaml")
    checkpoint, = Path("checkpoints/fixture").glob("*/mejor_val.pt")
    digest = file_sha256(checkpoint)
    execute("scripts/verify_speech_checkpoint.py", "--config", "config.yaml", "--checkpoint", str(checkpoint),
            "--verification-config", str(ROOT / "configs/verificacion_checkpoint.yaml"))
    evidence = json.loads((checkpoint.parent / "verificacion_recarga.json").read_text())
    assert evidence["verified"] is True and evidence["optimizer_steps"] == 0
    assert evidence["split"] == "val" and evidence["test_evaluated"] is False
    assert set(evidence["metrics_by_exit"]) == {str(value) for value in expected_exits}
    for exit_depth in expected_exits:
        assert evidence["comparisons"][str(exit_depth)]["counts_equal"] is True
        assert evidence["comparisons"][str(exit_depth)]["per_equal"] is True
        assert evidence["comparisons"][str(exit_depth)]["ctc_close"] is True
        assert evidence["example"]["profiles"][str(exit_depth)]["executed_layers"] == list(range(1, exit_depth + 1))
        assert evidence["metrics_by_exit"][str(exit_depth)]["frases"] == 3
    predictions = json.loads((checkpoint.parent / "predicciones_val_ejemplo.json").read_text())
    assert predictions == evidence["example"]
    evaluation = execute("run.py", "--config", "config.yaml", "--solo-evaluar", "--split", "val",
                         "--checkpoint", str(checkpoint))
    assert "=== metricas (validacion) ===" in evaluation.stdout
    assert file_sha256(checkpoint) == digest
    execute("scripts/report_speech_training.py", "--run-dir", str(checkpoint.parent))
    summary_name = "resumen_salidas.json" if depth is None else "resumen_baseline.json"
    summary = json.loads((checkpoint.parent / summary_name).read_text())
    assert summary["available_exits"] == list(expected_exits)
    assert summary["model_layers"] == (5 if depth is None else depth)
    assert summary["seed"] == cfg["experimento"]["seed"]
    if depth is None:
        evaluation_cfg = {**cfg, "experimento": {**cfg["experimento"], "seed": 1}}
        Path("config_eval.yaml").write_text(yaml.safe_dump(evaluation_cfg))
        execute("scripts/evaluate_speech.py", "--config", "config_eval.yaml", "--checkpoint", str(checkpoint),
                "--evaluation-config", str(ROOT / "configs/speech_evaluacion.yaml"), "--output", "diagnostic")
        diagnostic = json.loads(Path("diagnostic/metricas.json").read_text())
        rows = [json.loads(line) for line in Path("diagnostic/predicciones.jsonl").read_text().splitlines()]
        assert len(rows) == 3 and diagnostic["test_evaluated"] is False
        assert diagnostic["optimizer_steps"] == 0 and diagnostic["checkpoint_sha256"] == digest
        assert diagnostic["training_seed"] == cfg["experimento"]["seed"]
        table_path, = Path("tracking").rglob("*.table.json")
        table = json.loads(table_path.read_text())
        assert table["columns"] == ["control", "salida", "sesion", "ctc_loss", "per", "errores", "fonemas", "frases", "completo"]
        assert len(table["data"]) == len(expected_exits) * 3
        for exit_depth in expected_exits:
            measured = diagnostic["global"]["original"][str(exit_depth)]
            previous = evidence["metrics_by_exit"][str(exit_depth)]
            assert measured["per"] == previous["per"] and measured["errores"] == previous["errores"]
            assert measured["ctc_loss"] == pytest.approx(previous["ctc_loss"], rel=1e-6)
            assert diagnostic["by_session"]["original"][str(exit_depth)]["fixture"] == measured
    # No se han creado señales ni etiquetas de test: cualquier acceso fallaría.
    assert not Path("prepared/test").exists() and not Path("labels/test").exists()


def test_raw_benchmark_cli_uses_same_checkpoint_and_only_validation(tmp_path, monkeypatch):
    from scipy.io import savemat

    cfg = _prepared_fixture(tmp_path, monkeypatch, width=256)
    lengths = [14, 12, 10] * 2
    features = [np.linspace(2 + i, 3 + i, length * 256).reshape(length, 256).astype(np.float32)
                for i, length in enumerate(lengths)]
    spikes, crossings = np.empty((6, 1), dtype=object), np.empty((6, 1), dtype=object)
    for i, values in enumerate(features):
        spikes[i, 0] = np.concatenate([values[:, 128:], np.ones_like(values[:, 128:])], axis=1)
        crossings[i, 0] = np.concatenate([values[:, :128], np.ones_like(values[:, :128])], axis=1)
    Path("raw/train").mkdir(parents=True)
    savemat("raw/train/fixture.mat", {"spikePow": spikes, "tx1": crossings,
                                    "sentenceText": np.array(["fixture"] * 6, dtype=object).reshape(6, 1),
                                    "blockIdx": np.array([[1], [1], [1], [2], [2], [2]])})
    protocol = json.loads(Path("split.json").read_text())
    protocol["sessions"][0]["sources"]["train"]["sha256"] = file_sha256(Path("raw/train/fixture.mat"))
    _write_json(Path("split.json"), protocol)
    cfg["protocolo"]["sha256"] = file_sha256(Path("split.json"))
    values = np.concatenate(features).astype(np.float64)
    mean, std = values[:36].mean(0), values[:36].std(0)
    np.save("prepared/train/fixture.npy", ((values - mean) / (std + cfg["preprocesamiento"]["normalizacion_epsilon"])).astype(np.float32))
    manifest = json.loads(Path("prepared/manifest.json").read_text())
    manifest.update(signature=preparation_signature(cfg), protocol_sha256=cfg["protocolo"]["sha256"])
    manifest["normalizations"]["fixture"].update(mean=mean.tolist(), std=std.tolist(), fit_protocol_sha256=cfg["protocolo"]["sha256"])
    manifest["files"][0].update(source_sha256=protocol["sessions"][0]["sources"]["train"]["sha256"],
                                 sha256=file_sha256(Path("prepared/train/fixture.npy")))
    _write_json(Path("prepared/manifest.json"), manifest)
    Path("config.yaml").write_text(yaml.safe_dump(cfg))
    spec = yaml.safe_load((ROOT / "configs/speech_benchmark.yaml").read_text())
    spec.update(dispositivo="cpu", calentamiento=1, repeticiones=2)
    Path("benchmark_spec.yaml").write_text(yaml.safe_dump(spec))
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}

    def execute(script, *args):
        result = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr

    execute("run.py", "--config", "config.yaml")
    checkpoint, = Path("checkpoints/fixture").glob("*/mejor_val.pt")
    digest = file_sha256(checkpoint)
    execute("scripts/benchmark_speech.py", "--config", "config.yaml", "--checkpoint", str(checkpoint),
            "--benchmark-config", "benchmark_spec.yaml", "--split", "val", "--output", "benchmark")
    result = json.loads(Path("benchmark/benchmark.json").read_text())
    assert result["checkpoint_sha256"] == digest == file_sha256(checkpoint)
    assert result["decoded_profiles_match"] is True and result["optimizer_steps"] == 0
    assert result["test_evaluated"] is False and result["training_seed"] == 0
    assert len(result["inputs"]["samples"]) == 2
    assert len(result["measurements"]) == 24
    assert result["model_load_seconds"] > 0
    assert result["inputs"]["raw_data_preparation_seconds"] > 0
    assert all(3 <= row["source_trial_index"] <= 5 for row in result["inputs"]["samples"])
    for depth in (2, 3, 5):
        assert result["profiles"][str(depth)]["executed_layers"] == list(range(1, depth + 1))
    assert Path("benchmark/mediciones.csv").is_file()
    execute("scripts/evaluate_speech.py", "--config", "config.yaml", "--checkpoint", str(checkpoint),
            "--evaluation-config", str(ROOT / "configs/speech_evaluacion.yaml"), "--output", "quality")
    plot_cfg = yaml.safe_load((ROOT / "configs/speech_figuras_referencia.yaml").read_text())
    plot_cfg["titulo"] = "Synthetic fixture; not T12 results"
    plot_cfg["alcance"] = "Unit integration fixture"
    plot_cfg["formatos"] = ["png"]
    plot_cfg["registros"] = [{"etiqueta": "fixture", "evaluacion": "quality/metricas.json",
                              "benchmark": "benchmark/benchmark.json", "historia": str(checkpoint.parent / "resultado.json")}]
    Path("plot_spec.yaml").write_text(yaml.safe_dump(plot_cfg))
    execute("scripts/plot_speech_results.py", "--config", "plot_spec.yaml", "--output", "figures")
    plotted = json.loads(Path("figures/datos_figuras.json").read_text())
    assert {point["exit"] for point in plotted["points"]} == {2, 3, 5}
    assert len(plotted["histories"]) == 3
    assert all(history["split"] == "val" for history in plotted["histories"])
    assert not Path("raw/test").exists() and not Path("prepared/test").exists()


def test_local_demo_reloads_v2_and_widget_executes_only_existing_profiles(tmp_path, monkeypatch):
    from scripts.demo_speech_profiles import load_demo, create_controls

    cfg = _prepared_fixture(tmp_path, monkeypatch, width=256)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--config", "config.yaml"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint, = Path("checkpoints/fixture").glob("*/mejor_val.pt")
    digest = file_sha256(checkpoint)
    demo_cfg = yaml.safe_load((ROOT / "configs/speech_demo_salidas.yaml").read_text())
    demo_cfg["checkpoint"] = str(checkpoint)
    demo_cfg["datos"].update(sesion="fixture", frase=0)
    demo_cfg["ejecucion"]["dispositivo"] = "cpu"
    demo_cfg["medicion"].update(calentamiento=1, repeticiones=2)
    demo_cfg["salida"]["directorio"] = "demo"
    demo = load_demo(demo_cfg)
    assert demo.available_exits == (2, 3, 5)
    assert demo.metadata["split"] == "val" and demo.metadata["source_trial_index"] == 3
    assert demo.metadata["training_updates"] == 0 and demo.metadata["tracking_run_opened"] is False
    assert demo.draw_activity().stat().st_size > 0
    panel = create_controls(demo)
    assert "application/vnd.jupyter.widget-view+json" in panel._repr_mimebundle_()
    selector, button = panel.children[0].children
    for depth in (2, 3, 5):
        selector.value = depth
        button.click()
        predicted = demo.last_prediction
        assert predicted is not None and predicted["exit"] == depth
        assert predicted["executed_layers"] == list(range(1, depth + 1))
        assert predicted["forward_ms_p50"] > 0 and len(predicted["forward_ms_samples"]) == 2
        saved = json.loads((demo.output / f"prediccion_capa_{depth}.json").read_text())
        assert saved == predicted
    assert file_sha256(checkpoint) == digest
    with pytest.raises(ValueError, match="dispone"):
        demo.predict(4)
    with (demo.output / "entrada_modelo.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Cambió"):
        demo.predict(5)
    demo.close()
    with pytest.raises(RuntimeError, match="cerrada"):
        demo.predict(5)
    assert not Path("prepared/test").exists() and not Path("labels/test").exists()
