"""Figuras trazables: unión por pesos y rechazo de comparaciones incompatibles."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.plot_speech_results import collect_plot_data, render_figures


def _fixture(tmp_path):
    cfg = yaml.safe_load(Path("configs/speech_figuras_referencia.yaml").read_text())
    entry = cfg["registros"][0]
    evaluation = json.loads(Path(entry["evaluacion"]).read_text())["evaluation"]
    benchmark = json.loads(Path(entry["benchmark"]).read_text())["benchmark"]
    return cfg, evaluation, benchmark


def _record(tmp_path, cfg, evaluation, benchmark, name="first", *, seed=None):
    evaluation, benchmark = copy.deepcopy(evaluation), copy.deepcopy(benchmark)
    if seed is not None:
        evaluation["training_seed"] = benchmark["training_seed"] = seed
        evaluation["checkpoint_sha256"] = benchmark["checkpoint_sha256"] = f"{seed:064x}"
    ev, bench = tmp_path / f"{name}_eval.json", tmp_path / f"{name}_bench.json"
    ev.write_text(json.dumps(evaluation))
    bench.write_text(json.dumps(benchmark))
    return {"etiqueta": "GRU original", "evaluacion": str(ev), "benchmark": str(bench)}


def test_reference_figures_export_measured_values_and_rounded_historical_curve(tmp_path):
    cfg, _, _ = _fixture(tmp_path)
    data = collect_plot_data(cfg)
    point, = data["points"]
    assert point["per_by_control"]["original"] == 6129 / 19024
    assert point["latency_ms_p50"] == pytest.approx(28.250568499061046)
    assert len(data["sessions"]) == 3 * 19
    assert len(data["histories"][0]["rows"]) == 9
    assert "redondeada" in data["histories"][0]["split"]
    files = render_figures(cfg, data, tmp_path / "figures")
    assert len(files) == 8
    assert all((tmp_path / "figures" / name).stat().st_size > 0 for name in files)
    assert (tmp_path / "figures/comparacion.csv").is_file()
    assert b"\r" not in (tmp_path / "figures/comparacion.csv").read_bytes()


@pytest.mark.parametrize("field", ["checkpoint_sha256", "checkpoint_step", "split", "training_seed", "data_signature"])
def test_quality_and_latency_from_different_weights_or_splits_are_rejected(tmp_path, field):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    evaluation[field] = 999 if field in ("checkpoint_step", "training_seed") else "different"
    cfg["registros"] = [_record(tmp_path, cfg, evaluation, benchmark)]
    with pytest.raises(ValueError, match="mismo checkpoint"):
        collect_plot_data(cfg)


@pytest.mark.parametrize("field", ["runtime", "input_signature", "spec", "protocol"])
def test_different_measurement_conditions_cannot_be_pooled(tmp_path, field):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    cfg["registros"] = [_record(tmp_path, cfg, evaluation, benchmark)]
    if field == "runtime":
        benchmark["runtime"]["gpu_uuid"] = "different-device"
    elif field == "input_signature":
        benchmark["inputs"]["input_signature"] = "different-inputs"
    elif field == "spec":
        benchmark["spec"]["repeticiones"] += 1
    else:
        evaluation["protocol_sha256"] = "different-protocol"
    cfg["registros"].append(_record(tmp_path, cfg, evaluation, benchmark, "second", seed=1))
    with pytest.raises(ValueError, match="mezclar"):
        collect_plot_data(cfg)


def test_multiple_training_seeds_remain_individual_points_and_duplicates_fail(tmp_path):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    first = _record(tmp_path, cfg, evaluation, benchmark)
    cfg["registros"] = [first, _record(tmp_path, cfg, evaluation, benchmark, "second", seed=1)]
    data = collect_plot_data(cfg)
    assert [point["seed"] for point in data["points"]] == [0, 1]
    assert len(data["points"]) == 2
    cfg["registros"].append(first)
    with pytest.raises(ValueError, match="duplicado"):
        collect_plot_data(cfg)


def test_inconsistent_or_partial_quality_is_rejected(tmp_path):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    evaluation["global"]["original"]["5"]["completo"] = 0
    cfg["registros"] = [_record(tmp_path, cfg, evaluation, benchmark)]
    with pytest.raises(ValueError, match="completa"):
        collect_plot_data(cfg)


@pytest.mark.parametrize("field", ["observations", "p95_ms"])
def test_inconsistent_benchmark_summary_is_rejected(tmp_path, field):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    benchmark["profiles"]["5"]["raw_to_phones"][field] = 1
    cfg["registros"] = [_record(tmp_path, cfg, evaluation, benchmark)]
    with pytest.raises(ValueError, match="observaciones"):
        collect_plot_data(cfg)


@pytest.mark.parametrize("matches", [True, False])
def test_single_exit_validation_history_binds_seed_and_selected_metrics(tmp_path, matches):
    cfg, evaluation, benchmark = _fixture(tmp_path)
    # Metadatos controlados para probar la rama de historia v2, sin presentarlos
    # como una evaluación nueva del checkpoint histórico usado como plantilla.
    evaluation["split"] = benchmark["split"] = "val"
    evaluation["protocol_sha256"] = "fixture_protocol"
    metrics = copy.deepcopy(evaluation["global"]["original"]["5"])
    history = {"best_val_step": evaluation["checkpoint_step"],
               "config": {"experimento": {"seed": 0}, "protocolo": {"sha256": "fixture_protocol"}},
               "best_val": metrics, "validation_history": [{"step": evaluation["checkpoint_step"], **metrics}]}
    if not matches:
        metrics["ctc_loss"] += 1
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history))
    entry = _record(tmp_path, cfg, evaluation, benchmark)
    entry["historia"] = str(path)
    cfg["registros"] = [entry]
    if matches:
        data = collect_plot_data(cfg)
        assert "semilla 0" in data["histories"][0]["label"]
    else:
        with pytest.raises(ValueError, match="reevaluación"):
            collect_plot_data(cfg)
