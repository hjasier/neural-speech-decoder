"""La tabla de referencia solo reúne lo que corresponde a cada checkpoint seleccionado."""

import json
from pathlib import Path

import pytest
import yaml

from scripts import evaluate_queue
from scripts.build_reference_table import build_table, render_markdown
from scripts.evaluate_queue import queue_evaluations, reference_spec, run_evaluation
from src.data.speech_dataset import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Plan pequeño en una carpeta vacía: dos semillas causales y una offline."""
    monkeypatch.chdir(tmp_path)
    Path("plan.yaml").write_text(yaml.safe_dump({"semillas": [0, 1], "variantes": {
        "causal": {"config": str(ROOT / "configs/final/t12_causal.yaml")},
        "offline": {"config": str(ROOT / "configs/final/t12_offline.yaml"), "semillas": [0]}}}))
    profile = yaml.safe_load((ROOT / "configs/final/referencia.yaml").read_text())
    profile.update(plan="plan.yaml", salida="evaluaciones", tabla="tabla/referencia")
    Path("referencia.yaml").write_text(yaml.safe_dump(profile))
    return reference_spec(Path("referencia.yaml"))


def _train(evaluation: evaluate_queue.Evaluation, per: float, step: int = 9801) -> Path:
    """Deja el resultado y los pesos que escribiría un entrenamiento terminado."""
    cfg = evaluation.job.config
    run = Path(cfg["salida"]["directorio"]) / f"run{evaluation.job.seed}"
    run.mkdir(parents=True)
    (run / "mejor_val.pt").write_bytes(f"pesos {evaluation.job.name}".encode())
    (run / "resultado.json").write_text(json.dumps({
        "completed_steps": cfg["entrenamiento"]["pasos"], "stopped_for_time": False, "test_evaluated": False,
        "best_val_step": step, "run_id": run.name, "training_seconds": 340.0, "hardware": "CPU",
        "best_val": {"per": per, "errores": int(per * 1000), "fonemas": 1000, "frases": 20,
                     "ctc_loss": 25.0, "completo": 1}}))
    return run


def _decode(evaluation: evaluate_queue.Evaluation, wer: float, **changes: object) -> None:
    """Deja las tres salidas de una evaluación con trigrama ya resumida."""
    cfg, output = evaluation.job.config, evaluation.output
    output.mkdir(parents=True)
    (output / "predicciones.jsonl").write_text(f"{evaluation.job.name}\n")
    counts = {"wer": wer, "per": 0.25, "frases": 20, "errores_palabras": int(wer * 200), "palabras": 200}
    (output / "metricas.json").write_text(json.dumps({
        "split": "val", "checkpoint_step": 9801, "training_seed": evaluation.job.seed,
        "protocol_sha256": cfg["protocolo"]["sha256"], "checkpoint_sha256": file_sha256(evaluation.checkpoint),
        "wandb_run_id": "eval", "language_config": {"busqueda": {"beam": 18.0}},
        "global": {"5": counts}, "timing": {"5": {"rtf": 0.5}}, **changes}))
    interval = {"rate": wer, "low": wer - 0.02, "high": wer + 0.02, "trials": 20}
    (output / "resumen_metricas.json").write_text(json.dumps({
        "split": "val", "trials": 20, "predictions_sha256": file_sha256(output / "predicciones.jsonl"),
        "config": {"bootstrap": {"confianza": 0.95}},
        "by_exit": {"5": {"global": {"per": {**interval, "rate": 0.25}, "wer": interval}}}}))


# Evaluar test aquí gastaría la única evaluación final reservada para la Fase 5.
def test_reference_profile_only_accepts_validation(spec: dict) -> None:
    Path("referencia.yaml").write_text(yaml.safe_dump({**spec, "split": "test"}))
    with pytest.raises(ValueError, match="solo usa val"):
        reference_spec(Path("referencia.yaml"))


# El motor por bloques rechaza el suavizado centrado: el offline no debe llegar al trigrama.
def test_only_trained_causal_jobs_are_evaluated_and_partial_runs_are_kept(
    spec: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, offline = queue_evaluations(spec)
    assert {first.state, second.state, offline.state} == {"sin_entrenar"}
    _train(first, 0.24)
    _train(offline, 0.20)
    first, second, offline = queue_evaluations(spec)
    assert (first.state, second.state, offline.state) == ("pendiente", "sin_entrenar", "no_causal")

    commands = []
    monkeypatch.setattr(evaluate_queue.subprocess, "run",
                        lambda command, **_: commands.append(command) or type("R", (), {"returncode": 0})())
    assert all(run_evaluation(item, spec) == 0 for item in (second, offline)) and not commands
    assert run_evaluation(first, spec) == 0
    assert [command[1] for command in commands] == ["scripts/evaluate_speech_trigram.py",
                                                    "scripts/summarize_speech_predictions.py"]
    assert commands[0][commands[0].index("--checkpoint") + 1] == str(first.checkpoint)
    assert commands[0][commands[0].index("--split") + 1] == "val"

    # Un corte deja la carpeta sin métricas: se aparta, no se borra ni se da por buena.
    first.output.mkdir(parents=True)
    (first.output / "predicciones.jsonl.tmp").write_text("a medias")
    first = queue_evaluations(spec)[0]
    assert first.state == "incompleto"
    commands.clear()
    run_evaluation(first, spec)
    assert len(commands) == 2 and not first.output.exists()
    assert (first.output.with_name("causal_seed0_incompleto_1") / "predicciones.jsonl.tmp").is_file()

    _decode(first, 0.30)
    first = queue_evaluations(spec)[0]
    commands.clear()
    assert first.state == "evaluado" and run_evaluation(first, spec) == 0 and not commands


# Una media entre semillas con WER de otro checkpoint daría una tabla incoherente sin avisar.
def test_table_aggregates_seeds_and_rejects_foreign_evaluations(spec: dict) -> None:
    with pytest.raises(ValueError, match="Ningún trabajo"):
        build_table(spec, Path("referencia.yaml"))
    first, second, offline = queue_evaluations(spec)
    for evaluation, per in ((first, 0.24), (second, 0.26), (offline, 0.20)):
        _train(evaluation, per)
    first, second, offline = queue_evaluations(spec)
    _decode(first, 0.30)
    table = build_table(spec, Path("referencia.yaml"))
    causal, centered = table["variantes"]
    assert causal["per_val"]["media"] == pytest.approx(0.25)
    assert causal["per_val"]["desviacion"] == pytest.approx(0.02 ** 0.5 / 10)
    assert causal["semillas_con_trigrama"] == [0] and causal["wer"]["desviacion"] is None
    assert centered["suavizado"] == "centered" and centered["wer"] is None
    assert [row["trigrama"] is None for row in table["trabajos"]] == [False, True, True]
    assert table["confianza"] == "95 %" and table["sin_entrenar"] == []
    markdown = render_markdown(table)
    assert "| causal | 512/14 causal | rodante | 2 | 25,00 ± 1,41 (24,00–26,00) | 30,00 ± — (30,00–30,00) | 1 |" in markdown
    assert "30,00 [28,00; 32,00]" in markdown

    _decode(second, 0.32, checkpoint_step=9751)
    with pytest.raises(ValueError, match="no corresponde a su checkpoint"):
        build_table(spec, Path("referencia.yaml"))
