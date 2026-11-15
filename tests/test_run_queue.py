"""La cola deriva configuraciones sin tocar hiperparámetros y no repite trabajo hecho."""

import json
from pathlib import Path

import pytest
import yaml

from scripts import run_queue
from scripts.run_queue import job_state, queue_jobs, run_job
from src.data.preprocessing import normalization_spec
from src.training.speech_resume import validate_resume_config

PLAN = Path("configs/final/cola_replica.yaml")


# Un cambio accidental de arquitectura o presupuesto entre semillas invalidaría la comparación.
def test_jobs_only_change_seed_name_output_and_declared_keys() -> None:
    jobs = queue_jobs(PLAN)
    assert len(jobs) == 5 + 5 + 3 + 3 and len({job.config["salida"]["directorio"] for job in jobs}) == len(jobs)
    base = yaml.safe_load(Path("configs/final/t12_causal.yaml").read_text())
    for job in jobs:
        validate_resume_config(job.config)
        if job.variant != "t12_causal":
            continue
        derived = {key: value for key, value in job.config.items() if key not in ("experimento", "salida", "cola")}
        assert derived == {key: value for key, value in base.items() if key not in ("experimento", "salida")}
        assert job.config["experimento"] == {"nombre": f"t12_causal_seed{job.seed}", "seed": job.seed}
    fixed = next(job for job in jobs if job.variant == "t12_causal_train_fija")
    assert normalization_spec(fixed.config) == {"modo": "train_fija"}
    assert fixed.config["datos"]["preparados"].endswith("train_fija")
    assert fixed.config["modelo"] == base["modelo"] and fixed.config["entrenamiento"] == base["entrenamiento"]


# Las variantes de normalización no pueden compartir caché: la señal preparada es distinta.
def test_each_normalization_mode_uses_its_own_prepared_folder() -> None:
    folders = {}
    for job in queue_jobs(PLAN):
        mode = normalization_spec(job.config)["modo"]
        assert folders.setdefault(job.config["datos"]["preparados"], mode) == mode
    assert len(folders) == 3


def _plan(tmp_path: Path, **variant: object) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump({"semillas": [0], "variantes": {
        "prueba": {"config": "configs/final/t12_causal.yaml", **variant}}}))
    return path


# La semilla y la salida las fija la cola; declararlas en el plan produciría runs que se pisan.
def test_plan_rejects_reserved_or_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="semilla, nombre y salida"):
        queue_jobs(_plan(tmp_path, cambios={"experimento.seed": 3}))
    with pytest.raises(ValueError, match="no existe"):
        queue_jobs(_plan(tmp_path, cambios={"modelos.capas": 2}))
    with pytest.raises(ValueError, match="semillas enteras únicas"):
        queue_jobs(_plan(tmp_path, semillas=[1, 1]))


# Relanzar la cola tras un corte debe retomar el estado guardado y saltar lo terminado.
def test_queue_resumes_interrupted_jobs_and_skips_completed_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = queue_jobs(_plan(tmp_path))[0]
    job.config["salida"]["directorio"] = str(tmp_path / "salida")
    job.config["datos"]["preparados"] = str(tmp_path / "preparados")
    commands = []
    monkeypatch.setattr(run_queue.subprocess, "run",
                        lambda command, **_: commands.append(command) or type("R", (), {"returncode": 0})())
    assert job_state(job.config) == ("nuevo", None)
    assert run_job(job, tmp_path / "cola") == 0 and "--reanudar" not in commands[-1]
    written = yaml.safe_load((tmp_path / "cola" / "prueba_seed0.yaml").read_text())
    assert written == job.config
    # Sin caché de señales, primero se prepara con la misma configuración derivada.
    assert [command[1] for command in commands] == ["scripts/prepare_speech_data.py", "run.py"]
    assert commands[0][-1] == commands[1][-1] == str(tmp_path / "cola" / "prueba_seed0.yaml")
    (tmp_path / "preparados").mkdir()
    (tmp_path / "preparados" / "manifest.json").write_text("{}")
    commands.clear()
    run_job(job, tmp_path / "cola")
    assert [command[1] for command in commands] == ["run.py"]

    run = tmp_path / "salida" / "abc123"
    run.mkdir(parents=True)
    (run / "entrenamiento.pt").write_bytes(b"estado")
    (run / "resultado.json").write_text(json.dumps({"completed_steps": 500, "stopped_for_time": False}))
    assert job_state(job.config) == ("reanudar", run / "entrenamiento.pt")
    run_job(job, tmp_path / "cola")
    assert commands[-1][-2:] == ["--reanudar", str(run / "entrenamiento.pt")]

    (run / "resultado.json").write_text(json.dumps({
        "completed_steps": job.config["entrenamiento"]["pasos"], "stopped_for_time": False}))
    assert job_state(job.config)[0] == "completo"
    commands.clear()
    assert run_job(job, tmp_path / "cola") == 0 and not commands
