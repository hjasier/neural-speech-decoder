"""Ejecuta en serie las variantes × semillas de un plan, reanudando lo interrumpido.

    python scripts/run_queue.py --plan configs/final/cola_replica.yaml --dry-run
    python scripts/run_queue.py --plan configs/final/cola_replica.yaml

Cada trabajo llama a run.py con una configuración derivada del perfil base:
solo cambian la semilla, el nombre, el directorio de salida y las claves que el
plan declara en `cambios`. Los trabajos terminados se saltan y los que dejaron
un entrenamiento.pt se retoman con --reanudar, así que la cola se puede relanzar
tras un corte. Si falta la carpeta de señales preparadas de un trabajo, se
genera antes con scripts/prepare_speech_data.py; las etiquetas de fonemas deben
existir ya. Semillas e hiperparámetros viven en configs/, no aquí.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    """Una combinación de variante y semilla con su configuración completa."""

    name: str
    variant: str
    seed: int
    config: dict[str, Any]


def _replace(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Sustituye entera la clave `a.b.c`; no mezcla diccionarios."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise ValueError(f"La clave {dotted} no existe en el perfil base.")
        node = node[key]
    node[leaf] = copy.deepcopy(value)


def queue_jobs(plan_path: Path) -> list[Job]:
    """Valida el plan y devuelve los trabajos en orden: variante, luego semilla."""
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    jobs = []
    for variant, spec in plan["variantes"].items():
        if isinstance(spec, str):
            spec = {"config": spec}
        if not isinstance(spec, dict) or set(spec) - {"config", "cambios", "semillas"} or "config" not in spec:
            raise ValueError(f"La variante {variant} admite config, cambios y semillas.")
        seeds = spec.get("semillas", plan["semillas"])
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
            raise ValueError(f"La variante {variant} requiere semillas enteras únicas.")
        base = yaml.safe_load(Path(spec["config"]).read_text(encoding="utf-8"))
        for dotted, value in spec.get("cambios", {}).items():
            if dotted.split(".")[0] in ("experimento", "salida", "cola"):
                raise ValueError("La cola fija semilla, nombre y salida; no se declaran en cambios.")
            _replace(base, dotted, value)
        if (not base.get("protocolo") or base["protocolo"]["seleccion_checkpoint"] != "per"
                or base["evaluacion"]["max_lotes"] is not None):
            raise ValueError(f"La variante {variant} requiere protocolo y validación completa.")
        for seed in seeds:
            cfg = copy.deepcopy(base)
            name = f"{variant}_seed{seed}"
            cfg["experimento"].update(seed=seed, nombre=name)
            cfg["salida"]["directorio"] = str(Path(base["salida"]["directorio"]) / variant / f"seed{seed}")
            # Sin hashes: editar el plan para añadir variantes no debe impedir reanudar.
            cfg["cola"] = {"variante": variant, "config_base": spec["config"]}
            jobs.append(Job(name, variant, seed, cfg))
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("El plan produce trabajos con el mismo nombre.")
    return jobs


def job_state(cfg: dict[str, Any]) -> tuple[str, Path | None]:
    """Devuelve completo, reanudar (con su estado más reciente) o nuevo."""
    output = Path(cfg["salida"]["directorio"])
    runs = sorted(path for path in output.glob("*") if path.is_dir()) if output.is_dir() else []
    for run in runs:
        result = run / "resultado.json"
        if result.is_file():
            state = json.loads(result.read_text(encoding="utf-8"))
            if state["completed_steps"] == cfg["entrenamiento"]["pasos"] and not state["stopped_for_time"]:
                return "completo", result
    states = [run / "entrenamiento.pt" for run in runs if (run / "entrenamiento.pt").is_file()]
    if states:
        if not cfg.get("reanudacion"):
            raise ValueError(f"Hay un estado en {output}, pero el perfil no declara reanudacion.")
        return "reanudar", max(states, key=lambda path: path.stat().st_mtime_ns)
    return "nuevo", None


def run_job(job: Job, workspace: Path) -> int:
    """Escribe la configuración derivada y lanza run.py; devuelve su código de salida."""
    state, path = job_state(job.config)
    if state == "completo":
        print(f"[cola] {job.name}: completo ({path}); se salta.", flush=True)
        return 0
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / f"{job.name}.yaml"
    temporary = config_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(job.config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(config_path)
    # Cada modo de normalización tiene su caché; se prepara una vez si falta.
    if not (Path(job.config["datos"]["preparados"]) / "manifest.json").is_file():
        prepare = [sys.executable, "scripts/prepare_speech_data.py", "--config", str(config_path)]
        print(f"[cola] {job.name}: preparando señales. {' '.join(prepare)}", flush=True)
        code = subprocess.run(prepare, cwd=ROOT, check=False).returncode
        if code != 0:
            return code
    command = [sys.executable, "run.py", "--config", str(config_path)]
    if state == "reanudar":
        command += ["--reanudar", str(path)]
    print(f"[cola] {job.name}: {state}. {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    """Recorre la cola; se detiene en el primer fallo salvo que se pida continuar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--solo", action="append", help="variante a ejecutar; se puede repetir")
    parser.add_argument("--dry-run", action="store_true", help="muestra el estado de cada trabajo sin lanzar nada")
    parser.add_argument("--continuar-si-falla", action="store_true")
    parser.add_argument("--workspace", type=Path, help="carpeta de configuraciones derivadas; por defecto runs/cola/PLAN")
    args = parser.parse_args()
    try:
        jobs = queue_jobs(args.plan)
        if args.solo:
            unknown = set(args.solo) - {job.variant for job in jobs}
            if unknown:
                raise ValueError(f"Variantes desconocidas: {sorted(unknown)}")
            jobs = [job for job in jobs if job.variant in args.solo]
        if args.dry_run:
            for job in jobs:
                state, path = job_state(job.config)
                print(f"{job.name}: {state}{f' ({path})' if path else ''}")
            return 0
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    workspace = args.workspace or Path("runs/cola") / args.plan.stem
    failed = []
    for job in jobs:
        code = run_job(job, workspace)
        if code != 0:
            failed.append(job.name)
            print(f"[cola] {job.name}: run.py terminó con código {code}.", file=sys.stderr, flush=True)
            if not args.continuar_si_falla:
                break
    print(f"[cola] trabajos con fallo: {failed or 'ninguno'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
