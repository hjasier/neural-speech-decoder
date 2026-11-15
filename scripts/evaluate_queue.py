"""Evalúa con el trigrama los checkpoints causales que la cola ya ha terminado.

    python scripts/evaluate_queue.py --config configs/final/referencia.yaml --dry-run
    python scripts/evaluate_queue.py --config configs/final/referencia.yaml

Por cada trabajo completo del plan llama a scripts/evaluate_speech_trigram.py
sobre su mejor_val.pt y después a scripts/summarize_speech_predictions.py. No
entrena. Los trabajos ya evaluados se saltan, así que se puede relanzar tras un
corte. Los modelos con suavizado centrado no pasan por el trigrama: el motor por
bloques solo admite checkpoints causales. Necesita el runtime y el grafo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_queue import Job, job_state, queue_jobs

SPEC_KEYS = {"plan", "split", "evaluacion", "lenguaje", "resumen", "salida", "tabla"}


@dataclass(frozen=True)
class Evaluation:
    """Un trabajo de la cola con su estado de evaluación y sus rutas."""

    job: Job
    state: str
    result: Path | None
    output: Path

    @property
    def checkpoint(self) -> Path:
        """Pesos seleccionados por PER de val, junto al resultado del entrenamiento."""
        if self.result is None:
            raise ValueError(f"{self.job.name} no tiene un entrenamiento completo.")
        return self.result.parent / "mejor_val.pt"


def reference_spec(path: Path) -> dict[str, Any]:
    """Lee el perfil de la tabla de referencia; solo admite val."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError(f"El perfil debe declarar exactamente: {sorted(SPEC_KEYS)}.")
    if spec["split"] != "val":
        raise ValueError("La tabla de referencia solo usa val; test se reserva para la evaluación final.")
    return spec


def queue_evaluations(spec: dict[str, Any]) -> list[Evaluation]:
    """Clasifica los trabajos: sin_entrenar, no_causal, evaluado, incompleto o pendiente."""
    evaluations = []
    for job in queue_jobs(Path(spec["plan"])):
        output = Path(spec["salida"]) / job.name
        training, result = job_state(job.config)
        if training != "completo":
            state, result = "sin_entrenar", None
        elif job.config["preprocesamiento"]["suavizado"].get("modo", "centered") != "causal":
            state = "no_causal"
        elif (output / "metricas.json").is_file() and (output / "resumen_metricas.json").is_file():
            state = "evaluado"
        elif output.exists():
            state = "incompleto"
        else:
            state = "pendiente"
        evaluations.append(Evaluation(job, state, result, output))
    return evaluations


def _set_aside(output: Path) -> Path:
    """Aparta una evaluación a medias sin borrarla; el evaluador exige carpeta nueva."""
    index = 1
    while (target := output.with_name(f"{output.name}_incompleto_{index}")).exists():
        index += 1
    output.rename(target)
    return target


def run_evaluation(evaluation: Evaluation, spec: dict[str, Any]) -> int:
    """Lanza evaluación y resumen de un trabajo; devuelve el primer código distinto de cero."""
    name, output = evaluation.job.name, evaluation.output
    if evaluation.state in ("sin_entrenar", "no_causal", "evaluado"):
        print(f"[evaluación] {name}: {evaluation.state}; se salta.", flush=True)
        return 0
    commands = []
    if evaluation.state == "incompleto" and (output / "metricas.json").is_file():
        print(f"[evaluación] {name}: falta solo el resumen.", flush=True)
    else:
        if evaluation.state == "incompleto":
            print(f"[evaluación] {name}: evaluación a medias apartada en {_set_aside(output)}.", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        commands.append([sys.executable, "scripts/evaluate_speech_trigram.py",
                         "--checkpoint", str(evaluation.checkpoint),
                         "--evaluation-config", spec["evaluacion"], "--language-config", spec["lenguaje"],
                         "--split", spec["split"], "--output", str(output)])
    commands.append([sys.executable, "scripts/summarize_speech_predictions.py",
                     "--predictions", str(output / "predicciones.jsonl"), "--config", spec["resumen"]])
    for command in commands:
        print(f"[evaluación] {name}: {' '.join(command)}", flush=True)
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        if code != 0:
            return code
    return 0


def main() -> int:
    """Recorre la cola; se detiene en el primer fallo salvo que se pida continuar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/referencia.yaml"))
    parser.add_argument("--solo", action="append", help="variante a evaluar; se puede repetir")
    parser.add_argument("--dry-run", action="store_true", help="muestra el estado de cada trabajo sin lanzar nada")
    parser.add_argument("--continuar-si-falla", action="store_true")
    args = parser.parse_args()
    try:
        spec = reference_spec(args.config)
        evaluations = queue_evaluations(spec)
        if args.solo:
            unknown = set(args.solo) - {evaluation.job.variant for evaluation in evaluations}
            if unknown:
                raise ValueError(f"Variantes desconocidas: {sorted(unknown)}")
            evaluations = [evaluation for evaluation in evaluations if evaluation.job.variant in args.solo]
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        for evaluation in evaluations:
            print(f"{evaluation.job.name}: {evaluation.state} ({evaluation.output})")
        return 0
    failed = []
    for evaluation in evaluations:
        code = run_evaluation(evaluation, spec)
        if code != 0:
            failed.append(evaluation.job.name)
            print(f"[evaluación] {evaluation.job.name}: terminó con código {code}.", file=sys.stderr, flush=True)
            if not args.continuar_si_falla:
                break
    print(f"[evaluación] trabajos con fallo: {failed or 'ninguno'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
