"""Reúne la tabla de referencia de la cola: PER de val y, si existe, WER con trigrama.

    python scripts/build_reference_table.py --config configs/final/referencia.yaml

Lee el resultado.json de cada entrenamiento terminado y, para los trabajos ya
evaluados con scripts/evaluate_queue.py, sus métricas y su resumen con
intervalos. Comprueba que cada evaluación corresponde al checkpoint
seleccionado. No carga modelos ni decodifica. Escribe un .json con hashes de
procedencia y un .md con las tablas. No interpreta los resultados.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_queue import Evaluation, queue_evaluations, reference_spec
from src.data.preprocessing import normalization_spec
from src.data.speech_dataset import file_sha256

GROUPS = ("global", "vocalizada", "silenciosa")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trigram(evaluation: Evaluation, training: dict[str, Any], split: str) -> dict[str, Any]:
    """Métricas con trigrama de la salida final, verificadas contra el entrenamiento."""
    cfg, output = evaluation.job.config, evaluation.output
    metrics, summary = _read(output / "metricas.json"), _read(output / "resumen_metricas.json")
    depth = str(cfg["modelo"]["capas"])
    predictions = file_sha256(output / "predicciones.jsonl")
    if (metrics["split"] != split or summary["split"] != split
            or metrics["checkpoint_step"] != training["best_val_step"]
            or metrics["training_seed"] != evaluation.job.seed
            or metrics["protocol_sha256"] != cfg["protocolo"]["sha256"]
            or metrics["checkpoint_sha256"] != file_sha256(evaluation.checkpoint)
            or summary["predictions_sha256"] != predictions
            or summary["trials"] != metrics["global"][depth]["frases"]):
        raise ValueError(f"La evaluación de {evaluation.job.name} no corresponde a su checkpoint seleccionado.")
    counts = metrics["global"][depth]
    return {"wandb_run_id": metrics["wandb_run_id"], "checkpoint_sha256": metrics["checkpoint_sha256"],
            "predicciones_sha256": predictions, "metricas_sha256": file_sha256(output / "metricas.json"),
            "resumen_sha256": file_sha256(output / "resumen_metricas.json"),
            "busqueda": metrics["language_config"]["busqueda"], "rtf": metrics["timing"][depth]["rtf"],
            "wer": counts["wer"], "per": counts["per"], "frases": counts["frases"],
            "errores_palabras": counts["errores_palabras"], "palabras": counts["palabras"],
            "intervalos": {group: {metric: {key: values[key] for key in ("rate", "low", "high", "trials")}
                                   for metric, values in summary["by_exit"][depth][group].items()}
                           for group in GROUPS if group in summary["by_exit"][depth]},
            "bootstrap": summary["config"]["bootstrap"]}


def job_row(evaluation: Evaluation, split: str) -> dict[str, Any] | None:
    """Fila de un trabajo terminado; None si su entrenamiento no está completo."""
    if evaluation.result is None:
        return None
    cfg, training = evaluation.job.config, _read(evaluation.result)
    best = training["best_val"]
    if training["test_evaluated"] is not False or best["completo"] != 1:
        raise ValueError(f"{evaluation.job.name} no conserva una selección por val completa.")
    return {"trabajo": evaluation.job.name, "variante": evaluation.job.variant, "seed": evaluation.job.seed,
            "normalizacion": normalization_spec(cfg)["modo"],
            "suavizado": cfg["preprocesamiento"]["suavizado"].get("modo", "centered"),
            "unidades": cfg["modelo"]["unidades"], "kernel": cfg["modelo"]["kernel"],
            "run_id": training["run_id"], "resultado_sha256": file_sha256(evaluation.result),
            "paso_seleccionado": training["best_val_step"], "pasos": training["completed_steps"],
            "per_val": best["per"], "errores_fonemas": best["errores"], "fonemas": best["fonemas"],
            "frases": best["frases"], "ctc_val": best["ctc_loss"],
            "segundos_entrenamiento": training["training_seconds"], "hardware": training["hardware"],
            "trigrama": _trigram(evaluation, training, split) if evaluation.state == "evaluado" else None}


def _spread(values: list[float]) -> dict[str, float | int | None] | None:
    """Media entre semillas con desviación muestral; sin desviación con una sola."""
    if not values:
        return None
    return {"n": len(values), "media": statistics.fmean(values),
            "desviacion": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrega por variante en el orden del plan; cada semilla pesa lo mismo."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variante"], []).append(row)
    table = []
    for variant, members in grouped.items():
        decoded = [row["trigrama"] for row in members if row["trigrama"] is not None]
        table.append({"variante": variant, "normalizacion": members[0]["normalizacion"],
                      "suavizado": members[0]["suavizado"], "unidades": members[0]["unidades"],
                      "kernel": members[0]["kernel"], "semillas": [row["seed"] for row in members],
                      "semillas_con_trigrama": [row["seed"] for row in members if row["trigrama"] is not None],
                      "per_val": _spread([row["per_val"] for row in members]),
                      "wer": _spread([entry["wer"] for entry in decoded]),
                      "per_bloques": _spread([entry["per"] for entry in decoded])})
    return table


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}".replace(".", ",")


def _cell(spread: dict[str, Any] | None) -> str:
    if spread is None:
        return "—"
    return f"{_percent(spread['media'])} ± {_percent(spread['desviacion'])} ({_percent(spread['min'])}–{_percent(spread['max'])})"


def _interval(entry: dict[str, Any] | None, group: str, metric: str) -> str:
    values = None if entry is None else entry["intervalos"].get(group, {}).get(metric)
    return "—" if values is None else f"{_percent(values['rate'])} [{_percent(values['low'])}; {_percent(values['high'])}]"


def render_markdown(table: dict[str, Any]) -> str:
    """Dos tablas: una fila por variante y una fila por trabajo. Porcentajes."""
    lines = ["# Tabla de referencia T12, partición " + table["protocolo"], "",
             f"Split: {table['split']}. Generada por `scripts/build_reference_table.py`; no editar a mano.",
             "PER de val: evaluación completa del entrenamiento. WER y PER por bloques: evaluación",
             "causal con el trigrama. Media ± desviación muestral entre semillas (mínimo–máximo).", "",
             "| Variante | Modelo | Normalización | Semillas | PER val (%) | WER (%) | Semillas con WER |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in table["variantes"]:
        lines.append(f"| {row['variante']} | {row['unidades']}/{row['kernel']} {row['suavizado']} | "
                     f"{row['normalizacion']} | {len(row['semillas'])} | {_cell(row['per_val'])} | "
                     f"{_cell(row['wer'])} | {len(row['semillas_con_trigrama'])} |")
    heading = "Por trabajo."
    if table["confianza"]:
        heading += f" Intervalos al {table['confianza']}, bootstrap sobre frases."
    lines += ["", heading, "",
              "| Trabajo | Run | Paso | PER val (%) | WER (%) | WER vocalizada (%) | WER silenciosa (%) | PER por bloques (%) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table["trabajos"]:
        entry = row["trigrama"]
        lines.append(f"| {row['trabajo']} | {row['run_id']} | {row['paso_seleccionado']} | {_percent(row['per_val'])} | "
                     f"{_interval(entry, 'global', 'wer')} | {_interval(entry, 'vocalizada', 'wer')} | "
                     f"{_interval(entry, 'silenciosa', 'wer')} | {_interval(entry, 'global', 'per')} |")
    pending = table["sin_entrenar"]
    if pending:
        lines += ["", "Trabajos del plan sin entrenamiento completo: " + ", ".join(pending) + "."]
    return "\n".join(lines) + "\n"


def build_table(spec: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Construye la tabla completa; falla si no hay ningún trabajo terminado."""
    evaluations = queue_evaluations(spec)
    rows = [row for evaluation in evaluations if (row := job_row(evaluation, spec["split"])) is not None]
    if not rows:
        raise ValueError("Ningún trabajo del plan tiene un entrenamiento completo.")
    protocols = {evaluation.job.config["protocolo"]["nombre"] for evaluation in evaluations}
    confidences = {row["trigrama"]["bootstrap"]["confianza"] for row in rows if row["trigrama"] is not None}
    searches = [row["trigrama"]["busqueda"] for row in rows if row["trigrama"] is not None]
    if len(protocols) != 1 or len(confidences) > 1 or any(search != searches[0] for search in searches):
        raise ValueError("Los trabajos mezclan protocolos, confianzas o parámetros del trigrama.")
    return {"scope": "cifras observadas en val; sin interpretación estadística ni uso de test",
            "split": spec["split"], "protocolo": protocols.pop(),
            "confianza": f"{100 * confidences.pop():g} %" if confidences else None,
            "busqueda_trigrama": searches[0] if searches else None,
            "sources_sha256": {str(config_path): file_sha256(config_path), spec["plan"]: file_sha256(Path(spec["plan"]))},
            "sin_entrenar": [evaluation.job.name for evaluation in evaluations if evaluation.result is None],
            "variantes": variant_rows(rows), "trabajos": rows}


def main() -> int:
    """Escribe TABLA.json y TABLA.md en el prefijo que declara el perfil."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/referencia.yaml"))
    args = parser.parse_args()
    try:
        spec = reference_spec(args.config)
        table = build_table(spec, args.config)
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    prefix = Path(spec["tabla"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown = render_markdown(table)
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Tabla: {prefix.with_suffix('.json')} y {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
