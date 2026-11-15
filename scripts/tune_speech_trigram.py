"""Ajusta en val la búsqueda del trigrama sobre los checkpoints causales de la cola.

    python scripts/tune_speech_trigram.py --dry-run
    python scripts/tune_speech_trigram.py

Por cada semilla calcula una vez los logits causales de val y los decodifica
con cada combinación de la rejilla, sin repetir la GRU ni recargar el grafo.
Elige la combinación de menor WER agregado entre semillas (la primera en caso
de empate). No entrena ni usa test. Guarda cada combinación al terminarla, así
que se puede relanzar tras un corte. Necesita el runtime y el grafo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_queue import Job, job_state, queue_jobs
from src.data.speech_dataset import SpeechDataset, file_sha256
from src.inference.speech import OnlineSpeechDecoder
from src.inference.trigram import TrigramDecoder
from src.training.speech_online_evaluation import iter_online_trials
from src.training.speech_trigram_evaluation import collect_final_logits, search_word_errors
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run, registrar

SPEC_KEYS = {"plan", "variante", "semillas", "split", "evaluacion", "lenguaje", "rejilla", "salida", "tabla"}


def tuning_spec(path: Path) -> dict[str, Any]:
    """Lee el perfil del ajuste; solo admite val."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError(f"El perfil debe declarar exactamente: {sorted(SPEC_KEYS)}.")
    if spec["split"] != "val":
        raise ValueError("El trigrama se ajusta solo en val; test se reserva para la evaluación final.")
    return spec


def search_grid(base: dict[str, Any], grid: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Producto cartesiano de la rejilla; la búsqueda de partida va siempre, y la primera.

    Decodificarla antes permite comprobar pronto que reproduce el WER de la
    evaluación por bloques, antes de gastar horas en el resto.
    """
    if not grid or set(grid) - (set(base) - {"nbest"}):
        raise ValueError("La rejilla solo admite claves de la búsqueda de partida, salvo nbest.")
    for key, values in grid.items():
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
            raise ValueError(f"La rejilla de {key} necesita valores numéricos únicos.")
    combinations = [dict(zip(grid, values)) for values in product(*grid.values())]
    start = {key: base[key] for key in grid}
    return [start, *(combination for combination in combinations if combination != start)]


def tuning_jobs(spec: dict[str, Any]) -> list[tuple[Job, Path]]:
    """Trabajos causales terminados de la variante, con su checkpoint seleccionado."""
    jobs = {job.seed: job for job in queue_jobs(Path(spec["plan"])) if job.variant == spec["variante"]}
    seeds = spec["semillas"]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(jobs):
        raise ValueError("Las semillas deben ser únicas y existir en la variante del plan.")
    selected = []
    for seed in seeds:
        job = jobs[seed]
        state, result = job_state(job.config)
        if state != "completo" or job.config["preprocesamiento"]["suavizado"].get("modo", "centered") != "causal":
            raise ValueError(f"{job.name} necesita un entrenamiento causal completo.")
        selected.append((job, result.parent / "mejor_val.pt"))
    return selected


def _key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True)


def read_results(path: Path) -> list[dict[str, Any]]:
    """Combinaciones ya decodificadas; un archivo ausente equivale a ninguna."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(
    rows: list[dict[str, Any]], combinations: list[dict[str, Any]], jobs: list[str], start: dict[str, Any],
) -> dict[str, Any]:
    """Agrega errores y palabras entre semillas; exige todas las combinaciones de todos los trabajos."""
    table = {(row["trabajo"], _key(row["parametros"])): row for row in rows}
    entries = []
    for parameters in combinations:
        members = [table.get((job, _key(parameters))) for job in jobs]
        if any(member is None for member in members):
            raise ValueError("Faltan combinaciones por decodificar; relanza el ajuste para completarlas.")
        errors, words = sum(row["errores"] for row in members), sum(row["palabras"] for row in members)
        entries.append({"parametros": parameters, "errores": errors, "palabras": words, "wer": errors / words,
                        "wer_por_semilla": {str(row["seed"]): row["errores"] / row["palabras"] for row in members},
                        "segundos_busqueda": sum(row["segundos"] for row in members),
                        "partida": parameters == start})
    best = min(range(len(entries)), key=lambda index: entries[index]["wer"])
    for index, entry in enumerate(entries):
        entry["elegida"] = index == best
    return {"scope": "ajuste en val; la cifra elegida es optimista hasta evaluarla en test",
            "criterio": "mínimo WER agregado entre semillas; primera combinación en caso de empate",
            "trabajos": jobs, "frases_por_trabajo": len(rows[0]["errores_por_frase"]),
            "partida": start, "elegida": entries[best]["parametros"],
            "wer_partida": next(entry["wer"] for entry in entries if entry["partida"]),
            "wer_elegida": entries[best]["wer"], "combinaciones": entries}


def _number(value: float) -> str:
    return f"{value:.4g}".replace(".", ",")


def _percent(value: float) -> str:
    return f"{100 * value:.2f}".replace(".", ",")


def render_markdown(summary: dict[str, Any]) -> str:
    """Una fila por combinación, en el orden de la rejilla. WER en porcentaje."""
    keys = list(summary["partida"])
    lines = ["# Ajuste del trigrama en val", "",
             "Generada por `scripts/tune_speech_trigram.py`; no editar a mano. Criterio: "
             f"{summary['criterio']}.", f"Trabajos: {', '.join(summary['trabajos'])}; "
             f"{summary['frases_por_trabajo']} frases cada uno. El WER elegido es optimista: se",
             "selecciona y se mide sobre las mismas frases.", "",
             "| " + " | ".join(keys) + " | WER (%) | WER por semilla, mín–máx (%) | Búsqueda (s) | |",
             "|" + "---:|" * (len(keys) + 3) + "---|"]
    for entry in summary["combinaciones"]:
        seeds = entry["wer_por_semilla"].values()
        mark = " y ".join(label for label, active in (("partida", entry["partida"]), ("elegida", entry["elegida"])) if active)
        lines.append("| " + " | ".join(_number(entry["parametros"][key]) for key in keys)
                     + f" | {_percent(entry['wer'])} | {_percent(min(seeds))}–{_percent(max(seeds))} | "
                     f"{entry['segundos_busqueda']:.0f} | {mark} |")
    return "\n".join(lines) + "\n"


def _signature(checkpoint_hash: str, evaluation: dict[str, Any], language: dict[str, Any]) -> str:
    payload = json.dumps({"checkpoint": checkpoint_hash, "evaluacion": evaluation, "lenguaje": language}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    """Decodifica lo que falte de la rejilla, escribe la tabla y la registra en W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/ajuste_trigrama.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="muestra trabajos y combinaciones pendientes sin decodificar")
    args = parser.parse_args()
    try:
        spec = tuning_spec(args.config)
        evaluation = yaml.safe_load(Path(spec["evaluacion"]).read_text(encoding="utf-8"))
        language_spec = yaml.safe_load(Path(spec["lenguaje"]).read_text(encoding="utf-8"))
        evaluation["split"] = spec["split"]
        base = language_spec["busqueda"]
        combinations = search_grid(base, spec["rejilla"])
        jobs = tuning_jobs(spec)
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output = Path(spec["salida"])
    results_path = output / "resultados.jsonl"
    rows = read_results(results_path)
    done = {(row["trabajo"], _key(row["parametros"])) for row in rows}
    pending = {job.name: [item for item in combinations if (job.name, _key(item)) not in done] for job, _ in jobs}
    if args.dry_run:
        for job, checkpoint in jobs:
            print(f"{job.name}: {len(pending[job.name])} de {len(combinations)} combinaciones pendientes ({checkpoint})")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = evaluation["permitir_tf32"]
    torch.backends.cudnn.allow_tf32 = evaluation["permitir_tf32"]
    torch.set_num_threads(evaluation["hilos_cpu"])
    device = torch.device(evaluation["dispositivo"])
    output.mkdir(parents=True, exist_ok=True)
    language = None
    decoded = 0
    try:
        for job, checkpoint in jobs:
            signature = _signature(file_sha256(checkpoint), evaluation, language_spec)
            if any(row["firma"] != signature for row in rows if row["trabajo"] == job.name):
                raise ValueError(f"{results_path} contiene {job.name} con otro checkpoint o perfil; usa otra salida.")
            if not pending[job.name]:
                continue
            decoder = OnlineSpeechDecoder.from_checkpoint(checkpoint, device)
            fijar_seed(decoder.checkpoint.config["experimento"]["seed"])
            dataset = SpeechDataset(copy.deepcopy(decoder.checkpoint.config), spec["split"])
            started = perf_counter()
            items = collect_final_logits(decoder, iter_online_trials(dataset, decoder), evaluation,
                                         expected_trials=len(dataset))
            print(f"[ajuste] {job.name}: logits de {len(items)} frases en {perf_counter() - started:.0f} s.", flush=True)
            vocabulary = (tuple(decoder.checkpoint.phones), decoder.checkpoint.blank_id)
            if language is None:
                language = TrigramDecoder(language_spec, *vocabulary, log_path=output / "trigram_worker.log")
                shared = vocabulary
            elif vocabulary != shared:
                raise ValueError("Los checkpoints no comparten vocabulario de fonemas.")
            del decoder
            with results_path.open("a", encoding="utf-8") as stream:
                for parameters in pending[job.name]:
                    language.configure({**base, **parameters})
                    started = perf_counter()
                    errors, words = search_word_errors(language, items)
                    row = {"firma": signature, "trabajo": job.name, "seed": job.seed, "parametros": parameters,
                           "errores": sum(errors), "palabras": sum(words), "segundos": perf_counter() - started,
                           "errores_por_frase": errors}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    decoded += 1
                    print(f"[ajuste] {job.name} {_key(parameters)}: WER={row['errores'] / row['palabras']:.2%} "
                          f"en {row['segundos']:.0f} s.", flush=True)
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if language is not None:
            language.close()
    start = {key: base[key] for key in spec["rejilla"]}
    summary = summarize(rows, combinations, [job.name for job, _ in jobs], start)
    summary.update(config=spec, busqueda_base=base, results_sha256=file_sha256(results_path),
                   config_sha256=file_sha256(args.config), language_config_sha256=file_sha256(Path(spec["lenguaje"])))
    prefix = Path(spec["tabla"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(summary)
    if decoded or not prefix.with_suffix(".json").is_file():
        cfg = copy.deepcopy(jobs[0][0].config)
        cfg["experimento"]["nombre"] = f"{spec['variante']}_ajuste_trigrama_{spec['split']}"
        cfg["ajuste_trigrama"] = {"perfil": spec, "busqueda_base": base}
        run = abrir_run(cfg)
        try:
            from wandb import Table

            keys = list(start)
            registrar(run, {"ajuste_trigrama/rejilla": Table(
                columns=[*keys, "wer", "segundos_busqueda", "partida", "elegida"],
                data=[[*(entry["parametros"][key] for key in keys), entry["wer"], entry["segundos_busqueda"],
                       entry["partida"], entry["elegida"]] for entry in summary["combinaciones"]])})
            run.summary.update({"ajuste_trigrama/wer_partida": summary["wer_partida"],
                                "ajuste_trigrama/wer_elegida": summary["wer_elegida"],
                                "ajuste_trigrama/actualizaciones": 0, "ajuste_trigrama/test_evaluado": False,
                                **{f"ajuste_trigrama/elegida/{key}": value for key, value in summary["elegida"].items()}})
            summary["wandb_run_id"] = run.id
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish()
    else:
        summary["wandb_run_id"] = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8")).get("wandb_run_id")
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Partida: WER {summary['wer_partida']:.2%}. Elegida: {summary['elegida']} con WER {summary['wer_elegida']:.2%}.")
    print(f"Tabla: {prefix.with_suffix('.json')} y {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
