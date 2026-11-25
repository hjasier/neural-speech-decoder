"""Exporta los pares fonemas predichos → frase de la Línea A por ajuste cruzado.

    python scripts/export_phoneme_pairs.py --dry-run
    python scripts/export_phoneme_pairs.py

Train: cada modelo de configs/final/cola_pliegues.yaml predice los bloques que
no vio, así que las 6.900 frases salen con errores de un modelo que no las
memorizó. Val: cada semilla de la referencia causal predice val completa.
Test no se toca. Solo infiere; no entrena ni registra en W&B.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_queue import Job, job_state, queue_jobs
from src.data.preprocessing import build_smoother
from src.data.speech_dataset import SpeechDataset, collate_speech, file_sha256
from src.data.speech_protocol import fold_blocks, read_speech_protocol, train_fold_spec
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint, read_speech_checkpoint
from src.training.speech_pairs import predict_phonemes
from src.utils.seed import crear_generador, fijar_seed

SPEC_KEYS = {"pliegues", "referencia", "dispositivo", "salida", "tabla"}
# Claves que la cola cambia por trabajo; el resto de la receta debe coincidir.
JOB_KEYS = ("experimento", "salida", "cola")


def pairs_spec(path: Path) -> dict[str, Any]:
    """Lee el perfil y exige sus claves; los pares solo salen de train y val."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(spec, dict) or set(spec) != SPEC_KEYS
            or not isinstance(spec["referencia"], dict) or set(spec["referencia"]) != {"plan", "variante"}):
        raise ValueError(f"El perfil de pares requiere exactamente {sorted(SPEC_KEYS)}.")
    return spec


def _recipe(cfg: dict[str, Any]) -> dict[str, Any]:
    recipe = {key: value for key, value in copy.deepcopy(cfg).items() if key not in JOB_KEYS}
    recipe["datos"].pop("pliegue", None)
    return recipe


def source_jobs(spec: dict[str, Any]) -> tuple[list[Job], list[Job]]:
    """Trabajos de pliegues y de referencia, con una receta común salvo el pliegue."""
    folds = queue_jobs(Path(spec["pliegues"]))
    reference = [job for job in queue_jobs(Path(spec["referencia"]["plan"]))
                 if job.variant == spec["referencia"]["variante"]]
    if not reference:
        raise ValueError(f"El plan de referencia no contiene {spec['referencia']['variante']}.")
    specs = [train_fold_spec(job.config) for job in folds]
    if any(item is None for item in specs) or len({item["pliegues"] for item in specs}) != 1:
        raise ValueError("Todos los trabajos de pliegues deben declarar datos.pliegue con el mismo número.")
    if sorted(item["excluido"] for item in specs) != list(range(specs[0]["pliegues"])):
        raise ValueError("Los trabajos de pliegues deben cubrir cada pliegue exactamente una vez.")
    recipe = _recipe(reference[0].config)
    if any(train_fold_spec(job.config) is not None for job in reference) or any(
            _recipe(job.config) != recipe for job in folds + reference):
        raise ValueError("Pliegues y referencia deben compartir receta; solo cambia datos.pliegue.")
    return folds, reference


def _checkpoint(job: Job) -> Path:
    state, result = job_state(job.config)
    if state != "completo":
        raise ValueError(f"{job.name} no ha terminado de entrenar ({state}).")
    return result.parent / "mejor_val.pt"


def export_job(job: Job, split: str, device: torch.device, sink: Any) -> dict[str, Any]:
    """Predice un split con el mejor checkpoint del trabajo y escribe sus filas."""
    path = _checkpoint(job)
    digest = file_sha256(path)
    cfg = read_speech_checkpoint(path)["config"]
    if cfg["experimento"]["nombre"] != job.name or train_fold_spec(cfg) != train_fold_spec(job.config):
        raise ValueError(f"El checkpoint {path} no corresponde a {job.name}.")
    fijar_seed(cfg["experimento"]["seed"])
    dataset = SpeechDataset(cfg, split)
    model = construir_modelo(cfg)
    state = load_checkpoint(path, model, cfg, dataset)
    model.to(device)
    loader = DataLoader(dataset, batch_size=cfg["datos"]["batch_size"], shuffle=False,
                        num_workers=cfg["datos"]["num_workers"], collate_fn=collate_speech,
                        generator=crear_generador(cfg["experimento"]["seed"]))
    fold = train_fold_spec(cfg)
    errors = phones = rows = 0
    for row in predict_phonemes(model, loader, cfg, build_smoother(cfg).to(device), device):
        row.update(model=job.name, seed=job.seed, fold=None if fold is None else fold["excluido"])
        sink(row)
        errors, phones, rows = errors + row["errors"], phones + row["reference_length"], rows + 1
    if file_sha256(path) != digest:
        raise ValueError(f"El checkpoint {path} cambió durante la exportación.")
    result = json.loads((path.parent / "resultado.json").read_text(encoding="utf-8"))
    return {"job": job.name, "seed": job.seed, "fold": None if fold is None else fold["excluido"],
            "split": "train" if split == "pliegue" else split, "checkpoint": str(path), "checkpoint_sha256": digest,
            "checkpoint_step": state["step"], "rows": rows, "errors": errors, "phones": phones,
            "per": errors / phones, "val_per_training": result["best_val"]["per"]}


def check_train_coverage(cfg: dict[str, Any], keys: list[tuple[str, int]]) -> None:
    """Exige que los pliegues cubran train de la partición, cada frase una vez."""
    protocol = read_speech_protocol(cfg)
    expected = sorted((entry["session"], index) for entry in protocol["sessions"] for index in entry["train"])
    if sorted(keys) != expected:
        raise ValueError("Los pliegues no cubren train exactamente una vez.")


def render_markdown(manifest: dict[str, Any]) -> str:
    """Tabla con el PER de cada modelo sobre lo que exporta, sin interpretación."""
    def percent(value: float) -> str:
        return f"{100 * value:.2f}".replace(".", ",")

    lines = ["# Línea A: pares fonemas → frase por ajuste cruzado", "",
             f"Partición `{manifest['protocol']}`; split de val y bloques de train que cada modelo no vio.",
             "Train y val salen de la misma receta causal; solo cambia el pliegue apartado.",
             "Train apartado usa la normalización por bloque de train; val, la rodante.", "",
             "| Modelo | Pliegue | Split | Frases | PER exportado (%) | PER de val en train (%) |",
             "|---|---:|---|---:|---:|---:|"]
    for source in manifest["sources"]:
        fold = "—" if source["fold"] is None else str(source["fold"])
        lines.append(f"| {source['job']} | {fold} | {source['split']} | {source['rows']} | "
                     f"{percent(source['per'])} | {percent(source['val_per_training'])} |")
    lines += ["", "| Split | Frases | PER agregado (%) |", "|---|---:|---:|"]
    for split, total in manifest["totals"].items():
        lines.append(f"| {split} | {total['rows']} | {percent(total['per'])} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Exporta train.jsonl, val.jsonl y un manifiesto con hashes de procedencia."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/pares_ajuste_cruzado.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="comprueba los trabajos sin cargar modelos")
    args = parser.parse_args()
    try:
        spec = pairs_spec(args.config)
        folds, reference = source_jobs(spec)
        if args.dry_run:
            for job in folds + reference:
                state, result = job_state(job.config)
                print(f"{job.name}: {state}{f' ({result.parent})' if result else ''}")
            return 0
        output, prefix = Path(spec["salida"]), Path(spec["tabla"])
        if output.exists():
            raise ValueError(f"{output} ya existe; usa otra salida para no mezclar exportaciones.")
        device = torch.device(spec["dispositivo"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidió CUDA pero no está disponible.")
        temporary = output.with_name(output.name + ".tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        sources, keys = [], []
        for split, jobs, dataset_split in (("train", folds, "pliegue"), ("val", reference, "val")):
            with (temporary / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                def sink(row: dict[str, Any]) -> None:
                    if split == "train":
                        keys.append((row["session"], row["source_trial_index"]))
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

                for job in jobs:
                    sources.append(export_job(job, dataset_split, device, sink))
                    print(f"[pares] {job.name}: {sources[-1]['rows']} frases de {split}, "
                          f"PER {sources[-1]['per']:.2%}.", flush=True)
        cfg = folds[0].config
        check_train_coverage(cfg, keys)
        protocol = read_speech_protocol(cfg)
        count = train_fold_spec(cfg)["pliegues"]
        totals = {}
        for split in ("train", "val"):
            chosen = [source for source in sources if source["split"] == split]
            errors, phones = sum(item["errors"] for item in chosen), sum(item["phones"] for item in chosen)
            totals[split] = {"rows": sum(item["rows"] for item in chosen), "errors": errors,
                             "phones": phones, "per": errors / phones}
        manifest = {"format_version": 1, "scope": "pares de train por ajuste cruzado y de val; sin test",
                    "protocol": cfg["protocolo"]["nombre"], "protocol_sha256": cfg["protocolo"]["sha256"],
                    "config": spec, "config_sha256": file_sha256(args.config),
                    "plans_sha256": {plan: file_sha256(Path(plan)) for plan in (spec["pliegues"], spec["referencia"]["plan"])},
                    "fold_blocks": {entry["session"]: [fold_blocks(entry["train_blocks"], count, fold, rotation=position)
                                                       for fold in range(count)]
                                    for position, entry in enumerate(protocol["sessions"])},
                    "files": {name: file_sha256(temporary / name) for name in ("train.jsonl", "val.jsonl")},
                    "sources": sources, "totals": totals,
                    "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "torch_version": str(torch.__version__)}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
        temporary.rename(output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_markdown(manifest)
        prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps({key: value for key, value in manifest.items()
                                                           if key != "fold_blocks"}, indent=2, ensure_ascii=False)
                                               + "\n", encoding="utf-8")
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(markdown)
    print(f"Pares: {output}. Tabla: {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
