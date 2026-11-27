"""Línea A: decodificador seq2seq de fonemas predichos a texto, con WER de val.

Recibe el modelo y el tokenizador ya construidos, así que los tests usan un
T5 diminuto sin descargar pesos. Train son los pares por ajuste cruzado,
opcionalmente mezclados con pares sintéticos; val, las predicciones de las
cinco semillas de la referencia. Test no entra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.speech_dataset import file_sha256
from src.training.bootstrap import bootstrap_rate_interval
from src.training.speech_trigram_evaluation import word_error_counts, word_tokens
from src.utils.seed import crear_generador
from src.utils.tracking import registrar

SPLITS = ("train", "val")


def read_pair_split(root: Path, split: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Lee `<split>.jsonl` de una carpeta de pares tras comprobar su hash en el manifiesto."""
    path = root / f"{split}.jsonl"
    if file_sha256(path) != manifest["files"][path.name]:
        raise ValueError(f"Los pares han cambiado desde su exportación: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"{path} debe contener solo frases de {split}.")
    return rows


def read_pairs(cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Lee train.jsonl y val.jsonl tras comprobar los hashes del manifiesto.

    Con `datos.sinteticos` lee además su train.jsonl como `sinteticos`; el
    manifiesto devuelto es el de los pares reales.
    """
    root = Path(cfg["datos"]["pares"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != {f"{split}.jsonl" for split in SPLITS}:
        raise ValueError("El manifiesto de pares debe listar solo train.jsonl y val.jsonl.")
    pairs = {split: read_pair_split(root, split, manifest) for split in SPLITS}
    synthetic = cfg["datos"].get("sinteticos")
    if synthetic is not None:
        if set(synthetic) != {"pares", "proporcion"} or not 0 < synthetic["proporcion"] < 1:
            raise ValueError("datos.sinteticos requiere pares y una proporcion entre 0 y 1.")
        root = Path(synthetic["pares"])
        extra = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if set(extra["files"]) != {"train.jsonl"}:
            raise ValueError("El manifiesto de pares sintéticos debe listar solo train.jsonl.")
        pairs["sinteticos"] = read_pair_split(root, "train", extra)
    return pairs, manifest


def target_text(text: str) -> str:
    """Texto objetivo con la misma limpieza que el WER: minúsculas y sin puntuación."""
    return " ".join(word_tokens(text))


def encode_pairs(tokenizer: Any, rows: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, list[int]]]:
    """Tokeniza fonemas y texto; rechaza lo que exceda los máximos en vez de truncar."""
    data = cfg["datos"]
    inputs = tokenizer([data["prefijo"] + row["phones"] for row in rows])["input_ids"]
    labels = tokenizer([target_text(row["reference_text"]) for row in rows])["input_ids"]
    for kind, values, limit in (("entrada", inputs, data["max_tokens_entrada"]),
                                ("salida", labels, data["max_tokens_salida"])):
        longest = max(len(item) for item in values)
        if longest > limit:
            raise ValueError(f"Una frase ocupa {longest} tokens de {kind}; max_tokens_{kind} es {limit}.")
    return [{"input_ids": source, "labels": target} for source, target in zip(inputs, labels, strict=True)]


def collate_pairs(items: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Rellena entradas con pad y objetivos con -100, que la pérdida ignora."""
    width = max(len(item["input_ids"]) for item in items)
    height = max(len(item["labels"]) for item in items)
    inputs = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), height), -100, dtype=torch.long)
    for row, item in enumerate(items):
        inputs[row, :len(item["input_ids"])] = torch.tensor(item["input_ids"])
        labels[row, :len(item["labels"])] = torch.tensor(item["labels"])
    return {"input_ids": inputs, "attention_mask": inputs.ne(pad_id).long(), "labels": labels}


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """Subida lineal durante el calentamiento y bajada lineal hasta cero en `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def _autocast(cfg: dict[str, Any], device: torch.device) -> Any:
    enabled = bool(cfg["ejecucion"]["bf16"]) and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.inference_mode()
def generate_texts(
    model: nn.Module, tokenizer: Any, encoded: Sequence[dict[str, list[int]]], cfg: dict[str, Any],
    device: torch.device, num_beams: int,
) -> list[str]:
    """Genera el texto de cada frase en orden, por lotes, sin mirar el objetivo."""
    model.eval()
    texts: list[str] = []
    size = cfg["evaluacion"]["batch_size"]
    for start in range(0, len(encoded), size):
        batch = collate_pairs(encoded[start:start + size], tokenizer.pad_token_id)
        with _autocast(cfg, device):
            output = model.generate(input_ids=batch["input_ids"].to(device),
                                    attention_mask=batch["attention_mask"].to(device),
                                    max_new_tokens=cfg["datos"]["max_tokens_salida"], num_beams=num_beams)
        texts.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return texts


def score_texts(rows: Sequence[dict[str, Any]], hypotheses: Sequence[str]) -> dict[str, Any]:
    """WER agregado (errores totales entre palabras totales), global y por modelo de origen."""
    counts = [word_error_counts(row["reference_text"], hypothesis) for row, hypothesis in zip(rows, hypotheses, strict=True)]
    by_model: dict[str, list[int]] = {}
    for row, (errors, words) in zip(rows, counts):
        total = by_model.setdefault(row["model"], [0, 0])
        total[0] += errors
        total[1] += words
    errors, words = sum(item[0] for item in counts), sum(item[1] for item in counts)
    return {"wer": errors / words, "errores": errors, "palabras": words, "frases": len(rows),
            "por_modelo": {name: {"wer": value[0] / value[1], "errores": value[0], "palabras": value[1]}
                           for name, value in by_model.items()},
            "conteos": counts}


def train_text_decoder(
    model: nn.Module, tokenizer: Any, pairs: dict[str, list[dict[str, Any]]], cfg: dict[str, Any],
    run: Any, device: torch.device, *, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ajusta el modelo, elige el paso de menor WER de val y lo deja cargado.

    Desempate: el primer paso evaluado. La selección genera con
    `num_beams_seleccion`; el elegido se evalúa después con `num_beams_final`.
    """
    training, evaluation = cfg["entrenamiento"], cfg["evaluacion"]
    train = encode_pairs(tokenizer, pairs["train"], cfg)
    val = encode_pairs(tokenizer, pairs["val"], cfg)
    collate = partial(collate_pairs, pad_id=tokenizer.pad_token_id)
    generator = crear_generador(cfg["experimento"]["seed"])
    if "sinteticos" in pairs:
        # Cada ejemplo es sintético con probabilidad `proporcion`; con reemplazo,
        # una sola pasada del muestreador cubre todos los pasos.
        share = cfg["datos"]["sinteticos"]["proporcion"]
        synthetic = encode_pairs(tokenizer, pairs["sinteticos"], cfg)
        weights = torch.cat([torch.full((len(train),), (1 - share) / len(train), dtype=torch.double),
                             torch.full((len(synthetic),), share / len(synthetic), dtype=torch.double)])
        sampler = WeightedRandomSampler(weights, training["pasos"] * training["batch_size"], replacement=True,
                                        generator=generator)
        loader = DataLoader(train + synthetic, batch_size=training["batch_size"], sampler=sampler, collate_fn=collate)
    else:
        loader = DataLoader(train, batch_size=training["batch_size"], shuffle=True, generator=generator,
                            collate_fn=collate)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(
        step, training["calentamiento_pasos"], training["pasos"]))
    best: dict[str, Any] | None = None
    history = []
    step, started = 0, perf_counter()
    while step < training["pasos"]:
        for batch in loader:
            model.train()
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(cfg, device):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), training["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            step += 1
            if step % training["imprimir_cada"] == 0 or step == 1:
                rate = optimizer.param_groups[0]["lr"]
                registrar(run, {"train/loss": loss.item(), "train/learning_rate": rate,
                                "train/norma_gradiente": float(norm)}, step)
                log(f"paso {step}/{training['pasos']}  loss={loss.item():.4f}  lr={rate:.2e}")
            if step % evaluation["cada_pasos"] == 0 or step == training["pasos"]:
                score = score_texts(pairs["val"], generate_texts(model, tokenizer, val, cfg, device,
                                                                 evaluation["num_beams_seleccion"]))
                history.append({"step": step, "wer": score["wer"]})
                registrar(run, {"val/wer": score["wer"], "val/errores": score["errores"],
                                "val/palabras": score["palabras"]}, step)
                log(f"val: WER={score['wer']:.2%}  paso {step}")
                if best is None or score["wer"] < best["wer"]:
                    best = {"step": step, "wer": score["wer"],
                            "state": {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}}
            if step == training["pasos"]:
                break
    model.load_state_dict(best["state"])
    return {"best_step": best["step"], "best_val_wer_seleccion": best["wer"], "history": history,
            "completed_steps": step, "training_seconds": perf_counter() - started}


def final_report(
    model: nn.Module, tokenizer: Any, pairs: dict[str, list[dict[str, Any]]], cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evalúa el modelo elegido en val con el haz final; WER e intervalo por modelo de origen."""
    rows = pairs["val"]
    hypotheses = generate_texts(model, tokenizer, encode_pairs(tokenizer, rows, cfg), cfg, device,
                                cfg["evaluacion"]["num_beams_final"])
    score = score_texts(rows, hypotheses)
    spec = cfg["bootstrap"]
    intervals = {}
    for name in score["por_modelo"]:
        chosen = [count for row, count in zip(rows, score["conteos"]) if row["model"] == name]
        intervals[name] = bootstrap_rate_interval([item[0] for item in chosen], [item[1] for item in chosen],
                                                  resamples=spec["remuestreos"], confidence=spec["confianza"],
                                                  seed=spec["semilla"])
    rates = np.array([value["wer"] for value in score["por_modelo"].values()])
    predictions = [{"model": row["model"], "seed": row["seed"], "session": row["session"],
                    "source_trial_index": row["source_trial_index"], "phones": row["phones"],
                    "reference_text": row["reference_text"], "hypothesis": hypothesis,
                    "word_errors": count[0], "words": count[1]}
                   for row, hypothesis, count in zip(rows, hypotheses, score["conteos"], strict=True)]
    train_texts = {target_text(row["reference_text"]) for row in pairs["train"]}
    report = {"split": "val", "num_beams": cfg["evaluacion"]["num_beams_final"],
              "wer_agregado": score["wer"], "errores": score["errores"], "palabras": score["palabras"],
              "por_modelo": {name: {**value, "intervalo": intervals[name]} for name, value in score["por_modelo"].items()},
              "media_modelos": float(rates.mean()),
              "desviacion_modelos": float(rates.std(ddof=1)) if len(rates) > 1 else None,
              "frases_val_con_texto_en_train": sum(target_text(row["reference_text"]) in train_texts
                                                   for row in rows if row["model"] == rows[0]["model"])}
    return report, predictions
