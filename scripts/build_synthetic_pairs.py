"""Genera los pares sintéticos texto externo → fonemas con errores de la Línea A.

    python scripts/build_synthetic_pairs.py --dry-run
    python scripts/build_synthetic_pairs.py

Toma frases del corpus fijado (scripts/get_text_corpus.py), retira las que
coinciden con cualquier frase de T12, las pasa a fonemas con el g2p-en de las
etiquetas y les aplica un modelo de errores aprendido de los pares de train por
ajuste cruzado. No lee pares de val ni toca señal neuronal; no entrena.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.get_text_corpus import sha256_of
from src.data.phoneme_noise import PhonemeNoise, fit_error_model, phone_error_rate
from src.data.speech_dataset import file_sha256
from src.data.text_corpus import candidate_sentences, normalized_text
from src.training.text_decoder import read_pair_split
from src.utils.seed import crear_generador_numpy

SPEC_KEYS = {"corpus", "frases", "etiquetas", "excluir", "errores", "semilla", "procesos", "comprobacion_frases",
             "ejemplos", "salida", "tabla"}
BATCH_ROWS = 2000

_ENCODER: Any = None


def synthetic_spec(path: Path) -> dict[str, Any]:
    """Lee el perfil y exige exactamente sus claves."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        raise ValueError(f"El perfil de pares sintéticos requiere exactamente {sorted(SPEC_KEYS)}.")
    if spec["frases"]["maximo"] < 1 or spec["procesos"] < 1:
        raise ValueError("frases.maximo y procesos deben ser al menos 1.")
    return spec


def t12_texts(root: Path) -> set[str]:
    """Textos normalizados de todas las frases de T12 preparadas (train y test)."""
    files = sorted(root.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No hay etiquetas de T12 en {root}.")
    return {normalized_text(target["reference"]) for path in files
            for target in json.loads(path.read_text(encoding="utf-8"))["targets"]}


def corpus_documents(spec: dict[str, Any]) -> Iterator[str]:
    """Documentos del parquet, por lotes, tras comprobar su hash."""
    import pyarrow.parquet as pq

    corpus = spec["corpus"]
    path = Path(corpus["archivo"])
    if not path.is_file() or sha256_of(path) != corpus["sha256"]:
        raise ValueError(f"Falta o ha cambiado {path}. Ejecuta scripts/get_text_corpus.py.")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=[corpus["columna"]]):
        yield from (text for text in batch.column(0).to_pylist() if text)


def _start_encoder(labels: str) -> None:
    global _ENCODER
    from src.data.phonemes import PhonemeEncoder

    _ENCODER = PhonemeEncoder(yaml.safe_load(Path(labels).read_text(encoding="utf-8")))


def _encode(sentence: str) -> list[str] | None:
    try:
        return list(_ENCODER.encode(sentence).phones)
    except ValueError:
        return None


def encode_sentences(sentences: list[str], spec: dict[str, Any], counts: dict[str, int]) -> list[tuple[str, list[str]]]:
    """Fonemas limpios en el orden dado hasta reunir `frases.maximo` válidas."""
    limit, wanted = spec["frases"]["max_fonemas"], spec["frases"]["maximo"]
    kept: list[tuple[str, list[str]]] = []
    # spawn: torch ya ha abierto hilos y un fork podría bloquear a los hijos.
    context = multiprocessing.get_context("spawn")
    with context.Pool(spec["procesos"], initializer=_start_encoder, initargs=(spec["etiquetas"],)) as pool:
        for sentence, phones in zip(sentences, pool.imap(_encode, sentences, chunksize=64)):
            if phones is None:
                counts["descartada_g2p"] = counts.get("descartada_g2p", 0) + 1
            elif len(phones) > limit:
                counts["descartada_fonemas"] = counts.get("descartada_fonemas", 0) + 1
            else:
                kept.append((sentence, phones))
                if len(kept) == wanted:
                    pool.terminate()
                    break
    return kept


def render_markdown(result: dict[str, Any]) -> str:
    """Tabla de conteos y comprobaciones de PER, sin interpretación."""
    def percent(value: float) -> str:
        return f"{100 * value:.2f}".replace(".", ",")

    check, counts = result["comprobacion"], result["conteos"]
    lines = ["# Línea A: pares sintéticos", "",
             f"Corpus `{result['corpus']['nombre']}` (`{Path(result['corpus']['archivo']).name}`, "
             f"SHA-256 `{result['corpus']['sha256'][:12]}…`). Errores ajustados con los "
             f"{check['pares_reales']} pares de train por ajuste cruzado.", "",
             "| Conteo | Valor |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in counts.items()]
    lines += ["", "| PER (%) | Valor |", "|---|---:|",
              f"| Pares reales de train | {percent(check['per_real'])} |",
              f"| Modelo de errores sobre las referencias reales de train | {percent(check['per_ruido_referencias_reales'])} |",
              f"| Pares sintéticos (primeras {check['frases_sinteticas_medidas']} frases) | "
              f"{percent(check['per_sinteticos'])} |", "", "Ejemplos:", ""]
    for row in result["ejemplos"]:
        lines += [f"- {row['reference_text']}", f"  - limpio: `{row['reference_phones']}`",
                  f"  - con errores: `{row['phones']}`"]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Extrae frases, las pasa a fonemas, aplica errores y escribe los pares."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/pares_sinteticos.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="comprueba perfil, corpus y pares reales")
    args = parser.parse_args()
    try:
        spec = synthetic_spec(args.config)
        output = Path(spec["salida"])
        if output.exists():
            raise ValueError(f"{output} ya existe; cada generación usa una carpeta nueva.")
        real_manifest = json.loads((Path(spec["errores"]) / "manifest.json").read_text(encoding="utf-8"))
        real = {"train": read_pair_split(Path(spec["errores"]), "train", real_manifest)}
        excluded = t12_texts(Path(spec["excluir"]))
        print(f"Pares reales de train: {len(real['train'])}. Frases de T12 excluidas: {len(excluded)}.", flush=True)
        if args.dry_run:
            next(corpus_documents(spec))
            return 0

        references = [(row["reference_phones"].split(), row["phones"].split()) for row in real["train"]]
        model = fit_error_model(references)
        noise = PhonemeNoise(model)
        pick_rng, noise_rng, check_rng = crear_generador_numpy(spec["semilla"]).spawn(3)
        check = {"pares_reales": len(references), "per_real": model["per"],
                 "per_ruido_referencias_reales": phone_error_rate(
                     (reference, noise.apply(reference, check_rng)) for reference, _ in references)}
        print(f"PER real {model['per']:.2%}; ruido sobre referencias reales "
              f"{check['per_ruido_referencias_reales']:.2%}.", flush=True)

        counts: dict[str, int] = {}
        unique = candidate_sentences(corpus_documents(spec), spec["frases"], excluded, counts)
        counts["candidatas_unicas"] = len(unique)
        sentences = list(unique.values())
        sentences = [sentences[index] for index in pick_rng.permutation(len(sentences))]
        print(f"Candidatas únicas: {len(unique)}. Pasando a fonemas …", flush=True)
        encoded = encode_sentences(sentences, spec, counts)
        counts["pares"] = len(encoded)
        if not encoded:
            raise ValueError("No queda ninguna frase sintética.")

        rows = []
        for sentence, phones in encoded:
            rows.append({"split": "train", "source": spec["corpus"]["nombre"], "phones": " ".join(noise.apply(phones, noise_rng)),
                         "reference_phones": " ".join(phones), "reference_text": sentence,
                         "model": "sintetico", "seed": None, "fold": None})
        measured = rows[:spec["comprobacion_frases"]]
        check["frases_sinteticas_medidas"] = len(measured)
        check["per_sinteticos"] = phone_error_rate((row["reference_phones"].split(), row["phones"].split())
                                                   for row in measured)

        partial = output.with_name(output.name + ".part")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        with (partial / "train.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (partial / "modelo_errores.json").write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
        result = {"corpus": spec["corpus"], "spec": spec, "spec_sha256": file_sha256(args.config),
                  "conteos": counts, "comprobacion": check, "ejemplos": rows[:spec["ejemplos"]],
                  "pares_errores_manifest_sha256": file_sha256(Path(spec["errores"]) / "manifest.json"),
                  "pares_errores_files": real_manifest["files"]}
        manifest = {**result, "files": {"train.jsonl": file_sha256(partial / "train.jsonl")},
                    "modelo_errores_sha256": file_sha256(partial / "modelo_errores.json")}
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        partial.replace(output)
        prefix = Path(spec["tabla"])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_markdown(result)
        prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps({**result, "files": manifest["files"]}, indent=2,
                                                          ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(markdown)
    print(f"Pares: {output}. Tabla: {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
