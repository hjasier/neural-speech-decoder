"""Grafo mínimo CTC + léxico + trigrama para el runtime WFST del proyecto."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import log
import os
from pathlib import Path
import struct
import subprocess
from typing import Any

import yaml

from src.data.speech_dataset import file_sha256


def preparar_trigrama_smoke(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Construye un ConstFst pequeño y lo carga con el mismo binding de la demo.

    Solo requiere el runtime ya instalado; no lee ni recorta el grafo de
    125.000 palabras. La serialización VectorFst sigue OpenFst (vector v2):
    https://www.openfst.org/doxygen/fst/html/vector-fst_8h_source.html
    """
    options = cfg["smoke"]["lenguaje"]
    spec = yaml.safe_load(Path(options["config_runtime"]).read_text(encoding="utf-8"))
    runtime = Path(spec["python_runtime"])
    if not runtime.is_file():
        raise FileNotFoundError(
            "Falta el runtime CPU del trigrama. Instálalo una vez con:\n"
            "  python -m pip install uv==0.9.4\n"
            "  python scripts/prepare_speech_trigram.py "
            "--config configs/entrenos_online/speech_trigrama_compacto.yaml --build-runtime\n"
            "También puedes restaurar el respaldo CPU de Colab. "
            "Instrucciones y dependencias del sistema: docs/instalacion_runtime_trigrama.md."
        )
    model = root / "trigrama"
    model.mkdir()
    vocabulary = list(options["lexico"])
    labels = {phone: index + 2 for index, phone in enumerate(
        ["SIL", *(phone for phone in cfg["datos"]["fonemas"] if phone != "SIL")],
    )}
    lexicon = {tuple(phones + ["SIL"]): word for word, phones in options["lexico"].items()}
    prefixes = {phones[:length] for phones in lexicon for length in range(len(phones))}
    counts: Counter = Counter()
    for sentence in options["corpus"]:
        history: tuple[str, ...] = ()
        for word in sentence.split():
            counts[(history, word)] += 1
            history = (*history, word)[-2:]
    states = [((), (), 0)]
    indices = {states[0]: 0}
    graph = []
    for history, prefix, previous in states:
        arcs = []

        def arc(label: int, output: int, weight: float, following: tuple) -> None:
            if following not in indices:
                indices[following] = len(states)
                states.append(following)
            arcs.append((label, output, weight, indices[following]))

        arc(1, 0, 0.0, (history, prefix, 0))  # blank
        for phone, label in labels.items():
            if label == previous or (phone == "SIL" and not prefix):
                arc(label, 0, 0.0, (history, prefix, label))
                continue
            following = (*prefix, phone)
            if following in lexicon:
                word = lexicon[following]
                total = sum(counts[(history, candidate)] for candidate in vocabulary)
                probability = (counts[(history, word)] + options["suavizado"]) / (
                    total + options["suavizado"] * len(vocabulary)
                )
                arc(label, vocabulary.index(word) + 1, -log(probability), ((*history, word)[-2:], (), label))
            elif following in prefixes:
                arc(label, 0, 0.0, (history, following, label))
        graph.append((0.0 if not prefix else float("inf"), arcs))
    vector = model / "vector.fst"
    with vector.open("wb") as stream:
        stream.write(struct.pack("<i", 2125659606))
        for name in (b"vector", b"standard"):
            stream.write(struct.pack("<i", len(name)) + name)
        stream.write(struct.pack("<iiQqqq", 2, 0, 0, 0, len(graph), sum(len(arcs) for _, arcs in graph)))
        for final, arcs in graph:
            stream.write(struct.pack("<fq", final, len(arcs)))
            for arc_values in sorted(arcs):
                stream.write(struct.pack("<iifi", *arc_values))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(spec["hilos_cpu"]),
                   "MKL_NUM_THREADS": str(spec["hilos_cpu"]), "OPENBLAS_NUM_THREADS": str(spec["hilos_cpu"])}
    converted = subprocess.run(
        [str(runtime), "-c", "import torch, lm_decoder, sys; lm_decoder.WriteConstFst(sys.argv[1], sys.argv[2])",
         str(vector), str(model / "TLG.fst")], env=environment, capture_output=True, text=True,
        timeout=options["timeout_segundos"], check=False,
    )
    if converted.returncode:
        raise RuntimeError(f"No se pudo crear el grafo del smoke: {converted.stderr}")
    (model / "words.txt").write_text(
        "<eps> 0\n" + "".join(f"{word} {index + 1}\n" for index, word in enumerate(vocabulary)), encoding="utf-8",
    )
    provenance = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    manifest = {"archive_sha256": provenance, "source": "synthetic smoke; not the published language model",
                "graph_phones": cfg["datos"]["fonemas"],
                "files": {name: {"sha256": file_sha256(model / name)} for name in ("TLG.fst", "words.txt")}}
    (model / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    spec.update(modelo=str(model), archivo_sha256=provenance, fonemas_grafo=cfg["datos"]["fonemas"],
                timeout_segundos=options["timeout_segundos"])
    spec["busqueda"].update(options["busqueda"])
    return spec
