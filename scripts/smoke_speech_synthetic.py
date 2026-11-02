"""Entradas sintéticas para comprobar el pipeline en CI sin descargar T12."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from src.data.speech import RECORDED_CHANNELS
from src.data.speech_dataset import file_sha256
from src.utils.seed import crear_generador_numpy


def preparar_datos_sinteticos_smoke(cfg: dict[str, Any]) -> None:
    """Escribe señales, etiquetas y metadatos de prueba dentro del run del smoke."""
    counts = cfg["smoke"]["frases_por_sesion"]
    options = cfg["smoke"]["datos_sinteticos"]
    cfg["datos"]["sesiones"] = options["sesiones"].copy()
    cfg["experimento"]["nombre"] = options["nombre_experimento"]
    generator = crear_generador_numpy(cfg["experimento"]["seed"])
    files, totals = [], {"train": 0, "test": 0}
    for session in cfg["datos"]["sesiones"]:
        for split, count in (("train", counts["train"] + counts["val"]), ("test", counts["test"])):
            crossings, powers = np.empty((1, count), dtype=object), np.empty((1, count), dtype=object)
            texts, targets, blocks = [], [], []
            for index in range(count):
                length = options["longitudes_bins"][index % len(options["longitudes_bins"])]
                crossings[0, index] = generator.random((length, RECORDED_CHANNELS), dtype=np.float32)
                powers[0, index] = generator.random((length, RECORDED_CHANNELS), dtype=np.float32)
                phrase = cfg["smoke"]["frases_control"][index % len(cfg["smoke"]["frases_control"])]
                texts.append(phrase["texto"])
                ids = [cfg["datos"]["fonemas"].index(phone) for phone in phrase["fonemas"]]
                targets.append({"trial_index": index, "reference": phrase["texto"], "ids": ids,
                                "minimum_ctc_steps": len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))})
                blocks.append(1 if split == "test" or index < counts["train"] else 2)
            source = Path(cfg["datos"]["raiz"]) / split / f"{session}.mat"
            source.parent.mkdir(parents=True, exist_ok=True)
            savemat(source, {"tx1": crossings, "spikePow": powers, "sentenceText": np.asarray(texts), "blockIdx": blocks})
            relative = f"{split}/{session}.json"
            label = Path(cfg["etiquetas"]["salida"]) / relative
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(json.dumps({
                "session": session, "split": split, "targets": targets,
                "transcriptions_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
            }, indent=2) + "\n", encoding="utf-8")
            files.append({"path": relative, "sha256": file_sha256(label), "trials": count})
            totals[split] += count
    manifest = {"format_version": 1, "sessions": cfg["datos"]["sesiones"], "phones": cfg["datos"]["fonemas"],
                "blank_id": len(cfg["datos"]["fonemas"]), "label_config": cfg["etiquetas"], "totals": totals, "files": files}
    (Path(cfg["etiquetas"]["salida"]) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
