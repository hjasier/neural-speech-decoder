"""Prepara señales T12 normalizadas en archivos NPY para lectura por fragmentos.

    python scripts/prepare_speech_data.py --config configs/speech_comparacion_t4.yaml

Procesa una sesión cada vez, conserva todos sus ensayos y no suaviza ni añade
ruido. Por defecto las estadísticas proceden únicamente de train y se guardan
para test y el futuro checkpoint. Con preprocesamiento.normalizacion en modo
bloque o rodante cada frase usa las suyas (señal sin etiquetas) y la carpeta
preparada debe ser distinta. Los archivos quedan en data/, fuera de Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    extract_model_features, fit_train_normalization, normalization_spec, session_trial_normalizations,
)
from src.data.speech import load_session
from src.data.speech_dataset import file_sha256, preparation_signature, read_label_manifest
from src.data.speech_protocol import read_speech_protocol


def prepare_signals(cfg: dict[str, Any]) -> dict[str, int]:
    """Valida referencias y escribe cada sesión sin concatenar todo el corpus."""
    labels = read_label_manifest(cfg)
    label_root = Path(cfg["etiquetas"]["salida"])
    source_root = Path(cfg["datos"]["raiz"])
    output_root = Path(cfg["datos"]["preparados"])
    protocol = read_speech_protocol(cfg)
    partitions = {entry["session"]: entry for entry in protocol["sessions"]} if protocol else {}
    # Impedir que un perfil nuevo sobrescriba preparaciones de otra firma.
    existing = output_root / "manifest.json"
    if existing.is_file() and json.loads(existing.read_text())["signature"] != preparation_signature(cfg):
        raise ValueError("La carpeta preparada contiene otra firma; utiliza una carpeta distinta.")
    label_files = {entry["path"]: entry for entry in labels["files"]}
    files = []
    normalizations = {}
    totals = {"train": 0, "test": 0}
    mode = normalization_spec(cfg)["modo"]
    for name in cfg["datos"]["sesiones"]:
        part = partitions.get(name)
        if part:
            for split, source in part["sources"].items():
                if file_sha256(source_root / source["path"]) != source["sha256"]:
                    raise ValueError(f"El archivo original {split}/{name} ha cambiado.")
        stats = fit_train_normalization(
            source_root, name, cfg["preprocesamiento"]["normalizacion_epsilon"],
            trial_indices=part["train"] if part else None,
        )
        normalizations[name] = stats.to_dict()
        if part:
            normalizations[name].update({"fit_trial_indices": part["train"],
                                         "fit_protocol_sha256": cfg["protocolo"]["sha256"]})
        # En bloque/rodante cada frase tiene su propia media y desviación; las
        # de train se conservan en el manifiesto como referencia de la sesión.
        per_trial = None if mode == "train_fija" else session_trial_normalizations(
            cfg, name, train_indices=part["train"] if part else None,
        )
        for split in totals:
            relative = f"{split}/{name}.json"
            label_path = label_root / relative
            if file_sha256(label_path) != label_files[relative]["sha256"]:
                raise ValueError(f"Las etiquetas han cambiado: {label_path}")
            targets = json.loads(label_path.read_text(encoding="utf-8"))
            source = source_root / split / f"{name}.mat"
            session = load_session(source)
            if part and split == "train":
                for logical in ("train", "val"):
                    actual = [trial.index for trial in session.trials if trial.block in part[f"{logical}_blocks"]]
                    if actual != part[logical]:
                        raise ValueError("Los bloques reales no corresponden a la partición fijada.")
            if targets["session"] != name or targets["split"] != split or len(targets["targets"]) != len(session.trials):
                raise ValueError(f"Señales y etiquetas no corresponden: {source}")
            offsets = [0]
            for trial, target in zip(session.trials, targets["targets"], strict=True):
                if target["trial_index"] != trial.index or target["reference"] != trial.text:
                    raise ValueError(f"Referencia desalineada: {source}, ensayo {trial.index}")
                offsets.append(offsets[-1] + len(trial.spike_power))
            destination = output_root / split / f"{name}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".npy.part")
            array = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32, shape=(offsets[-1], cfg["modelo"]["entrada"]),
            )
            for index, trial in enumerate(session.trials):
                array[offsets[index]:offsets[index + 1]] = (
                    stats.transform(trial, name) if per_trial is None
                    else per_trial[split][index].transform_features(extract_model_features(trial), name)
                )
            array.flush()
            del array
            partial.replace(destination)
            files.append({
                "split": split, "session": name, "path": destination.relative_to(output_root).as_posix(),
                "source": source.as_posix(), "source_sha256": file_sha256(source),
                "sha256": file_sha256(destination), "offsets": offsets,
                "blocks": [trial.block for trial in session.trials],
                "transcriptions_sha256": targets["transcriptions_sha256"],
            })
            totals[split] += len(session.trials)
            del session
        print(f"Señales preparadas: {name}", flush=True)
    if totals != labels["totals"]:
        raise ValueError("El número de señales no coincide con las etiquetas.")
    manifest = {
        "format_version": 2 if protocol else 1, "signature": preparation_signature(cfg),
        "normalizations": normalizations, "totals": protocol["totals"] if protocol else totals, "files": files,
    }
    if protocol:
        manifest["protocol_sha256"] = cfg["protocolo"]["sha256"]
        manifest["source_totals"] = totals
    if mode != "train_fija":
        manifest["normalization"] = normalization_spec(cfg)
    partial_manifest = output_root / "manifest.json.part"
    partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial_manifest.replace(output_root / "manifest.json")
    return manifest["totals"]


def main() -> int:
    """Carga la configuración y prepara los dos splits etiquetados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        totals = prepare_signals(cfg)
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Señales listas: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
