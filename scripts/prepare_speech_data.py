"""Prepara señales T12 normalizadas en archivos NPY para lectura por fragmentos.

    python scripts/prepare_speech_data.py --config configs/speech_original.yaml

Procesa una sesión cada vez, conserva todos sus ensayos y no suaviza ni añade
ruido. Las estadísticas proceden únicamente de train y se guardan para test
y el futuro checkpoint. Los archivos preparados quedan en data/, fuera de Git.
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

from src.data.preprocessing import fit_train_normalization
from src.data.speech import load_session
from src.data.speech_dataset import file_sha256, preparation_signature, read_label_manifest


def prepare_signals(cfg: dict[str, Any]) -> dict[str, int]:
    """Valida referencias y escribe cada sesión sin concatenar todo el corpus."""
    labels = read_label_manifest(cfg)
    label_root = Path(cfg["etiquetas"]["salida"])
    source_root = Path(cfg["datos"]["raiz"])
    output_root = Path(cfg["datos"]["preparados"])
    label_files = {entry["path"]: entry for entry in labels["files"]}
    files = []
    normalizations = {}
    totals = {"train": 0, "test": 0}
    for name in cfg["datos"]["sesiones"]:
        stats = fit_train_normalization(
            source_root, name, cfg["preprocesamiento"]["normalizacion_epsilon"],
        )
        normalizations[name] = stats.to_dict()
        for split in totals:
            relative = f"{split}/{name}.json"
            label_path = label_root / relative
            if file_sha256(label_path) != label_files[relative]["sha256"]:
                raise ValueError(f"Las etiquetas han cambiado: {label_path}")
            targets = json.loads(label_path.read_text(encoding="utf-8"))
            source = source_root / split / f"{name}.mat"
            session = load_session(source)
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
                array[offsets[index]:offsets[index + 1]] = stats.transform(trial, name)
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
        "format_version": 1, "signature": preparation_signature(cfg),
        "normalizations": normalizations, "totals": totals, "files": files,
    }
    partial_manifest = output_root / "manifest.json.part"
    partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial_manifest.replace(output_root / "manifest.json")
    return totals


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
