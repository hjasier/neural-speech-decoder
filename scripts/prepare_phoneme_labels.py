"""Genera etiquetas de fonemas para los splits etiquetados de T12.

    python scripts/prepare_phoneme_labels.py --config configs/speech_original.yaml

Lee únicamente sentenceText de los .mat: no abre matrices neuronales,
no utiliza competitionHoldOut y no entrena. Guarda etiquetas y procedencia
en la carpeta de data/ declarada en la configuración.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any

from scipy.io import loadmat
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phonemes import PhonemeEncoder


def write_json(path: Path, data: dict[str, Any]) -> str:
    """Escribe un JSON mediante un archivo parcial y devuelve su SHA-256."""
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_bytes(payload)
    partial.replace(path)
    return hashlib.sha256(payload).hexdigest()


def prepare_labels(cfg: dict[str, Any]) -> dict[str, int]:
    """Convierte todas las referencias configuradas y publica su manifiesto."""
    sessions = cfg["datos"]["sesiones"]
    if not sessions or len(sessions) != len(set(sessions)):
        raise ValueError("Las sesiones deben ser una lista no vacía sin duplicados.")
    encoder = PhonemeEncoder(cfg)
    source_root = Path(cfg["datos"]["raiz"])
    output_root = Path(cfg["etiquetas"]["salida"])
    totals = {"train": 0, "test": 0}
    files = []
    for session in sessions:
        for split in totals:
            source = source_root / split / f"{session}.mat"
            texts = [str(text).strip() for text in loadmat(
                source, variable_names=["sentenceText"],
            )["sentenceText"].ravel()]
            targets = []
            for index, text in enumerate(texts):
                try:
                    target = encoder.encode(text)
                except ValueError as error:
                    raise ValueError(f"{split}/{session}, ensayo {index}: {error}") from error
                targets.append({
                    "trial_index": index,
                    "reference": text,
                    "text": target.text,
                    "phones": target.phones,
                    "ids": target.ids,
                    "minimum_ctc_steps": target.minimum_ctc_steps,
                })
            relative = Path(split) / f"{session}.json"
            transcription_hash = hashlib.sha256(
                json.dumps(texts, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            digest = write_json(output_root / relative, {
                "session": session,
                "split": split,
                "source": source.as_posix(),
                "transcriptions_sha256": transcription_hash,
                "targets": targets,
            })
            files.append({"path": relative.as_posix(), "sha256": digest, "trials": len(targets)})
            totals[split] += len(targets)
        print(f"Etiquetas preparadas: {session}", flush=True)
    write_json(output_root / "manifest.json", {
        "format_version": 1,
        "sessions": sessions,
        "phones": cfg["datos"]["fonemas"],
        "blank_id": encoder.blank_id,
        "label_config": cfg["etiquetas"],
        "versions": {name: version(name) for name in ("g2p-en", "nltk", "inflect", "numpy", "regex")},
        "totals": totals,
        "files": files,
    })
    return totals


def main() -> int:
    """Carga la configuración y genera los archivos de etiquetas reproducibles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        totals = prepare_labels(cfg)
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Etiquetas listas: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
