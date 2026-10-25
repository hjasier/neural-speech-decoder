"""Fija una partición T12 sin entrenar ni cargar matrices neuronales.

    python scripts/freeze_speech_split.py --config configs/speech_comparacion.yaml

No sobrescribe particiones distintas. Copiar el SHA-256 impreso al YAML
antes de preparar señales. Las variantes reutilizan el mismo manifiesto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.speech_protocol import build_speech_protocol


def main() -> int:
    """Crea una partición nueva o comprueba que la existente es idéntica."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = build_speech_protocol(cfg)
    payload = (json.dumps(state, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path = Path(cfg["protocolo"]["manifiesto"])
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Ya existe una partición distinta; no se sobrescribe.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                      "totals": state["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
