#!/usr/bin/env python3
"""Prepara EEG de escucha y secuencias de fonemas para nuestro SpeechGRU + CTC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.alice_ctc import prepare_alice_ctc  # noqa: E402


def main() -> int:
    """Verifica originales y recursos; publica señales y manifiesto reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alice_ctc.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    manifest = prepare_alice_ctc(cfg, force=args.force)
    print(json.dumps({"destino": cfg["datos"]["preparados"], "secuencias": manifest["counts"],
                      "canales": len(manifest["channels"]), "excluidos": manifest["excluded_channels"],
                      "blank_id": manifest["blank_id"], "signature": manifest["signature"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
