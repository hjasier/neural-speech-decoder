"""Genera una partición T12 nueva por bloques y muestra el hash que fija la configuración.

    python scripts/build_speech_protocol.py --config configs/final/t12_causal.yaml

Solo lee blockIdx y los hashes de los .mat de las sesiones configuradas; no usa
señales, textos ni resultados. Nunca sobrescribe un manifiesto existente: una
partición publicada queda congelada y una nueva necesita otro nombre de archivo.
El SHA-256 impreso se copia a protocolo.sha256 de las configuraciones que la usan.
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


def write_protocol(cfg: dict, *, overwrite: bool = False) -> tuple[Path, str, dict[str, int]]:
    """Escribe el manifiesto de la configuración y devuelve ruta, hash y recuentos."""
    path = Path(cfg["protocolo"]["manifiesto"])
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe {path}; una partición fijada no se regenera.")
    protocol = build_speech_protocol(cfg)
    payload = (json.dumps(protocol, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path, hashlib.sha256(payload).hexdigest(), protocol["totals"]


def main() -> int:
    """Lee sesiones, método y fracción del YAML y escribe el manifiesto nuevo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        path, digest, totals = write_protocol(cfg)
    except (OSError, ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Partición: {path}\nsha256: {digest}\nrecuentos: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
