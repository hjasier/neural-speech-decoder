"""Descarga y verifica los recursos NLTK fijados para g2p-en.

    python scripts/get_phoneme_resources.py --config configs/speech_original.yaml
    python scripts/get_phoneme_resources.py --config configs/speech_original.yaml --verify-only

No descarga señales ni entrena. Los recursos quedan dentro de data/, ignorado
por Git. Después de este paso, la conversión a fonemas funciona sin red.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phonemes import verify_phoneme_resources


def main() -> int:
    """Descarga solo recursos ausentes, comprueba hashes y extrae los ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["etiquetas"]["recursos"])
    try:
        if not args.verify_only:
            for resource in cfg["etiquetas"]["recursos_nltk"]:
                archive = root / resource["archivo"]
                if archive.exists():
                    payload = archive.read_bytes()
                else:
                    with urlopen(resource["url"], timeout=cfg["etiquetas"]["timeout_descarga"]) as response:
                        payload = response.read()
                if hashlib.sha256(payload).hexdigest() != resource["sha256"]:
                    raise ValueError(f"El SHA-256 no coincide: {archive}")
                if not archive.exists():
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    partial = archive.with_suffix(".zip.part")
                    partial.write_bytes(payload)
                    partial.replace(archive)
                with ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        relative = Path(member.filename)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError(f"Ruta no válida en el ZIP: {member.filename}")
                    bundle.extractall(archive.parent)
                print(f"Verificado y extraído: {resource['archivo']}", flush=True)
        verify_phoneme_resources(cfg)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("Recursos de fonemas preparados y verificados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
