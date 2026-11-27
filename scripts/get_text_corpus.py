"""Descarga y verifica el shard de texto externo de los pares sintéticos.

    python scripts/get_text_corpus.py --config configs/final/pares_sinteticos.yaml
    python scripts/get_text_corpus.py --config configs/final/pares_sinteticos.yaml --verify-only

El archivo queda en data/, ignorado por Git, fijado por commit y SHA-256.
No descarga de nuevo si ya existe; solo comprueba su hash.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

import yaml

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    """SHA-256 de un archivo leído por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str, timeout: float) -> None:
    """Descarga a un .part, comprueba el hash y solo entonces lo renombra."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urlopen(url, timeout=timeout) as response, partial.open("wb") as stream:
        for block in iter(lambda: response.read(CHUNK), b""):
            digest.update(block)
            stream.write(block)
    if digest.hexdigest() != expected:
        partial.unlink()
        raise ValueError(f"El SHA-256 descargado no coincide con el fijado: {url}")
    partial.replace(target)


def main() -> int:
    """Descarga el shard si falta y verifica siempre su hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/pares_sinteticos.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    corpus = yaml.safe_load(args.config.read_text(encoding="utf-8"))["corpus"]
    target = Path(corpus["archivo"])
    try:
        if not target.exists():
            if args.verify_only:
                raise FileNotFoundError(f"Falta {target}.")
            print(f"Descargando {corpus['url']} …", flush=True)
            download(corpus["url"], target, corpus["sha256"], corpus["timeout_descarga"])
        if sha256_of(target) != corpus["sha256"]:
            raise ValueError(f"El SHA-256 no coincide: {target}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Corpus verificado: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
