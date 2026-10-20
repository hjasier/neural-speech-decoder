"""Integridad del subconjunto fijado de NEMAR, sin cargar señales."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, Any]:
    """Lee el catálogo versionado y rechaza rutas fuera del destino."""
    catalog = json.loads(path.read_text(encoding="utf-8"))
    for item in catalog["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Ruta no permitida: {relative}")
        if item["checksum_algorithm"] not in {"git", "sha256"}:
            raise ValueError("Algoritmo de integridad desconocido.")
    return catalog


def verify_file(path: Path, item: dict[str, Any]) -> str:
    """Comprueba tamaño y hash oficial; devuelve además SHA-256 del contenido."""
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise ValueError(f"Falta el archivo o su tamaño no coincide: {path}")
    official = (hashlib.sha256() if item["checksum_algorithm"] == "sha256"
                else hashlib.sha1(f"blob {item['size']}\0".encode()))
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            official.update(chunk)
            sha256.update(chunk)
    if official.hexdigest() != item["checksum"]:
        raise ValueError(f"El hash oficial no coincide: {path}")
    return sha256.hexdigest()


def verify_catalog(root: Path, catalog: dict[str, Any]) -> dict[str, str]:
    """Verifica los archivos científicos usados por el piloto y su procedencia."""
    return {item["path"]: verify_file(root / item["path"], item)
            for item in catalog["files"]}
