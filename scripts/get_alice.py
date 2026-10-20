#!/usr/bin/env python3
"""Descarga y verifica el subconjunto de Alice fijado por el catálogo elegido."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.alice_catalog import load_catalog, verify_file  # noqa: E402


def main() -> int:
    """Descarga reanudable y verificación git-blob/SHA-256 de NEMAR v1.1.3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/alice_eeg"))
    parser.add_argument("--catalogo", type=Path, default=Path("docs/datasets/alice_eeg/catalogo.json"))
    parser.add_argument("--sujetos", nargs="+", help="limita la descarga a sujetos del catálogo y fuentes compartidas")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    catalog = load_catalog(ROOT / args.catalogo)
    if args.sujetos:
        allowed = catalog.get("subjects") or [catalog.get("subject")]
        if not set(args.sujetos).issubset(allowed):
            parser.error("hay sujetos fuera del catálogo")
        catalog["files"] = [e for e in catalog["files"] if not e["path"].startswith("sub-")
                            or e["path"].split("/")[0] in args.sujetos]
    if args.list:
        for item in catalog["files"]:
            print(f"{item['size']:>10}  {item['path']}")
        return 0
    root = args.dest if args.dest.is_absolute() else ROOT / args.dest
    try:
        root.mkdir(parents=True, exist_ok=True)
        pending = sum(item["size"] for item in catalog["files"] if not (root / item["path"]).exists())
        if not args.verify_only and shutil.disk_usage(root).free < pending * 2:
            raise ValueError("No cabe el subconjunto dejando margen para un parcial.")
        for item in catalog["files"]:
            path = root / item["path"]
            if not path.exists() and not args.verify_only:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".part")
                print(f"Descargando {item['path']} ({item['size'] / 1e6:.2f} MB)", flush=True)
                subprocess.run([
                    "curl", "--fail", "--location", "--show-error", "--retry", "2",
                    "--connect-timeout", "20", "--max-time", "300", "--continue-at", "-",
                    "--output", str(partial), item["url"],
                ], check=True)
                verify_file(partial, item)
                partial.replace(path)
            verify_file(path, item)
            print(f"Verificado: {item['path']}", flush=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
