#!/usr/bin/env python3
"""Descarga reanudable y verificación del catálogo fijado de Dryad.

    python scripts/get_data.py --list-all
    python scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz
    python scripts/get_data.py --dataset speech_2024_t15 --files t15_copyTask_neuralData.zip

Los manifiestos fijan publicación, tamaño y SHA-256 oficial. No extrae ni
convierte los datos; conserva los originales y los parciales para reanudar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROYECTO = Path(__file__).resolve().parents[1]
MANIFIESTOS = PROYECTO / "docs" / "datasets"


def sha256(ruta: Path) -> str:
    """Calcula el SHA-256 sin cargar todo el archivo en memoria."""
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_manifiesto(dataset: str) -> dict[str, Any]:
    """Lee los metadatos oficiales fijados de una publicación."""
    with (MANIFIESTOS / dataset / "manifest.json").open(encoding="utf-8") as archivo:
        return json.load(archivo)


def verificar(ruta: Path, fuente: dict[str, Any]) -> None:
    """Comprueba tamaño y SHA-256 contra los metadatos de Dryad."""
    if not ruta.is_file():
        raise ValueError(f"falta {ruta}")
    if ruta.stat().st_size != fuente["size_bytes"]:
        raise ValueError(f"tamaño incorrecto: {ruta}")
    if not fuente.get("sha256") or sha256(ruta) != fuente["sha256"]:
        raise ValueError(f"SHA-256 incorrecto o no declarado: {ruta}")


def descargar(fuente: dict[str, Any], carpeta: Path) -> Path:
    """Reanuda con curl en .part y publica solo tras verificar integridad."""
    nombre = fuente["name"]
    if Path(nombre).name != nombre or nombre in {".", ".."}:
        raise ValueError(f"nombre no permitido: {nombre}")
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / nombre
    if destino.exists():
        verificar(destino, fuente)
        print(f"ok: {nombre} ya descargado y verificado", flush=True)
        return destino
    parcial = destino.with_name(destino.name + ".part")
    pendiente = fuente["size_bytes"] - (parcial.stat().st_size if parcial.exists() else 0)
    if pendiente < 0:
        raise ValueError(f"el archivo parcial supera el tamaño esperado: {parcial}")
    if pendiente > 0:
        if shutil.which("curl") is None:
            raise ValueError("se necesita curl para descargar y reanudar")
        if shutil.disk_usage(carpeta).free < pendiente + 5 * (1 << 30):
            raise ValueError("espacio insuficiente: se reservan 5 GiB libres además de la descarga")
        print(f"descargando {nombre} ({fuente['size_bytes'] / 1e9:.2f} GB)", flush=True)
        resultado = subprocess.run([
            "curl", "--fail", "--location", "--show-error", "--continue-at", "-",
            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
            "--output", str(parcial), fuente["url"],
        ], check=False)
        if resultado.returncode:
            raise ValueError(
                f"descarga fallida de {nombre} (curl {resultado.returncode}); "
                "se conserva el parcial. Si Dryad devuelve 403, descarga desde su web "
                "y usa --verify-only; no se ha descargado un archivo válido."
            )
    verificar(parcial, fuente)
    parcial.rename(destino)
    print(f"ok: {nombre}, tamaño y SHA-256 oficiales verificados", flush=True)
    return destino


def main() -> int:
    """Lista, descarga o verifica los archivos de las publicaciones catalogadas."""
    datasets = sorted(p.parent.name for p in MANIFIESTOS.glob("*/manifest.json"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=datasets, default="speech_2023")
    parser.add_argument("--files", nargs="+", default=["competitionData.tar.gz"],
                        help="nombres exactos; 'all' selecciona toda la publicación")
    parser.add_argument("--dest", type=Path, default=PROYECTO / "data",
                        help="directorio base; cada publicación tiene su subcarpeta")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-all", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.list or args.list_all:
        for clave in datasets if args.list_all else [args.dataset]:
            catalogo = leer_manifiesto(clave)
            print(f"\n{clave}: {catalogo['title']} ({catalogo['publication_date']})")
            for fuente in catalogo["files"]:
                print(f"  {fuente['size_bytes'] / 1e9:7.3f} GB  {fuente['name']}")
            print(f"  Total: {sum(f['size_bytes'] for f in catalogo['files']) / 1e9:.3f} GB")
        return 0
    catalogo = leer_manifiesto(args.dataset)
    disponibles = {fuente["name"]: fuente for fuente in catalogo["files"]}
    nombres = list(disponibles) if args.files == ["all"] else list(dict.fromkeys(args.files))
    desconocidos = set(nombres) - disponibles.keys()
    if desconocidos:
        parser.error(f"archivos desconocidos: {', '.join(sorted(desconocidos))}; consulta --list")
    carpeta = args.dest / args.dataset
    try:
        if not args.verify_only:
            carpeta.mkdir(parents=True, exist_ok=True)
            pendiente = sum(max(0, disponibles[n]["size_bytes"] - (
                (carpeta / (n + ".part")).stat().st_size
                if (carpeta / (n + ".part")).exists() else 0
            )) for n in nombres if not (carpeta / n).exists())
            if shutil.disk_usage(carpeta).free < pendiente + 5 * (1 << 30):
                raise ValueError("la selección no cabe dejando 5 GiB libres; cambia --dest")
        for nombre in nombres:
            if args.verify_only:
                verificar(carpeta / nombre, disponibles[nombre])
                print(f"ok: {nombre}, tamaño y SHA-256 oficiales verificados")
            else:
                descargar(disponibles[nombre], carpeta)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
