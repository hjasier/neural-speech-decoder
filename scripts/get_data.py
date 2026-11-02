#!/usr/bin/env python3
"""Descarga reanudable desde Drive con verificación del catálogo oficial.

    python scripts/get_data.py --list-all
    python scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz
    python scripts/get_data.py --files competitionData.tar.gz languageModel.tar.gz

Los manifiestos fijan publicación, tamaño y SHA-256 oficial. Extrae los
.tar.gz verificados y conserva los originales y los parciales para reanudar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import gdown
from filelock import FileLock, Timeout
from requests import RequestException

PROYECTO = Path(__file__).resolve().parents[1]
MANIFIESTOS = PROYECTO / "docs" / "datasets"


class _DescargaDriveError(ValueError):
    """La transferencia ha fallado; permite mostrar la descarga manual en la CLI."""


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
    """Comprueba tamaño y SHA-256 contra los metadatos del catálogo."""
    if not ruta.is_file():
        raise ValueError(f"falta {ruta}")
    if ruta.stat().st_size != fuente["size_bytes"]:
        raise ValueError(f"tamaño incorrecto: {ruta}")
    if not fuente.get("sha256") or sha256(ruta) != fuente["sha256"]:
        raise ValueError(f"SHA-256 incorrecto o no declarado: {ruta}")


def _url_drive(fuente: dict[str, Any]) -> str:
    for item in fuente.get("mirrors", []):
        if item["provider"] == "google_drive":
            return item["url"]
    raise ValueError(f"No hay enlace de Drive configurado para {fuente['name']}; consulta --list.")


def _instrucciones_manuales(catalogo: dict[str, Any], base: Path) -> str:
    disponibles = [item for item in catalogo["files"] if any(
        mirror["provider"] == "google_drive" for mirror in item.get("mirrors", [])
    )]
    destino = base.relative_to(PROYECTO) if base.is_relative_to(PROYECTO) else base
    cantidad = "estos dos archivos" if len(disponibles) == 2 else "estos archivos"
    lines = [f"Descarga {cantidad} de Drive y déjalos en {destino}/:"]
    for item in disponibles:
        lines.extend([f"  {item['name']}", f"  {_url_drive(item)}"])
    lines.append("Después de dejarlos en esa carpeta, vuelve a ejecutar el mismo comando para verificarlos y descomprimirlos.")
    return "\n".join(lines)


def _parcial(carpeta: Path, nombre: str) -> Path | None:
    staging = carpeta / f".{nombre}.download"
    candidates = [carpeta / f"{nombre}.part", staging / "payload", *sorted(staging.glob("payload????????.part"))]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Hay varios parciales para {nombre}; se conservan. Usa --restart para empezar de cero.")
    return existing[0] if existing else None


def _pendiente(
    fuente: dict[str, Any], carpeta: Path, reiniciar: bool, archivo_local: Path | None = None,
) -> int:
    if (carpeta / fuente["name"]).exists() or (archivo_local is not None and archivo_local.is_file()):
        return 0
    partial = None if reiniciar else _parcial(carpeta, fuente["name"])
    size = partial.stat().st_size if partial else 0
    if size > fuente["size_bytes"]:
        raise ValueError(f"El parcial de {fuente['name']} supera el tamaño esperado; usa --restart.")
    return fuente["size_bytes"] - size


def _descargar_bloqueado(
    fuente: dict[str, Any], carpeta: Path, *, reintentos: int,
    timeout: float, reserva: int, reiniciar: bool, archivo_local: Path | None,
) -> Path:
    nombre = fuente["name"]
    destino = carpeta / nombre
    if destino.exists():
        verificar(destino, fuente)
        print(f"ok: {nombre} ya descargado y verificado", flush=True)
        return destino
    if archivo_local is not None and archivo_local.is_file():
        verificar(archivo_local, fuente)
        destino.symlink_to(os.path.relpath(archivo_local.resolve(), carpeta.resolve()))
        print(f"ok: {nombre} descargado manualmente, verificado y preparado", flush=True)
        return destino
    url = _url_drive(fuente)
    staging = carpeta / f".{nombre}.download"
    legacy = carpeta / f"{nombre}.part"
    if reiniciar:
        if staging.exists():
            shutil.rmtree(staging)
        legacy.unlink(missing_ok=True)
    pendiente = _pendiente(fuente, carpeta, False)
    if pendiente and shutil.disk_usage(carpeta).free < pendiente + reserva:
        raise ValueError("Espacio insuficiente para la descarga y la reserva; cambia --dest.")
    parcial = _parcial(carpeta, nombre)
    if parcial and pendiente == 0:
        verificar(parcial, fuente)
        parcial.replace(destino)
        print(f"ok: {nombre}, parcial completo verificado", flush=True)
        return destino
    staging.mkdir(exist_ok=True)
    payload = staging / "payload"
    if legacy.is_file():
        # gdown reconoce sus temporales por ocho caracteres entre nombre y .part.
        legacy.replace(staging / "payloadresume00.part")

    def progreso(actual: int, total: int | None) -> None:
        if actual > fuente["size_bytes"] or (total is not None and total != fuente["size_bytes"]):
            raise ValueError(f"El tamaño remoto no coincide con el catálogo de {nombre}; no se publica la descarga.")

    print(f"descargando {nombre} desde Drive ({fuente['size_bytes'] / 1e9:.2f} GB)", flush=True)
    try:
        result = gdown.download(url=url, output=str(payload), quiet=False, resume=True,
                                use_cookies=False, retries=reintentos, timeout=timeout, progress=progreso)
        if result is None:
            raise gdown.DownloadError("No se ha recibido el archivo.")
    except (gdown.DownloadError, RequestException) as error:
        message = ("Drive bloquea la descarga automática." if isinstance(error, gdown.FileURLRetrievalError)
                   else f"No se ha podido completar la descarga de {nombre} desde Drive: {error}. "
                        "Se conservan los parciales para reanudar.")
        raise _DescargaDriveError(message) from error
    try:
        verificar(payload, fuente)
    except ValueError as error:
        raise ValueError(f"{error}; se conserva {staging}. Usa --restart para descartar el parcial.") from error
    payload.replace(destino)
    print(f"ok: {nombre}, tamaño y SHA-256 oficiales verificados", flush=True)
    return destino


def descargar(
    fuente: dict[str, Any], carpeta: Path, *, reintentos: int = 3,
    timeout: float = 30, reserva: int = 5 * (1 << 30), reiniciar: bool = False,
    archivo_local: Path | None = None,
) -> Path:
    """Descarga con gdown, conserva parciales y publica solo tras verificar el SHA-256.

    Solo usa enlaces públicos de Drive. No lee cookies personales ni
    sobrescribe originales existentes. Un archivo local válido se enlaza a
    la carpeta del dataset conservando el original, sin duplicarlo en disco.
    """
    nombre = fuente["name"]
    if Path(nombre).name != nombre or nombre in {".", ".."} or "\\" in nombre:
        raise ValueError(f"nombre no permitido: {nombre}")
    carpeta.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(carpeta / f".{nombre}.lock"), timeout=0):
            return _descargar_bloqueado(fuente, carpeta, reintentos=reintentos,
                                       timeout=timeout, reserva=reserva, reiniciar=reiniciar,
                                       archivo_local=archivo_local)
    except Timeout as error:
        raise ValueError(f"Ya hay otra descarga de {nombre} en esta carpeta.") from error


def _extraer_tar(ruta: Path, carpeta: Path, reserva: int) -> None:
    """Extrae un tar.gz ya verificado, conservando los archivos que existan."""
    base = carpeta.resolve()
    nuevos: list[Path] = []
    directorios: set[Path] = set()
    vistos: set[Path] = set()
    print(f"descomprimiendo {ruta.name}", flush=True)
    with TemporaryDirectory(prefix=".extraccion_", dir=base) as temporal:
        staging = Path(temporal)
        with tarfile.open(ruta, "r|gz") as archivo:
            for miembro in archivo:
                relativa = PurePosixPath(miembro.name)
                if relativa.is_absolute() or ".." in relativa.parts or "\\" in miembro.name:
                    raise ValueError(f"Ruta no permitida en {ruta.name}: {miembro.name}")
                destino = base.joinpath(*relativa.parts)
                if not destino.resolve().is_relative_to(base):
                    raise ValueError(f"La ruta sale de la carpeta de datos: {miembro.name}")
                if miembro.isdir():
                    if destino.exists() and not destino.is_dir():
                        raise ValueError(f"Ya existe un archivo donde debe ir una carpeta: {destino}")
                    directorios.add(destino)
                    continue
                if not miembro.isfile() or destino in vistos:
                    raise ValueError(f"Entrada no permitida o repetida en {ruta.name}: {miembro.name}")
                vistos.add(destino)
                existente = destino.exists()
                if existente and (not destino.is_file() or destino.stat().st_size != miembro.size):
                    raise ValueError(f"El archivo existente es distinto y se conserva: {destino}")
                if not existente and shutil.disk_usage(base).free < miembro.size + reserva:
                    raise ValueError("Espacio insuficiente para descomprimir conservando la reserva; cambia --dest.")
                with archivo.extractfile(miembro) as origen:
                    if existente:
                        with destino.open("rb") as actual:
                            for bloque in iter(lambda: origen.read(1 << 20), b""):
                                if actual.read(len(bloque)) != bloque:
                                    raise ValueError(f"El archivo existente es distinto y se conserva: {destino}")
                    else:
                        parcial = staging.joinpath(*relativa.parts)
                        parcial.parent.mkdir(parents=True, exist_ok=True)
                        with parcial.open("xb") as salida:
                            shutil.copyfileobj(origen, salida, length=1 << 20)
                        nuevos.append(parcial.relative_to(staging))
        # Publicar solo archivos completos; los originales existentes no se sobrescriben.
        for directorio in sorted(directorios, key=lambda path: len(path.parts)):
            directorio.mkdir(parents=True, exist_ok=True)
        for relativa in nuevos:
            destino = base / relativa
            destino.parent.mkdir(parents=True, exist_ok=True)
            if not destino.parent.resolve().is_relative_to(base):
                raise ValueError(f"La ruta sale de la carpeta de datos: {relativa}")
            os.link(staging / relativa, destino)
    print(f"ok: {ruta.name} descomprimido ({len(nuevos)} archivos nuevos, {len(vistos) - len(nuevos)} ya existentes)", flush=True)


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
    parser.add_argument("--retries", type=int, default=3, help="reintentos por cortes de red")
    parser.add_argument("--timeout", type=float, default=30, help="segundos máximos sin respuesta")
    parser.add_argument("--reserve-gib", type=float, default=5, help="espacio libre que se conserva")
    parser.add_argument("--restart", action="store_true", help="descarta solo los parciales y empieza de cero")
    args = parser.parse_args()
    if (args.retries < 0 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0):
        parser.error("retries/reserve-gib deben ser no negativos y timeout debe ser positivo; usa valores finitos")
    if args.restart and args.verify_only:
        parser.error("--restart no se combina con --verify-only")
    if args.list or args.list_all:
        for clave in datasets if args.list_all else [args.dataset]:
            catalogo = leer_manifiesto(clave)
            print(f"\n{clave}: {catalogo['title']} ({catalogo['publication_date']})")
            for fuente in catalogo["files"]:
                available = any(item["provider"] == "google_drive" for item in fuente.get("mirrors", []))
                status = "Drive" if available else "sin enlace de Drive"
                print(f"  {fuente['size_bytes'] / 1e9:7.3f} GB  {fuente['name']} [{status}]")
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
            pendiente = sum(_pendiente(disponibles[n], carpeta, args.restart, args.dest / n) for n in nombres)
            if pendiente and shutil.disk_usage(carpeta).free < pendiente + args.reserve_gib * (1 << 30):
                raise ValueError(f"la selección no cabe dejando {args.reserve_gib:g} GiB libres; cambia --dest")
        for nombre in nombres:
            if args.verify_only:
                verificar(carpeta / nombre, disponibles[nombre])
                print(f"ok: {nombre}, tamaño y SHA-256 oficiales verificados")
            else:
                ruta = descargar(disponibles[nombre], carpeta, reintentos=args.retries,
                                 timeout=args.timeout, reserva=int(args.reserve_gib * (1 << 30)),
                                 reiniciar=args.restart, archivo_local=args.dest / nombre)
                if nombre.endswith(".tar.gz"):
                    with FileLock(str(carpeta / f".{nombre}.lock"), timeout=0):
                        _extraer_tar(ruta, carpeta, int(args.reserve_gib * (1 << 30)))
    except _DescargaDriveError as error:
        print(f"error: {error}\n{_instrucciones_manuales(catalogo, args.dest)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Proceso interrumpido; vuelve a ejecutar el comando para continuar.", file=sys.stderr)
        return 130
    except (OSError, ValueError, tarfile.TarError, EOFError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
