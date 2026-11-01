"""Restaura el runtime CPU de Colab sin compilar ni cargar el grafo.

Verifica el respaldo y reubica sus enlaces dentro de un directorio nuevo.
El lanzador conserva el Python y los paquetes aislados del trigrama. La prueba
final solo importa el binding; la inferencia local se comprueba por separado.

Desde la raíz del repositorio:
    python scripts/restore_speech_trigram_runtime.py --backup CARPETA_RESPALDO --sha256 HASH_DE_COLAB
Añadir --verify-only para comprobar la descarga sin extraer ni ejecutar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from typing import Any

import yaml


PROBE = r'''
import json, platform, sys
from pathlib import Path
import numpy, torch, lm_decoder
print(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
    "numpy": numpy.__version__, "cuda_build": torch.version.cuda,
    "mapped_graphs": hasattr(lm_decoder, "SetFstReadMode"),
    "compact_writer": hasattr(lm_decoder, "WriteConstFst"), "machine": platform.machine(),
    "modules": {"numpy": numpy.__file__, "torch": torch.__file__, "lm_decoder": lm_decoder.__file__},
    "maps": Path("/proc/self/maps").read_text()}))
'''


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member_path(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2
            or relative.parts[0] not in ("rootfs", "evidence") or str(relative) != name):
        raise ValueError(f"Ruta no válida en el respaldo: {name}")
    return relative


def _rootfs_path(original: str) -> PurePosixPath:
    path = PurePosixPath(original)
    if not path.is_absolute() or ".." in path.parts or str(path) != original:
        raise ValueError(f"Ruta de origen no válida: {original}")
    return PurePosixPath("rootfs", *path.parts[1:])


def verify_backup(backup: Path, expected_sha256: str | None) -> tuple[dict[str, Any], Path]:
    """Comprueba tamaño y SHA-256 del paquete antes de extraer archivos."""
    inventory = json.loads((backup / "respaldo_verificado.json").read_text())
    if inventory.get("format_version") != 1 or inventory.get("verified") is not True:
        raise ValueError("El respaldo no tiene el formato verificado esperado.")
    name = inventory["archive"]
    if Path(name).name != name or name != "runtime_trigrama_cpu.tar.gz":
        raise ValueError("Nombre de archivo de respaldo no válido.")
    archive = backup / name
    expected = inventory["archive_sha256"]
    sidecar = (backup / (name + ".sha256")).read_text().split()
    if (not sidecar or sidecar[0] != expected or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or expected_sha256 is not None and expected_sha256 != expected):
        raise ValueError("No coinciden los hashes declarados del respaldo.")
    if archive.stat().st_size != inventory["archive_size_bytes"] or _digest(archive) != expected:
        raise ValueError("El respaldo está incompleto o su SHA-256 no coincide; conservar la descarga.")
    for member, record in inventory["files"].items():
        _member_path(member)
        if record["type"] not in ("file", "symlink"):
            raise ValueError(f"Tipo de entrada no válido: {member}")
        if record["type"] == "file" and (type(record["size_bytes"]) is not int or record["size_bytes"] < 0):
            raise ValueError(f"Tamaño no válido: {member}")
    return inventory, archive


def extract_backup(archive: Path, records: dict[str, Any], destination: Path) -> None:
    """Extrae archivos verificados y transforma los enlaces de Colab a rutas internas."""
    destination.mkdir(exist_ok=False)
    links: list[tuple[Path, PurePosixPath]] = []
    seen: set[str] = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            relative = _member_path(member.name)
            if member.name not in records or member.name in seen:
                raise ValueError(f"Entrada inesperada o repetida: {member.name}")
            seen.add(member.name)
            record = records[member.name]
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if record["type"] == "symlink":
                if not member.issym() or member.linkname != record["target"] or relative.parts[0] != "rootfs":
                    raise ValueError(f"Enlace distinto en el respaldo: {member.name}")
                original = PurePosixPath(member.linkname)
                linked = (_rootfs_path(str(original)) if original.is_absolute()
                          else _member_path(posixpath.normpath(str(relative.parent / original))))
                if linked.parts[0] != "rootfs":
                    raise ValueError(f"El enlace sale del runtime: {member.name}")
                links.append((target, linked))
                continue
            if not member.isfile() or member.size != record["size_bytes"]:
                raise ValueError(f"Archivo distinto en el respaldo: {member.name}")
            digest = hashlib.sha256()
            with stream.extractfile(member) as source, target.open("xb") as output:
                while block := source.read(1 << 20):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 distinto en el respaldo: {member.name}")
            target.chmod(member.mode & 0o755)
    if seen != set(records):
        raise ValueError("Faltan archivos del inventario en el respaldo.")
    # Crear los enlaces al final impide escribir a través de ellos al extraer.
    for target, linked in links:
        target.symlink_to(os.path.relpath(destination.joinpath(*linked.parts), target.parent))
    for target, _ in links:
        if not target.resolve().is_relative_to(destination / "rootfs") or not target.exists():
            raise ValueError(f"Enlace fuera del runtime o sin destino: {target.relative_to(destination)}")


def build_launcher(destination: Path, inventory: dict[str, Any], threads: int) -> Path:
    """Genera un ejecutable CPU que usa el Python y las bibliotecas reubicados."""
    probe = inventory["probe"]
    base = _rootfs_path(probe["base_prefix"])
    prefix = _rootfs_path(probe["prefix"])
    executable = _rootfs_path(probe["executable"])
    binding = _rootfs_path(probe["binding"])
    version = ".".join(probe["python"].split(".")[:2])
    site = prefix / "lib" / ("python" + version) / "site-packages"
    if not all(destination.joinpath(*path.parts).exists() for path in (base, site, executable, binding)):
        raise ValueError("Faltan el Python, los paquetes o el binding dentro del respaldo.")
    native = [site / "torch/lib", base / "lib"]
    for name, record in sorted(inventory["files"].items()):
        if record["type"] != "file" or ".so" not in PurePosixPath(name).name:
            continue
        relative = _member_path(name)
        # Las bibliotecas del sistema de Colab quedan como evidencia; el host usa su libc.
        if relative.parts[0] == "rootfs" and relative.parts[1] not in ("usr", "lib", "lib64", "bin", "sbin"):
            native.append(relative.parent)
    native = list(dict.fromkeys(path for path in native if destination.joinpath(*path.parts).is_dir()))

    def shell_path(path: PurePosixPath) -> str:
        return '"$runtime_root"/' + shlex.quote(str(path))

    variables = {"PYTHONHOME": shell_path(base),
                 "PYTHONPATH": ":".join(shell_path(path) for path in dict.fromkeys((site, binding.parent))),
                 "LD_LIBRARY_PATH": ":".join(shell_path(path) for path in native)}
    launcher = destination / "bin/python"
    launcher.parent.mkdir(exist_ok=True)
    lines = ["#!/bin/sh", "set -eu", 'runtime_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)',
             "export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1",
             f"export OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} MKL_NUM_THREADS={threads}"]
    lines.extend(f"export {name}={value}" for name, value in variables.items())
    lines.append("exec " + shell_path(executable) + ' "$@"')
    launcher.write_text("\n".join(lines) + "\n")
    launcher.chmod(0o755)
    return launcher


def probe_runtime(launcher: Path, inventory: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Importa el binding sin abrir el grafo y detecta dependencias que sigan apuntando fuera."""
    result = subprocess.run([str(launcher.resolve()), "-c", PROBE], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=spec["timeout_segundos"])
    if result.returncode:
        raise RuntimeError(f"No se puede importar el runtime restaurado (código {result.returncode}):\n{result.stderr[-8000:]}")
    actual = json.loads(result.stdout)
    expected = inventory["probe"]
    if (any(actual[key] != expected[key] for key in ("python", "torch", "numpy", "machine"))
            or actual["torch"] != spec["torch_runtime"] or actual["cuda_build"] is not None
            or not actual["mapped_graphs"] or not actual["compact_writer"]):
        raise ValueError("Las versiones o capacidades del runtime restaurado no coinciden.")
    root = launcher.resolve().parent.parent / "rootfs"
    if any(not Path(path).resolve().is_relative_to(root) for path in actual["modules"].values()):
        raise ValueError("Se han importado módulos fuera del runtime restaurado.")
    loaded = []
    for line in actual.pop("maps").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            path = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[5])).resolve()
            if not path.is_relative_to(root) and path.parts[1] not in ("usr", "lib", "lib64"):
                raise ValueError(f"El runtime ha cargado una biblioteca externa: {path}")
            loaded.append(str(path))
    return {**actual, "loaded_libraries": sorted(set(loaded))}


def restore_runtime(backup: Path, spec: dict[str, Any], expected_sha256: str | None, verify_only: bool) -> None:
    """Verifica el respaldo y prepara otro directorio, sin modificar un runtime existente."""
    inventory, archive = verify_backup(backup, expected_sha256)
    print("SHA-256 del respaldo verificado:", inventory["archive_sha256"], flush=True)
    if verify_only:
        return
    if platform.system() != "Linux" or platform.machine() != inventory["probe"]["machine"]:
        raise ValueError("El respaldo requiere Linux y la misma arquitectura de CPU.")
    if type(spec["hilos_cpu"]) is not int or spec["hilos_cpu"] <= 0:
        raise ValueError("hilos_cpu debe ser un entero positivo.")
    destination = Path(spec["python_runtime"]).absolute().parent.parent
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"El destino ya existe y se conserva: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(record.get("size_bytes", 0) for record in inventory["files"].values())
    reserve = spec["compactacion"]["reserva_disco_gib"] * (1 << 30)
    if shutil.disk_usage(destination.parent).free < required + reserve:
        raise ValueError("No cabe el runtime conservando la reserva de disco configurada.")
    with TemporaryDirectory(prefix=".trigram_restore_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "runtime"
        extract_backup(archive, inventory["files"], staging)
        build_launcher(staging, inventory, spec["hilos_cpu"])
        staging.rename(destination)
    report = {"scope": "runtime_import_only; sin cargar el grafo ni el checkpoint",
              "runtime_import_verified": False, "local_demo_verified": False,
              "archive_sha256": inventory["archive_sha256"], "backup": str(backup.resolve())}
    try:
        report["runtime"] = probe_runtime(destination / "bin/python", inventory, spec)
        report["runtime_import_verified"] = True
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        raise
    finally:
        (destination / "runtime_restore.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Runtime CPU preparado:", destination / "bin/python")
    print("Importación verificada; queda pendiente la prueba de GRU + trigrama en CPU.")


def main() -> int:
    """Restaura únicamente el entorno aislado; no instala ni compila dependencias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/entrenos_online/speech_trigrama_compacto.yaml"))
    parser.add_argument("--sha256", help="SHA-256 mostrado por Colab, además del inventario y su fichero .sha256.")
    parser.add_argument("--verify-only", action="store_true", help="Comprueba la descarga sin extraer ni ejecutar el runtime.")
    args = parser.parse_args()
    try:
        restore_runtime(args.backup, yaml.safe_load(args.config.read_text()), args.sha256, args.verify_only)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
