"""Prepara el grafo oficial y compila su runtime CPU aislado, previsto para Colab.

python scripts/get_data.py --dataset speech_2023 --files languageModel.tar.gz
python scripts/prepare_speech_trigram.py --prepare-model --build-runtime

No entrena ni modifica el entorno de la GRU. Solo extrae TLG.fst y words.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Calcula el hash de un archivo por bloques, sin cargarlo en RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_model(spec: dict[str, Any], archive_path: Path) -> None:
    """Verifica el archivo publicado y extrae únicamente los recursos del trigrama."""
    catalog = json.loads((ROOT / "docs/datasets/speech_2023/manifest.json").read_text())
    entry = next(item for item in catalog["files"] if item["name"] == "languageModel.tar.gz")
    if spec["archivo_sha256"] != entry["sha256"]:
        raise ValueError("La configuración no identifica el archivo oficial del trigrama.")
    destination = Path(spec["modelo"])
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["archive_sha256"] != entry["sha256"] or any(
            digest(destination / name) != manifest["files"][name]["sha256"] for name in ("TLG.fst", "words.txt")
        ):
            raise ValueError("Ya existen recursos distintos o incompletos; se conservan para revisión.")
        print("Grafo ya preparado y verificado:", destination)
        return
    if archive_path.stat().st_size != entry["size_bytes"] or digest(archive_path) != entry["sha256"]:
        raise ValueError("El archivo del trigrama no supera el tamaño y SHA-256 oficiales.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or parts[-2] != "languageModel" or parts[-1] not in ("TLG.fst", "words.txt"):
                    continue
                name = parts[-1]
                if not member.isfile() or name in files:
                    raise ValueError("El archivo no contiene un único grafo regular del trigrama.")
                if shutil.disk_usage(staging).free < member.size + 5 * (1 << 30):
                    raise ValueError("El grafo no cabe conservando 5 GiB libres.")
                print("Extrayendo:", member.name, member.size, "bytes", flush=True)
                with archive.extractfile(member) as source, (staging / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                files[name] = {"source_member": member.name, "size_bytes": member.size,
                               "sha256": digest(staging / name)}
        if set(files) != {"TLG.fst", "words.txt"}:
            raise ValueError("Faltan languageModel/TLG.fst o languageModel/words.txt.")
        manifest = {"format_version": 1, "archive_sha256": entry["sha256"], "source_url": entry["url"],
                    "files": files, "graph_phones": spec["fonemas_grafo"]}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    print("Trigrama verificado:", destination)


def build_runtime(spec: dict[str, Any]) -> None:
    """Compila el binding original con Python 3.10 y libtorch 1.13.1 en otro entorno."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Falta uv; usa el notebook Colab, que instala la versión fijada.")
    prep = spec["preparacion"]
    sources = Path(prep["fuentes"]).resolve()
    sources.parent.mkdir(parents=True, exist_ok=True)
    if not sources.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", prep["repositorio"], str(sources)], check=True)
        subprocess.run(["git", "-C", str(sources), "checkout", "--detach", prep["commit"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(sources), "rev-parse", "HEAD"], text=True).strip()
    if revision != prep["commit"]:
        raise ValueError("Las fuentes existentes corresponden a otro commit; se conservan.")
    subprocess.run(["git", "-C", str(sources), "diff", "--exit-code", "HEAD"], check=True)
    python = Path(spec["python_runtime"]).absolute()
    environment_root = python.parent.parent
    if not python.exists():
        subprocess.run([uv, "venv", "--python", prep["python"], "--seed", str(environment_root)], check=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "CMAKE_BUILD_PARALLEL_LEVEL": str(spec["hilos_cpu"]),
                   "PATH": str(python.parent) + os.pathsep + os.environ["PATH"]}
    subprocess.run([uv, "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match",
                    "-r", str(ROOT / "requirements-trigram-runtime.txt")], env=environment, check=True)
    subprocess.run([str(python), "setup.py", "install"],
                   cwd=sources / "LanguageModelDecoder/runtime/server/x86", env=environment, check=True)
    probe = subprocess.check_output([str(python), "-c",
        "import json, platform, torch, numpy, lm_decoder; "
        "print(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),"
        "'numpy':numpy.__version__,'binding':lm_decoder.__file__}))"], env=environment, text=True)
    runtime = json.loads(probe)
    if runtime["torch"] != spec["torch_runtime"] or runtime["python"] != prep["python"]:
        raise ValueError("El runtime compilado no corresponde a las versiones declaradas.")
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True, env=environment)
    marker = {"source_commit": revision, "runtime": runtime, "pip_freeze": freeze,
              "requirements_sha256": digest(ROOT / "requirements-trigram-runtime.txt")}
    (environment_root / "runtime_build.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print("Runtime CPU preparado:", python)


def main() -> int:
    """Prepara solo los componentes solicitados; nunca carga datos ni pesos neuronales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/entrenos_online/speech_trigrama.yaml"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--build-runtime", action="store_true")
    args = parser.parse_args()
    if not (args.prepare_model or args.build_runtime):
        parser.error("Indica --prepare-model, --build-runtime o ambos.")
    spec = yaml.safe_load(args.config.read_text())
    if args.prepare_model:
        prepare_model(spec, args.archive or Path(spec["preparacion"]["archivo"]))
    if args.build_runtime:
        build_runtime(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
