"""Prepara el grafo oficial y compila su runtime CPU aislado, previsto para Colab.

python scripts/get_data.py --dataset speech_2023 --files languageModel.tar.gz
python scripts/prepare_speech_trigram.py --prepare-model --build-runtime

No entrena ni modifica el entorno de la GRU. Solo extrae TLG.fst y words.txt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def runtime_build_threads(sources: Path, threads: int) -> Iterator[None]:
    """Limita también el make de OpenFst y restaura después el CMake original."""
    path = sources / "LanguageModelDecoder/runtime/server/x86/CMakeLists.txt"
    original = path.read_bytes()
    text = original.decode("utf-8")
    command = "BUILD_COMMAND     make -j 4"
    if text.count(command) != 1:
        raise ValueError("El CMake oficial ha cambiado; no se aplica el límite de hilos.")
    try:
        path.write_text(text.replace(command, f"BUILD_COMMAND     make -j {threads}"), encoding="utf-8")
        yield
    finally:
        path.write_bytes(original)


@contextmanager
def mapped_binding(sources: Path, enabled: bool) -> Iterator[None]:
    """Añade el selector OpenFst al compilar y restaura después la fuente original."""
    if not enabled:
        yield
        return
    path = sources / "LanguageModelDecoder/runtime/server/x86/python/lm_decoder.cc"
    original = path.read_bytes()
    text = original.decode("utf-8")
    declaration = '#include "utils/log.h"'
    binding = '    m.def("DecodeNumpy", &DecodeNumpy);'
    if text.count(declaration) != 1 or text.count(binding) != 1:
        raise ValueError("El binding oficial ha cambiado; no se aplica el parche de lectura.")
    text = text.replace(declaration, declaration + '\n#include <fstream>\n#include <stdexcept>\n\nDECLARE_string(fst_read_mode);')
    text = text.replace(binding, binding + '''
    m.def("SetFstReadMode", [](const std::string &mode) {
        if (mode != "read" && mode != "map") {
            throw std::invalid_argument("Expected read or map");
        }
        FLAGS_fst_read_mode = mode;
    });
    m.def("WriteConstFst", [](const std::string &source, const std::string &destination) {
        std::unique_ptr<fst::Fst<fst::StdArc>> graph(fst::Fst<fst::StdArc>::Read(source));
        if (!graph) throw std::runtime_error("Cannot read source FST");
        std::ofstream output(destination, std::ios::out | std::ios::binary);
        fst::FstWriteOptions options(destination);
        options.align = true;
        // Write directly: constructing ConstFst(graph) would allocate another full graph.
        if (!fst::ConstFst<fst::StdArc>::WriteFst(*graph, output, options)) {
            throw std::runtime_error("Cannot write aligned ConstFst");
        }
        output.close();
        if (!output) throw std::runtime_error("Cannot close aligned ConstFst");
    });''')
    try:
        path.write_text(text, encoding="utf-8")
        yield
    finally:
        path.write_bytes(original)


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


def _probe_runtime(python: Path, spec: dict[str, Any], environment: dict[str, str]) -> dict[str, Any]:
    """Comprueba versiones e importación sin cargar el grafo ni instalar paquetes."""
    probe = subprocess.check_output([str(python), "-c",
        "import json, platform, torch, numpy, lm_decoder; "
        "print(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),"
        "'numpy':numpy.__version__,'binding':lm_decoder.__file__,'cuda_build':torch.version.cuda,"
        "'mapped_graphs':hasattr(lm_decoder,'SetFstReadMode'),"
        "'compact_writer':hasattr(lm_decoder,'WriteConstFst')}))"],
        env=environment, text=True, stderr=subprocess.STDOUT, timeout=spec["timeout_segundos"])
    runtime = json.loads(probe)
    if runtime["torch"] != spec["torch_runtime"] or runtime["python"] != spec["preparacion"]["python"]:
        raise ValueError("El runtime no corresponde a las versiones declaradas.")
    if runtime["cuda_build"] is not None:
        raise ValueError("El runtime debe usar PyTorch CPU.")
    if spec.get("lectura_grafo", "read") == "map" and not (runtime["mapped_graphs"] and runtime["compact_writer"]):
        raise ValueError("El runtime no incluye el selector de lectura map y el escritor ConstFst.")
    return runtime


def _verify_virtualenv(python: Path, spec: dict[str, Any], environment: dict[str, str]) -> None:
    """Evita instalar dependencias en un Python base o en un runtime restaurado."""
    probe = subprocess.check_output([str(python), "-c",
        "import json, platform, sys; "
        "print(json.dumps({'python':platform.python_version(),'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"],
        env=environment, text=True, stderr=subprocess.STDOUT, timeout=spec["timeout_segundos"])
    interpreter = json.loads(probe)
    if (interpreter["python"] != spec["preparacion"]["python"]
            or interpreter["prefix"] == interpreter["base_prefix"]
            or Path(interpreter["prefix"]).resolve() != python.parent.parent.resolve()):
        raise ValueError("El intérprete existente no es el venv aislado esperado; se conserva sin cambios.")


def build_runtime(spec: dict[str, Any]) -> None:
    """Reutiliza un runtime válido o compila el binding en un venv aislado."""
    prep = spec["preparacion"]
    python = Path(spec["python_runtime"]).absolute()
    environment_root = python.parent.parent
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "CMAKE_BUILD_PARALLEL_LEVEL": str(spec["hilos_cpu"]),
                   "OMP_NUM_THREADS": str(spec["hilos_cpu"]), "OPENBLAS_NUM_THREADS": str(spec["hilos_cpu"]),
                   "MKL_NUM_THREADS": str(spec["hilos_cpu"]), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PATH": str(python.parent) + os.pathsep + os.environ["PATH"]}
    for variable in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
    # gflags ya se compila con FetchContent. Evita que glog importe además
    # los targets del sistema, incluso al reintentar una configuración fallida.
    environment["CMAKE_ARGS"] = " ".join(filter(None, (
        environment.get("CMAKE_ARGS"), "-Ugflags_DIR -DCMAKE_DISABLE_FIND_PACKAGE_gflags=ON",
    )))
    if python.exists():
        try:
            _probe_runtime(python, spec, environment)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            if (environment_root / "runtime_restore.json").exists() or not (environment_root / "pyvenv.cfg").is_file():
                detail = error.output if isinstance(error, subprocess.CalledProcessError) else str(error)
                raise RuntimeError(
                    f"El runtime existente no supera la comprobación: {detail}\n"
                    "Se conserva sin instalar ni sobrescribir archivos. Restaura un respaldo válido "
                    "o elige otra ruta python_runtime en la configuración para compilar uno nuevo. "
                    "Consulta docs/instalacion_runtime_trigrama.md."
                ) from error
        else:
            print("Runtime CPU ya preparado y verificado:", python)
            return
        _verify_virtualenv(python, spec, environment)
    elif environment_root.exists():
        raise RuntimeError(
            f"Ya existe {environment_root}, pero falta su intérprete. Se conserva sin cambios; "
            "elige otra ruta python_runtime en la configuración."
        )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Falta uv; instálalo con: python -m pip install uv==0.9.4")
    sources = Path(prep["fuentes"]).resolve()
    sources.parent.mkdir(parents=True, exist_ok=True)
    if not sources.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", prep["repositorio"], str(sources)], check=True)
        subprocess.run(["git", "-C", str(sources), "checkout", "--detach", prep["commit"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(sources), "rev-parse", "HEAD"], text=True).strip()
    if revision != prep["commit"]:
        raise ValueError("Las fuentes existentes corresponden a otro commit; se conservan.")
    subprocess.run(["git", "-C", str(sources), "diff", "--exit-code", "HEAD"], check=True)
    if not python.exists():
        subprocess.run([uv, "venv", "--python", prep["python"], "--seed", str(environment_root)], env=environment, check=True)
        _verify_virtualenv(python, spec, environment)
    subprocess.run([uv, "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match",
                    "-r", str(ROOT / "requirements-trigram-runtime.txt")], env=environment, check=True)
    mapped = spec.get("lectura_grafo", "read") == "map"
    with runtime_build_threads(sources, spec["hilos_cpu"]), mapped_binding(sources, mapped):
        subprocess.run([str(python), "setup.py", "install"],
                       cwd=sources / "LanguageModelDecoder/runtime/server/x86", env=environment, check=True)
    runtime = _probe_runtime(python, spec, environment)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True, env=environment)
    marker = {"source_commit": revision, "runtime": runtime, "pip_freeze": freeze,
              "mapped_graph_patch": "const-map-v1" if mapped else None,
              "preparation_script_sha256": digest(Path(__file__)),
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
    try:
        if args.prepare_model:
            prepare_model(spec, args.archive or Path(spec["preparacion"]["archivo"]))
        if args.build_runtime:
            build_runtime(spec)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
