"""Convierte el trigrama oficial a ConstFst alineado y comprueba su carga map.

La conversión usa CPU con RAM alta (Colab); nunca se carga el original en
la máquina de la demo. Los pesos, símbolos y arcos los conserva OpenFst.
El archivo en disco puede crecer: se reduce la necesidad de cargarlo entero.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_speech_trigram import build_runtime, digest
from src.inference.fst import read_fst_header, require_mappable_fst


def available_memory() -> int:
    """Obtiene MemAvailable del host Linux en bytes, sin consultar CUDA."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("No se pudo consultar la memoria disponible.")


def cpu_environment(spec: dict[str, Any]) -> dict[str, str]:
    """Oculta CUDA y limita a los hilos CPU declarados en configuración."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(spec["hilos_cpu"])
    return env


def openfst_tools(spec: dict[str, Any]) -> Path:
    """Localiza los binarios ya compilados por el runtime del commit fijado."""
    sources = Path(spec["preparacion"]["fuentes"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(sources), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["preparacion"]["commit"]:
        raise ValueError("Las fuentes del runtime no corresponden al commit declarado.")
    subprocess.run(["git", "-C", str(sources), "diff", "--exit-code", "HEAD"], check=True)
    tools = sources / "LanguageModelDecoder/runtime/server/x86/fc_base/openfst-subbuild/openfst-populate-prefix/bin"
    if not all((tools / name).is_file() for name in ("fstcompile", "fstequal")):
        raise RuntimeError("Faltan los binarios OpenFst del runtime; ejecuta --build-runtime en Colab.")
    return tools


def write_command(spec: dict[str, Any], source: Path, destination: Path) -> list[str]:
    """Escribe ConstFst directamente, evitando una segunda copia completa en RAM."""
    return [str(Path(spec["python_runtime"]).absolute()), "-c",
            "import sys, torch, lm_decoder; lm_decoder.WriteConstFst(sys.argv[1], sys.argv[2])",
            str(source), str(destination)]


def check_converter(spec: dict[str, Any], tools: Path, directory: Path, env: dict[str, str]) -> None:
    """Comprueba símbolos, pesos, arcos repetidos y finales con un FST pequeño."""
    text = directory / "prueba.txt"
    text.write_text("0 1 1 7 0.25\n0 1 2 8 1.5\n1 2 0 0 0.75\n2 3 1 9 0.125\n3 0.5\n")
    original, mapped = directory / "prueba_vector.fst", directory / "prueba_const.fst"
    subprocess.run([str(tools / "fstcompile"), str(text), str(original)], env=env, check=True)
    subprocess.run(write_command(spec, original, mapped), env=env, check=True)
    require_mappable_fst(mapped)
    subprocess.run([str(tools / "fstequal"), "--fst_read_mode=map", str(original), str(mapped)],
                   env=env, check=True)


def prepare_compact_model(spec: dict[str, Any]) -> None:
    """Publica atómicamente otra representación del grafo, sin sustituir el original."""
    source, destination = Path(spec["compactacion"]["origen"]).resolve(), Path(spec["modelo"]).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Origen y destino deben ser carpetas independientes.")
    if destination.exists():
        raise FileExistsError(f"Se conserva el destino existente: {destination}")
    minimum = spec["compactacion"]["minimo_disponible_preparacion_gib"] * (1 << 30)
    if available_memory() < minimum:
        raise RuntimeError("La conversión requiere RAM alta; ejecútala en Colab, no en la máquina de la demo.")
    manifest_path = source / "manifest.json"
    original = json.loads(manifest_path.read_text())
    if (original["archive_sha256"] != spec["archivo_sha256"]
            or original["graph_phones"] != spec["fonemas_grafo"] or original.get("transformation")):
        raise ValueError("Se necesita el trigrama original verificado y su vocabulario original.")
    header = read_fst_header(source / "TLG.fst")
    if header["fst_type"] != "vector" or header["arc_type"] != "standard" or header["version"] != 2:
        raise ValueError("El grafo de origen no es el VectorFst estándar publicado.")
    tools, env = openfst_tools(spec), cpu_environment(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # VectorFst serializa 12 bytes/estado; ConstFst estándar usa 20, más alineación.
    expected = (source / "TLG.fst").stat().st_size + 8 * header["states"] + 4096
    reserve = spec["compactacion"]["reserva_disco_gib"] * (1 << 30)
    if shutil.disk_usage(destination.parent).free < expected + reserve:
        raise RuntimeError("No hay espacio para el grafo compacto conservando la reserva de disco.")
    print("Verificando el origen antes de convertir; el original se conserva.", flush=True)
    stamps = {}
    for name in ("TLG.fst", "words.txt"):
        if digest(source / name) != original["files"][name]["sha256"]:
            raise ValueError(f"El recurso de origen ha cambiado: {name}")
        stamps[name] = (source / name).stat()
    with TemporaryDirectory(dir=destination.parent, prefix="compactacion_") as temporary:
        staging = Path(temporary)
        check_converter(spec, tools, staging, env)
        command = write_command(spec, source / "TLG.fst", staging / "TLG.fst")
        print("Convirtiendo el grafo en CPU; esta preparación necesita RAM alta.", flush=True)
        subprocess.run(command, env=env, check=True)
        converted = require_mappable_fst(staging / "TLG.fst")
        if (converted["states"], converted["start"]) != (header["states"], header["start"]):
            raise ValueError("La conversión cambió los estados o el estado inicial.")
        shutil.copyfile(source / "words.txt", staging / "words.txt")
        files = {name: {"size_bytes": (staging / name).stat().st_size, "sha256": digest(staging / name)}
                 for name in ("TLG.fst", "words.txt")}
        for name, stamp in stamps.items():
            current = (source / name).stat()
            if (current.st_size, current.st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
                raise ValueError("El origen cambió durante la conversión; no se publica el resultado.")
        if files["words.txt"]["sha256"] != original["files"]["words.txt"]["sha256"]:
            raise ValueError("La copia de words.txt no conserva los símbolos originales.")
        result = {"format_version": 2, "archive_sha256": original["archive_sha256"],
                  "source_url": original["source_url"], "graph_phones": original["graph_phones"], "files": files,
                  "transformation": {"type": "openfst_const_aligned", "probability_pruning": False,
                      "source_manifest_sha256": digest(manifest_path),
                      "source_graph_sha256": original["files"]["TLG.fst"]["sha256"],
                      "source_commit": spec["preparacion"]["commit"], "openfst_version": "1.6.5",
                      "runtime_build_sha256": digest(Path(spec["python_runtime"]).absolute().parent.parent / "runtime_build.json"),
                      "writer": "ConstFst<StdArc>::WriteFst", "align": True, "small_fst_equal": True},
                  "fst_header": converted}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        for name in ("prueba.txt", "prueba_vector.fst", "prueba_const.fst"):
            (staging / name).unlink()
        staging.rename(destination)
    print("Grafo preparado:", destination, flush=True)


def check_loading(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Mide solo la carga CPU del trigrama; no certifica aún la inferencia con la GRU."""
    graph = Path(spec["modelo"]).resolve() / "TLG.fst"
    require_mappable_fst(graph)
    limits = spec["compactacion"]
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"scope": "solo_carga_trigrama", "verified": False,
        "peak_worker_rss_bytes": 0, "stopped_by_probe": None, "graph_bytes": graph.stat().st_size,
        "limits": limits, "manifest_sha256": digest(graph.parent / "manifest.json")}
    started = last_print = time.monotonic()
    stdout, stderr = output / "worker_stdout.log", output / "worker_stderr.log"
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen([str(Path(spec["python_runtime"]).absolute()), "-u", "-X", "faulthandler",
            str(ROOT / "scripts/trigram_worker.py")], stdin=subprocess.PIPE, stdout=out, stderr=err,
            text=True, env=cpu_environment(spec))
        try:
            process.stdin.write(json.dumps({"op": "init", "model": str(graph.parent),
                "options": spec["busqueda"], "threads": spec["hilos_cpu"], "read_mode": "map"}) + "\n")
            process.stdin.close()
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                except FileNotFoundError:
                    continue
                rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:")), 0)
                report["peak_worker_rss_bytes"] = max(report["peak_worker_rss_bytes"], rss)
                now, available = time.monotonic(), available_memory()
                if (rss >= limits["limite_rss_carga_gib"] * (1 << 30)
                        or available < limits["reserva_sistema_gib"] * (1 << 30)):
                    report["stopped_by_probe"] = "límite preventivo de memoria"
                elif now - started >= limits["timeout_carga_segundos"]:
                    report["stopped_by_probe"] = "límite de tiempo"
                if report["stopped_by_probe"]:
                    break
                if now - last_print >= 5:
                    print(f"Carga: RSS={rss / (1 << 30):.2f} GiB; disponible={available / (1 << 30):.2f} GiB", flush=True)
                    last_print = now
                time.sleep(0.1)
        except BrokenPipeError:
            pass
        finally:
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    report.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started)
    if process.returncode < 0:
        try:
            report["signal"] = signal.Signals(-process.returncode).name
        except ValueError:
            report["signal"] = str(-process.returncode)
    report["stdout"] = stdout.read_text(errors="replace")[-8192:]
    report["stderr"] = stderr.read_text(errors="replace")[-8192:]
    try:
        replies = [json.loads(line) for line in report["stdout"].splitlines()]
    except json.JSONDecodeError as error:
        report["protocol_error"] = str(error)
        replies = []
    if len(replies) == 2:
        report["peak_worker_rss_bytes"] = max(report["peak_worker_rss_bytes"], replies[1].get("peak_rss_bytes", 0))
    report["verified"] = bool(process.returncode == 0 and not report["stopped_by_probe"] and len(replies) == 2
        and all(reply.get("ok") for reply in replies) and replies[0].get("mapped_graphs")
        and replies[1].get("read_mode") == "map" and replies[1].get("mapped_graph_bytes", 0) > 0
        and report["peak_worker_rss_bytes"] < limits["limite_rss_carga_gib"] * (1 << 30))
    (output / "comprobacion_carga.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["verified"]:
        raise RuntimeError(f"La carga no está verificada; revisar {output / 'comprobacion_carga.json'}")
    return report


def main() -> int:
    """Prepara en RAM alta y/o comprueba la carga limitada, siempre en CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/entrenos_online/speech_trigrama_compacto.yaml"))
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--check-loading", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not any((args.build_runtime, args.prepare_model, args.check_loading)):
        parser.error("Indica --build-runtime, --prepare-model o --check-loading.")
    if args.check_loading and args.output is None:
        parser.error("--check-loading necesita --output.")
    spec = yaml.safe_load(args.config.read_text())
    if spec.get("lectura_grafo") != "map":
        raise ValueError("Este preparador requiere lectura_grafo: map.")
    if args.build_runtime:
        build_runtime(spec)
    if args.prepare_model:
        prepare_compact_model(spec)
    if args.check_loading:
        check_loading(spec, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
