"""Comprueba GRU + trigrama compacto con señal real y un límite conjunto de RAM.

Usa las primeras frases de val declaradas en configuración. Es una comprobación
de funcionamiento y recursos; no evalúa WER del corpus ni certifica otro host.
El supervisor no importa PyTorch y oculta CUDA antes de abrir los trabajadores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compact_speech_trigram import available_memory, cpu_environment


def tree_memory(pid: int) -> dict[str, Any]:
    """Desglosa el RSS de cada descendiente en memoria anónima, de archivos y compartida."""
    names = {"VmRSS:": "rss_bytes", "RssAnon:": "anonymous_bytes",
             "RssFile:": "file_bytes", "RssShmem:": "shared_memory_bytes"}
    total: dict[str, Any] = {**dict.fromkeys(names.values(), 0), "processes": []}
    pending, seen = [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            values = dict.fromkeys(names.values(), 0)
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                fields = line.split()
                if fields and fields[0] in names:
                    values[names[fields[0]]] = int(fields[1]) * 1024
            total["processes"].append({"pid": current, **values})
            for key, value in values.items():
                total[key] += value
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(map(int, children))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _stop_requested(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt("Comprobación interrumpida; se cerrará el grupo de procesos.")


def run_worker(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    """Procesa un subconjunto acotado con pesos congelados, siempre en CPU."""
    import copy
    from itertools import islice
    import numpy as np
    import torch

    from src.data.speech import BIN_SECONDS
    from src.data.speech_dataset import SpeechDataset, file_sha256
    from src.inference.speech import OnlineSpeechDecoder
    from src.inference.trigram import TrigramDecoder
    from src.training.speech_online_evaluation import iter_online_trials
    from src.training.speech_trigram_evaluation import _profile, check_trigram_streaming
    from src.utils.seed import fijar_seed

    torch.set_num_threads(spec["hilos_cpu"])
    policy = spec["comprobacion_cpu"]
    print("Cargando el checkpoint en CPU y comprobando la partición val.", flush=True)
    checkpoint_hash = file_sha256(args.checkpoint)
    decoder = OnlineSpeechDecoder.from_checkpoint(args.checkpoint, "cpu")
    cfg = copy.deepcopy(decoder.checkpoint.config)
    fijar_seed(cfg["experimento"]["seed"])
    dataset = SpeechDataset(cfg, "val")
    trials = list(islice(iter_online_trials(dataset, decoder), policy["max_frases"]))
    if len(trials) != policy["max_frases"]:
        raise ValueError("No hay suficientes frases de val para la comprobación configurada.")
    depth = decoder.checkpoint.model.num_layers
    evaluation = {"split": "val", "salidas": [depth], "bloque_bins": policy["bloque_bins"],
                  "calentamiento": 0, "logits_atol": 0.0, "logits_rtol": 0.0}
    print("Verificando los hashes del grafo compacto y abriendo el trabajador CPU.", flush=True)
    language = TrigramDecoder(spec, decoder.checkpoint.phones, decoder.checkpoint.blank_id,
                              log_path=args.output / "trigram_worker.log")
    try:
        verification = check_trigram_streaming(decoder, language, trials[0], evaluation)
        (args.output / "comprobacion_incremental.json").write_text(json.dumps(verification, indent=2) + "\n")
        if not verification["verified"]:
            raise ValueError("Los mismos logits agrupados y por bloques producen textos distintos.")
        rows = []
        for trial in trials:
            print(f"Procesando {trial.session}, ensayo {trial.trial_index}, en bloques.", flush=True)
            profile = _profile(decoder, language, trial, depth, policy["bloque_bins"])
            signal_ms = len(trial.raw) * BIN_SECONDS * 1000
            rows.append({"session": trial.session, "source_trial_index": trial.trial_index,
                         "input_bins": len(trial.raw), "signal_ms": signal_ms, **profile})
            # Guardado progresivo: una parada preventiva conserva las frases terminadas.
            (args.output / "frases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("El checkpoint cambió durante la comprobación.")
        packet_ms = [packet["processing_ms"] for row in rows for packet in row["packets"]]
        processing_ms = sum(row["processing_ms"] for row in rows)
        signal_ms = sum(row["signal_ms"] for row in rows)
        result = {"scope": "subconjunto_val_cpu; requiere comprobación adicional en el host de la demo",
                  "optimizer_steps": 0, "test_evaluated": False, "checkpoint_sha256": checkpoint_hash,
                  "checkpoint_step": decoder.checkpoint.step, "trial_selection": "primeras frases de val, en orden",
                  "trials": len(rows), "depth": depth, "streaming_verified": verification["verified"],
                  "language_manifest_sha256": file_sha256(Path(spec["modelo"]) / "manifest.json"),
                  "worker_runtime": language.runtime, "language_resources": language.resources,
                  "mapped_graph_cache": language.graph_cache,
                  "runtime": {"device": "cpu", "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                              "machine": platform.machine(), "processor": platform.processor()},
                  "timing": {"processing_ms": processing_ms, "signal_ms": signal_ms,
                             "rtf": processing_ms / signal_ms, "packet_count": len(packet_ms),
                             "packet_p50_ms": float(np.median(packet_ms)),
                             "packet_p95_ms": float(np.percentile(packet_ms, 95))},
                  "timing_scope": "preprocesamiento, GRU, CTC, IPC y WFST; excluye carga y comprobación inicial",
                  "cache_scope": "caché de disco sin controlar; primera frase verificada antes de medir",
                  "arrival_clock": "simulado, sin HTTP ni sleep; no representa la red de la demo"}
        (args.output / "inferencia_cpu.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("Inferencia CPU del subconjunto terminada.", flush=True)
    finally:
        language.close()


def supervise(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    """Controla memoria y tiempo del árbol completo y conserva el informe al fallar."""
    limits = spec["comprobacion_cpu"]
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"scope": "gru_y_trigrama_cpu_subconjunto_val", "verified": False,
        "local_demo_verified": False, "peak_tree_rss_bytes": 0, "stopped_by_probe": None, "limits": limits,
        "memory_scope": "máximo muestreado de la suma RSS del proceso principal y sus descendientes; conservador por páginas compartidas",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "language_config": spec, "python": sys.version, "machine": platform.machine()}
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint.resolve()),
               "--config", str(args.config.resolve()), "--output", str(args.output.resolve()), "--worker"]
    started = last_print = time.monotonic()
    with (args.output / "stdout.log").open("w") as out, (args.output / "stderr.log").open("w") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=cpu_environment(spec), stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                snapshot = tree_memory(process.pid)
                rss = snapshot["rss_bytes"]
                if rss > report["peak_tree_rss_bytes"]:
                    report["peak_tree_rss_bytes"] = rss
                    report["peak_tree_memory"] = snapshot
                report["last_tree_memory"] = snapshot
                now, available = time.monotonic(), available_memory()
                if rss >= limits["limite_rss_total_gib"] * (1 << 30) or available < limits["reserva_sistema_gib"] * (1 << 30):
                    report["stopped_by_probe"] = "límite preventivo de memoria del pipeline"
                elif now - started >= limits["timeout_segundos"]:
                    report["stopped_by_probe"] = "límite de tiempo del pipeline"
                if report["stopped_by_probe"]:
                    break
                if now - last_print >= 5:
                    print(f"Pipeline CPU: RSS total={rss / (1 << 30):.2f} GiB; disponible={available / (1 << 30):.2f} GiB", flush=True)
                    last_print = now
                time.sleep(limits["periodo_muestreo_segundos"])
        finally:
            # La sesión nueva contiene únicamente este trabajador y su WFST.
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                _signal_group(process.pid, signal.SIGKILL)
    report.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started)
    if process.returncode < 0:
        try:
            report["signal"] = signal.Signals(-process.returncode).name
        except ValueError:
            report["signal"] = str(-process.returncode)
    for stream in ("stdout", "stderr"):
        with (args.output / f"{stream}.log").open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 8192))
            report[f"{stream}_tail"] = log.read().decode("utf-8", errors="replace")
    result_path = args.output / "inferencia_cpu.json"
    if result_path.exists():
        report["inference"] = json.loads(result_path.read_text())
    report["verified"] = bool(process.returncode == 0 and not report["stopped_by_probe"]
        and report.get("inference", {}).get("streaming_verified")
        and report["inference"]["trials"] == limits["max_frases"]
        and report["peak_tree_rss_bytes"] < limits["limite_rss_total_gib"] * (1 << 30))
    (args.output / "comprobacion_cpu.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["verified"]:
        raise RuntimeError(f"Comprobación CPU no superada; revisar {args.output / 'comprobacion_cpu.json'}")
    return report


def main() -> int:
    """Valida el perfil y ejecuta una comprobación limitada, sin acceso a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/entrenos_online/speech_trigrama_compacto.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    limits = spec["comprobacion_cpu"]
    if (spec.get("lectura_grafo") != "map" or type(spec["hilos_cpu"]) is not int or spec["hilos_cpu"] <= 0
            or any(type(limits[key]) is not int or limits[key] <= 0 for key in ("max_frases", "bloque_bins"))
            or any(not isinstance(limits[key], (int, float)) or not 0 < limits[key] < float("inf")
                   for key in ("limite_rss_total_gib", "reserva_sistema_gib", "timeout_segundos", "periodo_muestreo_segundos"))):
        raise ValueError("Configuración de comprobación CPU no válida.")
    os.environ.update(cpu_environment(spec))
    if args.worker:
        run_worker(args, spec)
    else:
        signal.signal(signal.SIGTERM, _stop_requested)
        supervise(args, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
