"""Demo HTTP local: señal cruda de val al receptor GRU + trigrama en CPU.

Desde la raíz: python scripts/demo_speech_http.py
El supervisor limita la memoria del servidor y su trabajador WFST. Ctrl+C
cierra el grupo de procesos de esta demo. No entrena ni abre W&B.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_server(cfg: dict, language_spec: dict) -> None:
    """Carga los pesos una vez y conserva el grafo entre transmisiones HTTP."""
    import torch
    from src.app.speech_http import DemoController, DemoHTTPServer
    from src.app.speech_replay import DemoSamples
    from src.data.speech_dataset import file_sha256
    from src.inference.speech import OnlineSpeechDecoder
    from src.inference.trigram import TrigramDecoder
    from src.utils.seed import fijar_seed

    # Reservar el puerto antes de la lectura de 31 GB del grafo.
    with DemoHTTPServer((cfg["servidor"]["host"], cfg["servidor"]["puerto"]), None) as server:
        torch.set_num_threads(language_spec["hilos_cpu"])
        fijar_seed(cfg["seed"])
        output = Path(cfg["salida"]) / ("servidor_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=False)
        cfg = {**cfg, "salida": str(output), "hilos_cpu": language_spec["hilos_cpu"]}
        print("Cargando checkpoint causal en CPU…", flush=True)
        decoder = OnlineSpeechDecoder.from_checkpoint(Path(cfg["checkpoint"]), "cpu")
        print("Comprobando las muestras de validación…", flush=True)
        samples = DemoSamples(decoder.checkpoint, cfg["seed"])
        print("Verificando el grafo y abriendo el trigrama. Esta lectura inicial puede tardar…", flush=True)
        language = TrigramDecoder(language_spec, decoder.checkpoint.phones, decoder.checkpoint.blank_id,
                                  log_path=output / "trigram_worker.log")
        controller = DemoController(decoder, language, samples, cfg)
        server.controller = controller
        try:
            metadata = {"config": cfg, "checkpoint_sha256": file_sha256(Path(cfg["checkpoint"])),
                        "language_resources": language.resources, "device": "cpu",
                        "cpu_threads": language_spec["hilos_cpu"], "training_updates": 0,
                        "http_address": list(server.server_address), "pid": os.getpid(), "supervisor_pid": os.getppid()}
            (output / "servidor.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Demo lista: http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
            print("Registro:", output, flush=True)
            server.serve_forever(poll_interval=0.2)
        finally:
            controller.close()


def supervise(args: argparse.Namespace, language_spec: dict) -> int:
    """Vigila RSS y RAM libre sin importar PyTorch en el proceso supervisor."""
    from scripts.check_speech_trigram_cpu import tree_memory
    from scripts.compact_speech_trigram import available_memory

    limits = language_spec["comprobacion_cpu"]
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--worker"]
    if args.port is not None:
        command.extend(("--port", str(args.port)))
    print(f"Demo CPU · {language_spec['hilos_cpu']} hilos · límite conjunto {limits['limite_rss_total_gib']} GiB", flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), start_new_session=True)
    try:
        while process.poll() is None:
            if (tree_memory(process.pid)["rss_bytes"] >= limits["limite_rss_total_gib"] * (1 << 30)
                    or available_memory() < limits["reserva_sistema_gib"] * (1 << 30)):
                raise RuntimeError("Demo detenida por el límite preventivo de RAM.")
            time.sleep(limits["periodo_muestreo_segundos"])
        return process.returncode
    except KeyboardInterrupt:
        return 0
    finally:
        for signum, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 1)):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signum)
            try:
                process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
        # Incluye al trabajador nativo si el servidor terminó antes que él.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def _stop_requested(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    """Lee un perfil de demo y oculta CUDA antes de cargar cualquier modelo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/entrenos_online/speech_demo_http.yaml"))
    parser.add_argument("--port", type=int, help="puerto HTTP; 0 elige uno libre")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    language_spec = yaml.safe_load(Path(cfg["lenguaje"]).read_text())
    if args.port is not None:
        cfg["servidor"]["puerto"] = args.port
    if (cfg["servidor"]["host"] not in ("127.0.0.1", "localhost")
            or type(cfg["servidor"]["puerto"]) is not int or not 0 <= cfg["servidor"]["puerto"] <= 65535
            or language_spec.get("lectura_grafo") != "map"
            or any(type(cfg["transmision"][key]) is not int or cfg["transmision"][key] <= 0
                   for key in ("bloque_bins", "max_bloques_pendientes"))):
        parser.error("La demo requiere interfaz local, lectura map y bloques/cola positivos.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(language_spec["hilos_cpu"])
    os.environ["MPLBACKEND"] = "Agg"
    signal.signal(signal.SIGTERM, _stop_requested)
    try:
        if args.worker:
            run_server(cfg, language_spec)
            return 0
        return supervise(args, language_spec)
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
