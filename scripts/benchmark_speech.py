"""Mide carga, forward y señal cruda→fonemas de un checkpoint T12, sin entrenar."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import build_smoother
from src.data.speech_dataset import SpeechDataset, file_sha256
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.speech_benchmark import benchmark_profiles, prepare_benchmark_cases, validate_benchmark_spec
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run, registrar


def main() -> int:
    """Guarda observaciones por lote y resúmenes separados de cada etapa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, default=Path("configs/speech_benchmark.yaml"))
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    if args.split is not None:
        spec["split"] = args.split
    validate_benchmark_spec(spec)
    torch.set_num_threads(spec["hilos_cpu"])
    fijar_seed(cfg["experimento"]["seed"])
    device = torch.device(spec["dispositivo"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA para este perfil de benchmark.")
    initialization = perf_counter()
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.synchronize(device)
    cuda_initialization_seconds = perf_counter() - initialization
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    runtime = {"device": str(device), "hardware": properties.name if properties else platform.processor() or platform.machine(),
               "gpu_uuid": str(getattr(properties, "uuid", "unavailable")) if properties else None,
               "gpu_total_bytes": properties.total_memory if properties else None,
               "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "cpu_threads": torch.get_num_threads(),
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "tensor_dtype": "float32; no AMP"}
    cfg["experimento"]["nombre"] += "_benchmark"
    cfg["benchmark"] = spec
    run = abrir_run(cfg)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        opened = perf_counter()
        dataset = SpeechDataset(cfg, spec["split"])
        dataset_open_seconds = perf_counter() - opened
        cases, normalizers, inputs = prepare_benchmark_cases(dataset, cfg, spec)
        digest = file_sha256(args.checkpoint)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        model = construir_modelo(cfg)
        state = load_checkpoint(args.checkpoint, model, cfg, dataset)
        model.to(device).eval()
        smoother = build_smoother(cfg).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        load_seconds = perf_counter() - start
        load_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        state.pop("model_state")
        gc.collect()
        result = benchmark_profiles(model, smoother, cases, normalizers, spec, device)
        if file_sha256(args.checkpoint) != digest:
            raise ValueError("El checkpoint cambió durante el benchmark.")
        result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest, checkpoint_step=state["step"],
                      training_seed=state["config"]["experimento"]["seed"], comparison=state["config"].get("comparacion"),
                      split=spec["split"], test_evaluated=spec["split"] == "test", spec=spec, runtime=runtime,
                      inputs=inputs, model_load_seconds=load_seconds, load_gpu_peak_allocated_bytes=load_peak,
                      dataset_open_seconds=dataset_open_seconds, cuda_initialization_seconds=cuda_initialization_seconds,
                      load_scope="one observed load: construction, disk weights, compatibility checks, copy to device, smoother; filesystem cache uncontrolled",
                      wandb_run_id=run.id)
        temporary = args.output / "benchmark.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output / "benchmark.json")
        rows = result["measurements"]
        with (args.output / "mediciones.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        from wandb import Table

        logged = {f"benchmark/capa_{depth}/{stage}/{key}": value
                  for depth, profile in result["profiles"].items() for stage in ("forward", "raw_to_phones")
                  for key, value in profile[stage].items() if value is not None}
        logged["benchmark/model_load_seconds"] = load_seconds
        logged["benchmark/mediciones"] = Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])
        registrar(run, logged, state["step"])
        run.summary.update({"checkpoint/sha256": digest, "benchmark/input_signature": inputs["input_signature"],
                            "benchmark/actualizaciones": 0, "benchmark/semilla_entrenamiento": result["training_seed"]})
        for depth, profile in result["profiles"].items():
            print(f"Salida {depth}: mediana forward {profile['forward']['p50_ms']:.3f} ms; "
                  f"señal cruda→fonemas {profile['raw_to_phones']['p50_ms']:.3f} ms por lote.")
        print(f"Carga del modelo: {load_seconds:.3f} s. Resultados: {args.output}")
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
