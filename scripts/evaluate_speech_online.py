"""Evalúa PER y tiempos por bloques de un checkpoint causal, sin entrenar.

Desde la raíz del repositorio, con un checkpoint del entrenamiento online:
    python scripts/evaluate_speech_online.py --checkpoint PESOS.pt --output runs/online_val

Usa val por defecto; --split test reserva la evaluación final explícita.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.speech_dataset import SpeechDataset, file_sha256
from src.inference.speech import OnlineSpeechDecoder
from src.training.speech_online_evaluation import evaluate_online, iter_online_trials
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run, registrar


def main() -> int:
    """Usa la configuración del checkpoint y guarda métricas y emisiones por bloque."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/speech_online_evaluacion.yaml"))
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8"))
    if args.split is not None:
        spec["split"] = args.split
    torch.set_num_threads(spec["hilos_cpu"])
    device = torch.device(spec["dispositivo"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("La evaluación online admite CPU o CUDA explícitos.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA pero no está disponible.")
    digest = file_sha256(args.checkpoint)
    start = perf_counter()
    decoder = OnlineSpeechDecoder.from_checkpoint(args.checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = perf_counter() - start
    cfg = copy.deepcopy(decoder.checkpoint.config)
    cfg["experimento"]["nombre"] += f"_online_{spec['split']}"
    cfg["evaluacion_online"] = spec
    fijar_seed(cfg["experimento"]["seed"])
    dataset = SpeechDataset(cfg, spec["split"])
    args.output.mkdir(parents=True, exist_ok=False)
    run = abrir_run(cfg)
    try:
        temporary = args.output / "predicciones.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            def write_prediction(row: dict) -> None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

            result = evaluate_online(decoder, iter_online_trials(dataset, decoder), spec,
                                     expected_trials=len(dataset), prediction_sink=write_prediction)
        if file_sha256(args.checkpoint) != digest:
            raise ValueError("El checkpoint cambió durante la evaluación.")
        result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
                      checkpoint_step=decoder.checkpoint.step, training_seed=cfg["experimento"]["seed"],
                      protocol_sha256=cfg.get("protocolo", {}).get("sha256"),
                      data_signature=decoder.checkpoint.data_signature,
                      prepared_manifest_sha256=file_sha256(dataset.root / "manifest.json"),
                      evaluation_config=spec, model_load_seconds=load_seconds,
                      runtime={"device": str(device), "hardware": torch.cuda.get_device_name(device)
                               if device.type == "cuda" else platform.processor() or platform.machine(),
                               "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads()},
                      phones=list(decoder.checkpoint.phones), blank_id=decoder.checkpoint.blank_id,
                      wandb_run_id=run.id)
        temporary.replace(args.output / "predicciones.jsonl")
        (args.output / "metricas.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        logged = {f"online/{spec['split']}/capa_{depth}/{key}": value
                  for depth, metrics in result["global"].items() for key, value in metrics.items()}
        logged.update({f"online/tiempo/capa_{depth}/rtf": metrics["rtf"]
                       for depth, metrics in result["timing"].items()})
        logged.update({f"online/tiempo/capa_{depth}/{stage}/{key}": value
                       for depth, metrics in result["timing"].items()
                       for stage in ("packet_processing", "first_phone_input", "first_phone_ready")
                       for key, value in metrics[stage].items() if value is not None})
        registrar(run, logged, decoder.checkpoint.step)
        run.summary.update({"checkpoint/sha256": digest, "online/actualizaciones": 0,
                            "online/test_evaluado": result["test_evaluated"]})
        for depth, metrics in result["global"].items():
            print(f"Salida {depth}: PER={metrics['per']:.2%} ({metrics['errores']}/{metrics['fonemas']}); "
                  f"RTF={result['timing'][depth]['rtf']:.4f}")
        print(f"Evaluación online guardada: {args.output}; cero actualizaciones.")
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
