"""Evalúa un checkpoint T12 por sesión y con controles de señal; no entrena."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import build_smoother
from src.data.speech_dataset import SpeechDataset, collate_speech, file_sha256
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.speech_evaluation import evaluate_signal_controls
from src.utils.seed import fijar_seed, crear_generador
from src.utils.tracking import abrir_run, registrar


def main() -> int:
    """Carga un único split, escribe predicciones y registra el diagnóstico en W&B."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/speech_evaluacion.yaml"))
    parser.add_argument("--split", choices=("val", "test"), help="sobrescribe el split del diagnóstico de forma explícita")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8"))
    if args.split is not None:
        spec["split"] = args.split
    cfg["experimento"]["nombre"] += f"_diagnostico_{spec['split']}"
    cfg["diagnostico"] = spec
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    fijar_seed(cfg["experimento"]["seed"])
    requested = cfg["ejecucion"]["dispositivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA pero no está disponible.")
    run = abrir_run(cfg)
    try:
        dataset = SpeechDataset(cfg, spec["split"])
        loader = DataLoader(dataset, batch_size=cfg["datos"]["batch_size"], shuffle=False,
                            num_workers=cfg["datos"]["num_workers"], pin_memory=cfg["datos"]["pin_memory"],
                            collate_fn=collate_speech, generator=crear_generador(cfg["experimento"]["seed"]))
        digest = file_sha256(args.checkpoint)
        model = construir_modelo(cfg)
        state = load_checkpoint(args.checkpoint, model, cfg, dataset)
        model.to(device).eval()
        smoother = build_smoother(cfg).to(device)
        args.output.mkdir(parents=True, exist_ok=False)
        temporary = args.output / "predicciones.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            def write_prediction(row: dict) -> None:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

            result = evaluate_signal_controls(model, loader, cfg, spec, smoother, device, prediction_sink=write_prediction)
        if file_sha256(args.checkpoint) != digest:
            raise ValueError("El checkpoint cambió durante la evaluación.")
        result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest, checkpoint_step=state["step"],
                      protocol_sha256=cfg.get("protocolo", {}).get("sha256"), comparison=state["config"].get("comparacion"),
                      training_seed=state["config"]["experimento"]["seed"], evaluation_config=spec,
                      data_signature=state["data_signature"], phones=state["phones"], blank_id=state["blank_id"],
                      hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      torch_version=str(torch.__version__), wandb_run_id=run.id)
        temporary.replace(args.output / "predicciones.jsonl")
        with (args.output / "por_sesion.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["control", "salida", "sesion", "ctc_loss", "per", "errores", "fonemas", "frases", "completo"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            session_rows = []
            for control, depths in result["by_session"].items():
                for depth, sessions in depths.items():
                    for name, metrics in sessions.items():
                        row = {"control": control, "salida": depth, "sesion": name, **metrics}
                        writer.writerow(row)
                        session_rows.append([row[key] for key in fields])
        (args.output / "metricas.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        logged = {f"diagnostico/{spec['split']}/{control}/capa_{depth}/{key}": value
                  for control, depths in result["global"].items() for depth, metrics in depths.items() for key, value in metrics.items()}
        from wandb import Table

        logged["diagnostico/por_sesion"] = Table(columns=fields, data=session_rows)
        registrar(run, logged, state["step"])
        run.summary.update({"checkpoint/sha256": digest, "checkpoint/paso": state["step"],
                            "checkpoint/semilla_entrenamiento": result["training_seed"],
                            "diagnostico/split": spec["split"], "diagnostico/actualizaciones": 0,
                            "diagnostico/test_evaluado": result["test_evaluated"]})
        for control, depths in result["global"].items():
            for depth, metrics in depths.items():
                print(f"{control} · salida {depth}: PER={metrics['per']:.2%} ({metrics['errores']}/{metrics['fonemas']}); frases={metrics['frases']}")
        print(f"Diagnóstico guardado: {args.output}; cero actualizaciones.")
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
