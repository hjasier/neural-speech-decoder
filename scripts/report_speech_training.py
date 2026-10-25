"""Resume un entrenamiento T12 y dibuja validación, sin cargar ni evaluar test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Fijarlo antes del import evita heredar un backend inline del kernel remoto.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main() -> int:
    """Comprueba el checkpoint seleccionado y representa sus métricas guardadas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "resultado.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    history = report["validation_history"]
    checkpoint = args.run_dir / "mejor_val.pt"
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    selected = min(history, key=lambda point: point["per"])
    if (report["test_evaluated"] is not False or sidecar["split"] != "val"
            or report["best_val_step"] != selected["step"]
            or sidecar["step"] != selected["step"]
            or any(sidecar[key] != value for key, value in report["best_val"].items())
            or any(selected[key] != value for key, value in report["best_val"].items())
            or any(point["completo"] != 1 for point in history)):
        raise ValueError("El resumen no corresponde al mínimo PER de validación completa.")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {
        "run_id": report["run_id"], "completed_steps": report["completed_steps"],
        "planned_steps": cfg["entrenamiento"]["pasos"],
        "completed_budget": report["completed_steps"] == cfg["entrenamiento"]["pasos"],
        "best_val_step": report["best_val_step"], "best_val": report["best_val"],
        "last_val": report["last_val"], "test_evaluated": False,
        "hardware": report["hardware"], "model_parameters": report["model_parameters"],
        "training_seconds": report["training_seconds"], "loop_seconds": report["loop_seconds"],
        "cuda_memory": report["cuda_memory"], "torch_version": report["torch_version"],
        "protocol_sha256": cfg["protocolo"]["sha256"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "scope": "presupuesto y métricas observadas; la convergencia requiere revisión del autor",
    }
    (args.run_dir / "resumen_baseline.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.suptitle(f"T12 · validación · {report['run_id']}")
    steps = [point["step"] for point in history]
    axes[0].plot(steps, [100 * point["per"] for point in history])
    axes[0].set(ylabel="PER (%)", title="Selección por PER de validación")
    axes[1].plot(steps, [point["ctc_loss"] for point in history])
    axes[1].set(ylabel="CTC media por frase", title="Pérdida de validación")
    for axis in axes:
        axis.axvline(report["best_val_step"], color="#b44026", linestyle="--", label="Checkpoint elegido")
        axis.set_xlabel("Actualizaciones")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(args.run_dir / "curvas_validacion.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
