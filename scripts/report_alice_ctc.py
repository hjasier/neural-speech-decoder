#!/usr/bin/env python3
"""Figura de validación y PER de un run Alice CTC guardado, sin entrenar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.speech_dataset import file_sha256  # noqa: E402
from src.training.metrics import edit_distance  # noqa: E402


def main() -> int:
    """Representa las medidas guardadas y las referencias sin EEG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "resultado.json").read_text())
    manifest_path = Path(report["config"]["datos"]["preparados"]) / "manifest.json"
    if file_sha256(manifest_path) != report["data_manifest_sha256"]:
        raise ValueError("La preparación ya no corresponde al run que se representa.")
    manifest = json.loads(manifest_path.read_text())
    predictions = json.loads((args.run_dir / "predicciones_test.json").read_text())
    references = manifest["records"]["test"]
    if len(predictions) != len(references) or [p["sequence"] for p in predictions] != list(range(len(references))):
        raise ValueError("Las predicciones no cubren el test en el orden esperado.")
    silence = manifest["phones"].index(report["config"]["etiquetas"]["silencio"])
    filtered_refs = [[i for i in row["ids"] if i != silence] for row in references]
    filtered_hyps = [[i for i in row["hypothesis_ids"] if i != silence] for row in predictions]
    errors = sum(edit_distance(a, b) for a, b in zip(filtered_refs, filtered_hyps, strict=True))
    phones = sum(map(len, filtered_refs))
    diagnostics = {"tipo": "medición descriptiva posterior; no usada para seleccionar el modelo",
                   "errores_sin_sil": errors, "fonemas_sin_sil": phones, "per_sin_sil": errors / phones,
                   "fonemas_predichos": dict(Counter(p for row in predictions for p in row["hypothesis_phones"])),
                   "nota": "SIL se inserta como separador entre palabras siguiendo el código T12; no es silencio acústico anotado."}
    (args.run_dir / "diagnostico_fonemas.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    fig.suptitle(f"Alice · SpeechGRU + CTC · run {report['run_id']} · {report['counts']['test']} fragmentos test")
    history = report["validation_history"]
    if history:
        axes[0].plot([x["step"] for x in history], [x["ctc_loss"] for x in history], color="#2c718e")
        axes[1].plot([x["step"] for x in history], [100 * x["per"] for x in history], color="#2c718e")
        for ax in axes[:2]:
            ax.axvline(report["best_val_step"], ls="--", color="#b44026", label="Checkpoint seleccionado")
            ax.set_xlabel("Actualizaciones")
        axes[0].legend(fontsize=7)
    axes[0].set(title="Selección por CTC de validación", ylabel="CTC media por fragmento")
    axes[1].set(title="PER de validación", ylabel="PER (%)")
    names = ["GRU + EEG", "EEG cero", "Prior train", "Vacío"]
    values = [100 * report["test"]["per"], 100 * report["zero_eeg_control"]["per"],
              100 * report["phone_prior_control"]["per"], 100 * report["empty_prediction_per"]]
    bars = axes[2].bar(names, values, color=["#2c718e", "#a6b9c2", "#b44026", "#d9dee0"])
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[2].set(title="Test final · menor PER es mejor", ylabel="PER (%)", ylim=(0, max(values) * 1.15))
    axes[2].tick_params(axis="x", labelsize=8)
    fig.savefig(args.run_dir / "comparacion_ctc.png", dpi=160)
    plt.close(fig)
    print(args.run_dir / "comparacion_ctc.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
