#!/usr/bin/env python3
"""Genera una figura descriptiva del run Alice existente; no entrena ni selecciona."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main() -> int:
    """Dibuja datos test, métricas medidas y controles temporales ya calculados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "resultado.json").read_text())
    with (args.run_dir / "predicciones_test.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    y = np.array([float(row["real_log10"]) for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    fig.suptitle(f"Alice · sub-001 · test por segmentos · run {report['run_id']}", fontsize=16)
    axes[0, 0].plot(y, color="#202a35", alpha=.7, lw=1, label="Referencia acústica")
    axes[0, 0].plot([float(r["gru_log10"]) for r in rows], color="#b44026", lw=1, label="GRU desde EEG")
    boundary = next(i for i, r in enumerate(rows) if r["segmento"] != rows[0]["segmento"])
    axes[0, 0].axvline(boundary, ls=":", color="gray")
    axes[0, 0].set(xlabel="Palabra retenida de test (orden temporal)", ylabel="log10(sound_power)",
                   title="Palabras de los segmentos 11 y 12")
    axes[0, 0].legend(fontsize=8)
    for name, color in (("ridge", "#2c718e"), ("gru", "#b44026")):
        axes[0, 1].scatter(y, [float(row[f"{name}_log10"]) for row in rows], s=10, alpha=.45,
                           label=f"{name}: r={report['test'][name]['pearson']:.3f}", color=color)
    axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], color="gray", ls="--", lw=1)
    axes[0, 1].set(xlabel="Referencia log10(sound_power)", ylabel="Predicción log10(sound_power)",
                   title=f"{len(rows)} palabras, excluidas del ajuste")
    axes[0, 1].legend(fontsize=8)
    for axis, name, color in ((axes[1, 0], "ridge", "#2c718e"), (axes[1, 1], "gru", "#b44026")):
        control = report["controles_test"][name]
        axis.hist(control["pearson_desplazados"], bins=20, color="#b8c8ce", edgecolor="white")
        axis.axvline(control["pearson_observado"], color=color, lw=2, label="r sin desplazar")
        axis.set(xlabel="Correlación tras desplazar etiquetas dentro de cada segmento", ylabel="Repeticiones",
                 title=f"{name} · {len(control['pearson_desplazados'])} controles · p exploratoria={control['p_empirica_unilateral']:.3f}")
        axis.legend(fontsize=8)
    fig.savefig(args.run_dir / "comparacion.png", dpi=160)
    plt.close(fig)
    print(args.run_dir / "comparacion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
