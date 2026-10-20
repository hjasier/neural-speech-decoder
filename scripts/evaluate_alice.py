#!/usr/bin/env python3
"""Recarga GRU/ridge de Alice y comprueba las predicciones guardadas, sin entrenar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.alice import prepare_alice  # noqa: E402
from src.models.factory import construir_modelo  # noqa: E402
from src.utils.seed import fijar_seed  # noqa: E402


def main() -> int:
    """Restaura normalización y pesos; verifica integridad y predicciones test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir
    checkpoint = torch.load(output / "gru.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    fijar_seed(cfg["experimento"]["seed"])
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    prepared = prepare_alice(cfg, normalization=checkpoint["normalization"])
    if prepared.report["archivos_sha256"] != checkpoint["data_report"]["archivos_sha256"]:
        raise ValueError("Los archivos difieren de los usados al entrenar.")
    test = prepared.splits["test"]
    model = construir_modelo(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    batch = cfg["entrenamiento"]["batch_size"]
    with torch.no_grad():
        gru = torch.cat([model(torch.from_numpy(test.features[i:i + batch]))
                         for i in range(0, len(test.targets), batch)]).numpy()
    ridge = np.load(output / "ridge.npz", allow_pickle=False)
    for key in ("xmean", "xstd"):
        if not np.array_equal(ridge[key], checkpoint["normalization"][key]):
            raise ValueError("La normalización de ridge y GRU no coincide.")
    pooled = np.concatenate([part.mean(axis=1) for part in np.array_split(
        test.features, cfg["ridge"]["bins_temporales"], axis=1)], axis=1).astype(np.float64)
    predictions = {"gru": gru, "ridge": (pooled - ridge["input_mean"]) @ ridge["weights"]}
    with (output / "predicciones_test.csv").open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    if [e["word_index"] for e in test.events] != [e["word_index"] for e in reference]:
        raise ValueError("Las palabras no coinciden con las del test guardado.")
    norm = checkpoint["normalization"]
    errors = {}
    metrics = {}
    for name, values in predictions.items():
        expected = np.array([float(row[f"{name}_log10"]) for row in reference])
        actual = values * norm["ystd"] + norm["ymean"]
        errors[name] = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(f"La recarga de {name} no reproduce las predicciones: {errors[name]}")
        y = test.targets.astype(np.float64)
        prediction = values.astype(np.float64)
        mse = float(np.mean((y - prediction) ** 2))
        metrics[name] = {"pearson": float(np.corrcoef(y, prediction)[0, 1]),
                         "r2": 1 - mse / float(np.var(y)), "mse_z": mse}
    report = {"recarga_verificada": True, "dispositivo_reevaluacion": "cpu",
              "ejemplos": len(test.targets), "test": metrics,
              "error_maximo_log10_frente_csv": errors,
              "checkpoint_sha256": hashlib.sha256((output / "gru.pt").read_bytes()).hexdigest(),
              "ridge_sha256": hashlib.sha256((output / "ridge.npz").read_bytes()).hexdigest()}
    (output / "reevaluacion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
