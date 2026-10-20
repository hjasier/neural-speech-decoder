"""Piloto propio EEG→potencia acústica: media, ridge, GRU y control temporal."""

from __future__ import annotations

import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.alice import prepare_alice
from src.models.factory import construir_modelo
from src.utils.seed import crear_generador, crear_generador_numpy
from src.utils.tracking import registrar


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Predicciones no finitas.")
    y, prediction = y.astype(np.float64), prediction.astype(np.float64)
    error = float(np.mean((y - prediction) ** 2))
    pearson = float(np.corrcoef(y, prediction)[0, 1]) if np.std(y) > 0 and np.std(prediction) > 0 else None
    variance = float(np.var(y))
    return {"pearson": pearson, "r2": 1 - error / variance if variance > 0 else None, "mse_z": error}


def _predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu()
                          for i in range(0, len(x), batch_size)]).numpy()


def _control(y: np.ndarray, prediction: np.ndarray, events: list[dict], cfg: dict) -> dict:
    rng = crear_generador_numpy(cfg["experimento"]["seed"])
    labels = np.array([int(e["segment"]) for e in events])
    groups = [np.flatnonzero(labels == seg) for seg in np.unique(labels)]
    minimum = cfg["evaluacion"]["minimo_desplazamiento_palabras"]
    if any(len(g) <= 2 * minimum for g in groups):
        raise ValueError("No hay suficientes palabras para el control de desplazamientos.")
    observed = _metrics(y, prediction)["pearson"]
    null = []
    for _ in range(cfg["evaluacion"]["desplazamientos"]):
        shifted = y.copy()
        for group in groups:
            offset = int(rng.integers(minimum, len(group) - minimum + 1))
            shifted[group] = np.roll(y[group], offset)
        null.append(_metrics(shifted, prediction)["pearson"])
    valid = [value for value in null if value is not None]
    return {"pearson_observado": observed, "pearson_desplazados": valid,
            "p_empirica_unilateral": ((1 + sum(v >= observed for v in valid)) / (1 + len(valid))
                                      if observed is not None and valid else None),
            "tipo": "rotación de etiquetas dentro de cada segmento test; exploratorio"}


def ejecutar_alice(cfg: dict[str, Any], run: Any) -> dict[str, float]:
    """Compara modelos con selección en val y evalúa test solo al final."""
    if run is None:
        raise ValueError("El experimento requiere registro W&B online u offline.")
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    requested = cfg["ejecucion"]["dispositivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA y no está disponible.")
    start = perf_counter()
    prepared = prepare_alice(cfg)
    train, val, test = (prepared.splits[k] for k in ("train", "val", "test"))
    output = Path(cfg["salida"]["directorio"]) / run.id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    (output / "datos.json").write_text(json.dumps(prepared.report, indent=2, ensure_ascii=False) + "\n")
    print(f"Alice {cfg['datos']['sujeto']}: {prepared.report['ejemplos']}  "
          f"ventana={prepared.report['forma_ventana']}  dispositivo={device}", flush=True)
    print("Entradas: solo EEG; excluidos:", prepared.report["canales_excluidos"], flush=True)
    bins = cfg["ridge"]["bins_temporales"]
    if not 0 < bins <= train.features.shape[1]:
        raise ValueError("Número de bins temporales de ridge no válido.")
    pooled = {name: np.concatenate([part.mean(axis=1) for part in np.array_split(split.features, bins, axis=1)], axis=1).astype(np.float64)
              for name, split in prepared.splits.items()}
    xmean = pooled["train"].mean(axis=0)
    centered = pooled["train"] - xmean
    gram = centered.T @ centered
    rhs = centered.T @ train.targets.astype(np.float64)
    candidates = []
    ridge_start = perf_counter()
    for alpha in cfg["ridge"]["alphas"]:
        if alpha <= 0:
            raise ValueError("Los alphas de ridge deben ser positivos.")
        weights = np.linalg.solve(gram + alpha * np.eye(len(gram)), rhs)
        prediction = (pooled["val"] - xmean) @ weights
        candidates.append((float(_metrics(val.targets, prediction)["mse_z"]), alpha, weights))
    _, best_alpha, weights = min(candidates, key=lambda item: item[0])
    ridge_seconds = perf_counter() - ridge_start
    np.savez_compressed(output / "ridge.npz", weights=weights, input_mean=xmean,
                        xmean=np.array(prepared.normalization["xmean"]),
                        xstd=np.array(prepared.normalization["xstd"]),
                        target_mean=prepared.normalization["ymean"], target_std=prepared.normalization["ystd"])
    print(f"Ridge seleccionado en val: alpha={best_alpha:g}, {ridge_seconds:.2f} s", flush=True)
    model = construir_modelo(cfg).to(device)
    training = cfg["entrenamiento"]
    loader = DataLoader(TensorDataset(torch.from_numpy(train.features), torch.from_numpy(train.targets)),
                        batch_size=training["batch_size"], shuffle=True, num_workers=0,
                        generator=crear_generador(cfg["experimento"]["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    gru_start = perf_counter()
    for epoch in range(1, training["epocas"] + 1):
        model.train()
        loss_sum = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("La pérdida GRU no es finita.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["clip_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(targets)
        validation = _metrics(val.targets, _predict(model, val.features, training["batch_size"], device))
        epoch_metrics = {"epoca": epoch, "train_mse": loss_sum / len(train.targets), **validation}
        history.append(epoch_metrics)
        registrar(run, {"gru/epoca": epoch, "gru/train_mse": epoch_metrics["train_mse"],
                        **{f"gru/val_{k}": v for k, v in validation.items() if v is not None}}, epoch)
        print(f"Época {epoch}: train MSE={epoch_metrics['train_mse']:.4f}  val MSE={validation['mse_z']:.4f}  "
              f"val r={validation['pearson']}", flush=True)
        if validation["mse_z"] < best_loss:
            best_loss, best_epoch, stale = validation["mse_z"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= training["paciencia"] or perf_counter() - gru_start >= training["max_segundos"]:
            break
    if best_state is None:
        raise ValueError("No se ha completado una época válida.")
    model.load_state_dict(best_state)
    gru_seconds = perf_counter() - gru_start
    predictions = {"media": np.full_like(test.targets, train.targets.mean()),
                   "ridge": (pooled["test"] - xmean) @ weights,
                   "gru": _predict(model, test.features, training["batch_size"], device)}
    results = {name: _metrics(test.targets, pred) for name, pred in predictions.items()}
    controls = {name: _control(test.targets, predictions[name], test.events, cfg) for name in ("ridge", "gru")}
    report = {"tarea": "EEG a log10(sound_power) por palabra; escucha, no habla intentada",
              "run_id": run.id, "hardware": str(device), "datos": prepared.report,
              "reproducibilidad": {
                  "python": platform.python_version(), "sistema": platform.platform(),
                  "paquetes": {name: version(name) for name in ("torch", "numpy", "scipy", "wandb", "pyyaml")},
                  "fuentes_sha256": {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in [Path("run.py"), *sorted(Path("src").rglob("*.py"))]},
                  "parametros_gru": sum(p.numel() for p in model.parameters())},
              "normalizacion": prepared.normalization, "ridge_alpha": best_alpha,
              "ridge_validacion": [{"alpha": a, "mse_z": score} for score, a, _ in candidates],
              "gru_mejor_epoca": best_epoch, "gru_epocas_completadas": epoch, "historial": history,
              "test": results, "controles_test": controls,
              "segundos": {"ridge_ajuste": ridge_seconds, "gru_ajuste": gru_seconds,
                           "total": perf_counter() - start},
              "limites": ["Un participante y un audiolibro; no generalización entre personas.",
                          "Sin limpieza ICA ni rechazo manual de artefactos.",
                          "Onsets de palabras conocidos; no es un decodificador autónomo de texto.",
                          "Las épocas pueden solaparse dentro de cada partición.",
                          "El control temporal es exploratorio, no una conclusión poblacional."]}
    torch.save({"model_state": best_state, "config": cfg, "normalization": prepared.normalization,
                "data_report": prepared.report, "best_epoch": best_epoch}, output / "gru.pt")
    (output / "resultado.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "predicciones_test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["word_index", "word_referencia_no_entrada", "segmento", "real_log10", "ridge_log10", "gru_log10"])
        norm = prepared.normalization
        for i, event in enumerate(test.events):
            writer.writerow([event["word_index"], event["word"], event["segment"],
                             test.targets[i] * norm["ystd"] + norm["ymean"],
                             predictions["ridge"][i] * norm["ystd"] + norm["ymean"],
                             predictions["gru"][i] * norm["ystd"] + norm["ymean"]])
    flat = {f"alice/test_{name}_{key}": value for name, values in results.items()
            for key, value in values.items() if value is not None}
    flat.update({f"alice/{name}_p_desplazamiento": values["p_empirica_unilateral"]
                 for name, values in controls.items() if values["p_empirica_unilateral"] is not None})
    run.summary.update({**flat, "alice/ridge_alpha": best_alpha, "alice/gru_mejor_epoca": best_epoch,
                        "alice/resultado": str(output / "resultado.json"), "alice/segundos": report["segundos"]["total"]})
    registrar(run, flat, epoch + 1)
    print("Resultado:", output / "resultado.json", flush=True)
    return flat
