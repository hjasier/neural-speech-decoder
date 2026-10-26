"""Medición sincronizada de perfiles T12 con señal cruda y muestras comunes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from src.data.preprocessing import TrainingNormalization, extract_model_features
from src.data.speech import SpeechTrial, load_session
from src.data.speech_dataset import SpeechBatch, SpeechDataset, SpeechSample, collate_speech, file_sha256, preparation_signature
from src.models.speech_gru import SpeechGRU
from src.training.metrics import decode_ctc
from src.utils.seed import crear_generador


@dataclass(frozen=True)
class BenchmarkCase:
    """Un ensayo crudo y sus metadatos, conservados en CPU para repetir el pipeline."""

    dataset_index: int
    session: str
    raw: SpeechTrial
    sample: SpeechSample


def validate_benchmark_spec(spec: dict[str, Any]) -> None:
    """Exige cohortes, lotes y repeticiones explícitos en la configuración."""
    if spec["split"] not in ("val", "test"):
        raise ValueError("El benchmark requiere val o test.")
    for key in ("ensayos_por_sesion", "batch_size", "calentamiento", "repeticiones", "hilos_cpu"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError(f"{key} debe ser un entero positivo.")
    if type(spec["semilla_muestras"]) is not int or not 0 <= spec["semilla_muestras"] < 2**32:
        raise ValueError("La semilla de muestras debe ser un entero válido.")


def prepare_benchmark_cases(
    dataset: SpeechDataset, cfg: dict[str, Any], spec: dict[str, Any],
) -> tuple[list[BenchmarkCase], dict[str, TrainingNormalization], dict[str, Any]]:
    """Selecciona ensayos por sesión y verifica sus .mat sin ajustar estadísticas.

    Lee una sesión cada vez y copia solo los ensayos seleccionados. La firma
    incluye señales crudas, estadísticas, preprocesamiento y orden del lote.
    """
    validate_benchmark_spec(spec)
    if dataset.split != spec["split"]:
        raise ValueError("El dataset no corresponde al split del benchmark.")
    names = cfg["datos"]["sesiones"] if spec["sesiones"] is None else spec["sesiones"]
    if (not names or len(set(names)) != len(names)
            or any(name not in dataset.sessions for name in names)):
        raise ValueError("Las sesiones del benchmark deben existir y ser únicas.")
    source_split = "train" if dataset.split == "val" else "test"
    manifest = json.loads((dataset.root / "manifest.json").read_text())
    sources = {(entry["split"], entry["session"]): entry for entry in manifest["files"]}
    generator = crear_generador(spec["semilla_muestras"])
    cases, rows, normalizers = [], [], {}
    start = perf_counter()
    for name in names:
        day = dataset.sessions.index(name)
        indices = dataset.session_indices[day]
        count = spec["ensayos_por_sesion"]
        if len(indices) < count:
            raise ValueError(f"La sesión {name} no tiene suficientes ensayos para el benchmark.")
        selected = sorted(indices[int(i)] for i in torch.randperm(len(indices), generator=generator)[:count])
        path = Path(cfg["datos"]["raiz"]) / source_split / f"{name}.mat"
        source_hash = file_sha256(path)
        if source_hash != sources[(source_split, name)]["source_sha256"]:
            raise ValueError(f"La grabación cruda ha cambiado: {path}")
        recording = load_session(path)
        stats = TrainingNormalization.from_dict(dataset.normalization_states[name])
        normalizers[name] = stats
        for index in selected:
            sample = dataset[index]
            trial = recording.trials[sample.trial_index]
            if trial.index != sample.trial_index or trial.block != sample.block:
                raise ValueError("El ensayo crudo no corresponde a los metadatos preparados.")
            raw = replace(trial, spike_power=trial.spike_power.copy(), threshold_crossings=trial.threshold_crossings.copy())
            normalized = stats.transform(raw, name)
            if not np.array_equal(normalized, sample.features.numpy()):
                raise ValueError("La normalización desde señal cruda no reproduce la caché de entrenamiento.")
            cases.append(BenchmarkCase(index, name, raw, sample))
            rows.append({"dataset_index": index, "session": name, "source_trial_index": trial.index,
                         "block": trial.block, "input_bins": len(normalized), "source_sha256": source_hash,
                         "raw_features_sha256": hashlib.sha256(extract_model_features(raw).tobytes()).hexdigest()})
        del recording
        print(f"Benchmark: {name}, {count} ensayos crudos verificados.", flush=True)
    if len(cases) % spec["batch_size"]:
        raise ValueError("El número de ensayos debe ser divisible por batch_size; no se descartan frases.")
    identity = {"split": dataset.split, "data_signature": preparation_signature(cfg),
                "normalization": dataset.normalization_states, "preprocessing": cfg["preprocesamiento"],
                "batch_size": spec["batch_size"], "samples": rows}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return cases, normalizers, {"input_signature": signature, "samples": rows,
                                "raw_data_preparation_seconds": perf_counter() - start,
                                "data_signature": identity["data_signature"]}


def _raw_batch(cases: list[BenchmarkCase], stats: dict[str, TrainingNormalization]) -> SpeechBatch:
    return collate_speech([replace(case.sample, features=torch.from_numpy(stats[case.session].transform(case.raw, case.session)))
                           for case in cases])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    _sync(device)
    start = perf_counter()
    result = call()
    _sync(device)
    return (perf_counter() - start) * 1000, result


def _clear(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [row["elapsed_ms"] for row in rows]
    return {"observations": len(times), "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "phrases_per_second": sum(row["batch_size"] for row in rows) / (sum(times) / 1000),
            **{key: max(row[key] for row in rows) if rows[0][key] is not None else None
               for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "gpu_extra_allocated_bytes")}}


@torch.inference_mode()
def benchmark_profiles(
    model: SpeechGRU, smoother: nn.Module, cases: list[BenchmarkCase],
    stats: dict[str, TrainingNormalization], spec: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    """Mide forward y señal cruda→fonemas por separado, con warmup y sincronización.

    Las lecturas .mat y la carga de pesos quedan fuera. Los perfiles conservan
    todos los pesos residentes del modelo. Se verifica la parada con hooks y
    la igualdad de fonemas entre ambas rutas, sin utilizar sus referencias.
    """
    validate_benchmark_spec(spec)
    if not cases or len(cases) % spec["batch_size"]:
        raise ValueError("Se necesitan lotes completos y no vacíos.")
    exits = model.available_exits if spec["salidas"] is None else tuple(spec["salidas"])
    if (not exits or len(set(exits)) != len(exits)
            or any(type(depth) is not int or depth not in model.available_exits for depth in exits)):
        raise ValueError("El perfil solicitado no existe.")
    model.eval()
    batches = [cases[i:i + spec["batch_size"]] for i in range(0, len(cases), spec["batch_size"])]
    blank = model.output_layer.out_features - 1
    expected = []
    for batch_cases in batches:
        batch = _raw_batch(batch_cases, stats).to(device)
        outputs = model.forward_exits(smoother(batch.features), batch.session_ids, layers=exits)
        expected.append({depth: decode_ctc(logits, model.output_lengths(batch.input_lengths), blank)
                         for depth, logits in outputs.items()})
        del outputs, batch
    records, profiles = [], {}
    for depth in exits:
        called = []
        handles = [layer.register_forward_pre_hook(lambda _m, _a, n=i: called.append(n))
                   for i, layer in enumerate(model.gru_layers, start=1)]
        try:
            probe = _raw_batch(batches[0], stats).to(device)
            value = model(smoother(probe.features), probe.session_ids, exit_layer=depth)
            del value, probe
        finally:
            for handle in handles:
                handle.remove()
        if called != list(range(1, depth + 1)):
            raise ValueError("El perfil no detiene el cálculo en su capa.")
        for stage in ("forward", "raw_to_phones"):
            _clear(device)
            for batch_index, batch_cases in enumerate(batches):
                if stage == "forward":
                    batch = _raw_batch(batch_cases, stats).to(device)
                    features, sessions = smoother(batch.features), batch.session_ids
                    del batch

                    def invoke() -> Any:
                        return model(features, sessions, exit_layer=depth)
                else:
                    def invoke() -> Any:
                        batch = _raw_batch(batch_cases, stats).to(device)
                        logits = model(smoother(batch.features), batch.session_ids, exit_layer=depth)
                        return decode_ctc(logits, model.output_lengths(batch.input_lengths), blank)

                for _ in range(spec["calentamiento"]):
                    value = invoke()
                    del value
                _sync(device)
                for repeat in range(spec["repeticiones"]):
                    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    milliseconds, value = _timed(invoke, device)
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                    if stage == "raw_to_phones" and value != expected[batch_index][depth]:
                        raise ValueError("El pipeline crudo no reproduce los fonemas de la entrada precalculada.")
                    records.append({"exit": depth, "stage": stage, "batch_index": batch_index,
                                    "repeat": repeat, "batch_size": len(batch_cases),
                                    "max_input_bins": max(len(case.raw.spike_power) for case in batch_cases),
                                    "elapsed_ms": milliseconds, "gpu_peak_allocated_bytes": peak,
                                    "gpu_peak_reserved_bytes": reserved,
                                    "gpu_extra_allocated_bytes": peak - baseline if peak is not None else None})
                    del value
                if stage == "forward":
                    del features, sessions
            profiles[depth] = {"executed_layers": called,
                               **{stage_name: _summary([row for row in records if row["exit"] == depth and row["stage"] == stage_name])
                                  for stage_name in ("forward", "raw_to_phones")
                                  if any(row["exit"] == depth and row["stage"] == stage_name for row in records)}}
        print(f"Benchmark: perfil {depth}, {len(batches)} lotes, forward y señal cruda→fonemas completados.", flush=True)
    return {"profiles": profiles, "measurements": records, "optimizer_steps": 0,
            "decoded_profiles_match": True, "batch_size": spec["batch_size"],
            "timing_unit": "milliseconds per batch; warm synchronized wall time",
            "resident_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "weight_memory_scope": "all model layers and heads remain resident for every exit"}
