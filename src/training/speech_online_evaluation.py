"""PER y tiempos por perfil con reproducción causal de bloques T12."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from src.data.preprocessing import extract_model_features
from src.data.speech import BIN_SECONDS, load_session
from src.data.speech_dataset import SpeechDataset, file_sha256, preparation_signature
from src.inference.ctc import decode_ctc
from src.inference.speech import OnlineSpeechDecoder
from src.training.metrics import edit_distance


@dataclass(frozen=True)
class OnlineTrial:
    """Señal cruda y referencia separadas para evaluar una frase completa."""

    raw: NDArray[np.float32]
    reference_ids: tuple[int, ...]
    session: str
    trial_index: int
    block: int
    reference_text: str


def iter_online_trials(dataset: SpeechDataset, decoder: OnlineSpeechDecoder) -> Iterator[OnlineTrial]:
    """Lee una sesión cruda cada vez y verifica señal, partición y normalización.

    La caché preparada aporta referencias y selección de ensayos; su señal
    normalizada solo se usa para verificar la correspondencia con el original.
    El motor recibe siempre tx1/spikePow crudo, sin normalizar dos veces.
    """
    checkpoint = decoder.checkpoint
    cfg = checkpoint.config
    if (dataset.split not in ("val", "test") or dataset.sessions != checkpoint.sessions
            or preparation_signature(cfg) != checkpoint.data_signature):
        raise ValueError("Los datos no corresponden al checkpoint o al split de evaluación.")
    manifest = json.loads((dataset.root / "manifest.json").read_text(encoding="utf-8"))
    source_split = "train" if dataset.split == "val" else "test"
    sources = {entry["session"]: entry for entry in manifest["files"] if entry["split"] == source_split}
    for day, name in enumerate(dataset.sessions):
        path = Path(cfg["datos"]["raiz"]) / source_split / f"{name}.mat"
        if file_sha256(path) != sources[name]["source_sha256"]:
            raise ValueError(f"La grabación cruda ha cambiado: {path}")
        recording = load_session(path)
        for index in dataset.session_indices[day]:
            sample = dataset[index]
            trial = recording.trials[sample.trial_index]
            raw = extract_model_features(trial)
            normalized = checkpoint.normalization[name].transform_features(raw, name)
            if (trial.index != sample.trial_index or trial.block != sample.block
                    or not np.array_equal(normalized, sample.features.numpy())):
                raise ValueError("La señal cruda y sus estadísticas no reproducen el ensayo preparado.")
            yield OnlineTrial(raw, tuple(sample.targets.tolist()), name, trial.index, trial.block, sample.reference)
        del recording


def _timed(call: Callable[[], Any], device: torch.device) -> tuple[float, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    value = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (perf_counter() - start) * 1000, value


def _run_profile(
    decoder: OnlineSpeechDecoder, trial: OnlineTrial, depth: int, block_bins: int,
) -> dict[str, Any]:
    device = decoder.checkpoint.model.initial_state.device
    opening_ms, stream = _timed(lambda: decoder.start(trial.session, layers=(depth,)), device)
    processing_ms = opening_ms
    ready_ms = opening_ms
    first_input_ms = first_ready_ms = None
    packets = []
    for start in range(0, len(trial.raw), block_bins):
        stop = min(start + block_bins, len(trial.raw))
        packet = trial.raw[start:stop]
        elapsed_ms, updates = _timed(lambda: stream.push(packet), device)
        input_ms = stop * BIN_SECONDS * 1000
        # Un único consumidor: si el cálculo tarda más que el bloque, acumula cola.
        ready_ms = max(ready_ms, input_ms) + elapsed_ms
        processing_ms += elapsed_ms
        if updates[depth] and first_input_ms is None:
            first_input_ms, first_ready_ms = input_ms, ready_ms
        packets.append({"stop_bin": stop, "input_ms": input_ms, "processing_ms": elapsed_ms,
                        "ready_ms": ready_ms, "new_ids": updates[depth]})
    closing_ms, updates = _timed(stream.finish, device)
    processing_ms += closing_ms
    ready_ms += closing_ms
    if updates[depth] and first_input_ms is None:
        first_input_ms = len(trial.raw) * BIN_SECONDS * 1000
        first_ready_ms = ready_ms
    return {"ids": stream.hypotheses[depth], "packets": packets,
            "opening_ms": opening_ms, "closing_ms": closing_ms, "processing_ms": processing_ms,
            "first_phone_input_ms": first_input_ms, "first_phone_ready_ms": first_ready_ms}


def _counts() -> dict[str, int]:
    return {"errores": 0, "fonemas": 0, "frases": 0}


def _per(counts: dict[str, int]) -> dict[str, float | int]:
    return {**counts, "per": counts["errores"] / counts["fonemas"], "completo": 1}


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {"observations": len(values),
            "p50_ms": float(np.median(values)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None}


@torch.inference_mode()
def evaluate_online(
    decoder: OnlineSpeechDecoder, trials: Iterable[OnlineTrial], spec: dict[str, Any],
    *, expected_trials: int, prediction_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evalúa cada prefijo por separado y contrasta CTC con la pasada causal completa.

    PER suma errores y fonemas. Los tiempos excluyen lectura, referencia
    completa, PER y escritura. El reloj de llegada se simula sin esperas reales:
    cada bloque está disponible cuando han llegado todos sus bins de 20 ms.
    El primer fonema (incluido SIL) se mide desde el inicio de la frase,
    sin alineación fonética ni transporte HTTP.
    """
    model = decoder.checkpoint.model
    exits = model.available_exits if spec["salidas"] is None else tuple(spec["salidas"])
    if (spec["split"] not in ("val", "test") or expected_trials <= 0
            or type(spec["bloque_bins"]) is not int or spec["bloque_bins"] <= 0
            or type(spec["calentamiento"]) is not int or spec["calentamiento"] < 0
            or not exits or len(set(exits)) != len(exits)
            or any(type(depth) is not int or depth not in model.available_exits for depth in exits)):
        raise ValueError("Split, bloques, calentamiento o salidas de evaluación no válidos.")
    totals = {depth: _counts() for depth in exits}
    sessions: dict[int, dict[str, dict[str, int]]] = {depth: {} for depth in exits}
    timing = {depth: {"processing_ms": 0.0, "signal_ms": 0.0,
                      "packet_ms": [], "first_input_ms": [], "first_ready_ms": []} for depth in exits}
    count = 0
    for trial in trials:
        if not trial.reference_ids or any(phone < 0 or phone >= decoder.checkpoint.blank_id
                                          for phone in trial.reference_ids):
            raise ValueError("La referencia debe contener fonemas válidos, sin blank.")
        session_id = decoder.checkpoint.sessions.index(trial.session)
        normalized = decoder.checkpoint.normalization[trial.session].transform_features(trial.raw, trial.session)
        features = torch.from_numpy(normalized).unsqueeze(0).to(model.initial_state)
        outputs = model.forward_exits(decoder.checkpoint.smoother(features),
                                      torch.tensor([session_id], device=features.device), layers=exits)
        complete = {depth: decode_ctc(logits, torch.tensor([logits.shape[1]]), decoder.checkpoint.blank_id)[0]
                    for depth, logits in outputs.items()}
        del outputs, features
        row = {"split": spec["split"], "source_split": "train" if spec["split"] == "val" else "test",
               "session": trial.session, "source_trial_index": trial.trial_index, "block": trial.block,
               "input_bins": len(trial.raw), "reference_ids": list(trial.reference_ids),
               "reference_text": trial.reference_text, "profiles": {}}
        signal_ms = len(trial.raw) * BIN_SECONDS * 1000
        for depth in exits:
            if count == 0:
                for _ in range(spec["calentamiento"]):
                    _run_profile(decoder, trial, depth, spec["bloque_bins"])
            profile = _run_profile(decoder, trial, depth, spec["bloque_bins"])
            if profile["ids"] != complete[depth]:
                raise ValueError(f"Bloques y secuencia causal completa difieren: {trial.session}, ensayo {trial.trial_index}, salida {depth}.")
            errors = edit_distance(trial.reference_ids, profile["ids"])
            for total in (totals[depth], sessions[depth].setdefault(trial.session, _counts())):
                total["errores"] += errors
                total["fonemas"] += len(trial.reference_ids)
                total["frases"] += 1
            measured = timing[depth]
            measured["processing_ms"] += profile["processing_ms"]
            measured["signal_ms"] += signal_ms
            measured["packet_ms"].extend(packet["processing_ms"] for packet in profile["packets"])
            if profile["first_phone_input_ms"] is not None:
                measured["first_input_ms"].append(profile["first_phone_input_ms"])
                measured["first_ready_ms"].append(profile["first_phone_ready_ms"])
            profile.update(errors=errors, per=errors / len(trial.reference_ids),
                           rtf=profile["processing_ms"] / signal_ms, matches_complete=True)
            row["profiles"][depth] = profile
        count += 1
        if prediction_sink is not None:
            prediction_sink(row)
    if count != expected_trials:
        raise ValueError("La evaluación online no recorrió exactamente la partición completa.")
    return {"split": spec["split"], "test_evaluated": spec["split"] == "test", "optimizer_steps": 0,
            "available_exits": list(model.available_exits), "evaluated_exits": list(exits),
            "complete_sequence_matches": count, "bin_ms": BIN_SECONDS * 1000,
            "global": {depth: _per(total) for depth, total in totals.items()},
            "by_session": {depth: {name: _per(total) for name, total in days.items()}
                           for depth, days in sessions.items()},
            "timing": {depth: {"processing_ms": measured["processing_ms"], "signal_ms": measured["signal_ms"],
                               "rtf": measured["processing_ms"] / measured["signal_ms"],
                               "packet_processing": _distribution(measured["packet_ms"]),
                               "first_phone_input": _distribution(measured["first_input_ms"]),
                               "first_phone_ready": _distribution(measured["first_ready_ms"]),
                               "phrases_without_emission": count - len(measured["first_input_ms"])}
                       for depth, measured in timing.items()},
            "timing_scope": "perfiles independientes, pesos compartidos residentes; apertura, normalización, filtro, ventanas, GRU, CTC y cierre; sin disco, pasada completa, PER ni escritura",
            "arrival_clock": "simulado sin sleep ni red; bloque disponible en stop_bin * bin_ms; consumidor secuencial con cola",
            "equivalence_reference": "mismo checkpoint causal, pasada completa y CTC voraz"}
