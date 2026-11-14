"""PER y tiempos por perfil con reproducción causal de bloques T12."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from src.data.preprocessing import (
    TrialNormalization, extract_model_features, normalization_spec, session_trial_normalizations,
)
from src.data.speech import BIN_SECONDS, load_session
from src.data.speech_dataset import SpeechDataset, file_sha256, preparation_signature
from src.data.speech_protocol import read_speech_protocol
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
    # Estadísticas de la frase en los modos bloque y rodante; None con normalización fija.
    normalization: TrialNormalization | None = None


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
    fixed = normalization_spec(cfg)["modo"] == "train_fija"
    protocol = read_speech_protocol(cfg)
    partitions = {entry["session"]: entry["train"] for entry in protocol["sessions"]} if protocol else {}
    for day, name in enumerate(dataset.sessions):
        path = Path(cfg["datos"]["raiz"]) / source_split / f"{name}.mat"
        if file_sha256(path) != sources[name]["source_sha256"]:
            raise ValueError(f"La grabación cruda ha cambiado: {path}")
        recording = load_session(path)
        per_trial = None if fixed else session_trial_normalizations(
            cfg, name, train_indices=partitions.get(name),
        )[source_split]
        for index in dataset.session_indices[day]:
            sample = dataset[index]
            trial = recording.trials[sample.trial_index]
            raw = extract_model_features(trial)
            statistics = None if per_trial is None else per_trial[sample.trial_index]
            normalized = (statistics or checkpoint.normalization[name]).transform_features(raw, name)
            if (trial.index != sample.trial_index or trial.block != sample.block
                    or not np.array_equal(normalized, sample.features.numpy())):
                raise ValueError("La señal cruda y sus estadísticas no reproducen el ensayo preparado.")
            yield OnlineTrial(raw, tuple(sample.targets.tolist()), name, trial.index, trial.block,
                              sample.reference, statistics)
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
    opening_ms, stream = _timed(lambda: decoder.start(trial.session, layers=(depth,), normalization=trial.normalization), device)
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


def _complete_logits(
    decoder: OnlineSpeechDecoder, trial: OnlineTrial, exits: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    checkpoint = decoder.checkpoint
    session_id = checkpoint.sessions.index(trial.session)
    statistics = trial.normalization or checkpoint.normalization[trial.session]
    normalized = statistics.transform_features(trial.raw, trial.session)
    features = torch.from_numpy(normalized).unsqueeze(0).to(checkpoint.model.initial_state)
    return checkpoint.model.forward_exits(checkpoint.smoother(features),
                                         torch.tensor([session_id], device=features.device), layers=exits)


def _stream_logits(
    decoder: OnlineSpeechDecoder, trial: OnlineTrial, depth: int, block_bins: int,
) -> tuple[torch.Tensor, list[int]]:
    # Captura solo durante el diagnóstico, nunca durante las medidas de tiempo.
    model = decoder.checkpoint.model
    head = model.output_layer if depth == model.num_layers else model.intermediate_heads[str(depth)]
    chunks = []

    def capture(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        chunks.append(output.detach().clone())

    handle = head.register_forward_hook(capture)
    try:
        stream = decoder.start(trial.session, layers=(depth,), normalization=trial.normalization)
        for start in range(0, len(trial.raw), block_bins):
            stream.push(trial.raw[start:start + block_bins])
        stream.finish()
    finally:
        handle.remove()
    logits = torch.cat(chunks, dim=1) if chunks else model.initial_state.new_empty((1, 0, head.out_features))
    return logits, stream.hypotheses[depth]


def _compare_logits(complete: torch.Tensor, streamed: torch.Tensor, spec: dict[str, Any]) -> dict[str, Any]:
    result = {"complete_shape": list(complete.shape), "streamed_shape": list(streamed.shape),
              "logits_atol": spec["logits_atol"], "logits_rtol": spec["logits_rtol"], "logits_close": False}
    if complete.shape != streamed.shape:
        return result
    result["finite"] = bool(torch.isfinite(complete).all() & torch.isfinite(streamed).all())
    if not result["finite"]:
        return result
    difference = (complete - streamed).abs()
    result.update(logits_close=bool(torch.allclose(streamed, complete, atol=spec["logits_atol"], rtol=spec["logits_rtol"])),
                  max_abs_error=float(difference.max()) if difference.numel() else 0.0,
                  mean_abs_error=float(difference.mean()) if difference.numel() else 0.0,
                  argmax_different_frames=int((complete.argmax(-1) != streamed.argmax(-1)).sum()))
    return result


def _decode_logits(logits: torch.Tensor, blank_id: int) -> list[int]:
    return decode_ctc(logits, torch.tensor([logits.shape[1]]), blank_id)[0]


def _evaluation_exits(decoder: OnlineSpeechDecoder, spec: dict[str, Any]) -> tuple[int, ...]:
    available = decoder.checkpoint.model.available_exits
    exits = available if spec["salidas"] is None else tuple(spec["salidas"])
    if (spec["split"] not in ("val", "test")
            or type(spec["bloque_bins"]) is not int or spec["bloque_bins"] <= 0
            or type(spec["calentamiento"]) is not int or spec["calentamiento"] < 0
            or not exits or len(set(exits)) != len(exits)
            or any(type(depth) is not int or depth not in available for depth in exits)
            or any(not math.isfinite(spec[key]) or spec[key] < 0 for key in ("logits_atol", "logits_rtol"))):
        raise ValueError("Split, bloques, calentamiento, salidas o tolerancias de evaluación no válidos.")
    return exits


@torch.inference_mode()
def diagnose_online_trial(
    decoder: OnlineSpeechDecoder, trial: OnlineTrial, spec: dict[str, Any],
) -> dict[str, Any]:
    """Compara logits completos y por bloques de una frase, sin medir PER ni tiempos."""
    exits = _evaluation_exits(decoder, spec)
    complete = _complete_logits(decoder, trial, exits)
    profiles = {}
    for depth in exits:
        streamed, ids = _stream_logits(decoder, trial, depth, spec["bloque_bins"])
        reference = _decode_logits(complete[depth], decoder.checkpoint.blank_id)
        profiles[depth] = {**_compare_logits(complete[depth], streamed, spec),
                           "complete_ids": reference, "streamed_ids": ids, "matches_complete": ids == reference,
                           "incremental_matches_collapsed": ids == _decode_logits(streamed, decoder.checkpoint.blank_id)}
    return {"session": trial.session, "source_trial_index": trial.trial_index, "split": spec["split"],
            "optimizer_steps": 0, "profiles": profiles,
            "all_logits_close": all(profile["logits_close"] for profile in profiles.values()),
            "all_checks_passed": all(profile["logits_close"] and profile["incremental_matches_collapsed"]
                                     for profile in profiles.values())}


@torch.inference_mode()
def evaluate_online(
    decoder: OnlineSpeechDecoder, trials: Iterable[OnlineTrial], spec: dict[str, Any],
    *, expected_trials: int, prediction_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evalúa cada prefijo por separado y contrasta CTC con la pasada causal completa.

    PER suma errores y fonemas. Los tiempos excluyen lectura, referencia
    completa, diagnóstico de equivalencia, PER y escritura. El reloj de llegada se simula sin esperas reales:
    cada bloque está disponible cuando han llegado todos sus bins de 20 ms.
    El primer fonema (incluido SIL) se mide desde el inicio de la frase,
    sin alineación fonética ni transporte HTTP.
    """
    model = decoder.checkpoint.model
    exits = _evaluation_exits(decoder, spec)
    if expected_trials <= 0:
        raise ValueError("La partición de evaluación debe contener frases.")
    totals = {depth: _counts() for depth in exits}
    sessions: dict[int, dict[str, dict[str, int]]] = {depth: {} for depth in exits}
    timing = {depth: {"processing_ms": 0.0, "signal_ms": 0.0,
                      "packet_ms": [], "first_input_ms": [], "first_ready_ms": []} for depth in exits}
    count = exact_matches = 0
    ctc_mismatches = {depth: 0 for depth in exits}
    for trial in trials:
        if not trial.reference_ids or any(phone < 0 or phone >= decoder.checkpoint.blank_id
                                          for phone in trial.reference_ids):
            raise ValueError("La referencia debe contener fonemas válidos, sin blank.")
        outputs = _complete_logits(decoder, trial, exits)
        complete = {depth: _decode_logits(logits, decoder.checkpoint.blank_id)
                    for depth, logits in outputs.items()}
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
            matches = profile["ids"] == complete[depth]
            if not matches:
                streamed, replay_ids = _stream_logits(decoder, trial, depth, spec["bloque_bins"])
                comparison = _compare_logits(outputs[depth], streamed, spec)
                comparison.update(complete_ids=complete[depth], replay_matches_measured=replay_ids == profile["ids"],
                                  incremental_matches_collapsed=replay_ids == _decode_logits(streamed, decoder.checkpoint.blank_id))
                profile["equivalence"] = comparison
                if not all(comparison[key] for key in
                           ("logits_close", "replay_matches_measured", "incremental_matches_collapsed")):
                    raise ValueError(f"Diferencia de logits o repetición online inestable: {trial.session}, "
                                     f"ensayo {trial.trial_index}, salida {depth}. "
                                     f"Diagnóstico: {json.dumps(comparison, ensure_ascii=False)}")
                ctc_mismatches[depth] += 1
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
                           rtf=profile["processing_ms"] / signal_ms, matches_complete=matches)
            row["profiles"][depth] = profile
        exact_matches += all(profile["matches_complete"] for profile in row["profiles"].values())
        count += 1
        if prediction_sink is not None:
            prediction_sink(row)
    if count != expected_trials:
        raise ValueError("La evaluación online no recorrió exactamente la partición completa.")
    return {"split": spec["split"], "test_evaluated": spec["split"] == "test", "optimizer_steps": 0,
            "available_exits": list(model.available_exits), "evaluated_exits": list(exits),
            "complete_sequence_matches": exact_matches, "ctc_mismatches_with_close_logits": ctc_mismatches,
            "bin_ms": BIN_SECONDS * 1000,
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
            "timing_scope": "perfiles independientes, pesos compartidos residentes; apertura, normalización, filtro, ventanas, GRU, CTC y cierre; sin disco, pasada completa, diagnóstico de equivalencia, PER ni escritura",
            "arrival_clock": "simulado sin sleep ni red; bloque disponible en stop_bin * bin_ms; consumidor secuencial con cola",
            "equivalence_reference": "mismo checkpoint causal, pasada completa y CTC voraz; si CTC difiere, se exige repetición estable y logits dentro de tolerancia"}
