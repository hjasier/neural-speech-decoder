"""Recuentos de PER/WER y tiempos de la GRU causal conectada al trigrama."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import re
from typing import Any

import torch

from src.data.speech import BIN_SECONDS
from src.inference.speech import OnlineSpeechDecoder
from src.inference.trigram import TrigramDecoder
from src.training.metrics import edit_distance
from src.training.speech_online_evaluation import OnlineTrial, _distribution, _evaluation_exits, _timed


def word_tokens(text: str) -> list[str]:
    """Usa la limpieza inglesa de las etiquetas: minúsculas, apóstrofos y guiones."""
    return re.sub(r"[^a-zA-Z\- ']", "", text.strip()).replace("--", "").lower().split()


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Cuenta edición entre palabras, conservando el denominador de referencia."""
    expected, actual = word_tokens(reference), word_tokens(hypothesis)
    if not expected:
        raise ValueError("La referencia WER no contiene palabras; no se omite el ensayo.")
    vocabulary = {word: index for index, word in enumerate(dict.fromkeys(expected + actual))}
    return edit_distance([vocabulary[word] for word in expected], [vocabulary[word] for word in actual]), len(expected)


@torch.inference_mode()
def check_trigram_streaming(
    decoder: OnlineSpeechDecoder, language: TrigramDecoder, trial: OnlineTrial, spec: dict[str, Any],
) -> dict[str, Any]:
    """Decodifica los mismos logits agrupados o por bloques, aislando el estado WFST."""
    profiles = {}
    for depth in _evaluation_exits(decoder, spec):
        neural = decoder.start(trial.session, layers=(depth,))
        incremental = language.start()
        chunks = []
        for start in range(0, len(trial.raw), spec["bloque_bins"]):
            logits = neural.push_with_logits(trial.raw[start:start + spec["bloque_bins"]]).logits[depth]
            incremental.push(logits)
            if logits.shape[0]:
                chunks.append(logits.cpu())
        neural.finish()
        if not chunks:
            raise ValueError("El ensayo no produjo logits para comprobar el WFST incremental.")
        full = language.start()
        full.push(torch.cat(chunks))
        complete_text, streamed_text = full.finish(), incremental.finish()
        profiles[depth] = {"matches": complete_text == streamed_text, "complete": complete_text,
                           "streamed": streamed_text, "frames": sum(len(chunk) for chunk in chunks)}
    return {"split": spec["split"], "session": trial.session, "source_trial_index": trial.trial_index,
            "profiles": profiles, "verified": all(profile["matches"] for profile in profiles.values()),
            "optimizer_steps": 0, "scope": "mismos logits causales; cambia solo su agrupación al entrar al WFST"}


def _profile(
    decoder: OnlineSpeechDecoder, language: TrigramDecoder, trial: OnlineTrial, depth: int, block_bins: int,
) -> dict[str, Any]:
    device = decoder.checkpoint.model.initial_state.device

    def open_streams() -> tuple:
        return decoder.start(trial.session, layers=(depth,)), language.start()

    opening_ms, (neural, words) = _timed(open_streams, device)
    processing_ms = ready_ms = opening_ms
    packets = []
    first_input = first_ready = None
    for start in range(0, len(trial.raw), block_bins):
        stop = min(start + block_bins, len(trial.raw))

        def push() -> str:
            return words.push(neural.push_with_logits(trial.raw[start:stop]).logits[depth])

        elapsed, text = _timed(push, device)
        input_ms = stop * BIN_SECONDS * 1000
        ready_ms = max(ready_ms, input_ms) + elapsed
        processing_ms += elapsed
        if text and first_input is None:
            first_input, first_ready = input_ms, ready_ms
        packets.append({"stop_bin": stop, "input_ms": input_ms, "processing_ms": elapsed,
                        "ready_ms": ready_ms, "partial_text": text})

    def finish() -> str:
        neural.finish()
        return words.finish()

    closing_ms, final_text = _timed(finish, device)
    processing_ms += closing_ms
    ready_ms += closing_ms
    if final_text and first_input is None:
        first_input, first_ready = len(trial.raw) * BIN_SECONDS * 1000, ready_ms
    return {"text": final_text, "phone_ids": neural.hypotheses[depth], "packets": packets,
            "opening_ms": opening_ms, "closing_ms": closing_ms, "processing_ms": processing_ms,
            "first_partial_input_ms": first_input, "first_partial_ready_ms": first_ready}


@torch.inference_mode()
def evaluate_trigram(
    decoder: OnlineSpeechDecoder, language: TrigramDecoder, trials: Iterable[OnlineTrial], spec: dict[str, Any],
    *, expected_trials: int, prediction_sink: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Evalúa el split completo; WER y PER agregan errores, sin promediar porcentajes."""
    exits = _evaluation_exits(decoder, spec)
    totals = {depth: {"errores_palabras": 0, "palabras": 0, "errores_fonemas": 0, "fonemas": 0, "frases": 0}
              for depth in exits}
    sessions = {depth: {} for depth in exits}
    timing = {depth: {"processing_ms": 0.0, "signal_ms": 0.0, "packet_ms": [], "first_input": [], "first_ready": []}
              for depth in exits}
    count = 0
    for trial in trials:
        if not trial.reference_ids:
            raise ValueError("Falta la referencia de fonemas del ensayo.")
        row = {"split": spec["split"], "source_split": "train" if spec["split"] == "val" else "test",
               "session": trial.session, "source_trial_index": trial.trial_index, "block": trial.block,
               "input_bins": len(trial.raw),
               "reference_text": trial.reference_text, "reference_ids": list(trial.reference_ids), "profiles": {}}
        for depth in exits:
            if count == 0:
                for _ in range(spec["calentamiento"]):
                    _profile(decoder, language, trial, depth, spec["bloque_bins"])
            profile = _profile(decoder, language, trial, depth, spec["bloque_bins"])
            errors, words = word_error_counts(trial.reference_text, profile["text"])
            phones = edit_distance(trial.reference_ids, profile["phone_ids"])
            counts = {"errores_palabras": errors, "palabras": words, "errores_fonemas": phones,
                      "fonemas": len(trial.reference_ids), "frases": 1}
            for aggregate in (totals[depth], sessions[depth].setdefault(trial.session, dict.fromkeys(counts, 0))):
                for key, value in counts.items():
                    aggregate[key] += value
            measured = timing[depth]
            measured["processing_ms"] += profile["processing_ms"]
            measured["signal_ms"] += len(trial.raw) * BIN_SECONDS * 1000
            measured["packet_ms"].extend(packet["processing_ms"] for packet in profile["packets"])
            if profile["first_partial_input_ms"] is not None:
                measured["first_input"].append(profile["first_partial_input_ms"])
                measured["first_ready"].append(profile["first_partial_ready_ms"])
            row["profiles"][depth] = {**profile, **counts}
        prediction_sink(row)
        count += 1
    if count != expected_trials or count == 0:
        raise ValueError("La evaluación del trigrama no recorrió exactamente el split completo.")

    def metrics(counts: dict[str, int]) -> dict[str, Any]:
        return {**counts, "wer": counts["errores_palabras"] / counts["palabras"],
                "per": counts["errores_fonemas"] / counts["fonemas"], "completo": 1}

    return {"split": spec["split"], "test_evaluated": spec["split"] == "test", "optimizer_steps": 0,
            "available_exits": list(decoder.checkpoint.model.available_exits), "evaluated_exits": list(exits),
            "global": {depth: metrics(value) for depth, value in totals.items()},
            "by_session": {depth: {name: metrics(value) for name, value in days.items()} for depth, days in sessions.items()},
            "timing": {depth: {"rtf": value["processing_ms"] / value["signal_ms"],
                               "processing_ms": value["processing_ms"], "signal_ms": value["signal_ms"],
                               "packet_processing": _distribution(value["packet_ms"]),
                               "first_partial_input": _distribution(value["first_input"]),
                               "first_partial_ready": _distribution(value["first_ready"])} for depth, value in timing.items()},
            "wer_normalization": "misma limpieza inglesa que etiquetas: minúsculas, letras, apóstrofos y guiones; split por espacios",
            "timing_scope": "GRU, normalización, filtro, ventanas, CTC, transferencia CPU, IPC y WFST; sin disco, métricas ni carga de modelos",
            "arrival_clock": "simulado sin red ni sleep; bloques disponibles en stop_bin * 20 ms; parciales revisables"}
