"""Carga autocontenida de pesos de habla sin abrir datos ni importar entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.data.preprocessing import (
    GaussianSmoothing, TrainingNormalization, build_smoother, normalization_spec,
)
from src.data.speech import BIN_SECONDS
from src.models.factory import construir_modelo
from src.models.speech_gru import SpeechGRU


def inference_format(cfg: dict[str, Any]) -> int:
    """Distingue pesos causales firmados de los formatos centrados históricos."""
    if cfg["preprocesamiento"]["suavizado"].get("modo", "centered") == "causal":
        return 3
    return 2 if cfg.get("protocolo") else 1


def streaming_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Describe las convenciones de señal, reinicio y cierre del formato causal."""
    mode = normalization_spec(cfg)["modo"]
    return {"feature_order": ["tx1_6v", "spikePow_6v"], "bin_seconds": BIN_SECONDS,
            "normalization": {"train_fija": "fixed_train_per_session", "bloque": "per_block",
                              "rodante": "rolling_previous_sentences"}[mode],
            "reset": "per_phrase",
            "initial_filter_history": "normalized_zeros", "finish": "discard_incomplete_windows"}


def inference_signature(state: dict[str, Any]) -> str:
    """Firma los metadatos que pueden cambiar predicciones con los mismos pesos.

    Incluye arquitectura, filtro, estadísticas y correspondencias de sesiones
    y fonemas. La firma de datos preparada sigue separada: esa caché no se
    suaviza y puede compartirse con los modelos centrados anteriores.
    """
    payload = {key: state[key] for key in (
        "format_version", "streaming", "normalization", "sessions", "phones", "blank_id", "data_signature",
    )}
    payload["config"] = {key: state["config"][key] for key in ("modelo", "preprocesamiento")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_speech_checkpoint(path: Path) -> dict[str, Any]:
    """Lee con weights_only y comprueba la coherencia de sesiones y vocabulario."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format_version", "model_state", "config", "normalization",
        "sessions", "phones", "blank_id", "step", "data_signature",
    }
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("El archivo no es un checkpoint de inferencia de habla completo.")
    cfg = state["config"]
    model = cfg["modelo"]
    if (
        model["nombre"] != "speech_gru"
        or state["format_version"] != inference_format(cfg)
        or state["sessions"] != cfg["datos"]["sesiones"]
        or state["phones"] != cfg["datos"]["fonemas"]
        or state["blank_id"] != len(state["phones"])
        or (cfg.get("protocolo") and state.get("val_metrics", {}).get("completo") != 1)
    ):
        raise ValueError("El checkpoint contiene configuración, sesiones o vocabulario inconsistentes.")
    if state["format_version"] == 3 and (
        state.get("streaming") != streaming_spec(cfg)
        or state.get("inference_sha256") != inference_signature(state)
    ):
        raise ValueError("La configuración temporal o los metadatos no coinciden con la firma de inferencia.")
    exits = model.get("salidas_intermedias", [])
    if exits and (
        state.get("selection_exit") != model["capas"]
        or set(state.get("exit_metrics", {})) != set(exits) | {model["capas"]}
        or any(value.get("completo") != 1 for value in state["exit_metrics"].values())
        or state.get("ctc_weights") != cfg["entrenamiento"]["ctc_pesos"]
    ):
        raise ValueError("Las salidas conjuntas no coinciden con la selección guardada.")
    return state


@dataclass(frozen=True)
class SpeechCheckpoint:
    """Modelo evaluable y metadatos necesarios para procesar señal cruda."""

    model: SpeechGRU
    smoother: GaussianSmoothing
    config: dict[str, Any]
    sessions: tuple[str, ...]
    phones: tuple[str, ...]
    blank_id: int
    normalization: dict[str, TrainingNormalization]
    step: int
    data_signature: str
    format_version: int = 2
    inference_sha256: str | None = None


def load_inference_checkpoint(
    path: Path, device: str | torch.device = "cpu",
) -> SpeechCheckpoint:
    """Reconstruye desde la configuración guardada, sin leer rutas del dataset.

    Conserva el modo temporal original; cargar pesos centrados no los convierte
    a causales. El dispositivo se elige explícitamente y por defecto es CPU.
    """
    state = read_speech_checkpoint(path)
    cfg = state["config"]
    model = construir_modelo(cfg)
    if not isinstance(model, SpeechGRU):
        raise ValueError("Se necesita un modelo de habla.")
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    smoother = build_smoother(cfg).to(device).eval()
    sessions = tuple(state["sessions"])
    normalization = {
        name: TrainingNormalization.from_dict(values)
        for name, values in state["normalization"].items()
    }
    if set(normalization) != set(sessions) or any(
        values.session != name
        or values.epsilon != cfg["preprocesamiento"]["normalizacion_epsilon"]
        or len(values.mean) != model.input_features
        for name, values in normalization.items()
    ):
        raise ValueError("La normalización guardada no corresponde a las sesiones del modelo.")
    return SpeechCheckpoint(
        model=model, smoother=smoother, config=cfg, sessions=sessions,
        phones=tuple(state["phones"]), blank_id=state["blank_id"],
        normalization=normalization, step=state["step"],
        data_signature=state["data_signature"],
        format_version=state["format_version"], inference_sha256=state.get("inference_sha256"),
    )
