"""Carga autocontenida de pesos de habla sin abrir datos ni importar entrenamiento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.preprocessing import GaussianSmoothing, TrainingNormalization, build_smoother
from src.models.factory import construir_modelo
from src.models.speech_gru import SpeechGRU


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
        or state["format_version"] != (2 if cfg.get("protocolo") else 1)
        or state["sessions"] != cfg["datos"]["sesiones"]
        or state["phones"] != cfg["datos"]["fonemas"]
        or state["blank_id"] != len(state["phones"])
        or (cfg.get("protocolo") and state.get("val_metrics", {}).get("completo") != 1)
    ):
        raise ValueError("El checkpoint contiene configuración, sesiones o vocabulario inconsistentes.")
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
    )
