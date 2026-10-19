"""Checkpoints de habla autocontenidos para evaluación e inferencia."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.speech_dataset import SpeechDataset, preparation_signature
from src.models.speech_gru import SpeechGRU


def save_checkpoint(
    path: Path, model: SpeechGRU, cfg: dict[str, Any], dataset: SpeechDataset,
    step: int, metrics: dict[str, float | int],
) -> None:
    """Guarda pesos CPU, configuración, normalización de train y procedencia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": cfg,
        "normalization": dataset.normalization_states,
        "data_signature": preparation_signature(cfg),
        "sessions": list(dataset.sessions),
        "phones": list(cfg["datos"]["fonemas"]),
        "blank_id": len(cfg["datos"]["fonemas"]),
        "step": step,
        "test_metrics": metrics,
        "torch_version": str(torch.__version__),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps({"step": step, "split": "test", **metrics}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path, model: SpeechGRU, cfg: dict[str, Any], dataset: SpeechDataset,
) -> dict[str, Any]:
    """Carga con weights_only y rechaza modelos, vocabularios o datos distintos."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    original = state["config"]
    if (
        state["format_version"] != 1
        or original["modelo"] != cfg["modelo"]
        or original["preprocesamiento"] != cfg["preprocesamiento"]
        or state["sessions"] != list(dataset.sessions)
        or state["phones"] != cfg["datos"]["fonemas"]
        or state["blank_id"] != len(cfg["datos"]["fonemas"])
        or state["data_signature"] != preparation_signature(cfg)
        or state["normalization"] != dataset.normalization_states
    ):
        raise ValueError("El checkpoint no corresponde al modelo, vocabulario o datos configurados.")
    model.load_state_dict(state["model_state"], strict=True)
    return state
