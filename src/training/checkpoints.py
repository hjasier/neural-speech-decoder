"""Checkpoints de habla autocontenidos para evaluación e inferencia."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.speech_dataset import SpeechDataset, preparation_signature
from src.inference.checkpoints import read_speech_checkpoint
from src.models.speech_gru import SpeechGRU


def save_checkpoint(
    path: Path, model: SpeechGRU, cfg: dict[str, Any], dataset: SpeechDataset,
    step: int, metrics: dict[str, float | int],
    *, split: str = "test", exit_metrics: dict[int, dict[str, float | int]] | None = None,
) -> None:
    """Guarda pesos CPU, configuración, normalización de train y procedencia."""
    if split not in ("val", "test"):
        raise ValueError("Las métricas del checkpoint deben ser de val o test.")
    if cfg.get("protocolo") and (split != "val" or metrics.get("completo") != 1):
        raise ValueError("El protocolo requiere guardar métricas de validación completa.")
    exits = cfg["modelo"].get("salidas_intermedias", [])
    if exits:
        final = cfg["modelo"]["capas"]
        if (exit_metrics is None or set(exit_metrics) != set(exits) | {final}
                or any(value.get("completo") != 1 for value in exit_metrics.values())
                or any(metrics[key] != value for key, value in exit_metrics[final].items())):
            raise ValueError("El checkpoint debe conservar todas las salidas y seleccionar por la final.")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 2 if cfg.get("protocolo") else 1,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": cfg,
        "normalization": dataset.normalization_states,
        "data_signature": preparation_signature(cfg),
        "sessions": list(dataset.sessions),
        "phones": list(cfg["datos"]["fonemas"]),
        "blank_id": len(cfg["datos"]["fonemas"]),
        "step": step,
        f"{split}_metrics": metrics,
        "torch_version": str(torch.__version__),
    }
    if exit_metrics is not None:
        state.update(exit_metrics=exit_metrics, selection_exit=cfg["modelo"]["capas"],
                     ctc_weights=cfg["entrenamiento"]["ctc_pesos"])
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps({"step": step, "split": split, **metrics,
                    **({"salidas": exit_metrics} if exit_metrics is not None else {})}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path, model: SpeechGRU, cfg: dict[str, Any], dataset: SpeechDataset,
) -> dict[str, Any]:
    """Carga con weights_only y rechaza modelos, vocabularios o datos distintos."""
    state = read_speech_checkpoint(path)
    original = state["config"]
    if (
        state["format_version"] != (2 if cfg.get("protocolo") else 1)
        or (cfg.get("protocolo") and state.get("val_metrics", {}).get("completo") != 1)
        or (cfg["modelo"].get("salidas_intermedias")
            and (state.get("selection_exit") != cfg["modelo"]["capas"]
                 or set(state.get("exit_metrics", {})) != set(cfg["modelo"]["salidas_intermedias"]) | {cfg["modelo"]["capas"]}))
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
