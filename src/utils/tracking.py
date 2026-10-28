"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.env import cargar_clave_wandb


def abrir_run(cfg: dict) -> Any | None:
    """Abre un run de W&B, o devuelve None si esta desactivado."""
    wcfg = cfg.get("wandb", {})
    if wcfg.get("modo") == "disabled":
        if cfg["modelo"]["nombre"] == "speech_gru":
            raise ValueError("El entrenamiento de habla requiere W&B; usa modo offline u online.")
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Falta W&B; instala requirements.txt antes de entrenar.") from error

    cargar_clave_wandb()
    directory = wcfg.get("directorio")
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=wcfg.get("entidad"),
        name=cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
        dir=directory,
    )


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)
