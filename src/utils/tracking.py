"""Seguimiento de experimentos con Weights & Biases.

Aislado en un modulo para que el resto del codigo no dependa de wandb y para
que los tests puedan correr con modo='disabled'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.env import cargar_clave_wandb


def abrir_run(cfg: dict, *, resume: dict[str, Any] | None = None) -> Any | None:
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
    options: dict[str, Any] = {}
    if resume is not None and wcfg.get("modo") == "online":
        options = {"id": resume["logical_id"], "resume": "must"}
    elif cfg.get("reanudacion"):
        # W&B no admite resume en offline. Cada segmento tiene ID propio;
        # se vuelve a publicar el historial completo en el segmento final.
        options["id"] = wandb.util.generate_id()
        if wcfg.get("modo") == "online":
            options["resume"] = "never"
    run = wandb.init(
        project=wcfg.get("proyecto", "tiiac-2026"),
        entity=(resume.get("entity") or wcfg.get("entidad")) if resume is not None else wcfg.get("entidad"),
        name=cfg["experimento"]["nombre"],
        config=cfg,
        mode=wcfg.get("modo", "online"),
        dir=directory,
        **options,
    )
    if resume is not None:
        if wcfg.get("modo") == "online" and (run.id != resume["logical_id"] or not run.resumed):
            run.finish(exit_code=1)
            raise RuntimeError("W&B no ha reanudado el run original; se detiene antes de entrenar.")
        run.summary.update({"reanudacion/id_logico": resume["logical_id"],
                            "reanudacion/segmento_anterior": resume["last_segment_id"],
                            "reanudacion/offline_segmentado": wcfg.get("modo") == "offline"})
    return run


def registrar(run: Any | None, metricas: dict, paso: int | None = None) -> None:
    """Registra metricas si hay run abierto. Silencioso si no."""
    if run is not None:
        run.log(metricas, step=paso)
