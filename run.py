#!/usr/bin/env python3
"""
Punto de entrada unico del proyecto.

    python run.py --smoke                        # pipeline pequeño con datos reales T12
    python run.py --config configs/speech_comparacion_t4.yaml
    python run.py --config configs/speech_comparacion_t4.yaml --solo-evaluar --checkpoint RUTA

Este fichero es deliberadamente corto: orquesta, no implementa. La logica vive
en src/. Si run.py crece por encima de un centenar de lineas, algo tendria que
haber bajado a un modulo.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
from pathlib import Path

import yaml

from src.data.loaders import construir_loaders
from src.models.factory import construir_modelo
from src.training.loop import entrenar, evaluar
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run


def cargar_config(ruta: Path) -> dict:
    """Lee la configuración YAML del experimento."""
    with ruta.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--smoke", action="store_true", help="pipeline de habla pequeño con datos reales T12 en CPU")
    parser.add_argument(
        "--solo-evaluar", action="store_true", help="evalúa sin entrenar"
    )
    parser.add_argument("--checkpoint", type=Path, help="pesos para --solo-evaluar")
    parser.add_argument(
        "--reanudar", type=Path, help="estado completo entrenamiento.pt"
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        help="partición de evaluación; por defecto test",
    )
    args = parser.parse_args()
    if args.smoke and (args.solo_evaluar or args.checkpoint or args.reanudar or args.split):
        parser.error("--smoke no se combina con evaluación ni reanudación")
    if args.solo_evaluar and args.checkpoint is None:
        parser.error("--solo-evaluar requiere --checkpoint")
    if args.checkpoint is not None and not args.solo_evaluar:
        parser.error(
            "--checkpoint requiere --solo-evaluar; para continuar usa --reanudar"
        )
    if args.reanudar is not None and (
        args.solo_evaluar or args.checkpoint is not None or args.split is not None
    ):
        parser.error(
            "--reanudar no se combina con --solo-evaluar, --checkpoint ni --split"
        )
    if args.split is not None and not args.solo_evaluar:
        parser.error("--split requiere --solo-evaluar")
    for path in (args.checkpoint, args.reanudar):
        if path is not None and not path.is_file():
            parser.error(f"no existe el archivo {path}")
    return args


def _estado_reanudacion(cfg: dict, args: argparse.Namespace) -> dict | None:
    if not (args.reanudar is not None or cfg.get("reanudacion")):
        return None
    from src.training.speech_resume import (
        read_training_checkpoint,
        validate_resume_config,
    )

    if cfg["modelo"]["nombre"] != "speech_gru":
        raise ValueError("La reanudación solo está disponible para T12.")
    validate_resume_config(cfg)
    if args.reanudar is None:
        return None
    state = read_training_checkpoint(args.reanudar, cfg)
    if args.reanudar.parent.name != state["tracking"]["logical_id"]:
        raise ValueError(
            "Conserva el nombre de la carpeta del run al copiar el estado completo."
        )
    return state


def _ejecutar(
    cfg: dict, args: argparse.Namespace, run: Any, resume_state: dict | None
) -> dict:
    if cfg["modelo"]["nombre"] == "speech_gru":
        from src.training.speech import ejecutar_habla

        return ejecutar_habla(
            cfg,
            run,
            checkpoint=args.checkpoint,
            solo_evaluar=args.solo_evaluar,
            split_evaluacion=args.split,
            resume_state=resume_state,
            resume_path=args.reanudar,
        )
    train, val = construir_loaders(cfg)
    modelo = construir_modelo(cfg)
    entrenar(modelo, train, val, cfg, run)
    return evaluar(modelo, val, cfg)


def main() -> int:
    """Carga la configuración, abre W&B y ejecuta el modo solicitado."""
    args = _argumentos()
    ruta = Path("configs/smoke.yaml") if args.smoke else args.config
    if not ruta.is_file():
        print(f"error: no existe {ruta}", file=sys.stderr)
        return 2
    cfg = cargar_config(ruta)
    smoke = bool(cfg.get("smoke"))
    if smoke:
        if args.solo_evaluar or args.checkpoint or args.reanudar or args.split:
            raise ValueError("El smoke no admite evaluación ni reanudación externas.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("WANDB_SILENT", "true")
        from scripts.smoke_speech import preparar_smoke

        try:
            cfg = preparar_smoke(cfg)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    if cfg.get("tarea") is not None:
        raise ValueError(
            "Las exploraciones anteriores están retiradas; usa un perfil T12 o el smoke."
        )
    speech = cfg["modelo"]["nombre"] == "speech_gru"
    if (args.solo_evaluar or args.split is not None) and not speech:
        raise ValueError(
            "La evaluación desde checkpoint solo está disponible para T12."
        )
    resume_state = _estado_reanudacion(cfg, args)
    fijar_seed(cfg["experimento"]["seed"])
    run = (
        abrir_run(cfg, resume=resume_state["tracking"])
        if resume_state is not None
        else abrir_run(cfg)
    )
    try:
        metricas = _ejecutar(cfg, args, run, resume_state)
        if smoke:
            from scripts.smoke_speech import verificar_smoke

            verificar_smoke(cfg, run, metricas)
        split = args.split or (
            "val" if cfg.get("protocolo") and not args.solo_evaluar else "test"
        )
        label = f"smoke, datos {cfg['smoke']['origen_datos']}" if smoke else split
        print(f"\n=== métricas ({label}) ===" if speech else "\n=== métricas ===")
        for key, value in metricas.items():
            print(f"{key}: {value:.4f}")
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
