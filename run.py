#!/usr/bin/env python3
"""
Punto de entrada unico del proyecto.

    python run.py --smoke                        # plantilla sintética anterior
    python run.py --config configs/speech_original.yaml
    python run.py --config configs/speech_original.yaml --solo-evaluar --checkpoint RUTA

Este fichero es deliberadamente corto: orquesta, no implementa. La logica vive
en src/. Si run.py crece por encima de un centenar de lineas, algo tendria que
haber bajado a un modulo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import fijar_seed          # noqa: E402
from src.utils.tracking import abrir_run       # noqa: E402
from src.data.loaders import construir_loaders  # noqa: E402
from src.models.factory import construir_modelo  # noqa: E402
from src.training.loop import entrenar, evaluar  # noqa: E402


def cargar_config(ruta: Path) -> dict:
    """Lee los parámetros del experimento desde YAML."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    """Orquesta datos, modelo, seguimiento y entrenamiento o evaluación."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="fichero de configuracion")
    ap.add_argument("--smoke", action="store_true",
                    help="usa configs/smoke.yaml: prueba corta de la plantilla sintética")
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="carga el checkpoint y evalua, sin entrenar")
    ap.add_argument("--checkpoint", type=Path,
                    help="checkpoint de habla para --solo-evaluar")
    args = ap.parse_args()
    if args.solo_evaluar and args.checkpoint is None:
        ap.error("--solo-evaluar requiere --checkpoint")
    if args.checkpoint is not None and not args.solo_evaluar:
        ap.error("--checkpoint requiere --solo-evaluar; no se admite reanudar entrenamiento todavía")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        ap.error(f"no existe el checkpoint {args.checkpoint}")

    ruta = Path("configs/smoke.yaml") if args.smoke else args.config
    if not ruta.is_file():
        print(f"error: no existe {ruta}", file=sys.stderr)
        return 2
    cfg = cargar_config(ruta)
    alice_ctc = cfg.get("tarea") == "alice_ctc"
    speech = cfg["modelo"]["nombre"] == "speech_gru" and not alice_ctc
    alice = cfg.get("tarea") == "alice_power"
    if args.solo_evaluar and not (speech or alice_ctc):
        ap.error("la evaluación desde checkpoint está implementada para habla y Alice CTC")

    # 1 · reproducibilidad, antes que nada
    fijar_seed(cfg["experimento"]["seed"])

    # 2 · seguimiento del experimento
    run = abrir_run(cfg)
    try:
        if alice_ctc:
            from src.training.alice_ctc import ejecutar_alice_ctc

            metricas = ejecutar_alice_ctc(cfg, run, checkpoint=args.checkpoint, solo_evaluar=args.solo_evaluar)
        elif alice:
            from src.training.alice import ejecutar_alice

            metricas = ejecutar_alice(cfg, run)
        elif speech:
            from src.training.speech import ejecutar_habla

            metricas = ejecutar_habla(cfg, run, checkpoint=args.checkpoint, solo_evaluar=args.solo_evaluar)
        else:
            train_loader, val_loader = construir_loaders(cfg)
            modelo = construir_modelo(cfg)
            entrenar(modelo, train_loader, val_loader, cfg, run)
            metricas = evaluar(modelo, val_loader, cfg)
        label = "validacion; checkpoint seleccionado" if cfg.get("protocolo") and not args.solo_evaluar else "test publicado"
        print(f"\n=== metricas ({label}) ===" if speech else "\n=== metricas ===")
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
