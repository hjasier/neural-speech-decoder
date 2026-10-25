"""Verifica recarga y predicciones de un checkpoint T12 sin entrenar ni usar test."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verification import verify_speech_checkpoint
from src.utils.tracking import abrir_run


def main() -> int:
    """Carga en otro proceso y guarda la evidencia tras superar las comprobaciones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verification-config", type=Path, default=Path("configs/verificacion_checkpoint.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["experimento"]["nombre"] += "_recarga_val"
    spec = yaml.safe_load(args.verification_config.read_text(encoding="utf-8"))
    run = abrir_run(cfg)
    try:
        result = verify_speech_checkpoint(cfg, args.checkpoint, spec, run)
        result["wandb_run_id"] = run.id
        output = args.checkpoint.parent / "verificacion_recarga.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        predictions = args.checkpoint.parent / "predicciones_val_ejemplo.json"
        predictions.write_text(json.dumps(result["example"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Recarga verificada en validación: paso {result['checkpoint_step']}; cero actualizaciones.")
        print(f"Evidencia: {output}")
        for depth, profile in result["example"]["profiles"].items():
            print(f"Salida {depth}: {' '.join(profile['hypothesis_phones'])}")
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
