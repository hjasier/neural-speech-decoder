"""Añade persistencia a una variante común sin alterar sus hiperparámetros."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.configure_speech_comparison import comparison_config
from src.training.speech_resume import validate_resume_config


def main() -> int:
    """Genera una configuración reanudable con el mismo protocolo y presupuesto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/speech_comparativa.yaml"))
    parser.add_argument("--resume-config", type=Path, default=Path("configs/speech_reanudacion.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = comparison_config(args.plan, args.variant, args.seed_index)
    spec = yaml.safe_load(args.resume_config.read_text(encoding="utf-8"))
    cfg["reanudacion"] = spec["reanudacion"]
    validate_resume_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Perfil reanudable: {args.variant}; semilla {cfg['experimento']['seed']}; "
          f"presupuesto {cfg['entrenamiento']['pasos']} pasos. Configuración: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
