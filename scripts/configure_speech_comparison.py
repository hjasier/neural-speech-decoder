"""Prepara una variante/semilla y exige las condiciones comunes de comparación."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


def comparison_config(plan_path: Path, variant: str, seed_index: int) -> dict[str, Any]:
    """Valida los perfiles y devuelve una configuración completa, sin entrenar.

    Las rutas del plan se interpretan desde la raíz del proyecto, igual que
    las de run.py. Solo pueden diferir profundidad, cabezas, pesos CTC,
    nombre y directorio de salida. La selección siempre es por PER final.
    """
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    seeds = plan["semillas"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or type(seed_index) is not int or not 0 <= seed_index < len(seeds)):
        raise ValueError("El plan requiere semillas enteras únicas y un índice válido.")
    variants = plan["variantes"]
    if variant not in variants or plan["referencia"] not in variants:
        raise ValueError("La variante o referencia no existe en el plan.")
    configurations = {name: yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                      for name, path in variants.items()}
    reference = configurations[plan["referencia"]]

    def common(cfg: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(cfg)
        value.pop("experimento")
        value.pop("salida")
        value["modelo"].pop("capas")
        value["modelo"].pop("salidas_intermedias", None)
        value["entrenamiento"].pop("ctc_pesos", None)
        return value

    for name, cfg in configurations.items():
        if (common(cfg) != common(reference) or not cfg.get("protocolo")
                or cfg["protocolo"]["seleccion_checkpoint"] != "per"
                or cfg["evaluacion"]["max_lotes"] is not None
                or cfg["entrenamiento"].get("max_segundos") is not None):
            raise ValueError(f"La variante {name} no respeta los datos/presupuesto/protocolo comunes.")
    cfg = copy.deepcopy(configurations[variant])
    seed = seeds[seed_index]
    cfg["experimento"].update(seed=seed, nombre=f"{cfg['experimento']['nombre']}_seed{seed}")
    cfg["comparacion"] = {
        "variante": variant, "plan": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "config_base": variants[variant],
        "config_base_sha256": hashlib.sha256(Path(variants[variant]).read_bytes()).hexdigest(),
    }
    return cfg


def main() -> int:
    """Escribe el perfil seleccionado; las semillas e hiperparámetros vienen de YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/speech_comparativa.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = comparison_config(args.plan, args.variant, args.seed_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Configuración: {args.output}; variante: {args.variant}; semilla: {cfg['experimento']['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
