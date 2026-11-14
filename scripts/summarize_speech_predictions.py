"""Resume PER y WER con intervalos de confianza desde las predicciones guardadas.

    python scripts/summarize_speech_predictions.py --predictions runs/trigrama_val/predicciones.jsonl

Lee el predicciones.jsonl de scripts/evaluate_speech_trigram.py; no carga el
modelo ni vuelve a decodificar. Separa habla vocalizada y silenciosa y escribe
resumen_metricas.json junto a las predicciones. No interpreta los resultados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.bootstrap import bootstrap_rate_interval

METRICS = {"per": ("errores_fonemas", "fonemas"), "wer": ("errores_palabras", "palabras")}


def summarize_predictions(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Agrupa por salida y tipo de habla; cada grupo conserva su número de frases."""
    if not rows or len({row["split"] for row in rows}) != 1:
        raise ValueError("Se necesitan predicciones de una única partición.")
    silent = set(spec["sesiones_silenciosas"])
    groups = {"global": rows,
              "vocalizada": [row for row in rows if row["session"] not in silent],
              "silenciosa": [row for row in rows if row["session"] in silent]}
    options = {"resamples": spec["bootstrap"]["remuestreos"], "confidence": spec["bootstrap"]["confianza"],
               "seed": spec["bootstrap"]["seed"]}
    summary: dict[str, Any] = {}
    for depth in rows[0]["profiles"]:
        summary[depth] = {
            group: {metric: bootstrap_rate_interval(
                [row["profiles"][depth][errors] for row in selected],
                [row["profiles"][depth][total] for row in selected], **options,
            ) for metric, (errors, total) in METRICS.items()}
            for group, selected in groups.items() if selected
        }
    return {"split": rows[0]["split"], "trials": len(rows), "sessions": sorted({row["session"] for row in rows}),
            "by_exit": summary}


def main() -> int:
    """Calcula el resumen y lo guarda con los hashes de sus dos entradas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/final/resumen_metricas.yaml"))
    parser.add_argument("--output", type=Path, help="por defecto, resumen_metricas.json junto a las predicciones")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = args.predictions.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    try:
        result = summarize_predictions(rows, spec)
    except (ValueError, LookupError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    result.update(config=spec, predictions_sha256=hashlib.sha256(payload).hexdigest(),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    output = args.output or args.predictions.with_name("resumen_metricas.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for depth, groups in result["by_exit"].items():
        for group, metrics in groups.items():
            print(f"salida {depth} · {group} ({metrics['per']['trials']} frases): " + "  ".join(
                f"{name.upper()} {value['rate']:.2%} [{value['low']:.2%}, {value['high']:.2%}]"
                for name, value in metrics.items()))
    print(f"Resumen: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
