"""Entrena el decodificador T5 de fonemas a texto de la Línea A y mide su WER en val.

    python scripts/train_text_decoder.py --dry-run
    python scripts/train_text_decoder.py

Usa los pares de scripts/export_phoneme_pairs.py y, si el perfil declara
datos.sinteticos, los de scripts/build_synthetic_pairs.py. Elige el paso de menor WER
de val y lo evalúa con el haz final, con intervalo por bootstrap para cada
semilla de la GRU que generó los fonemas. Registra en W&B; no usa test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.speech_dataset import file_sha256
from src.training.text_decoder import final_report, read_pairs, train_text_decoder
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run

SECTIONS = {"experimento", "datos", "modelo", "entrenamiento", "evaluacion", "bootstrap", "ejecucion", "salida", "wandb"}


def load_config(path: Path) -> dict[str, Any]:
    """Lee el perfil y exige sus secciones y un modelo seq2seq de texto."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != SECTIONS or cfg["modelo"]["nombre"] != "t5_texto":
        raise ValueError(f"El perfil requiere exactamente {sorted(SECTIONS)} y modelo.nombre t5_texto.")
    if cfg["evaluacion"]["num_beams_seleccion"] < 1 or cfg["evaluacion"]["num_beams_final"] < 1:
        raise ValueError("Los haces de generación deben ser al menos 1.")
    return cfg


def render_markdown(result: dict[str, Any]) -> str:
    """Tabla de WER de val por semilla de la GRU, sin interpretación."""
    def percent(value: float) -> str:
        return f"{100 * value:.2f}".replace(".", ",")

    report = result["val"]
    deviation = report["desviacion_modelos"]
    lines = ["# Línea A: decodificador T5 de fonemas a texto", "",
             f"Modelo `{result['modelo']['base']}` (commit `{result['modelo']['commit']}`), "
             f"paso elegido {result['best_step']} de {result['completed_steps']} por WER de val.",
             f"Haz final {report['num_beams']}. Intervalos del {100 * result['bootstrap']['confianza']:g} % "
             "por bootstrap de frases. El WER es optimista: se elige y se mide en val.", ""]
    synthetic = result.get("config", {}).get("datos", {}).get("sinteticos")
    if synthetic is not None:
        lines[3:3] = [f"Train mezcla pares sintéticos de `{synthetic['pares']}` con proporción "
                      f"{synthetic['proporcion']:g}."]
    lines += ["| Fonemas de | WER (%) | Intervalo (%) | Errores | Palabras |", "|---|---:|---:|---:|---:|"]
    for name, value in report["por_modelo"].items():
        interval = value["intervalo"]
        lines.append(f"| {name} | {percent(value['wer'])} | {percent(interval['low'])}–{percent(interval['high'])} | "
                     f"{value['errores']} | {value['palabras']} |")
    lines += ["", f"Media entre semillas: {percent(report['media_modelos'])} ± "
                  f"{'—' if deviation is None else percent(deviation)} %.",
              f"Frases de val cuyo texto aparece también en train: {report['frases_val_con_texto_en_train']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Entrena, guarda el modelo elegido, sus predicciones de val y la tabla."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final/decodificador_t5.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="comprueba perfil y pares sin descargar ni entrenar")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        pairs, manifest = read_pairs(cfg)
        print(f"Pares: {len(pairs['train'])} de train y {len(pairs['val'])} de val ({cfg['datos']['pares']}).")
        synthetic = cfg["datos"].get("sinteticos")
        if synthetic is not None:
            print(f"Pares sintéticos: {len(pairs['sinteticos'])}, proporción {synthetic['proporcion']} "
                  f"({synthetic['pares']}).")
        if args.dry_run:
            return 0
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
        device = torch.device(cfg["ejecucion"]["dispositivo"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidió CUDA pero no está disponible.")
        fijar_seed(cfg["experimento"]["seed"])
        base, revision = cfg["modelo"]["base"], cfg["modelo"]["revision"]
        tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        model = T5ForConditionalGeneration.from_pretrained(base, revision=revision)
        commit = getattr(model.config, "_commit_hash", None)
        run = abrir_run(cfg)
        try:
            training = train_text_decoder(model, tokenizer, pairs, cfg, run, device,
                                          log=lambda line: print(line, flush=True))
            report, predictions = final_report(model, tokenizer, pairs, cfg, device)
            output = Path(cfg["salida"]["directorio"]) / (run.id if run is not None else datetime.now().strftime("%Y%m%d_%H%M%S"))
            output.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(output / "mejor_val")
            tokenizer.save_pretrained(output / "mejor_val")
            with (output / "predicciones_val.jsonl").open("w", encoding="utf-8") as stream:
                for row in predictions:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = {**training, "val": report, "test_evaluated": False, "config": cfg,
                      "config_sha256": file_sha256(args.config), "bootstrap": cfg["bootstrap"],
                      "pares_manifest_sha256": file_sha256(Path(cfg["datos"]["pares"]) / "manifest.json"),
                      "pares_files": manifest["files"],
                      "pares_sinteticos_manifest_sha256": None if synthetic is None else
                      file_sha256(Path(synthetic["pares"]) / "manifest.json"),
                      "modelo": {"base": base, "revision": revision, "commit": commit},
                      "wandb_run_id": None if run is None else run.id,
                      "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                      "torch_version": str(torch.__version__)}
            (output / "resultado.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
            if run is not None:
                run.summary.update({"val/wer_final": report["wer_agregado"], "val/wer_media_modelos": report["media_modelos"],
                                    "seleccion/paso": training["best_step"], "test_evaluado": False,
                                    **{f"val/wer_final/{name}": value["wer"] for name, value in report["por_modelo"].items()}})
        except BaseException:
            if run is not None:
                run.finish(exit_code=1)
            raise
        if run is not None:
            run.finish()
        prefix = Path(cfg["salida"]["tabla"])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_markdown(result)
        prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps({key: value for key, value in result.items() if key != "config"},
                                                          indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(markdown)
    print(f"Modelo: {output / 'mejor_val'}. Tabla: {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
