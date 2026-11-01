"""Evalúa el trigrama conectado a un checkpoint causal, sin entrenar.

Desde la raíz del proyecto, con el runtime y el grafo ya preparados:
    python scripts/evaluate_speech_trigram.py --checkpoint PESOS.pt --check-streaming --output runs/trigrama_comprobacion
    python scripts/evaluate_speech_trigram.py --checkpoint PESOS.pt --output runs/trigrama_val

Usa val por defecto; --split test reserva la evaluación final explícita.
"""

from __future__ import annotations

import argparse
import copy
from itertools import chain
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.speech_dataset import SpeechDataset, file_sha256
from src.inference.speech import OnlineSpeechDecoder
from src.inference.trigram import TrigramDecoder
from src.training.speech_online_evaluation import iter_online_trials
from src.training.speech_trigram_evaluation import check_trigram_streaming, evaluate_trigram
from src.utils.seed import fijar_seed
from src.utils.tracking import abrir_run, registrar


def main() -> int:
    """Verifica primero el WFST incremental; después registra PER, WER y procedencia."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/entrenos_online/speech_online_evaluacion.yaml"))
    parser.add_argument("--language-config", type=Path, default=Path("configs/entrenos_online/speech_trigrama.yaml"))
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--check-streaming", action="store_true", help="Solo una frase de comprobación; no calcula WER del corpus.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8"))
    language_spec = yaml.safe_load(args.language_config.read_text(encoding="utf-8"))
    if args.split is not None:
        spec["split"] = args.split
    if type(spec["permitir_tf32"]) is not bool or spec["split"] not in ("val", "test"):
        raise ValueError("Split o precisión no válidos.")
    torch.backends.cuda.matmul.allow_tf32 = spec["permitir_tf32"]
    torch.backends.cudnn.allow_tf32 = spec["permitir_tf32"]
    torch.set_num_threads(spec["hilos_cpu"])
    device = torch.device(spec["dispositivo"])
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("El dispositivo solicitado no está disponible.")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = file_sha256(args.checkpoint)
    started = perf_counter()
    decoder = OnlineSpeechDecoder.from_checkpoint(args.checkpoint, device)
    cfg = copy.deepcopy(decoder.checkpoint.config)
    fijar_seed(cfg["experimento"]["seed"])
    dataset = SpeechDataset(cfg, spec["split"])
    language = TrigramDecoder(language_spec, decoder.checkpoint.phones, decoder.checkpoint.blank_id,
                              log_path=args.output / "trigram_worker.log")
    load_seconds = perf_counter() - started
    run = None
    try:
        trials = iter(iter_online_trials(dataset, decoder))
        first = next(trials, None)
        if first is None:
            raise ValueError("La partición de evaluación debe contener frases.")
        verification = check_trigram_streaming(decoder, language, first, spec)
        verification.update(checkpoint_sha256=checkpoint_hash, language_resources=language.resources,
                            worker_runtime=language.runtime, evaluation_config=spec, language_config=language_spec)
        (args.output / "comprobacion_incremental.json").write_text(
            json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("El checkpoint ha cambiado durante la comprobación incremental.")
        if not verification["verified"]:
            raise ValueError("El WFST por bloques difiere con los mismos logits; revisar comprobacion_incremental.json.")
        if args.check_streaming:
            print("Comprobación incremental correcta:", args.output)
            return 0
        cfg["experimento"]["nombre"] += f"_trigrama_{spec['split']}"
        cfg["evaluacion_trigrama"] = {"evaluacion": spec, "lenguaje": language_spec}
        run = abrir_run(cfg)
        temporary = args.output / "predicciones.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            def write_prediction(row: dict) -> None:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

            result = evaluate_trigram(decoder, language, chain([first], trials), spec,
                                      expected_trials=len(dataset), prediction_sink=write_prediction)
        if file_sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("El checkpoint ha cambiado durante la evaluación.")
        source_names = ["scripts/evaluate_speech_trigram.py", "scripts/trigram_worker.py", "src/inference/speech.py",
                        "src/inference/trigram.py", "src/training/speech_trigram_evaluation.py",
                        "src/training/speech_online_evaluation.py"]
        sources = {name: file_sha256(ROOT / name) for name in source_names}
        sources.update(evaluation_config=file_sha256(args.evaluation_config),
                       language_config=file_sha256(args.language_config))
        result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                      checkpoint_step=decoder.checkpoint.step, training_seed=cfg["experimento"]["seed"],
                      protocol_sha256=cfg.get("protocolo", {}).get("sha256"),
                      checkpoint_format=decoder.checkpoint.format_version,
                      phones=list(decoder.checkpoint.phones), blank_id=decoder.checkpoint.blank_id,
                      inference_sha256=decoder.checkpoint.inference_sha256,
                      data_signature=decoder.checkpoint.data_signature,
                      prepared_manifest_sha256=file_sha256(dataset.root / "manifest.json"),
                      evaluation_config=spec, language_config=language_spec, language_resources=language.resources,
                      worker_runtime=language.runtime, setup_seconds=load_seconds,
                      runtime={"device": str(device), "torch": str(torch.__version__),
                               "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
                               else platform.processor() or platform.machine(),
                               "cpu_threads": torch.get_num_threads(),
                               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                      sources_sha256=sources, wandb_run_id=run.id)
        temporary.replace(args.output / "predicciones.jsonl")
        (args.output / "metricas.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        logged = {f"trigrama/{spec['split']}/capa_{depth}/{key}": value
                  for depth, values in result["global"].items() for key, value in values.items()}
        logged.update({f"trigrama/tiempo/capa_{depth}/rtf": value["rtf"] for depth, value in result["timing"].items()})
        logged.update({f"trigrama/tiempo/capa_{depth}/{stage}/{key}": value
                       for depth, metrics in result["timing"].items()
                       for stage in ("packet_processing", "first_partial_input", "first_partial_ready")
                       for key, value in metrics[stage].items() if value is not None})
        registrar(run, logged, decoder.checkpoint.step)
        run.summary.update({"checkpoint/sha256": checkpoint_hash, "trigrama/actualizaciones": 0,
                            "trigrama/test_evaluado": result["test_evaluated"]})
        for depth, values in result["global"].items():
            print(f"Salida {depth}: WER={values['wer']:.2%} ({values['errores_palabras']}/{values['palabras']}); "
                  f"PER={values['per']:.2%}; frases={values['frases']}")
        print("Evaluación del trigrama guardada:", args.output)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    finally:
        language.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
