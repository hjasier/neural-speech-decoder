"""Adaptación de nuestro SpeechGRU/CTC a Alice; selección exclusivamente en val."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.alice_ctc import AliceCTCDataset
from src.data.speech_dataset import SessionBatchSampler, SpeechSample, collate_speech, file_sha256
from src.models.factory import construir_modelo
from src.training.metrics import decode_ctc, edit_distance
from src.training.optim import TensorFlowAdam
# Reutilización literal del entrenamiento ya implementado para T12. El
# preprocesamiento EEG está en disco, por lo que aquí el smoother es Identity.
from src.training.speech import (
    _backward_microbatch, _criterion, _learning_rate, _regularization, _memory_metrics, evaluar_habla,
)
from src.utils.seed import crear_generador
from src.utils.tracking import registrar


class _ZeroEEG(Dataset):
    def __init__(self, dataset: AliceCTCDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """Conserva el mismo número de secuencias para el control."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> SpeechSample:
        """Sustituye solo la señal por ceros, conservando longitudes y objetivos."""
        sample = self.dataset[index]
        return replace(sample, features=torch.zeros_like(sample.features))


def _load(path: Path, model: nn.Module, cfg: dict, manifest_hash: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("format") != "alice_ctc_v1" or state["data_manifest_sha256"] != manifest_hash
            or state["config"]["modelo"] != cfg["modelo"]
            or state["config"]["preprocesamiento"] != cfg["preprocesamiento"]
            or state["phones"] != cfg["datos"]["fonemas"]):
        raise ValueError("El checkpoint no corresponde a estos datos/modelo de Alice CTC.")
    model.load_state_dict(state["model_state"], strict=True)
    return state


def ejecutar_alice_ctc(
    cfg: dict[str, Any], run: Any, *, checkpoint: Path | None, solo_evaluar: bool,
) -> dict[str, float | int]:
    """Entrena con nuestro bucle CTC, restaura el mejor val y evalúa test al final."""
    if run is None:
        raise ValueError("Alice CTC requiere W&B online u offline.")
    training = cfg["entrenamiento"]
    if (training["optimizador"] != "adam_tensorflow" or training["pasos"] <= 0
            or training["max_segundos"] <= 0 or cfg["evaluacion"]["max_lotes"] is not None
            or cfg["evaluacion"]["cada_pasos"] <= 0 or training["micro_batch_size"] <= 0
            or cfg["datos"]["batch_size"] <= 0 or cfg["ejecucion"]["hilos_cpu"] <= 0):
        raise ValueError("Presupuesto o configuración de entrenamiento Alice CTC no válida.")
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    requested = cfg["ejecucion"]["dispositivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Este perfil requiere CUDA; para CPU usa configs/alice_ctc_cpu.yaml.")
    manifest_path = Path(cfg["datos"]["preparados"]) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Falta preparar Alice: ejecuta scripts/prepare_alice_ctc.py --config con el mismo YAML.")
    manifest_hash = file_sha256(manifest_path)
    datasets = {split: AliceCTCDataset(cfg, split) for split in ("train", "val", "test")}
    manifest = datasets["train"].manifest
    common = {"num_workers": cfg["datos"]["num_workers"], "pin_memory": cfg["datos"]["pin_memory"],
              "collate_fn": collate_speech, "generator": crear_generador(cfg["experimento"]["seed"])}
    train_loader = DataLoader(datasets["train"], batch_sampler=SessionBatchSampler(
        datasets["train"].session_indices, cfg["datos"]["batch_size"], training["pasos"],
        crear_generador(cfg["experimento"]["seed"])), **common)
    val_loader, test_loader = [DataLoader(datasets[split], batch_size=cfg["datos"]["batch_size"],
                                         shuffle=False, **common) for split in ("val", "test")]
    output = Path(cfg["salida"]["directorio"]) / run.id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = construir_modelo(cfg).to(device)
    smoother = nn.Identity().to(device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Alice CTC · SpeechGRU reutilizada · {hardware} · secuencias {manifest['counts']}", flush=True)
    print(f"Participantes: {len(cfg['datos']['sesiones'])} · parámetros: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"EEG {len(manifest['channels'])} canales; blank={manifest['blank_id']}; normalización por participante con train", flush=True)
    best_metrics, best_step, step, history, stopped = None, 0, 0, [], False
    start = perf_counter()
    if not solo_evaluar:
        criterion = _criterion(cfg)
        optimizer = TensorFlowAdam(model.parameters(), lr=_learning_rate(cfg, 0),
                                  betas=tuple(training["betas"]), eps=training["epsilon"])
        best_loss = float("inf")
        for step, batch in enumerate(train_loader, start=1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = _learning_rate(cfg, step - 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ctc = sum(_backward_microbatch(model, small, cfg, smoother, criterion, device, len(batch.references))
                      for small in batch.microbatches(training["micro_batch_size"]))
            penalty = _regularization(model, batch.session_ids, training["l2"])
            penalty.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), training["clip_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            logged = {"train/ctc_loss": ctc, "train/l2": penalty.item(), "train/grad_norm": norm.item(),
                      "train/learning_rate": lr, "train/segundos": perf_counter() - start,
                      **_memory_metrics(device)}
            registrar(run, logged, step)
            if step == 1 or step % training["imprimir_cada"] == 0:
                memory = (f" · CUDA pico={logged['cuda/pico_asignado_mib']:.0f} MiB asignados / "
                          f"{logged['cuda/pico_reservado_mib']:.0f} MiB reservados" if device.type == "cuda" else "")
                print(f"paso {step}/{training['pasos']} · CTC={ctc:.4f} · lr={lr:.6f}{memory}", flush=True)
            stopped = perf_counter() - start >= training["max_segundos"] and step < training["pasos"]
            if step == 1 or step % cfg["evaluacion"]["cada_pasos"] == 0 or step == training["pasos"] or stopped:
                validation = evaluar_habla(model, val_loader, cfg, smoother, device)
                history.append({"step": step, **validation})
                registrar(run, {f"val/{k}": v for k, v in validation.items()}, step)
                print(f"val: PER={validation['per']:.2%} · CTC={validation['ctc_loss']:.4f}", flush=True)
                if validation["ctc_loss"] < best_loss:
                    best_loss, best_metrics, best_step = validation["ctc_loss"], validation, step
                    state = {"format": "alice_ctc_v1", "config": cfg,
                             "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                             "data_manifest_sha256": manifest_hash, "normalization": manifest["normalization"],
                             "channels": manifest["channels"], "phones": cfg["datos"]["fonemas"],
                             "subjects": cfg["datos"]["sesiones"],
                             "blank_id": len(cfg["datos"]["fonemas"]), "step": step, "val_metrics": validation}
                    temporary = output / "mejor_val.tmp"
                    torch.save(state, temporary)
                    temporary.replace(output / "mejor_val.pt")
            if stopped or (step < training["pasos"] and perf_counter() - start >= training["max_segundos"]):
                stopped = True
                break
        checkpoint = output / "mejor_val.pt"
    if checkpoint is None:
        raise ValueError("Falta un checkpoint Alice CTC para evaluar.")
    training_seconds = perf_counter() - start if not solo_evaluar else 0.0
    # Test se calcula usando los bytes del checkpoint seleccionado, no el último estado en RAM.
    state = _load(checkpoint, model, cfg, manifest_hash)
    best_step, best_metrics = state["step"], state["val_metrics"]
    test_metrics = evaluar_habla(model, test_loader, cfg, smoother, device)
    zeros = DataLoader(_ZeroEEG(datasets["test"]), batch_size=cfg["datos"]["batch_size"], shuffle=False, **common)
    control = evaluar_habla(model, zeros, cfg, smoother, device)
    # Referencia sin EEG: símbolo más frecuente en train, repetido su mediana
    # de apariciones por fragmento. No usa duración ni etiquetas de test para ajustar.
    counts = Counter(i for row in datasets["train"].rows for i in row["ids"])
    frequent = counts.most_common(1)[0][0]
    repetitions = max(1, round(median(row["ids"].count(frequent) for row in datasets["train"].rows)))
    prior = [frequent] * repetitions
    prior_errors = sum(edit_distance(row["ids"], prior) for row in datasets["test"].rows)
    prior_control = {"symbol": manifest["phones"][frequent], "repetitions": repetitions,
                     "errors": prior_errors, "phones": test_metrics["fonemas"],
                     "per": prior_errors / test_metrics["fonemas"],
                     "rule": "fonema más frecuente de train; repeticiones = mediana de su recuento por secuencia train"}
    predictions, subject_metrics = [], {}
    with torch.no_grad():
        for batch in test_loader:
            for small in batch.microbatches(cfg["evaluacion"]["micro_batch_size"], trim_padding=True):
                moved = small.to(device)
                paths = decode_ctc(model(moved.features, moved.session_ids), model.output_lengths(small.input_lengths), state["blank_id"])
                references = small.targets.split(small.target_lengths.tolist())
                for i, ids in enumerate(paths):
                    subject = cfg["datos"]["sesiones"][int(small.session_ids[i])]
                    reference = references[i].tolist()
                    metrics = subject_metrics.setdefault(subject, {"errores": 0, "fonemas": 0, "fragmentos": 0})
                    metrics["errores"] += edit_distance(reference, ids)
                    metrics["fonemas"] += len(reference)
                    metrics["fragmentos"] += 1
                    predictions.append({"sequence": small.trial_indices[i], "segment": small.blocks[i],
                                        "subject": subject, "reference": small.references[i], "hypothesis_ids": ids,
                                        "hypothesis_phones": [state["phones"][j] for j in ids]})
    for metrics in subject_metrics.values():
        metrics["per"] = metrics["errores"] / metrics["fonemas"]
    if (sum(m["errores"] for m in subject_metrics.values()) != test_metrics["errores"]
            or sum(m["fonemas"] for m in subject_metrics.values()) != test_metrics["fonemas"]):
        raise ValueError("Las predicciones por participante no reproducen los recuentos del test.")
    report = {"tarea": "EEG de escucha a fonemas; adaptación propia de SpeechGRU + CTC",
              "run_id": run.id, "hardware": hardware, "device": str(device), "config": cfg,
              "data_manifest_sha256": manifest_hash, "sources_sha256": manifest["sources_sha256"],
              "counts": manifest["counts"], "channels": manifest["channels"],
              "subjects": cfg["datos"]["sesiones"], "counts_by_subject": manifest.get("counts_by_subject"),
              "missing_segments": manifest.get("missing_segments", []),
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "reused": ["SpeechGRU", "PhonemeEncoder", "SpeechSample/SpeechBatch/collate_speech",
                         "SessionBatchSampler", "TensorFlowAdam", "bucle CTC/microlotes/L2/clipping", "evaluar_habla/PER"],
              "versions": {p: version(p) for p in ("torch", "numpy", "scipy", "g2p-en", "nltk", "wandb")},
              "source_code_sha256": {p.as_posix(): file_sha256(p) for p in [Path("run.py"), *sorted(Path("src").rglob("*.py"))]},
              "completed_steps": step, "best_val_step": best_step, "best_val": best_metrics,
              "stopped_for_time": stopped, "validation_history": history,
              "training_seconds_including_val_and_saves": training_seconds,
              "test": test_metrics, "zero_eeg_control": control, "phone_prior_control": prior_control,
              "test_by_subject": subject_metrics,
              "test_macro_per": sum(m["per"] for m in subject_metrics.values()) / len(subject_metrics),
              "cuda_memory": _memory_metrics(device),
              "empty_prediction_per": 1.0,
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "checkpoint_reloaded_before_test": True, "evaluation_only": solo_evaluar,
              "limits": [f"{len(cfg['datos']['sesiones'])} participantes conocidos y un audiolibro; texto reservado por segmentos.",
                         "No prueba generalización entre personas, habla intentada ni funcionamiento en tiempo real.",
                         "Referencias fonéticas generadas por g2p-en; no anotación fonética del audio.",
                         "Sin ICA ni rechazo manual de artefactos; el control de ceros no descarta todos los confusores.",
                         "Sin modelo de lenguaje ni WER; PER greedy agregado incluyendo SIL."]}
    (output / "resultado.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "predicciones_test.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False) + "\n")
    run.summary.update({"alice_ctc/best_val_step": best_step, "alice_ctc/completed_steps": step,
                        "alice_ctc/training_seconds": training_seconds,
                        "alice_ctc/stopped_for_time": stopped, "alice_ctc/resultado": str(output / "resultado.json"),
                        **{f"alice_ctc/test_{k}": v for k, v in test_metrics.items()},
                        "alice_ctc/zero_eeg_per": control["per"], "alice_ctc/prior_per": prior_control["per"],
                        "alice_ctc/empty_per": 1.0})
    run.summary.update({"alice_ctc/subjects": len(subject_metrics), "alice_ctc/test_macro_per": report["test_macro_per"],
                        **_memory_metrics(device),
                        **{f"alice_ctc/subjects/{subject}/per": m["per"] for subject, m in subject_metrics.items()}})
    registrar(run, {**{f"test_final/{k}": v for k, v in test_metrics.items()},
                    "control_ceros/per": control["per"], "control_prior/per": prior_control["per"]}, step + 1)
    print(f"test final: PER={test_metrics['per']:.2%} · EEG cero={control['per']:.2%} · "
          f"prior={prior_control['per']:.2%} · vacío=100%", flush=True)
    print(f"Resultado: {output / 'resultado.json'}", flush=True)
    return test_metrics
