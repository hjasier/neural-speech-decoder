"""Entrenamiento offline de speechBCI con CTC y evaluación PER."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data.loaders import construir_loaders
from src.data.preprocessing import GaussianSmoothing, build_smoother
from src.data.speech_dataset import SpeechBatch
from src.models.factory import construir_modelo
from src.models.speech_gru import SpeechGRU
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.metrics import decode_ctc, phoneme_errors
from src.training.optim import TensorFlowAdam
from src.utils.tracking import registrar


def _criterion(cfg: dict[str, Any]) -> nn.CTCLoss:
    # TF reduce_mean no divide cada frase por su longitud, a diferencia del
    # reduction='mean' de PyTorch. Un infinito debe detener la ejecución.
    return nn.CTCLoss(blank=len(cfg["datos"]["fonemas"]), reduction="none", zero_infinity=False)


def _ctc_losses(
    criterion: nn.CTCLoss, logits: Tensor, lengths: Tensor, batch: SpeechBatch,
) -> Tensor:
    losses = criterion(
        logits.log_softmax(-1).transpose(0, 1).contiguous(),
        batch.targets, lengths, batch.target_lengths,
    )
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("CTC contiene NaN o infinito; se detiene antes de actualizar pesos.")
    return losses


def _training_features(batch: SpeechBatch, cfg: dict[str, Any]) -> Tensor:
    noise = cfg["entrenamiento"]["ruido"]
    features = batch.features
    if noise["blanco_sd"] > 0:
        features = features + torch.randn_like(features) * noise["blanco_sd"]
    if noise["offset_sd"] > 0:
        offset = torch.randn_like(features[:, :1, :]) * noise["offset_sd"]
        features = features + offset
    # Como TF, el ruido se aplica al tensor rellenado antes de suavizar.
    return features


def _regularization(model: SpeechGRU, session_ids: Tensor, coefficient: float) -> Tensor:
    weights = [layer.weight_ih_l0 for layer in model.gru_layers]
    weights.extend(model.input_layers[day].weight for day in session_ids.unique().tolist())
    return coefficient * sum(weight.square().sum() for weight in weights)


def _learning_rate(cfg: dict[str, Any], completed_steps: int) -> float:
    schedule = cfg["entrenamiento"]["learning_rate"]
    fraction = min(completed_steps / schedule["decay_pasos"], 1.0)
    return (schedule["inicio"] - schedule["final"]) * (1 - fraction) ** schedule["potencia"] + schedule["final"]


def _backward_microbatch(
    model: SpeechGRU, batch: SpeechBatch, cfg: dict[str, Any],
    smoother: GaussianSmoothing, criterion: nn.CTCLoss,
    device: torch.device, effective_batch_size: int,
) -> float:
    # El grafo de cada grupo se libera antes de procesar el siguiente.
    batch = batch.to(device, non_blocking=cfg["datos"]["pin_memory"])
    logits = model(smoother(_training_features(batch, cfg)), batch.session_ids)
    losses = _ctc_losses(criterion, logits, model.output_lengths(batch.input_lengths), batch)
    # Ponderar por frases del lote completo también trata bien el último
    # grupo cuando batch_size no es divisible por micro_batch_size.
    loss = losses.sum() / effective_batch_size
    loss.backward()
    return loss.item()


def _memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda/pico_asignado_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "cuda/pico_reservado_mib": torch.cuda.max_memory_reserved(device) / (1 << 20),
    }


@torch.no_grad()
def evaluar_habla(
    model: SpeechGRU, loader: DataLoader, cfg: dict[str, Any],
    smoother: GaussianSmoothing, device: torch.device,
) -> dict[str, float | int]:
    """Evalúa sin ruido ni dropout; agrega CTC por frase y PER por fonema."""
    start = perf_counter()
    model.eval()
    criterion = _criterion(cfg)
    total_loss, errors, phones, trials = 0.0, 0, 0, 0
    limit = cfg["evaluacion"]["max_lotes"]
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        for microbatch in batch.microbatches(cfg["evaluacion"]["micro_batch_size"], trim_padding=True):
            microbatch = microbatch.to(device, non_blocking=cfg["datos"]["pin_memory"])
            lengths = model.output_lengths(microbatch.input_lengths)
            logits = model(smoother(microbatch.features), microbatch.session_ids)
            losses = _ctc_losses(criterion, logits, lengths, microbatch)
            hypotheses = decode_ctc(logits, lengths, criterion.blank)
            batch_errors, batch_phones = phoneme_errors(hypotheses, microbatch.targets, microbatch.target_lengths)
            total_loss += losses.sum().item()
            errors += batch_errors
            phones += batch_phones
            trials += len(microbatch.references)
            del logits, losses, microbatch
    if not trials or not phones:
        raise ValueError("La evaluación no contiene frases o fonemas.")
    return {
        "ctc_loss": total_loss / trials, "per": errors / phones,
        "errores": errors, "fonemas": phones, "frases": trials,
        "completo": int(trials == len(loader.dataset)), "segundos": perf_counter() - start,
    }


def entrenar_habla(
    model: SpeechGRU, train_loader: DataLoader, evaluation_loader: DataLoader,
    cfg: dict[str, Any], run: Any, smoother: GaussianSmoothing,
    device: torch.device, output: Path,
) -> dict[str, float | int]:
    """Entrena; el protocolo nuevo selecciona por PER de val sin evaluar test."""
    training = cfg["entrenamiento"]
    split = "val" if cfg.get("protocolo") else "test"
    best_metrics, best_step = None, 0
    history = []
    criterion = _criterion(cfg)
    optimizer = TensorFlowAdam(
        model.parameters(), lr=_learning_rate(cfg, 0),
        betas=tuple(training["betas"]), eps=training["epsilon"],
    )
    start = perf_counter()
    training_seconds = 0.0
    time_limit = training.get("max_segundos")
    stopped_for_time = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics: dict[str, float | int] = {}
    for step, batch in enumerate(train_loader, start=1):
        step_start = perf_counter()
        model.train()
        learning_rate = _learning_rate(cfg, step - 1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        ctc_loss = sum(
            _backward_microbatch(model, microbatch, cfg, smoother, criterion, device, len(batch.references))
            for microbatch in batch.microbatches(training["micro_batch_size"])
        )
        regularization = _regularization(model, batch.session_ids, training["l2"])
        if not bool(torch.isfinite(regularization)):
            raise FloatingPointError("La pérdida total no es finita.")
        # L2 y clipping se aplican una vez por actualización de 64 frases.
        regularization.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), training["clip_grad_norm"], error_if_nonfinite=True,
        )
        optimizer.step()
        # item sincroniza también el trabajo CUDA antes de medir el tiempo.
        logged = {
            "train/ctc_loss": ctc_loss, "train/l2": regularization.item(),
            "train/loss": ctc_loss + regularization.item(), "train/grad_norm": gradient_norm.item(),
            "train/learning_rate": learning_rate,
            **_memory_metrics(device),
        }
        training_seconds += perf_counter() - step_start
        logged["train/segundos_acumulados"] = training_seconds
        registrar(run, logged, step)
        if step == 1 or step % training["imprimir_cada"] == 0 or step == training["pasos"]:
            memory = (
                f"  pico CUDA asignado={logged['cuda/pico_asignado_mib']:.0f} MiB"
                f"  reservado={logged['cuda/pico_reservado_mib']:.0f} MiB"
                if device.type == "cuda" else ""
            )
            print(f"paso {step}/{training['pasos']}  ctc={logged['train/ctc_loss']:.4f}  "
                  f"l2={logged['train/l2']:.4f}  lr={learning_rate:.6f}{memory}", flush=True)
        time_reached = time_limit is not None and perf_counter() - start >= time_limit
        evaluation_due = (step - 1) % cfg["evaluacion"]["cada_pasos"] == 0 or step == training["pasos"] or time_reached
        if evaluation_due:
            metrics = evaluar_habla(model, evaluation_loader, cfg, smoother, device)
            if split == "val" and not metrics["completo"]:
                raise ValueError("La selección de checkpoint requiere toda la validación.")
            registrar(run, {f"{split}/{key}": value for key, value in metrics.items()}, step)
            print(f"{split}: PER={metrics['per']:.2%}  CTC={metrics['ctc_loss']:.4f}  "
                  f"frases={metrics['frases']}  completo={bool(metrics['completo'])}", flush=True)
            save_checkpoint(output / "ultimo.pt", model, cfg, train_loader.dataset, step, metrics, split=split)
            if split == "val":
                history.append({"step": step, **metrics})
                if best_metrics is None or metrics["per"] < best_metrics["per"]:
                    best_metrics, best_step = dict(metrics), step
                    save_checkpoint(output / "mejor_val.pt", model, cfg, train_loader.dataset, step, metrics, split="val")
        if evaluation_due and step < training["pasos"] and time_limit is not None and perf_counter() - start >= time_limit:
            stopped_for_time = True
            print(f"Límite de {time_limit:g} s alcanzado; parada tras el paso {step}, "
                  "con evaluación y checkpoint guardados.", flush=True)
            break
    if not metrics:
        raise ValueError("No se completó ningún paso con evaluación.")
    run.summary.update({
        "entrenamiento/pasos": step, "entrenamiento/segundos": training_seconds,
        "entrenamiento/pasos_previstos": training["pasos"],
        "entrenamiento/parada_por_tiempo": stopped_for_time,
        "ejecucion/segundos": perf_counter() - start,
        "checkpoint": str(output / ("mejor_val.pt" if split == "val" else "ultimo.pt")),
        **_memory_metrics(device),
        **{f"{split}_final/{key}": value for key, value in metrics.items()},
    })
    if split == "val":
        run.summary.update({"seleccion/metrica": "val/per", "seleccion/paso": best_step,
                            "seleccion/empates": "primer checkpoint",
                            "protocolo/sha256": cfg["protocolo"]["sha256"],
                            **{f"val_mejor/{key}": value for key, value in best_metrics.items()}})
        (output / "resultado.json").write_text(json.dumps({
            "config": cfg, "completed_steps": step, "best_val_step": best_step,
            "best_val": best_metrics, "last_val": metrics, "validation_history": history,
            "checkpoint": str(output / "mejor_val.pt"), "test_evaluated": False,
            "stopped_for_time": stopped_for_time,
        }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"checkpoint seleccionado: {output / 'mejor_val.pt'} (val PER; paso {best_step})", flush=True)
        return best_metrics
    print(f"checkpoint: {output / 'ultimo.pt'}", flush=True)
    return metrics


def ejecutar_habla(
    cfg: dict[str, Any], run: Any, *, checkpoint: Path | None, solo_evaluar: bool,
) -> dict[str, float | int]:
    """Construye el pipeline real y entrena o evalúa los pesos de un checkpoint."""
    if run is None:
        raise ValueError("El pipeline de habla requiere W&B: usa modo offline u online.")
    training = cfg["entrenamiento"]
    if cfg.get("protocolo") and cfg["evaluacion"]["max_lotes"] is not None:
        raise ValueError("El protocolo de comparación requiere validación/test completos: max_lotes=null.")
    schedule = training["learning_rate"]
    time_limit = training.get("max_segundos")
    if time_limit is not None and (
        isinstance(time_limit, bool) or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit) or time_limit <= 0
    ):
        raise ValueError("entrenamiento.max_segundos debe ser un número finito positivo o null.")
    if (
        training["optimizador"] != "adam_tensorflow" or training["pasos"] <= 0
        or training["epsilon"] <= 0 or training["clip_grad_norm"] <= 0 or training["l2"] < 0
        or training["imprimir_cada"] <= 0 or cfg["evaluacion"]["cada_pasos"] <= 0
        or training["micro_batch_size"] <= 0 or cfg["evaluacion"]["micro_batch_size"] <= 0
        or schedule["decay_pasos"] <= 0 or schedule["inicio"] <= 0
        or schedule["final"] < 0 or schedule["potencia"] <= 0
        or any(value < 0 for value in training["ruido"].values())
        or (cfg["evaluacion"]["max_lotes"] is not None and cfg["evaluacion"]["max_lotes"] <= 0)
    ):
        raise ValueError("Configuración de entrenamiento/evaluación de habla no válida.")
    requested = cfg["ejecucion"]["dispositivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA, pero no está disponible en este entorno.")
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    train_loader, evaluation_loader = construir_loaders(cfg, split_evaluacion="test" if solo_evaluar else None)
    split = evaluation_loader.dataset.split
    model = construir_modelo(cfg)
    if not isinstance(model, SpeechGRU):
        raise TypeError("El pipeline CTC requiere SpeechGRU.")
    if solo_evaluar:
        if checkpoint is None:
            raise ValueError("--solo-evaluar necesita --checkpoint con pesos entrenados.")
        state = load_checkpoint(checkpoint, model, cfg, evaluation_loader.dataset)
    model.to(device)
    smoother = build_smoother(cfg).to(device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    run.summary.update({"dispositivo": str(device), "hardware": hardware,
                        "train/frases_disponibles": len(train_loader.dataset),
                        f"{split}/frases_disponibles": len(evaluation_loader.dataset)})
    print(f"dispositivo: {device} ({hardware})  hilos CPU={cfg['ejecucion']['hilos_cpu']}  "
          f"train={len(train_loader.dataset)}  "
          f"{split}={len(evaluation_loader.dataset)}", flush=True)
    print(f"lote efectivo: {cfg['datos']['batch_size']}  "
          f"microlote train: {training['micro_batch_size']}  "
          f"microlote {split}: {cfg['evaluacion']['micro_batch_size']}", flush=True)
    if solo_evaluar:
        metrics = evaluar_habla(model, evaluation_loader, cfg, smoother, device)
        registrar(run, {f"test/{key}": value for key, value in metrics.items()}, state["step"])
        run.summary.update({"checkpoint": str(checkpoint), "checkpoint/paso": state["step"],
                            **{f"test_final/{key}": value for key, value in metrics.items()}})
        return metrics
    output = Path(cfg["salida"]["directorio"]) / run.id
    output.mkdir(parents=True, exist_ok=False)
    return entrenar_habla(model, train_loader, evaluation_loader, cfg, run, smoother, device, output)
