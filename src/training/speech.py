"""Entrenamiento offline de speechBCI con CTC y evaluación PER."""

from __future__ import annotations

from math import isclose, isfinite
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
from src.training.checkpoints import load_checkpoint
from src.training.metrics import decode_ctc, phoneme_errors
from src.utils.tracking import registrar


def _criterion(cfg: dict[str, Any]) -> nn.CTCLoss:
    # TF reduce_mean no divide cada frase por su longitud, a diferencia del
    # reduction='mean' de PyTorch. Un infinito debe detener la ejecución.
    return nn.CTCLoss(
        blank=len(cfg["datos"]["fonemas"]), reduction="none", zero_infinity=False
    )


def _ctc_losses(
    criterion: nn.CTCLoss,
    logits: Tensor,
    lengths: Tensor,
    batch: SpeechBatch,
) -> Tensor:
    losses = criterion(
        logits.log_softmax(-1).transpose(0, 1).contiguous(),
        batch.targets,
        lengths,
        batch.target_lengths,
    )
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError(
            "CTC contiene NaN o infinito; se detiene antes de actualizar pesos."
        )
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


def _regularization(
    model: SpeechGRU, session_ids: Tensor, coefficient: float
) -> Tensor:
    weights = [layer.weight_ih_l0 for layer in model.gru_layers]
    weights.extend(
        model.input_layers[day].weight for day in session_ids.unique().tolist()
    )
    return coefficient * sum(weight.square().sum() for weight in weights)


def _learning_rate(cfg: dict[str, Any], completed_steps: int) -> float:
    schedule = cfg["entrenamiento"]["learning_rate"]
    fraction = min(completed_steps / schedule["decay_pasos"], 1.0)
    return (schedule["inicio"] - schedule["final"]) * (1 - fraction) ** schedule[
        "potencia"
    ] + schedule["final"]


def _ctc_weights(cfg: dict[str, Any]) -> dict[int, float] | None:
    """Valida pesos explícitos normalizados; None conserva el objetivo original."""
    layers = cfg["modelo"].get("salidas_intermedias", [])
    supplied = cfg["entrenamiento"].get("ctc_pesos")
    if not layers:
        if supplied is not None:
            raise ValueError("ctc_pesos requiere configurar salidas_intermedias.")
        return None
    if not cfg.get("protocolo") or not isinstance(supplied, dict):
        raise ValueError(
            "La CTC conjunta requiere el protocolo de validación y ctc_pesos explícitos."
        )
    weights = {}
    for key, value in supplied.items():
        if (
            type(key) not in (str, int)
            or not str(key).isdigit()
            or str(int(key)) != str(key)
            or type(value) not in (float, int)
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "Cada peso CTC debe identificar una capa y ser positivo y finito."
            )
        weights[int(key)] = float(value)
    expected = set(layers) | {cfg["modelo"]["capas"]}
    if (
        len(weights) != len(supplied)
        or set(weights) != expected
        or not isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("ctc_pesos debe cubrir todas las salidas y sumar uno.")
    return weights


def _backward_microbatch(
    model: SpeechGRU,
    batch: SpeechBatch,
    cfg: dict[str, Any],
    smoother: GaussianSmoothing,
    criterion: nn.CTCLoss,
    device: torch.device,
    effective_batch_size: int,
    *,
    component_losses: dict[int, float] | None = None,
) -> float:
    # El grafo de cada grupo se libera antes de procesar el siguiente.
    weights = _ctc_weights(cfg)
    batch = batch.to(device, non_blocking=cfg["datos"]["pin_memory"])
    features = smoother(_training_features(batch, cfg))
    if weights is None:
        logits = model(features, batch.session_ids)
        losses = _ctc_losses(
            criterion, logits, model.output_lengths(batch.input_lengths), batch
        )
        loss = losses.sum() / effective_batch_size
    else:
        outputs = model.forward_exits(features, batch.session_ids)
        lengths = model.output_lengths(batch.input_lengths)
        components = {
            depth: _ctc_losses(criterion, logits, lengths, batch)
            for depth, logits in outputs.items()
        }
        loss = sum(
            weights[depth] * values.sum() / effective_batch_size
            for depth, values in components.items()
        )
    # Ponderar por frases del lote completo también trata bien el último
    # grupo cuando batch_size no es divisible por micro_batch_size.
    loss.backward()
    if weights is not None and component_losses is not None:
        for depth, values in components.items():
            component_losses[depth] = (
                component_losses.get(depth, 0.0)
                + values.detach().sum().item() / effective_batch_size
            )
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
    model: SpeechGRU,
    loader: DataLoader,
    cfg: dict[str, Any],
    smoother: GaussianSmoothing,
    device: torch.device,
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
        for microbatch in batch.microbatches(
            cfg["evaluacion"]["micro_batch_size"], trim_padding=True
        ):
            microbatch = microbatch.to(device, non_blocking=cfg["datos"]["pin_memory"])
            lengths = model.output_lengths(microbatch.input_lengths)
            logits = model(smoother(microbatch.features), microbatch.session_ids)
            losses = _ctc_losses(criterion, logits, lengths, microbatch)
            hypotheses = decode_ctc(logits, lengths, criterion.blank)
            batch_errors, batch_phones = phoneme_errors(
                hypotheses, microbatch.targets, microbatch.target_lengths
            )
            total_loss += losses.sum().item()
            errors += batch_errors
            phones += batch_phones
            trials += len(microbatch.references)
            del logits, losses, microbatch
    if not trials or not phones:
        raise ValueError("La evaluación no contiene frases o fonemas.")
    return {
        "ctc_loss": total_loss / trials,
        "per": errors / phones,
        "errores": errors,
        "fonemas": phones,
        "frases": trials,
        "completo": int(trials == len(loader.dataset)),
        "segundos": perf_counter() - start,
    }


@torch.no_grad()
def evaluar_salidas(
    model: SpeechGRU,
    loader: DataLoader,
    cfg: dict[str, Any],
    smoother: GaussianSmoothing,
    device: torch.device,
) -> tuple[dict[int, dict[str, float | int]], float]:
    """Evalúa todas las salidas en una pasada; devuelve métricas y tiempo conjunto.

    El tiempo corresponde a la evaluación de todas las cabezas, no a la
    latencia individual de un perfil. No hay ruido ni dropout.
    """
    start = perf_counter()
    model.eval()
    criterion = _criterion(cfg)
    totals = {
        depth: {"ctc_loss": 0.0, "errores": 0, "fonemas": 0, "frases": 0}
        for depth in model.available_exits
    }
    limit = cfg["evaluacion"]["max_lotes"]
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        for small in batch.microbatches(
            cfg["evaluacion"]["micro_batch_size"], trim_padding=True
        ):
            small = small.to(device, non_blocking=cfg["datos"]["pin_memory"])
            lengths = model.output_lengths(small.input_lengths)
            outputs = model.forward_exits(smoother(small.features), small.session_ids)
            for depth, logits in outputs.items():
                losses = _ctc_losses(criterion, logits, lengths, small)
                hypotheses = decode_ctc(logits, lengths, criterion.blank)
                errors, phones = phoneme_errors(
                    hypotheses, small.targets, small.target_lengths
                )
                total = totals[depth]
                total["ctc_loss"] += losses.sum().item()
                total["errores"] += errors
                total["fonemas"] += phones
                total["frases"] += len(small.references)
    results = {}
    for depth, total in totals.items():
        if not total["frases"] or not total["fonemas"]:
            raise ValueError("La evaluación no contiene frases o fonemas.")
        results[depth] = {
            **total,
            "ctc_loss": total["ctc_loss"] / total["frases"],
            "per": total["errores"] / total["fonemas"],
            "completo": int(total["frases"] == len(loader.dataset)),
        }
    return results, perf_counter() - start


def entrenar_habla(
    model: SpeechGRU,
    train_loader: DataLoader,
    evaluation_loader: DataLoader,
    cfg: dict[str, Any],
    run: Any,
    smoother: GaussianSmoothing,
    device: torch.device,
    output: Path,
    *,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    """Entrena y devuelve las métricas de los pesos seleccionados en validación."""
    from src.training.speech_trainer import SpeechTrainer

    return SpeechTrainer(
        model,
        train_loader,
        evaluation_loader,
        cfg,
        run,
        smoother,
        device,
        output,
        resume_state,
    ).fit()


def _validate_training_config(cfg: dict[str, Any]) -> None:
    """Comprueba los tamaños, límites y parámetros del optimizador."""
    training = cfg["entrenamiento"]
    if cfg.get("protocolo") and cfg["evaluacion"]["max_lotes"] is not None:
        raise ValueError(
            "El protocolo de comparación requiere validación/test completos: max_lotes=null."
        )
    schedule = training["learning_rate"]
    time_limit = training.get("max_segundos")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, (int, float))
        or not isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError(
            "entrenamiento.max_segundos debe ser un número finito positivo o null."
        )
    if (
        training["optimizador"] != "adam_tensorflow"
        or training["pasos"] <= 0
        or training["epsilon"] <= 0
        or training["clip_grad_norm"] <= 0
        or training["l2"] < 0
        or training["imprimir_cada"] <= 0
        or cfg["evaluacion"]["cada_pasos"] <= 0
        or training["micro_batch_size"] <= 0
        or cfg["evaluacion"]["micro_batch_size"] <= 0
        or schedule["decay_pasos"] <= 0
        or schedule["inicio"] <= 0
        or schedule["final"] < 0
        or schedule["potencia"] <= 0
        or any(value < 0 for value in training["ruido"].values())
        or (
            cfg["evaluacion"]["max_lotes"] is not None
            and cfg["evaluacion"]["max_lotes"] <= 0
        )
    ):
        raise ValueError(
            "Configuración de entrenamiento/evaluación de habla no válida."
        )


def _log_pipeline(
    cfg: dict[str, Any],
    run: Any,
    device: torch.device,
    train_loader: DataLoader,
    evaluation_loader: DataLoader,
) -> None:
    """Muestra el dispositivo, las frases disponibles y los tamaños de lote."""
    training = cfg["entrenamiento"]
    split = evaluation_loader.dataset.split
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    run.summary.update(
        {
            "dispositivo": str(device),
            "hardware": hardware,
            "train/frases_disponibles": len(train_loader.dataset),
            f"{split}/frases_disponibles": len(evaluation_loader.dataset),
        }
    )
    print(
        f"dispositivo: {device} ({hardware})  hilos CPU={cfg['ejecucion']['hilos_cpu']}  "
        f"train={len(train_loader.dataset)}  "
        f"{split}={len(evaluation_loader.dataset)}",
        flush=True,
    )
    print(
        f"lote efectivo: {cfg['datos']['batch_size']}  "
        f"microlote train: {training['micro_batch_size']}  "
        f"microlote {split}: {cfg['evaluacion']['micro_batch_size']}",
        flush=True,
    )


def _evaluate_checkpoint(
    model: SpeechGRU,
    loader: DataLoader,
    cfg: dict[str, Any],
    run: Any,
    smoother: GaussianSmoothing,
    device: torch.device,
    checkpoint: Path,
    step: int,
) -> dict[str, float | int]:
    """Evalúa los pesos cargados y registra las métricas sin actualizar el modelo."""
    split = loader.dataset.split
    weights = _ctc_weights(cfg)
    per_exit_logged = {}
    if weights is None:
        metrics = evaluar_habla(model, loader, cfg, smoother, device)
    else:
        exit_metrics, seconds = evaluar_salidas(model, loader, cfg, smoother, device)
        metrics = {**exit_metrics[cfg["modelo"]["capas"]], "segundos": seconds}
        per_exit_logged = {
            f"{split}/capa_{depth}/{key}": value
            for depth, values in exit_metrics.items()
            for key, value in values.items()
        }
        per_exit_logged[f"{split}/segundos_todas_salidas"] = seconds
    registrar(
        run,
        {
            **{f"{split}/{key}": value for key, value in metrics.items()},
            **per_exit_logged,
        },
        step,
    )
    run.summary.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint/paso": step,
            **{f"{split}_final/{key}": value for key, value in metrics.items()},
        }
    )
    return metrics


def ejecutar_habla(
    cfg: dict[str, Any],
    run: Any,
    *,
    checkpoint: Path | None,
    solo_evaluar: bool,
    split_evaluacion: str | None = None,
    resume_state: dict[str, Any] | None = None,
    resume_path: Path | None = None,
) -> dict[str, float | int]:
    """Construye el pipeline real y entrena o evalúa los pesos de un checkpoint."""
    if run is None:
        raise ValueError(
            "El pipeline de habla requiere W&B: usa modo offline u online."
        )
    if split_evaluacion is not None and (
        not solo_evaluar or split_evaluacion not in ("val", "test")
    ):
        raise ValueError(
            "La partición de evaluación solo puede elegirse en modo solo_evaluar."
        )
    if split_evaluacion == "val" and not cfg.get("protocolo"):
        raise ValueError(
            "Evaluar val requiere el protocolo T12 con validación independiente."
        )
    _ctc_weights(cfg)
    _validate_training_config(cfg)
    requested = cfg["ejecucion"]["dispositivo"]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió CUDA, pero no está disponible en este entorno.")
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    selected_split = (split_evaluacion or "test") if solo_evaluar else None
    train_loader, evaluation_loader = construir_loaders(
        cfg, split_evaluacion=selected_split
    )
    model = construir_modelo(cfg)
    if not isinstance(model, SpeechGRU):
        raise TypeError("El pipeline CTC requiere SpeechGRU.")
    if solo_evaluar:
        if checkpoint is None:
            raise ValueError(
                "--solo-evaluar necesita --checkpoint con pesos entrenados."
            )
        state = load_checkpoint(checkpoint, model, cfg, evaluation_loader.dataset)
    model.to(device)
    smoother = build_smoother(cfg).to(device)
    _log_pipeline(cfg, run, device, train_loader, evaluation_loader)
    if solo_evaluar:
        return _evaluate_checkpoint(
            model,
            evaluation_loader,
            cfg,
            run,
            smoother,
            device,
            checkpoint,
            state["step"],
        )
    output = Path(cfg["salida"]["directorio"]) / run.id
    if resume_state is not None:
        if solo_evaluar or resume_path is None:
            raise ValueError(
                "La reanudación requiere un estado de entrenamiento, sin solo_evaluar."
            )
        output = resume_path.parent
        if output.name != resume_state["tracking"]["logical_id"]:
            raise ValueError(
                "Conserva el nombre de la carpeta del run al copiar el estado completo."
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return entrenar_habla(
        model,
        train_loader,
        evaluation_loader,
        cfg,
        run,
        smoother,
        device,
        output,
        resume_state=resume_state,
    )
