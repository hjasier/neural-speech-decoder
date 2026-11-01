"""Estado de entrenamiento T12, separado de los checkpoints de inferencia."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.speech_dataset import SessionBatchSampler, preparation_signature
from src.utils.seed import capturar_azar, restaurar_azar


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    names = [
        "run.py",
        "src/data/loaders.py",
        "src/data/speech_dataset.py",
        "src/models/speech_gru.py",
        "src/models/factory.py",
        "src/utils/seed.py",
        "src/utils/tracking.py",
        "src/utils/env.py",
        "src/training/speech.py",
        "src/training/speech_trainer.py",
        "src/training/speech_resume.py",
        "src/training/checkpoints.py",
        "src/training/optim.py",
        "src/training/metrics.py",
        "src/inference/ctc.py",
        "src/inference/checkpoints.py",
        "src/data/preprocessing.py",
        "src/data/speech_protocol.py",
    ]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }


def _scientific_config(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value.pop(
        "reanudacion", None
    )  # Frecuencia de persistencia y duración de una sesión.
    value.get("wandb", {}).pop("directorio", None)
    return value


def validate_resume_config(cfg: dict[str, Any]) -> None:
    """Comprueba la partición fija y exige workers=0 para recuperar el muestreo."""
    spec = cfg.get("reanudacion")
    if (
        not cfg.get("protocolo")
        or not isinstance(spec, dict)
        or type(spec.get("cada_pasos")) is not int
        or spec["cada_pasos"] <= 0
        or cfg["datos"]["num_workers"] != 0
    ):
        raise ValueError(
            "La reanudación T12 requiere protocolo, cada_pasos positivo y num_workers=0."
        )
    limit = spec.get("max_pasos_sesion")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError(
            "reanudacion.max_pasos_sesion debe ser entero positivo o null."
        )


def read_training_checkpoint(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Lee antes de abrir W&B; los pesos antiguos nunca se migran implícitamente."""
    validate_resume_config(cfg)
    try:
        with ZipFile(path) as archive:
            damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Checkpoint dañado: CRC incorrecto en {damaged}.")
    except BadZipFile as error:
        raise ValueError(
            "Archivo de reanudación incompleto o con formato antiguo."
        ) from error
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("kind") != "t12_training_resume"
        or state.get("version") != 1
    ):
        raise ValueError(
            "Este archivo no permite reanudar: los pesos antiguos solo sirven para inferencia."
        )
    required = {
        "model_state",
        "optimizer_state",
        "sampler",
        "loaders",
        "rng",
        "progress",
        "tracking",
        "best_checkpoint",
        "last_checkpoint",
        "config",
        "runtime",
        "sources",
        "data_signature",
        "normalization",
        "sessions",
        "phones",
        "step",
    }
    if not required <= state.keys():
        raise ValueError("Estado de entrenamiento incompleto; no se abre otro run.")
    if (
        _scientific_config(cfg) != _scientific_config(state["config"])
        or state["sources"] != _sources()
        or state["data_signature"] != preparation_signature(cfg)
    ):
        raise ValueError(
            "La configuración, fuentes o datos cambiaron; no es una reanudación equivalente."
        )
    step = state["step"]
    progress = state["progress"]
    tracking = state["tracking"]
    if (
        type(step) is not int
        or not 0 <= step <= cfg["entrenamiento"]["pasos"]
        or state["sampler"]["completed_batches"] != step
        or not tracking["logical_id"]
        or Path(tracking["logical_id"]).name != tracking["logical_id"]
        or tracking["project"] != cfg["wandb"]["proyecto"]
        or tracking["mode"] != cfg["wandb"]["modo"]
        or any(
            group.get("iterations") != step
            for group in state["optimizer_state"]["param_groups"]
        )
        or [row["step"] for row in progress["metric_history"]]
        != list(range(1, step + 1))
        or any(point["completo"] != 1 for point in progress["history"])
        or any(point["step"] > step for point in progress["history"])
    ):
        raise ValueError("Paso, muestreo o historial inconsistentes en el checkpoint.")
    if step:
        selected = min(progress["history"], key=lambda point: point["per"])
        best = state["best_checkpoint"]
        last = state["last_checkpoint"]
        if (
            best is None
            or selected["step"] != progress["best_step"]
            or best["step"] != selected["step"]
            or any(
                selected[key] != value
                for key, value in progress["best_metrics"].items()
            )
            or best["val_metrics"] != progress["best_metrics"]
            or last is None
            or last["step"] != progress["history"][-1]["step"]
            or last["val_metrics"] != progress["metrics"]
        ):
            raise ValueError(
                "La selección del mejor checkpoint no coincide con el historial."
            )
    return state


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    """Escribe y sincroniza el temporal antes de sustituir el archivo anterior."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_inference(path: Path, state: dict[str, Any]) -> None:
    atomic_torch_save(path, state)
    metrics = state["val_metrics"]
    value = {"step": state["step"], "split": "val", **metrics}
    if "exit_metrics" in state:
        value["salidas"] = state["exit_metrics"]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path.with_suffix(".json"))


def runtime_state(device: torch.device) -> dict[str, Any]:
    """Registra el entorno numérico; CPU no consulta ni inicializa CUDA."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device_type": device.type,
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "threads": torch.get_num_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def restore_training(
    state: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train: DataLoader,
    validation: DataLoader,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    """Restaura progreso y alias de inferencia desde el estado autocontenido."""
    current = runtime_state(device)
    expected = state["runtime"]
    if any(
        current[key] != value for key, value in expected.items() if key != "hardware"
    ):
        raise ValueError(
            "El entorno numérico cambió (versiones, dispositivo o determinismo)."
        )
    if current["hardware"] != expected["hardware"]:
        print(
            "Aviso: cambió el modelo de GPU; se restauran los estados, sin prometer igualdad bit a bit.",
            flush=True,
        )
    dataset = train.dataset
    if (
        state["normalization"] != dataset.normalization_states
        or state["sessions"] != list(dataset.sessions)
        or state["phones"] != state["config"]["datos"]["fonemas"]
    ):
        raise ValueError(
            "Normalización, sesiones o vocabulario diferentes al checkpoint."
        )
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    train.batch_sampler.load_state_dict(state["sampler"])
    train.generator.set_state(state["loaders"]["train"])
    validation.generator.set_state(state["loaders"]["validation"])
    best = state["best_checkpoint"]
    if best is not None:
        # Reparar también un alias adelantado si el corte ocurrió antes del commit.
        _write_inference(output / "mejor_val.pt", best)
    if state["last_checkpoint"] is not None:
        _write_inference(output / "ultimo.pt", state["last_checkpoint"])
    restaurar_azar(state["rng"], device)
    return copy.deepcopy(state["progress"])


def flush_metrics(run: Any, history: list[dict[str, Any]], completed_step: int) -> None:
    """Publica únicamente tras persistir y recupera un prefijo que W&B no recibió."""
    next_step = int(getattr(run, "step", 0))
    if next_step > completed_step + 1:
        raise ValueError(
            "W&B está adelantado al checkpoint: selecciona el estado de entrenamiento más reciente."
        )
    for row in history:
        if row["step"] >= next_step:
            run.log(row["values"], step=row["step"], commit=True)


def save_training(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    train: DataLoader,
    validation: DataLoader,
    device: torch.device,
    progress: dict[str, Any],
    run: Any,
    logical_id: str,
) -> dict[str, Any]:
    """Confirma un paso completo y su selección antes de publicar métricas."""
    sampler = train.batch_sampler
    if not isinstance(sampler, SessionBatchSampler) or not sampler.resumable:
        raise ValueError("Se necesita el muestreador T12 reanudable.")
    dataset = train.dataset
    best = None
    last = None
    if progress["best_metrics"] is not None:
        best = torch.load(
            path.parent / "mejor_val.pt", map_location="cpu", weights_only=True
        )
        last = torch.load(
            path.parent / "ultimo.pt", map_location="cpu", weights_only=True
        )
    rng = capturar_azar(device)
    state = {
        "kind": "t12_training_resume",
        "version": 1,
        "step": sampler.completed_batches,
        "config": copy.deepcopy(cfg),
        "model_state": _cpu(model.state_dict()),
        "optimizer_state": _cpu(optimizer.state_dict()),
        "sampler": sampler.state_dict(),
        "loaders": {
            "train": train.generator.get_state(),
            "validation": validation.generator.get_state(),
        },
        "rng": rng,
        "progress": copy.deepcopy(progress),
        "best_checkpoint": best,
        "last_checkpoint": last,
        "data_signature": preparation_signature(cfg),
        "normalization": dataset.normalization_states,
        "sessions": list(dataset.sessions),
        "phones": list(cfg["datos"]["fonemas"]),
        "runtime": runtime_state(device),
        "sources": _sources(),
        "tracking": {
            "logical_id": logical_id,
            "last_segment_id": run.id,
            "project": cfg["wandb"]["proyecto"],
            "entity": getattr(run, "entity", None),
            "mode": cfg["wandb"]["modo"],
        },
    }
    atomic_torch_save(path, state)
    # El JSON es informativo; solo el .pt es la fuente autoritativa.
    sidecar = {
        "kind": state["kind"],
        "version": 1,
        "step": state["step"],
        "planned_steps": cfg["entrenamiento"]["pasos"],
        "run_id": logical_id,
        "best_val_step": progress["best_step"],
        "best_val": progress["best_metrics"],
        "variant": cfg.get("comparacion", {}).get("variante"),
        "seed": cfg["experimento"]["seed"],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(sidecar, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path.with_suffix(".json"))
    try:
        flush_metrics(run, progress["metric_history"], state["step"])
    finally:
        # El seguimiento no debe consumir azar del entrenamiento.
        restaurar_azar(rng, device)
    return state
