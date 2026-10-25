"""Recarga independiente de pesos T12 y predicción por perfil usando validación."""

from __future__ import annotations

from math import isclose, isfinite
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import build_smoother
from src.data.speech_dataset import SpeechDataset, collate_speech, file_sha256
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.metrics import decode_ctc, edit_distance
from src.training.speech import evaluar_salidas
from src.utils.seed import fijar_seed, crear_generador
from src.utils.tracking import registrar


def verify_speech_checkpoint(
    cfg: dict[str, Any], checkpoint: Path, spec: dict[str, Any], run: Any,
) -> dict[str, Any]:
    """Recarga, exige métricas de val equivalentes y predice un ejemplo por salida.

    No construye train/test ni optimizador. Cada perfil ejecuta su prefijo
    por separado y se contrasta con la salida compartida del mismo ejemplo.
    """
    if not cfg.get("protocolo"):
        raise ValueError("La verificación requiere el protocolo con validación independiente.")
    for key in ("ctc_atol", "ctc_rtol", "logits_atol", "logits_rtol"):
        if type(spec[key]) not in (float, int) or not isfinite(spec[key]) or spec[key] < 0:
            raise ValueError("Las tolerancias deben ser finitas y no negativas.")
    torch.set_num_threads(cfg["ejecucion"]["hilos_cpu"])
    fijar_seed(cfg["experimento"]["seed"])
    requested = cfg["ejecucion"]["dispositivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA para recargar este perfil.")
    digest = file_sha256(checkpoint)
    dataset = SpeechDataset(cfg, "val")
    loader = DataLoader(dataset, batch_size=cfg["datos"]["batch_size"], shuffle=False,
                        num_workers=cfg["datos"]["num_workers"], pin_memory=cfg["datos"]["pin_memory"],
                        collate_fn=collate_speech, generator=crear_generador(cfg["experimento"]["seed"]))
    model = construir_modelo(cfg)
    state = load_checkpoint(checkpoint, model, cfg, dataset)
    model.to(device).eval()
    smoother = build_smoother(cfg).to(device)
    measured, seconds = evaluar_salidas(model, loader, cfg, smoother, device)
    stored = state.get("exit_metrics", {cfg["modelo"]["capas"]: state["val_metrics"]})
    comparisons = {}
    for depth, metrics in measured.items():
        previous = stored[depth]
        counts_equal = all(metrics[key] == previous[key] for key in ("errores", "fonemas", "frases", "completo"))
        per_equal = metrics["per"] == previous["per"]
        ctc_close = isclose(metrics["ctc_loss"], previous["ctc_loss"], rel_tol=spec["ctc_rtol"], abs_tol=spec["ctc_atol"])
        comparisons[depth] = {"counts_equal": counts_equal, "per_equal": per_equal,
                              "ctc_close": ctc_close, "ctc_abs_difference": abs(metrics["ctc_loss"] - previous["ctc_loss"])}
        if not (counts_equal and per_equal and ctc_close and metrics["completo"] == 1):
            raise ValueError(f"La recarga no reproduce las métricas de validación de la salida {depth}.")
    index = spec["ejemplo_validacion"]
    if type(index) is not int or not 0 <= index < len(dataset):
        raise ValueError("El ejemplo debe ser un índice válido de validación.")
    example = dataset[index]
    batch = collate_speech([example]).to(device)
    profiles = {}
    with torch.inference_mode():
        features = smoother(batch.features)
        lengths = model.output_lengths(batch.input_lengths)
        shared = model.forward_exits(features, batch.session_ids)
        for depth in model.available_exits:
            called = []
            handles = [layer.register_forward_pre_hook(lambda _m, _a, n=i: called.append(n))
                       for i, layer in enumerate(model.gru_layers, start=1)]
            try:
                logits = model(features, batch.session_ids, exit_layer=depth)
            finally:
                for handle in handles:
                    handle.remove()
            if called != list(range(1, depth + 1)) or not torch.allclose(
                logits, shared[depth], atol=spec["logits_atol"], rtol=spec["logits_rtol"],
            ):
                raise ValueError(f"El perfil {depth} no coincide con su prefijo/salida compartida.")
            hypothesis = decode_ctc(logits, lengths, state["blank_id"])[0]
            if hypothesis != decode_ctc(shared[depth], lengths, state["blank_id"])[0]:
                raise ValueError("La predicción individual no coincide con la compartida.")
            errors = edit_distance(example.targets.tolist(), hypothesis)
            profiles[depth] = {"executed_layers": called, "hypothesis_ids": hypothesis,
                               "hypothesis_phones": [state["phones"][phone] for phone in hypothesis],
                               "errors": errors, "reference_phones": len(example.targets),
                               "example_per": errors / len(example.targets)}
    if file_sha256(checkpoint) != digest:
        raise ValueError("El checkpoint cambió durante la verificación; repetir con un archivo estable.")
    record = {
        "verified": True, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "checkpoint_step": state["step"], "split": "val", "optimizer_steps": 0,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "tolerances": spec,
        "metrics_by_exit": measured, "comparisons": comparisons,
        "shared_validation_seconds": seconds, "test_evaluated": False,
        "example": {"validation_index": index, "session": cfg["datos"]["sesiones"][example.session_id],
                    "source_trial_index": example.trial_index, "block": example.block,
                    "reference_text": example.reference, "reference_ids": example.targets.tolist(),
                    "reference_phones": [state["phones"][phone] for phone in example.targets.tolist()],
                    "profiles": profiles},
    }
    registrar(run, {f"val_recarga/capa_{depth}/{key}": value
                    for depth, metrics in measured.items() for key, value in metrics.items()}, state["step"])
    run.summary.update({"verificacion/correcta": True, "verificacion/actualizaciones": 0,
                        "checkpoint/sha256": digest, "checkpoint/paso": state["step"],
                        "verificacion/test_evaluado": False})
    return record
