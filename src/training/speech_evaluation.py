"""Calidad T12 por sesión y controles de señal, sin entrenamiento ni selección."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.speech_dataset import SpeechBatch
from src.models.speech_gru import SpeechGRU
from src.training.metrics import decode_ctc, edit_distance
from src.training.speech import _criterion, _ctc_losses
from src.utils.seed import crear_generador


def signal_control(batch: SpeechBatch, kind: str, seed: int, split: str) -> SpeechBatch:
    """Controla señal normalizada CPU antes del suavizado; conserva etiquetas.

    La permutación mueve vectores de todos los canales juntos, solo dentro
    de los bins válidos de cada frase. Su RNG depende de semilla, split,
    sesión e índice de ensayo: no depende de lote, perfil ni orden de visita.
    """
    if kind == "original":
        return batch
    if kind == "cero":
        return replace(batch, features=torch.zeros_like(batch.features))
    if kind != "permutada" or batch.features.device.type != "cpu":
        raise ValueError("El control temporal debe aplicarse en CPU antes de mover el lote.")
    features = batch.features.clone()
    for index, length in enumerate(batch.input_lengths.tolist()):
        identity = [seed, split, int(batch.session_ids[index]), batch.trial_indices[index]]
        digest = hashlib.sha256(json.dumps(identity).encode("utf-8")).digest()
        generator = crear_generador(int.from_bytes(digest[:8], "big"))
        order = torch.randperm(length, generator=generator)
        features[index, :length] = batch.features[index, :length][order]
    return replace(batch, features=features)


def _counts() -> dict[str, float | int]:
    return {"ctc_sum": 0.0, "errores": 0, "fonemas": 0, "frases": 0}


def _metrics(total: dict[str, float | int], complete: bool) -> dict[str, float | int]:
    if not total["frases"] or not total["fonemas"]:
        raise ValueError("La evaluación no contiene frases o fonemas.")
    return {"ctc_loss": total["ctc_sum"] / total["frases"],
            "per": total["errores"] / total["fonemas"],
            **{key: total[key] for key in ("errores", "fonemas", "frases")},
            "completo": int(complete)}


@torch.inference_mode()
def evaluate_signal_controls(
    model: SpeechGRU, loader: DataLoader, cfg: dict[str, Any], spec: dict[str, Any],
    smoother: nn.Module, device: torch.device,
    *, prediction_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evalúa todas las frases y agrega PER por fonemas, global y por sesión.

    El tiempo abarca controles, todas las salidas, CTC, decodificación y
    escritura de ejemplos; no representa latencia de un perfil. La referencia
    no entra en forward: se usa para CTC/PER y para el registro.
    """
    controls = spec["controles"]
    seed = spec["semilla_control"]
    split = spec["split"]
    if (split not in ("val", "test") or not isinstance(controls, list) or not controls
            or len(controls) != len(set(controls)) or "original" not in controls
            or any(control not in ("original", "cero", "permutada") for control in controls)
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Split, controles o semilla de diagnóstico no válidos.")
    if cfg["evaluacion"]["max_lotes"] is not None:
        raise ValueError("El diagnóstico requiere evaluar la partición completa: max_lotes=null.")
    if loader.dataset.split != split:
        raise ValueError("El loader no corresponde al split del diagnóstico.")
    exits = model.available_exits if spec["salidas"] is None else tuple(spec["salidas"])
    if (not exits or len(set(exits)) != len(exits)
            or any(type(depth) is not int or depth not in model.available_exits for depth in exits)):
        raise ValueError("Las salidas solicitadas deben existir en el modelo.")
    model.eval()
    criterion = _criterion(cfg)
    totals = {control: {depth: _counts() for depth in exits} for control in controls}
    sessions: dict[str, dict[int, dict[str, dict[str, float | int]]]] = {
        control: {depth: {} for depth in exits} for control in controls}
    start = perf_counter()
    for batch in loader:
        for small in batch.microbatches(cfg["evaluacion"]["micro_batch_size"], trim_padding=True):
            references = [part.tolist() for part in small.targets.split(small.target_lengths.tolist())]
            names = [cfg["datos"]["sesiones"][day] for day in small.session_ids.tolist()]
            rows = [{"split": split, "source_split": "train" if split == "val" else "test",
                     "session": name, "source_trial_index": small.trial_indices[i], "block": small.blocks[i],
                     "input_bins": int(small.input_lengths[i]), "reference_text": small.references[i],
                     "reference_ids": reference,
                     "reference_phones": [cfg["datos"]["fonemas"][phone] for phone in reference],
                     "predictions": {control: {} for control in controls}}
                    for i, (name, reference) in enumerate(zip(names, references, strict=True))]
            for control in controls:
                transformed = signal_control(small, control, seed, split).to(device)
                lengths = model.output_lengths(transformed.input_lengths)
                outputs = model.forward_exits(smoother(transformed.features), transformed.session_ids, layers=exits)
                for depth, logits in outputs.items():
                    losses = _ctc_losses(criterion, logits, lengths, transformed).cpu().tolist()
                    hypotheses = decode_ctc(logits, lengths, criterion.blank)
                    for i, (name, reference, hypothesis, loss) in enumerate(zip(names, references, hypotheses, losses, strict=True)):
                        errors = edit_distance(reference, hypothesis)
                        session = sessions[control][depth].setdefault(name, _counts())
                        for target in (totals[control][depth], session):
                            target["ctc_sum"] += loss
                            target["errores"] += errors
                            target["fonemas"] += len(reference)
                            target["frases"] += 1
                        rows[i]["predictions"][control][depth] = {
                            "ids": hypothesis, "phones": [cfg["datos"]["fonemas"][phone] for phone in hypothesis],
                            "ctc_loss": loss, "errors": errors, "per": errors / len(reference)}
                del outputs, transformed
            if prediction_sink is not None:
                for row in rows:
                    prediction_sink(row)
    complete = all(total["frases"] == len(loader.dataset) for values in totals.values() for total in values.values())
    if not complete:
        raise ValueError("El diagnóstico no recorrió la partición completa.")
    return {"split": split, "controls": controls, "control_seed": seed,
            "available_exits": list(model.available_exits), "evaluated_exits": list(exits),
            "global": {control: {depth: _metrics(total, complete) for depth, total in values.items()}
                       for control, values in totals.items()},
            "by_session": {control: {depth: {name: _metrics(total, complete) for name, total in values.items()}
                                     for depth, values in depths.items()} for control, depths in sessions.items()},
            "diagnostic_wall_seconds": perf_counter() - start,
            "timing_scope": "controles, salidas compartidas, CTC, greedy, agregación y escritura; no latencia de perfil",
            "optimizer_steps": 0, "test_evaluated": split == "test"}
