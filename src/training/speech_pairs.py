"""Pares fonemas predichos → frase para la Línea A, sin entrenar ni usar test."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.speech_gru import SpeechGRU
from src.training.metrics import decode_ctc, edit_distance


@torch.inference_mode()
def predict_phonemes(
    model: SpeechGRU, loader: DataLoader, cfg: dict[str, Any], smoother: nn.Module, device: torch.device,
) -> Iterator[dict[str, Any]]:
    """Decodifica con CTC voraz la salida final de cada frase del loader.

    Es la misma pasada causal y voraz que la validación del entrenamiento:
    frase completa, padding solo al final y sin ruido ni enmascarado.
    """
    split = loader.dataset.split
    if split == "test":
        raise ValueError("Los pares no se generan sobre test: queda reservado para la Fase 5.")
    phones = cfg["datos"]["fonemas"]
    depth = model.num_layers
    model.eval()
    for batch in loader:
        for small in batch.microbatches(cfg["evaluacion"]["micro_batch_size"], trim_padding=True):
            references = [part.tolist() for part in small.targets.split(small.target_lengths.tolist())]
            moved = small.to(device)
            lengths = model.output_lengths(moved.input_lengths)
            logits = model.forward_exits(smoother(moved.features), moved.session_ids, layers=(depth,))[depth]
            hypotheses = decode_ctc(logits, lengths, len(phones))
            for i, (reference, hypothesis) in enumerate(zip(references, hypotheses, strict=True)):
                yield {"split": "train" if split == "pliegue" else split,
                       "session": cfg["datos"]["sesiones"][int(small.session_ids[i])],
                       "block": small.blocks[i], "source_trial_index": small.trial_indices[i],
                       "phones": " ".join(phones[phone] for phone in hypothesis),
                       "reference_phones": " ".join(phones[phone] for phone in reference),
                       "reference_text": small.references[i],
                       "errors": edit_distance(reference, hypothesis), "reference_length": len(reference)}
