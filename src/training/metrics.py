"""Decodificación CTC y distancia de edición para el PER de habla."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

from src.inference.ctc import decode_ctc


def edit_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    """Cuenta inserciones, borrados y sustituciones con memoria lineal."""
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, predicted in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (expected != predicted),
            ))
        previous = current
    return previous[-1]


def phoneme_errors(
    hypotheses: Sequence[Sequence[int]], targets: Tensor, target_lengths: Tensor,
) -> tuple[int, int]:
    """Devuelve errores y fonemas de referencia para agregar PER por corpus."""
    references = targets.detach().cpu().split(target_lengths.tolist())
    errors = sum(
        edit_distance(reference.tolist(), hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    )
    return errors, targets.numel()
