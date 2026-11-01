"""Decodificación CTC voraz, completa o incremental, sin dependencias de entrenamiento."""

from __future__ import annotations

from torch import Tensor


def _collapse_path(
    path: Tensor, blank_id: int, previous: int | None = None,
) -> tuple[list[int], int | None]:
    runs = path.unique_consecutive().tolist()
    if not runs:
        return [], previous
    last = runs[-1]
    if runs[0] == previous:
        runs = runs[1:]
    return [phone for phone in runs if phone != blank_id], last


def decode_ctc(logits: Tensor, lengths: Tensor, blank_id: int) -> list[list[int]]:
    """Colapsa repeticiones antes de quitar blank; excluye el padding temporal."""
    paths = logits.detach().argmax(dim=-1).cpu()
    decoded: list[list[int]] = []
    for path, length in zip(paths, lengths.tolist(), strict=True):
        if not 0 < length <= path.numel():
            raise ValueError("Longitud de salida CTC no válida.")
        phones, _ = _collapse_path(path[:length], blank_id)
        decoded.append(phones)
    return decoded


class IncrementalCTC:
    """Conserva la última clase cruda, incluido blank, entre bloques de una frase."""

    def __init__(self, blank_id: int) -> None:
        """Fija el índice blank del vocabulario del checkpoint."""
        if type(blank_id) is not int or blank_id < 0:
            raise ValueError("blank_id debe ser un entero no negativo.")
        self.blank_id = blank_id
        self.reset()

    def reset(self) -> None:
        """Inicia otra frase sin arrastrar la última clase anterior."""
        self._previous: int | None = None

    def push(self, logits: Tensor) -> list[int]:
        """Consume tiempo × clases y devuelve únicamente los fonemas nuevos."""
        if logits.ndim != 2 or not self.blank_id < logits.shape[-1]:
            raise ValueError("Se necesitan logits tiempo × clases con blank incluido.")
        path = logits.detach().argmax(dim=-1).cpu()
        phones, self._previous = _collapse_path(path, self.blank_id, self._previous)
        return phones
