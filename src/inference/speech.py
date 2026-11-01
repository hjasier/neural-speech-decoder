"""Motor de fonemas por bloques con memoria independiente para cada flujo."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from src.inference.checkpoints import SpeechCheckpoint, load_inference_checkpoint
from src.inference.ctc import IncrementalCTC
from src.inference.preprocessing import OnlinePreprocessor


class OnlineSpeechDecoder:
    """Comparte pesos en evaluación; cada frase conserva su propio estado."""

    def __init__(self, checkpoint: SpeechCheckpoint) -> None:
        """Acepta exclusivamente el preprocesamiento causal guardado al entrenar."""
        if checkpoint.smoother.mode != "causal":
            raise ValueError("El motor online requiere un checkpoint entrenado con suavizado causal.")
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: str | torch.device = "cpu",
    ) -> OnlineSpeechDecoder:
        """Carga pesos y preprocesamiento autocontenidos, por defecto en CPU."""
        return cls(load_inference_checkpoint(path, device))

    def start(self, session: str, *, layers: Sequence[int] | None = None) -> SpeechStream:
        """Abre una frase de esa sesión con salidas fijadas hasta finish o reset."""
        return SpeechStream(self.checkpoint, session, layers=layers)


class SpeechStream:
    """Estado de una frase: filtro, ventanas, GRU y CTC independientes por salida."""

    def __init__(
        self, checkpoint: SpeechCheckpoint, session: str, *, layers: Sequence[int] | None = None,
    ) -> None:
        """Vincula la frase a la normalización y al adaptador de su sesión."""
        selected = checkpoint.model.available_exits if layers is None else tuple(layers)
        if (not selected or len(set(selected)) != len(selected)
                or any(type(depth) is not int or depth not in checkpoint.model.available_exits
                       for depth in selected)):
            raise ValueError("Las salidas deben ser únicas y existir en el checkpoint.")
        self._layers = tuple(selected)
        self._checkpoint = checkpoint
        self._preprocessor = OnlinePreprocessor(
            checkpoint.model, checkpoint.smoother, checkpoint.normalization[session], checkpoint.sessions,
        )
        self._ctc = {depth: IncrementalCTC(checkpoint.blank_id) for depth in self.layers}
        self.reset()

    @property
    def layers(self) -> tuple[int, ...]:
        """Devuelve las salidas elegidas para este flujo."""
        return self._layers

    @property
    def hypotheses(self) -> dict[int, list[int]]:
        """Devuelve una copia de los IDs acumulados por salida."""
        return {depth: list(phones) for depth, phones in self._hypotheses.items()}

    @property
    def phonemes(self) -> dict[int, list[str]]:
        """Traduce los IDs acumulados usando el vocabulario del checkpoint."""
        return {
            depth: [self._checkpoint.phones[phone] for phone in phones]
            for depth, phones in self._hypotheses.items()
        }

    def reset(self) -> None:
        """Inicia otra frase de la misma sesión y elimina todo el estado anterior."""
        self._preprocessor.reset()
        self._states: tuple[Tensor, ...] | None = None
        self._hypotheses: dict[int, list[int]] = {depth: [] for depth in self.layers}
        for decoder in self._ctc.values():
            decoder.reset()
        self._closed = False

    def _consume(self, windows: Tensor) -> dict[int, list[int]]:
        if windows.shape[1] == 0:
            return {depth: [] for depth in self.layers}
        logits, self._states = self._checkpoint.model.forward_windows(
            windows, self._states, layers=self.layers,
        )
        updates = {depth: self._ctc[depth].push(logits[depth][0]) for depth in self.layers}
        for depth, phones in updates.items():
            self._hypotheses[depth].extend(phones)
        return updates

    @torch.inference_mode()
    def push(self, features: NDArray[np.float32]) -> dict[int, list[int]]:
        """Recibe T × 256 crudo en orden tx1/spikePow y emite solo IDs nuevos."""
        if self._closed:
            raise ValueError("La frase está cerrada; inicia otra con reset().")
        return self._consume(self._preprocessor.push(features))

    @torch.inference_mode()
    def finish(self) -> dict[int, list[int]]:
        """Cierra sin inventar bins; conserva la hipótesis y no duplica emisiones."""
        if self._closed:
            return {depth: [] for depth in self.layers}
        updates = self._consume(self._preprocessor.finish())
        self._states = None
        self._closed = True
        return updates
