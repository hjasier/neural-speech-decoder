"""Convierte bloques de una frase T12 en ventanas para la GRU."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from src.data.preprocessing import GaussianSmoothing, TrainingNormalization
from src.models.speech_gru import SpeechGRU


class OnlinePreprocessor:
    """Normaliza, suaviza y forma ventanas con memoria propia por frase.

    Recibe matrices T × 256 sin normalizar, con tx1 antes de spikePow.
    El adaptador se obtiene del nombre de la normalización y del orden de
    sesiones guardado en el checkpoint; ambos quedan fijados al construirlo.
    Las ventanas devueltas se pueden pasar a SpeechGRU.forward_windows.
    """

    def __init__(
        self, model: SpeechGRU, smoother: GaussianSmoothing,
        normalization: TrainingNormalization, sessions: Sequence[str],
    ) -> None:
        """Vincula el flujo a una sesión y a un modelo en modo de evaluación."""
        if model.training:
            raise ValueError("El preprocesamiento incremental requiere model.eval().")
        if smoother.mode != "causal":
            raise ValueError("El flujo requiere un filtro causal con retardo explícito.")
        if (model.input_features != smoother.num_features
                or model.input_features != len(normalization.mean)):
            raise ValueError("Modelo, filtro y normalización deben usar las mismas características.")
        if len(sessions) != model.num_sessions or len(set(sessions)) != len(sessions):
            raise ValueError("Se necesita el orden de sesiones del checkpoint, sin duplicados.")
        if normalization.session not in sessions:
            raise ValueError("La sesión de normalización no pertenece al modelo.")
        session_id = sessions.index(normalization.session)
        self.model = model
        self.normalization = normalization
        self.smoother = smoother.to(model.initial_state.device)
        self._session_ids = torch.tensor([session_id], device=model.initial_state.device)
        self.reset()

    def reset(self) -> None:
        """Inicia otra frase de la misma sesión, sin historial ni ventanas pendientes."""
        self._history: Tensor | None = None
        self._buffer: Tensor | None = None
        self._skip_bins = 0
        self._closed = False

    def _empty_windows(self) -> Tensor:
        return self.model.initial_state.new_empty((1, 0, self.model.gru_layers[0].input_size))

    def _windows(self, features: Tensor) -> Tensor:
        skipped = min(self._skip_bins, features.shape[1])
        self._skip_bins -= skipped
        features = features[:, skipped:]
        if self._buffer is not None:
            features = torch.cat((self._buffer, features), dim=1)
        if features.shape[1] < self.model.kernel_size:
            self._buffer = features.clone()
            return self._empty_windows()
        windows = self.model.prepare_windows(features, self._session_ids)
        advance = windows.shape[1] * self.model.stride
        # Mantener también el salto pendiente si el stride supera a la ventana.
        self._skip_bins = max(advance - features.shape[1], 0)
        self._buffer = features[:, advance:].clone()
        return windows

    @torch.no_grad()
    def push(self, features: NDArray[np.float32]) -> Tensor:
        """Recibe un bloque crudo y devuelve cero o más ventanas, sin ejecutar GRU."""
        if self._closed:
            raise ValueError("La frase está cerrada; inicia otra con reset().")
        if self.model.training:
            raise ValueError("El preprocesamiento incremental requiere model.eval().")
        normalized = self.normalization.transform_features(features, self.normalization.session)
        values = torch.from_numpy(normalized).unsqueeze(0).to(
            device=self.model.initial_state.device, dtype=self.model.initial_state.dtype,
        )
        smoothed, self._history = self.smoother.forward_chunk(values, self._history)
        return self._windows(smoothed)

    def finish(self) -> Tensor:
        """Cierra sin añadir bins ni ventanas incompletas; repetirlo no emite nada.

        Cada push ya entrega todas las ventanas completas disponibles. La cola
        que no alcanza otra ventana se descarta como en el forward completo.
        """
        self.reset()
        self._closed = True
        return self._empty_windows()
