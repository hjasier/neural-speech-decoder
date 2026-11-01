"""Orden, normalización de train y suavizado de señales T12 para la GRU.

La normalización se puede guardar como datos JSON. En entrenamiento, el
suavizado se aplicará después de añadir ruido y antes de la red de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.data.speech import AREA_6V_CHANNELS, SpeechTrial, load_session


def extract_model_features(trial: SpeechTrial) -> NDArray[np.float32]:
    """Devuelve T × 256 en el orden original: tx1 de 6v, después spikePow."""
    return np.concatenate(
        (trial.threshold_crossings[:, :AREA_6V_CHANNELS],
         trial.spike_power[:, :AREA_6V_CHANNELS]),
        axis=1,
    ).astype(np.float32)


@dataclass(frozen=True)
class TrainingNormalization:
    """Estadísticas de una sesión de train para reutilizar en sus otros splits."""

    session: str
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    frames: int
    epsilon: float

    def __post_init__(self) -> None:
        """Rechaza estadísticas corruptas antes de usarlas sobre una señal."""
        expected = (2 * AREA_6V_CHANNELS,)
        if not self.session or self.frames <= 0 or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("La sesión, el número de bins o epsilon no son válidos.")
        if self.mean.shape != expected or self.std.shape != expected:
            raise ValueError("Se necesitan una media y una desviación por característica.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std < 0).any():
            raise ValueError("Las estadísticas contienen valores inválidos.")

    def transform(self, trial: SpeechTrial, session: str) -> NDArray[np.float32]:
        """Normaliza un ensayo del mismo día sin ajustar ninguna estadística."""
        return self.transform_features(extract_model_features(trial), session)

    def transform_features(
        self, features: NDArray[np.float32], session: str,
    ) -> NDArray[np.float32]:
        """Normaliza un bloque T × 256 en orden tx1/spikePow con estadísticas fijas."""
        if session != self.session:
            raise ValueError(f"Las estadísticas de {self.session} no corresponden a {session}.")
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != len(self.mean):
            raise ValueError("Se necesita un bloque no vacío con una columna por característica.")
        normalized = ((features - self.mean) / (self.std + self.epsilon)).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("La señal normalizada contiene NaN o infinito.")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Exporta valores simples para JSON o para el futuro checkpoint."""
        return {
            "session": self.session,
            "fitted_split": "train",
            "frames": self.frames,
            "epsilon": self.epsilon,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> TrainingNormalization:
        """Restaura estadísticas y comprueba su procedencia declarada y tamaños."""
        if state["fitted_split"] != "train":
            raise ValueError("Solo se admiten estadísticas ajustadas con train.")
        return cls(
            session=state["session"],
            mean=np.asarray(state["mean"], dtype=np.float64),
            std=np.asarray(state["std"], dtype=np.float64),
            frames=state["frames"],
            epsilon=state["epsilon"],
        )


def fit_train_normalization(
    corpus_root: Path, session_name: str, epsilon: float,
    trial_indices: Sequence[int] | None = None,
) -> TrainingNormalization:
    """Lee solo train/<sesión>.mat y ajusta media y desviación por canal.

    Combina medias y varianzas por ensayo sin concatenar una segunda
    copia de la sesión. Si se indican índices, solo ajusta con esos ensayos,
    excluyendo la validación antes del ajuste. Usa varianza poblacional.
    """
    if Path(session_name).name != session_name or not session_name:
        raise ValueError("Se necesita un nombre de sesión, sin directorios.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon debe ser positivo y finito.")
    session = load_session(corpus_root / "train" / f"{session_name}.mat")
    selected = list(range(len(session.trials))) if trial_indices is None else list(trial_indices)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < len(session.trials) for i in selected)):
        raise ValueError("Los índices para ajustar normalización deben ser únicos y válidos.")
    count = 0
    mean = np.zeros(2 * AREA_6V_CHANNELS, dtype=np.float64)
    squared_deviations = np.zeros_like(mean)
    for index in selected:
        trial = session.trials[index]
        features = extract_model_features(trial).astype(np.float64)
        trial_count = len(features)
        trial_mean = features.mean(axis=0)
        trial_variance = features.var(axis=0)
        total = count + trial_count
        delta = trial_mean - mean
        squared_deviations += (
            trial_variance * trial_count + delta ** 2 * count * trial_count / total
        )
        mean += delta * trial_count / total
        count = total
    return TrainingNormalization(
        session=session.name,
        mean=mean,
        std=np.sqrt(np.maximum(squared_deviations / count, 0)),
        frames=count,
        epsilon=epsilon,
    )


class GaussianSmoothing(nn.Module):
    """Suavizado gaussiano centrado o causal con padding cero.

    No mezcla canales ni ejemplos. Las señales se normalizan antes de
    formar el lote; el padding del lote debe contener ceros normalizados.
    El modo causal desplaza el centro del filtro delay_bins hacia el pasado.
    Conserva T salidas para T entradas, sin prolongar la señal al finalizar.
    """

    def __init__(
        self, *, num_features: int, sigma: float, cutoff: float,
        impulse_size: int, truncate: float, mode: str = "centered", delay_bins: int = 0,
    ) -> None:
        """Construye una vez el kernel con los parámetros del YAML."""
        super().__init__()
        if num_features <= 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("El número de características y sigma deben ser positivos.")
        if not np.isfinite(cutoff) or not 0 <= cutoff < 1:
            raise ValueError("El umbral del kernel debe estar entre 0 incluido y 1 excluido.")
        if mode not in ("centered", "causal") or type(delay_bins) is not int or delay_bins < 0:
            raise ValueError("El modo debe ser centered/causal y el retardo un entero no negativo.")
        if mode == "centered" and delay_bins != 0:
            raise ValueError("El suavizado centrado no admite retardo.")
        if not np.isfinite(truncate) or truncate <= 0:
            raise ValueError("La extensión del kernel debe ser positiva y finita.")
        radius = int(truncate * sigma + 0.5)
        if impulse_size < 2 * radius + 1:
            raise ValueError("El impulso es demasiado corto para alojar el kernel.")
        impulse = np.zeros(impulse_size, dtype=np.float32)
        impulse[impulse_size // 2] = 1
        kernel = gaussian_filter1d(impulse, sigma=sigma, truncate=truncate)
        kernel = kernel[kernel > cutoff]
        if not len(kernel) or len(kernel) % 2 == 0:
            raise ValueError("El recorte debe conservar un kernel no vacío e impar.")
        if mode == "causal" and delay_bins < len(kernel) // 2:
            raise ValueError("El retardo causal debe cubrir la mitad derecha del kernel.")
        kernel /= kernel.sum()
        self.num_features = num_features
        self.mode = mode
        self.delay_bins = delay_bins
        self.history_size = delay_bins + len(kernel) // 2
        weights = torch.from_numpy(kernel).reshape(1, 1, -1).repeat(num_features, 1, 1)
        self.register_buffer("kernel", weights)

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.num_features:
            raise ValueError("La entrada debe tener forma batch × tiempo × características.")
        if not features.is_floating_point() or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("Se necesita un lote no vacío de señales en coma flotante.")

    def _convolve(self, features: Tensor, padding: int = 0) -> Tensor:
        return F.conv1d(
            features.transpose(1, 2),
            self.kernel.to(dtype=features.dtype),
            padding=padding,
            groups=self.num_features,
        ).transpose(1, 2)

    def forward(self, features: Tensor) -> Tensor:
        """Suaviza B × T × F; el modo causal usa solo el pasado y conserva T."""
        self._validate_features(features)
        if self.mode == "centered":
            return self._convolve(features, padding=self.kernel.shape[-1] // 2)
        padded = F.pad(features, (0, 0, self.history_size, 0))
        return self._convolve(padded)[:, :features.shape[1]]

    def forward_chunk(
        self, features: Tensor, history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Suaviza un bloque causal y devuelve su historial B × history_size × F.

        None inicia una frase con ceros normalizados antes de su primer bin.
        El historial se conserva fuera del filtro y mantiene el orden del lote.
        """
        self._validate_features(features)
        if self.mode != "causal":
            raise ValueError("El procesamiento por bloques requiere suavizado causal.")
        expected = (features.shape[0], self.history_size, self.num_features)
        if history is None:
            history = features.new_zeros(expected)
        elif (history.shape != expected or history.device != features.device
              or history.dtype != features.dtype):
            raise ValueError("El historial no corresponde al lote, dispositivo o dtype del bloque.")
        context = torch.cat((history, features), dim=1)
        output = self._convolve(context)[:, :features.shape[1]]
        # Copiar solo el contexto evita retener el almacenamiento del paquete completo.
        following = context[:, -self.history_size:].clone() if self.history_size else context[:, :0].clone()
        return output, following


def build_smoother(cfg: dict[str, Any]) -> GaussianSmoothing:
    """Construye el suavizado con la configuración del experimento."""
    smoothing = cfg["preprocesamiento"]["suavizado"]
    return GaussianSmoothing(
        num_features=cfg["modelo"]["entrada"],
        sigma=smoothing["sigma"],
        cutoff=smoothing["umbral"],
        impulse_size=smoothing["longitud_impulso"],
        truncate=smoothing["truncate"],
        mode=smoothing.get("modo", "centered"),
        delay_bins=smoothing.get("retardo_bins", 0),
    )
