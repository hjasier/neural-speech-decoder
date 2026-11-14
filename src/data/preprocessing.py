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


NORMALIZATION_MODES = ("train_fija", "bloque", "rodante")
Statistics = tuple[NDArray[np.float64], NDArray[np.float64]]


def normalization_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Devuelve el modo de normalización; sin la clave, la fija ajustada con train.

    train_fija: una media y desviación por sesión, ajustadas solo con train.
    bloque: estadísticas del bloque de grabación de cada frase (suplemento 5.1).
    rodante: train por bloque; val y test con la estimación causal de la
    ecuación 2 del suplemento 5.2, que solo usa frases anteriores. Su
    desviación tiene un suelo relativo a la de train de la sesión: con pocas
    frases previas un canal sin actividad daría una desviación nula.
    Los modos bloque y rodante leen señal de evaluación, nunca sus etiquetas.
    """
    spec = cfg["preprocesamiento"].get("normalizacion")
    if spec is None:
        return {"modo": "train_fija"}
    error = ValueError(
        "preprocesamiento.normalizacion requiere modo train_fija, bloque o rodante; "
        "rodante necesita además frases_mezcla y ventana_frases enteros positivos "
        "y desviacion_minima_relativa entre cero y uno."
    )
    if not isinstance(spec, dict) or spec.get("modo") not in NORMALIZATION_MODES:
        raise error
    counts = ("frases_mezcla", "ventana_frases") if spec["modo"] == "rodante" else ()
    floor = ("desviacion_minima_relativa",) if spec["modo"] == "rodante" else ()
    if (set(spec) != {"modo", *counts, *floor}
            or any(type(spec[key]) is not int or spec[key] <= 0 for key in counts)
            or any(type(spec[key]) is not float or not 0 < spec[key] <= 1 for key in floor)):
        raise error
    return dict(spec)


@dataclass(frozen=True)
class TrialMoments:
    """Sumas por canal de una frase, suficientes para combinar medias y desviaciones."""

    block: int
    frames: int
    total: NDArray[np.float64]
    squares: NDArray[np.float64]


def trial_moments(trial: SpeechTrial) -> TrialMoments:
    """Acumula en float64 las características tx1/spikePow de 6v de un ensayo."""
    features = extract_model_features(trial).astype(np.float64)
    return TrialMoments(trial.block, len(features), features.sum(axis=0), np.square(features).sum(axis=0))


def pooled_statistics(moments: Sequence[TrialMoments]) -> Statistics:
    """Media y desviación poblacional sobre todos los bins de las frases dadas."""
    frames = sum(item.frames for item in moments)
    if not moments or frames <= 0:
        raise ValueError("Se necesita al menos una frase con bins para estimar estadísticas.")
    mean = sum(item.total for item in moments) / frames
    variance = sum(item.squares for item in moments) / frames - np.square(mean)
    return mean, np.sqrt(np.maximum(variance, 0))


def rolling_statistics(
    block: Sequence[TrialMoments], previous: Statistics, *, blend: int, window: int,
) -> list[Statistics]:
    """Estadísticas causales por frase de un bloque (suplemento 5.2, ecuación 2).

    La frase i, contada desde uno, nunca usa su propia señal ni la posterior.
    En las primeras `blend` frases pondera el bloque anterior con peso
    (blend + 1 - i) / blend y las frases previas del bloque actual con
    (i - 1) / blend. Después usa solo las últimas `window` frases previas.
    Las desviaciones se combinan igual que las medias.
    """
    if blend <= 0 or window <= 0:
        raise ValueError("frases_mezcla y ventana_frases deben ser positivos.")
    statistics = []
    for position in range(1, len(block) + 1):
        seen = block[:position - 1]
        if position > blend:
            statistics.append(pooled_statistics(seen[-window:]))
            continue
        current_weight = (position - 1) / blend
        if not seen:
            statistics.append((previous[0].copy(), previous[1].copy()))
            continue
        current = pooled_statistics(seen)
        statistics.append(tuple(
            (1 - current_weight) * before + current_weight * now
            for before, now in zip(previous, current, strict=True)
        ))
    return statistics


def session_statistics(
    splits: dict[str, Sequence[TrialMoments]], spec: dict[str, Any],
    *, train_indices: Sequence[int] | None,
) -> dict[str, list[Statistics]]:
    """Asigna media y desviación a cada frase de los archivos train/test de un día.

    `train_indices` son las frases de train del protocolo dentro del archivo
    train; None significa todo el archivo. En modo rodante esas frases usan
    su bloque, como al entrenar en el paper, y el resto (val y test) la
    estimación causal, cuyo bloque anterior se busca por número de bloque en
    los dos archivos del día. Su desviación no baja de
    desviacion_minima_relativa × la desviación de las frases de train del día;
    un canal que tampoco varía en train se anula en vez de dividir por epsilon.
    """
    if spec["modo"] not in ("bloque", "rodante") or set(splits) != {"train", "test"}:
        raise ValueError("Las estadísticas por frase requieren modo bloque o rodante y los splits train/test.")
    groups: dict[tuple[str, int], list[int]] = {}
    for split, moments in splits.items():
        for index, item in enumerate(moments):
            groups.setdefault((split, item.block), []).append(index)
    block_statistics = {
        key: pooled_statistics([splits[key[0]][index] for index in indices])
        for key, indices in groups.items()
    }
    result = {split: [block_statistics[(split, item.block)] for item in moments]
              for split, moments in splits.items()}
    if spec["modo"] == "bloque":
        return result
    fitted = set(range(len(splits["train"]))) if train_indices is None else set(train_indices)
    if not fitted:
        raise ValueError("La normalización rodante necesita frases de train para fijar el suelo de la desviación.")
    floor = spec["desviacion_minima_relativa"] * pooled_statistics(
        [splits["train"][index] for index in sorted(fitted)]
    )[1]
    for (split, block), indices in groups.items():
        causal = [index for index in indices if split != "train" or index not in fitted]
        if not causal:
            continue
        if causal != indices:
            raise ValueError("Un bloque mezcla frases de train y de evaluación; no admite normalización rodante.")
        earlier = [key for key in groups if key[1] < block]
        if not earlier:
            raise ValueError(f"El bloque {block} de {split} no tiene un bloque anterior para la normalización rodante.")
        previous = block_statistics[max(earlier, key=lambda key: key[1])]
        rolled = rolling_statistics(
            [splits[split][index] for index in indices], previous,
            blend=spec["frases_mezcla"], window=spec["ventana_frases"],
        )
        for index, (mean, std) in zip(indices, rolled, strict=True):
            scale = np.maximum(std, floor)
            scale[scale == 0] = np.inf
            result[split][index] = (mean, scale)
    return result


@dataclass(frozen=True)
class TrialNormalization:
    """Media y desviación de una frase concreta en los modos bloque y rodante."""

    session: str
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    epsilon: float

    def transform_features(
        self, features: NDArray[np.float32], session: str,
    ) -> NDArray[np.float32]:
        """Normaliza T × 256 en orden tx1/spikePow con las estadísticas de la frase."""
        if session != self.session:
            raise ValueError(f"Las estadísticas de {self.session} no corresponden a {session}.")
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != len(self.mean):
            raise ValueError("Se necesita un bloque no vacío con una columna por característica.")
        normalized = ((features - self.mean) / (self.std + self.epsilon)).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("La señal normalizada contiene NaN o infinito.")
        return normalized


def session_trial_normalizations(
    cfg: dict[str, Any], session_name: str, *, train_indices: Sequence[int] | None,
) -> dict[str, list[TrialNormalization]]:
    """Lee train/test crudos de un día y devuelve la normalización de cada frase.

    Es la única fuente de estas estadísticas: la preparación, la evaluación
    por bloques y el visor la llaman con la misma configuración y partición.
    """
    spec = normalization_spec(cfg)
    root = Path(cfg["datos"]["raiz"])
    moments = {
        split: [trial_moments(trial) for trial in load_session(root / split / f"{session_name}.mat").trials]
        for split in ("train", "test")
    }
    epsilon = cfg["preprocesamiento"]["normalizacion_epsilon"]
    return {
        split: [TrialNormalization(session_name, mean, std, epsilon) for mean, std in statistics]
        for split, statistics in session_statistics(moments, spec, train_indices=train_indices).items()
    }


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
