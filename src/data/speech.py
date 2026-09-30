"""Lectura de competitionData T12; preserva señales, orden y splits oficiales.

No ajusta estadísticas, no normaliza, no aumenta datos y no depende de torch.
Los 20 ms y las 256 columnas son propiedades del formato de Dryad.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, whosmat

BIN_SECONDS = 0.02
RECORDED_CHANNELS = 256
AREA_6V_CHANNELS = 128
OFFICIAL_SPLITS = ("train", "test", "competitionHoldOut")
REQUIRED_FIELDS = ("sentenceText", "blockIdx", "spikePow", "tx1")


@dataclass(frozen=True)
class SpeechTrial:
    """Un intento de frase, con matrices crudas tiempo × canal y texto opcional."""

    index: int
    block: int
    text: str | None
    spike_power: NDArray[Any]
    threshold_crossings: NDArray[Any]

    @property
    def duration_seconds(self) -> float:
        """Duración de los bins publicados; no es tiempo alineado de cada palabra."""
        return self.spike_power.shape[0] * BIN_SECONDS

    def features_6v(self) -> NDArray[np.float32]:
        """Concatena 128 spikePow y 128 tx1 de 6v, sin normalización: T × 256."""
        return np.concatenate(
            (self.spike_power[:, :AREA_6V_CHANNELS],
             self.threshold_crossings[:, :AREA_6V_CHANNELS]), axis=1,
        ).astype(np.float32)


@dataclass(frozen=True)
class SpeechSession:
    """Un .mat completo: ensayos en el mismo orden que los arrays originales."""

    name: str
    trials: tuple[SpeechTrial, ...]


def load_session(path: Path) -> SpeechSession:
    """Lee spikePow/tx1/texto/bloque y rechaza desalineación o señales inválidas.

    Traduce el marcador publicado HELD OUT (o texto vacío) a None para que
    nunca pueda usarse como una etiqueta de entrenamiento o de evaluación.
    No carga tx2–tx4: se pueden añadir después sin modificar los originales.
    """
    data = loadmat(path, variable_names=list(REQUIRED_FIELDS), chars_as_strings=True)
    missing = set(REQUIRED_FIELDS) - data.keys()
    if missing:
        raise ValueError(f"{path.name}: faltan campos {sorted(missing)}")
    spike_cells = data["spikePow"]
    crossing_cells = data["tx1"]
    if spike_cells.dtype != object or crossing_cells.dtype != object:
        raise ValueError(f"{path.name}: las señales deben ser arrays de celdas MATLAB")
    spikes = spike_cells.ravel()
    crossings = crossing_cells.ravel()
    texts = data["sentenceText"].ravel()
    blocks = data["blockIdx"].ravel()
    if not len(spikes) or len({len(spikes), len(crossings), len(texts), len(blocks)}) != 1:
        raise ValueError(f"{path.name}: texto, bloques y señales no están alineados")
    trials = []
    for index, (spike, crossing, raw_text, block) in enumerate(
        zip(spikes, crossings, texts, blocks, strict=True)
    ):
        if spike.ndim != 2 or spike.shape[0] == 0 or spike.shape[1] != RECORDED_CHANNELS:
            raise ValueError(f"{path.name}, ensayo {index}: spikePow debe ser T × 256")
        if crossing.shape != spike.shape:
            raise ValueError(f"{path.name}, ensayo {index}: tx1 y spikePow no coinciden")
        if not np.isfinite(spike).all() or not np.isfinite(crossing).all():
            raise ValueError(f"{path.name}, ensayo {index}: señal con NaN o infinito")
        if (spike < 0).any() or (crossing < 0).any():
            raise ValueError(f"{path.name}, ensayo {index}: potencia o conteos negativos")
        if not np.isfinite(block) or float(block) != int(block):
            raise ValueError(f"{path.name}, ensayo {index}: bloque inválido")
        text = str(raw_text).strip()
        trials.append(SpeechTrial(index, int(block),
                                  None if not text or text.upper() == "HELD OUT" else text,
                                  spike, crossing))
    return SpeechSession(path.stem, tuple(trials))


def inspect_corpus(root: Path) -> list[dict[str, Any]]:
    """Cuenta archivos y ensayos en los tres splits leyendo solo metadatos .mat.

    Este inventario no inspecciona todos los valores neuronales ni asegura
    que todas las frases tengan etiquetas utilizables.
    """
    records = []
    for split in OFFICIAL_SPLITS:
        paths = sorted((root / split).glob("*.mat"))
        if not paths:
            raise ValueError(f"no hay archivos .mat en {root / split}")
        for path in paths:
            variables = whosmat(path)
            shapes = {name: shape for name, shape, _ in variables}
            missing = set(REQUIRED_FIELDS) - shapes.keys()
            if missing:
                raise ValueError(f"{path.name}: faltan campos {sorted(missing)}")
            count = int(np.prod(shapes["spikePow"]))
            if count != int(np.prod(shapes["tx1"])) or count != int(np.prod(shapes["blockIdx"])):
                raise ValueError(f"{path.name}: señales y bloques no están alineados")
            records.append({
                "split": split, "session": path.stem, "path": path.relative_to(root).as_posix(),
                "trials": count, "file_bytes": path.stat().st_size,
                "variables": [{"name": name, "shape": list(shape), "matlab_class": kind}
                              for name, shape, kind in variables],
            })
    return records
