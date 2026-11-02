"""Muestras crudas de validación para el emisor HTTP; sin decodificar ni entrenar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from src.data.preprocessing import extract_model_features
from src.data.speech import BIN_SECONDS, load_session
from src.data.speech_dataset import SpeechDataset, file_sha256, preparation_signature
from src.inference.checkpoints import SpeechCheckpoint
from src.utils.seed import crear_generador_numpy


@dataclass(frozen=True)
class ReplaySample:
    """Señal del emisor y referencia de pantalla, separadas de la inferencia."""

    id: str
    session: str
    trial_index: int
    raw: NDArray[np.float32]
    reference: str
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    @property
    def sha256(self) -> str:
        """Identifica todos los bins crudos, en su orden de transmisión."""
        return hashlib.sha256(self.raw.astype('<f4', copy=False).tobytes()).hexdigest()

    def payload(self) -> dict[str, Any]:
        """Entrega al navegador los bins originales y las estadísticas de la vista."""
        return {**self.metadata(), "features": self.raw.tolist()}

    def metadata(self) -> dict[str, Any]:
        """Describe la vista sin entregar señal futura a los observadores."""
        return {"id": self.id, "session": self.session, "trial_index": self.trial_index,
                "split": "val", "bins": len(self.raw), "bin_ms": BIN_SECONDS * 1000,
                "sha256": self.sha256, "reference": self.reference,
                "view_mean": self.mean.tolist(), "view_scale": self.scale.tolist()}


class DemoSamples:
    """Selecciona uniformemente entre val y comprueba la señal contra la caché."""

    def __init__(self, checkpoint: SpeechCheckpoint, seed: int) -> None:
        """Reutiliza la partición y las estadísticas guardadas al entrenar."""
        self.checkpoint = checkpoint
        self.dataset = SpeechDataset(checkpoint.config, "val")
        if (self.dataset.sessions != checkpoint.sessions
                or preparation_signature(checkpoint.config) != checkpoint.data_signature):
            raise ValueError("Las muestras no corresponden al checkpoint online.")
        manifest = json.loads((self.dataset.root / "manifest.json").read_text())
        self.sources = {entry["session"]: entry for entry in manifest["files"] if entry["split"] == "train"}
        self._verified: dict[Path, tuple[int, int]] = {}
        self._rng = crear_generador_numpy(seed)

    def load(self) -> ReplaySample:
        """Carga una frase aleatoria de val sin normalizar la señal enviada por HTTP."""
        index = int(self._rng.integers(len(self.dataset)))
        sample = self.dataset[index]
        name = self.dataset.sessions[sample.session_id]
        path = Path(self.checkpoint.config["datos"]["raiz"]) / "train" / f"{name}.mat"
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if self._verified.get(path) != identity:
            if file_sha256(path) != self.sources[name]["source_sha256"]:
                raise ValueError(f"La grabación original ha cambiado: {name}")
            self._verified[path] = identity
        recording = load_session(path)
        trial = recording.trials[sample.trial_index]
        raw = extract_model_features(trial)
        normalization = self.checkpoint.normalization[name]
        if (trial.index != sample.trial_index or trial.block != sample.block
                or not np.array_equal(normalization.transform_features(raw, name), sample.features.numpy())):
            raise ValueError("La muestra cruda no reproduce la señal preparada del checkpoint.")
        return ReplaySample(uuid4().hex, name, trial.index, raw, sample.reference,
                            normalization.mean, normalization.std + normalization.epsilon)
