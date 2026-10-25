"""Dataset T12 en disco, lotes variables y muestreo por sesión para speechBCI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from src.data.speech_protocol import read_speech_protocol


def file_sha256(path: Path) -> str:
    """Calcula un SHA-256 sin cargar todo el archivo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lee el manifiesto de etiquetas y exige la configuración vigente."""
    path = Path(cfg["etiquetas"]["salida"]) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Faltan etiquetas: ejecuta scripts/prepare_phoneme_labels.py.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest["format_version"] != 1
        or manifest["sessions"] != cfg["datos"]["sesiones"]
        or manifest["phones"] != cfg["datos"]["fonemas"]
        or manifest["blank_id"] != len(cfg["datos"]["fonemas"])
        or manifest["label_config"] != cfg["etiquetas"]
    ):
        raise ValueError("Las etiquetas no corresponden a esta configuración; vuelve a prepararlas.")
    return manifest


def preparation_signature(cfg: dict[str, Any]) -> str:
    """Identifica datos, orden de sesiones, etiquetas y normalización del caché."""
    state = {
        "format_version": 1,
        "source": cfg["datos"]["raiz"],
        "sessions": cfg["datos"]["sesiones"],
        "feature_order": ["tx1_6v", "spikePow_6v"],
        "epsilon": cfg["preprocesamiento"]["normalizacion_epsilon"],
        "labels_sha256": file_sha256(Path(cfg["etiquetas"]["salida"]) / "manifest.json"),
    }
    if read_speech_protocol(cfg) is not None:
        state["protocol_sha256"] = cfg["protocolo"]["sha256"]
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpeechSample:
    """Un ensayo normalizado, sin suavizado ni ampliación de datos."""

    features: Tensor
    targets: Tensor
    session_id: int
    reference: str
    trial_index: int
    block: int


@dataclass(frozen=True)
class SpeechBatch:
    """Señales con padding cero y objetivos CTC concatenados, sin padding."""

    features: Tensor
    targets: Tensor
    input_lengths: Tensor
    target_lengths: Tensor
    session_ids: Tensor
    references: tuple[str, ...]
    trial_indices: tuple[int, ...]
    blocks: tuple[int, ...]

    def to(self, device: str | torch.device, non_blocking: bool = False) -> SpeechBatch:
        """Mueve señales, objetivos e IDs; conserva en CPU las longitudes de CTC."""
        return replace(
            self,
            features=self.features.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            session_ids=self.session_ids.to(device, non_blocking=non_blocking),
        )

    def microbatches(self, size: int, *, trim_padding: bool = False) -> Iterator[SpeechBatch]:
        """Divide frases y objetivos CTC alineados, conservando todos los ejemplos.

        Train conserva la extensión temporal del lote original, incluido el
        padding donde se añade ruido. Evaluación puede recortar ceros finales.
        """
        if size <= 0:
            raise ValueError("El tamaño del microlote debe ser positivo.")
        target_start = 0
        for start in range(0, len(self.references), size):
            stop = min(start + size, len(self.references))
            lengths = self.input_lengths[start:stop]
            target_lengths = self.target_lengths[start:stop]
            target_stop = target_start + int(target_lengths.sum().item())
            time_steps = int(lengths.max().item()) if trim_padding else self.features.shape[1]
            yield SpeechBatch(
                features=self.features[start:stop, :time_steps],
                targets=self.targets[target_start:target_stop],
                input_lengths=lengths, target_lengths=target_lengths,
                session_ids=self.session_ids[start:stop],
                references=self.references[start:stop],
                trial_indices=self.trial_indices[start:stop], blocks=self.blocks[start:stop],
            )
            target_start = target_stop

    def pin_memory(self) -> SpeechBatch:
        """Permite que DataLoader reserve memoria fijada para transferencias GPU."""
        return replace(
            self,
            features=self.features.pin_memory(),
            targets=self.targets.pin_memory(),
            input_lengths=self.input_lengths.pin_memory(),
            target_lengths=self.target_lengths.pin_memory(),
            session_ids=self.session_ids.pin_memory(),
        )


class SpeechDataset(Dataset[SpeechSample]):
    """Une señales preparadas y etiquetas verificadas sin cargar el corpus en RAM."""

    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        """Abre metadatos y comprueba integridad, vocabulario y longitudes CTC."""
        protocol = read_speech_protocol(cfg)
        allowed = ("train", "val", "test") if protocol else ("train", "test")
        if split not in allowed:
            raise ValueError(f"La configuración admite las particiones {allowed}.")
        self.split = split
        source_split = "train" if split == "val" else split
        partitions = {entry["session"]: entry for entry in protocol["sessions"]} if protocol else {}
        labels = read_label_manifest(cfg)
        self.root = Path(cfg["datos"]["preparados"])
        path = self.root / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("Faltan señales preparadas: ejecuta scripts/prepare_speech_data.py.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["format_version"] != (2 if protocol else 1) or manifest["signature"] != preparation_signature(cfg):
            raise ValueError("Las señales preparadas no corresponden a esta configuración.")
        if protocol and (manifest["protocol_sha256"] != cfg["protocolo"]["sha256"]
                         or manifest["totals"] != protocol["totals"]):
            raise ValueError("El manifiesto preparado no corresponde a la partición T12.")
        self.normalization_states = manifest["normalizations"]
        self.sessions = tuple(cfg["datos"]["sesiones"])
        self._rows: list[dict[str, Any]] = []
        self._arrays: dict[str, np.memmap] = {}
        self.session_indices: list[range] = []
        label_files = {entry["path"]: entry for entry in labels["files"]}
        feature_files = {(entry["split"], entry["session"]): entry for entry in manifest["files"]}
        blank_id = len(cfg["datos"]["fonemas"])
        for session_id, session in enumerate(self.sessions):
            start = len(self._rows)
            entry = feature_files[(source_split, session)]
            label_name = f"{source_split}/{session}.json"
            label_path = Path(cfg["etiquetas"]["salida"]) / label_name
            if file_sha256(label_path) != label_files[label_name]["sha256"]:
                raise ValueError(f"Las etiquetas han cambiado: {label_name}")
            targets = json.loads(label_path.read_text(encoding="utf-8"))
            if (
                targets["session"] != session or targets["split"] != source_split
                or targets["transcriptions_sha256"] != entry["transcriptions_sha256"]
            ):
                raise ValueError(f"Las señales y etiquetas no corresponden: {label_name}")
            feature_path = self.root / entry["path"]
            if file_sha256(feature_path) != entry["sha256"]:
                raise ValueError(f"Las señales preparadas han cambiado: {feature_path}")
            array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            offsets = entry["offsets"]
            records = targets["targets"]
            selected = None
            if protocol:
                part = partitions[session]
                norm = self.normalization_states[session]
                if (entry["source_sha256"] != part["sources"][source_split]["sha256"]
                        or len(records) != part["sources"][source_split]["trials"]
                        or norm["fit_trial_indices"] != part["train"]
                        or norm["fit_protocol_sha256"] != cfg["protocolo"]["sha256"]):
                    raise ValueError("Las señales o la normalización no corresponden al protocolo T12.")
                if source_split == "train":
                    if norm["frames"] != sum(offsets[i + 1] - offsets[i] for i in part["train"]):
                        raise ValueError("La normalización no cuenta exclusivamente los bins de train.")
                    for logical in ("train", "val"):
                        actual = [i for i, block in enumerate(entry["blocks"]) if block in part[f"{logical}_blocks"]]
                        if actual != part[logical]:
                            raise ValueError("Los índices no corresponden a los bloques reservados.")
                selected = set(part[split])
            if (
                array.ndim != 2 or array.shape[1] != cfg["modelo"]["entrada"]
                or array.dtype != np.float32 or len(offsets) != len(records) + 1
                or len(entry["blocks"]) != len(records)
                or offsets[0] != 0 or offsets[-1] != len(array)
            ):
                raise ValueError(f"Metadatos de señales no válidos: {feature_path}")
            for index, target in enumerate(records):
                if selected is not None and index not in selected:
                    continue
                length = offsets[index + 1] - offsets[index]
                ids = tuple(target["ids"])
                steps = (length - cfg["modelo"]["kernel"]) // cfg["modelo"]["stride"] + 1
                minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
                if (
                    target["trial_index"] != index or length <= 0 or not ids
                    or any(value < 0 or value >= blank_id for value in ids)
                    or minimum != target["minimum_ctc_steps"] or steps < minimum
                ):
                    raise ValueError(f"Ensayo no compatible con CTC: {split}/{session}, índice {index}")
                self._rows.append({
                    "path": entry["path"], "start": offsets[index], "stop": offsets[index + 1],
                    "targets": ids, "session_id": session_id, "reference": target["reference"],
                    "trial_index": index, "block": entry["blocks"][index],
                })
            self.session_indices.append(range(start, len(self._rows)))
            del array
        expected = protocol["totals"][split] if protocol else labels["totals"][split]
        if len(self) != expected:
            raise ValueError("El número de ensayos no coincide con el manifiesto de etiquetas.")

    def __len__(self) -> int:
        """Devuelve el número de ensayos de la partición."""
        return len(self._rows)

    def __getitem__(self, index: int) -> SpeechSample:
        """Lee solo un ensayo y devuelve una copia modificable de su señal."""
        row = self._rows[index]
        if row["path"] not in self._arrays:
            self._arrays[row["path"]] = np.load(
                self.root / row["path"], mmap_mode="r", allow_pickle=False,
            )
        features = self._arrays[row["path"]][row["start"]:row["stop"]].copy()
        return SpeechSample(
            features=torch.from_numpy(features), targets=torch.tensor(row["targets"], dtype=torch.int64),
            session_id=row["session_id"], reference=row["reference"],
            trial_index=row["trial_index"], block=row["block"],
        )

    def __getstate__(self) -> dict[str, Any]:
        """Reabre los memmaps en cada proceso en vez de serializar sus matrices."""
        return {**self.__dict__, "_arrays": {}}


def collate_speech(samples: Sequence[SpeechSample]) -> SpeechBatch:
    """Añade padding solo a las señales y conserva las longitudes y el ID 0."""
    if not samples:
        raise ValueError("No se puede formar un lote vacío.")
    return SpeechBatch(
        features=pad_sequence([sample.features for sample in samples], batch_first=True),
        targets=torch.cat([sample.targets for sample in samples]),
        input_lengths=torch.tensor([len(sample.features) for sample in samples], dtype=torch.int64),
        target_lengths=torch.tensor([len(sample.targets) for sample in samples], dtype=torch.int64),
        session_ids=torch.tensor([sample.session_id for sample in samples], dtype=torch.int64),
        references=tuple(sample.reference for sample in samples),
        trial_indices=tuple(sample.trial_index for sample in samples),
        blocks=tuple(sample.block for sample in samples),
    )


class SessionBatchSampler(Sampler[list[int]]):
    """Elige un día uniformemente por lote y recorre sus ensayos barajados."""

    def __init__(
        self, session_indices: Sequence[range], batch_size: int,
        num_batches: int, generator: torch.Generator,
    ) -> None:
        """Fija tamaño, número de lotes y un RNG creado en src/utils/seed.py."""
        self.sessions = tuple(indices for indices in session_indices if len(indices))
        if not self.sessions or batch_size <= 0 or num_batches <= 0:
            raise ValueError("Se necesitan sesiones con datos y tamaños positivos.")
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.generator = generator

    def __len__(self) -> int:
        """Devuelve el número de actualizaciones configuradas."""
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Reinicia una permutación de cada día cuando se agotan sus ensayos."""
        permutations: dict[int, list[int]] = {}
        cursors: dict[int, int] = {}
        for _ in range(self.num_batches):
            day = int(torch.randint(len(self.sessions), (), generator=self.generator).item())
            indices = self.sessions[day]
            batch: list[int] = []
            while len(batch) < self.batch_size:
                if day not in permutations or cursors[day] == len(indices):
                    permutations[day] = torch.randperm(len(indices), generator=self.generator).tolist()
                    cursors[day] = 0
                count = min(self.batch_size - len(batch), len(indices) - cursors[day])
                selected = permutations[day][cursors[day]:cursors[day] + count]
                batch.extend(indices[position] for position in selected)
                cursors[day] += count
            yield batch
