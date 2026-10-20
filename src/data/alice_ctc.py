"""Adapta EEG continuo de Alice a secuencias y lotes del decodificador CTC."""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset

from src.data.alice_catalog import load_catalog, verify_catalog
from src.data.phonemes import PhonemeEncoder
from src.data.speech_dataset import SpeechSample, file_sha256


def _table(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def alice_ctc_signature(cfg: dict[str, Any]) -> str:
    """Identifica selección, preprocesamiento y etiquetas; excluye el presupuesto."""
    data = {k: v for k, v in cfg["datos"].items()
            if k not in {"preparados", "batch_size", "num_workers", "pin_memory"}}
    state = {"format_version": 1, "datos": data, "preprocesamiento": cfg["preprocesamiento"],
             "etiquetas": cfg["etiquetas"], "catalogo_sha256": file_sha256(Path(data["catalogo"]))}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _tokens(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    # La anotación separa las contracciones: would + nt, I + ve, Rabbit + s.
    tokens: list[dict[str, Any]] = []
    for event in events:
        word = event["word"]
        if tokens and word.lower() in {"nt", "s", "d", "m", "re", "ve", "ll"}:
            tokens[-1]["text"] += "n't" if word.lower() == "nt" else "'" + word
            tokens[-1]["events"].append(event)
        else:
            tokens.append({"text": word, "events": [event]})
    return tokens


def prepare_alice_ctc(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Verifica fuentes, separa segmentos y guarda EEG/fonemas sin mezclar splits."""
    data, pre = cfg["datos"], cfg["preprocesamiento"]
    root, output = Path(data["raiz"]), Path(data["preparados"])
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != alice_ctc_signature(cfg):
            raise ValueError("El caché Alice no corresponde al YAML; usa otro destino o --force.")
        for name, entry in manifest["arrays"].items():
            if file_sha256(output / name) != entry["sha256"]:
                raise ValueError(f"Señal preparada modificada: {name}")
        return manifest
    catalog = load_catalog(Path(data["catalogo"]))
    hashes = verify_catalog(root, catalog)
    subject = data["sujeto"]
    if subject != catalog["subject"] or data["sesiones"] != [subject]:
        raise ValueError("Este catálogo y adaptador están fijados a un participante.")
    participant = next(e for e in _table(root / "participants.tsv") if e["participant_id"] == subject)
    if participant["analyzed"] != "yes":
        raise ValueError("El participante no se incluyó en el análisis original.")
    base = root / subject / "eeg" / f"{subject}_task-alicelistening"
    header = configparser.ConfigParser(interpolation=None)
    text = base.with_name(base.name + "_eeg.vhdr").read_text(encoding="utf-8-sig")
    header.read_string(text[text.index("[Common Infos]"):])
    info = header["Common Infos"]
    if (info["DataFormat"] != "BINARY" or info["DataOrientation"] != "MULTIPLEXED"
            or header["Binary Infos"]["BinaryFormat"] != "IEEE_FLOAT_32"):
        raise ValueError("Formato EEG distinto del BrainVision float32 verificado.")
    n_channels = int(info["NumberOfChannels"])
    fs = 1e6 / float(info["SamplingInterval"])
    names = [header["Channel Infos"][f"Ch{i + 1}"].split(",")[0] for i in range(n_channels)]
    channels = _table(base.with_name(base.name + "_channels.tsv"))
    if names != [e["name"] for e in channels]:
        raise ValueError("El TSV de canales y la cabecera no coinciden.")
    columns = [i for i, e in enumerate(channels) if e["type"] == "EEG" and e["status"] == "good"]
    if len(columns) != cfg["modelo"]["entrada"]:
        raise ValueError("El número de canales no coincide con modelo.entrada.")
    filename = Path(info["DataFile"])
    if filename.name != str(filename):
        raise ValueError("Nombre de binario EEG no permitido.")
    raw = np.memmap(base.parent / filename, dtype="<f4", mode="r").reshape(-1, n_channels)
    resolutions = np.array([float(header["Channel Infos"][f"Ch{i + 1}"].split(",")[2]) for i in columns])
    events = _table(base.with_name(base.name + "_events.tsv"))
    if any(float(a["onset"]) >= float(b["onset"]) for a, b in zip(events, events[1:])):
        raise ValueError("Los eventos no están en orden temporal estricto.")
    story = _table(root / "sourcedata/AliceChapterOne-EEG.csv", delimiter=",")
    canonical = {e["Order"]: e for e in story}
    positions = {e["Order"]: i for i, e in enumerate(story)}
    for event in events:
        source = canonical[event["word_index"]]
        if source["Word"] != event["word"] or source["Segment"] != event["segment"]:
            raise ValueError("La palabra o el segmento no coincide con el CSV original.")
    groups = {split: data[f"{split}_segmentos"] for split in ("train", "val", "test")}
    segments = sorted({int(e["segment"]) for e in events})
    assigned = sum(groups.values(), [])
    if len(assigned) != len(set(assigned)) or set(assigned) != set(segments):
        raise ValueError("Cada segmento debe pertenecer exactamente a una partición.")
    rate = pre["frecuencia_salida"]
    if fs != int(fs) or rate != int(rate) or rate <= 2 * max(pre["banda_hz"]):
        raise ValueError("Frecuencia de muestreo o banda no válida.")
    if (pre["margen_segmento_s"] <= 0 or pre["normalizacion_epsilon"] <= 0
            or pre["clip_z"] <= 0 or data["palabras_por_secuencia"] <= 0
            or min(data["contexto_pre_s"], data["contexto_post_s"]) < 0):
        raise ValueError("Márgenes, tamaño de secuencia o normalización no válidos.")
    sos = signal.butter(pre["orden_filtro"], pre["banda_hz"], fs=fs, btype="bandpass", output="sos")
    divisor = gcd(int(fs), int(rate))
    starts = {seg: min(float(e["onset"]) for e in events if int(e["segment"]) == seg) for seg in segments}
    blocks, block_info = {}, []
    for i, seg in enumerate(segments):
        start = int(np.ceil(starts[seg] * fs))
        end = int(np.floor(starts[segments[i + 1]] * fs)) if i + 1 < len(segments) else len(raw)
        if not 0 <= start < end <= len(raw):
            raise ValueError("Los eventos quedan fuera del binario EEG.")
        block = np.asarray(raw[start:end, columns], dtype=np.float64) * resolutions
        if not np.isfinite(block).all():
            raise ValueError("El EEG contiene NaN o infinito.")
        block -= block.mean(axis=1, keepdims=True)
        block = signal.sosfilt(sos, block, axis=0)
        blocks[seg] = signal.resample_poly(block, int(rate) // divisor, int(fs) // divisor, axis=0)
        selected = [e for e in events if int(e["segment"]) == seg]
        deltas = [float(e["onset"]) - float(canonical[e["word_index"]]["onset"]) for e in selected]
        if np.ptp(deltas) > data["tolerancia_alineacion_s"]:
            raise ValueError("La temporización EEG/audio no es consistente dentro del segmento.")
        block_info.append({"segmento": seg, "split": next(k for k, v in groups.items() if seg in v),
                           "inicio_s": start / fs, "fin_s": end / fs,
                           "desfase_eeg_audio_s": float(np.median(deltas)),
                           "rango_desfase_s": float(np.ptp(deltas))})
    margin = round(pre["margen_segmento_s"] * rate)
    training = [blocks[seg][margin:-margin] for seg in groups["train"]]
    count = sum(len(block) for block in training)
    if not count or any(not len(block) for block in training):
        raise ValueError("No hay EEG de entrenamiento después de aplicar los márgenes.")
    mean = sum(block.sum(axis=0) for block in training) / count
    variance = sum(((block - mean) ** 2).sum(axis=0) for block in training) / count
    std = np.maximum(np.sqrt(variance), pre["normalizacion_epsilon"])
    encoder = PhonemeEncoder(cfg)
    records, arrays, dropped = {k: [] for k in groups}, {}, 0
    output.mkdir(parents=True, exist_ok=True)
    for info in block_info:
        seg, split = info["segmento"], info["split"]
        block = np.clip((blocks[seg] - mean) / std, -pre["clip_z"], pre["clip_z"]).astype(np.float32)
        name = f"segmento_{seg:02d}.npy"
        np.save(output / name, block, allow_pickle=False)
        arrays[name] = {"sha256": file_sha256(output / name), "shape": list(block.shape)}
        selected = [e for e in events if int(e["segment"]) == seg]
        # Cortar también ante palabras ausentes: no asignar EEG de palabras sin referencia.
        runs, current = [], []
        for event in selected:
            if current and (current[-1]["sentence_id"] != event["sentence_id"]
                            or positions[event["word_index"]] != positions[current[-1]["word_index"]] + 1):
                runs.append(current)
                current = []
            current.append(event)
        if current:
            runs.append(current)
        for sentence in runs:
            words = _tokens(sentence)
            for offset in range(0, len(words), data["palabras_por_secuencia"]):
                chunk = words[offset:offset + data["palabras_por_secuencia"]]
                first, last = chunk[0]["events"][0], chunk[-1]["events"][-1]
                begin = float(first["onset"]) - data["contexto_pre_s"]
                end = float(last["onset"]) + float(last["duration"]) + data["contexto_post_s"]
                if begin < info["inicio_s"] + pre["margen_segmento_s"] or end > info["fin_s"] - pre["margen_segmento_s"]:
                    dropped += 1
                    continue
                a, b = round((begin - info["inicio_s"]) * rate), round((end - info["inicio_s"]) * rate)
                target = encoder.encode(" ".join(word["text"] for word in chunk))
                steps = (b - a - cfg["modelo"]["kernel"]) // cfg["modelo"]["stride"] + 1
                if not 0 <= a < b <= len(block) or steps < target.minimum_ctc_steps:
                    raise ValueError(f"Secuencia incompatible con CTC: {target.text}")
                records[split].append({"path": name, "start": a, "stop": b,
                                       "ids": list(target.ids), "reference": target.text,
                                       "minimum_ctc_steps": target.minimum_ctc_steps, "segment": seg,
                                       "word_indices": [e["word_index"] for word in chunk for e in word["events"]],
                                       "inicio_s": begin, "fin_s": end})
    if any(not rows for rows in records.values()):
        raise ValueError("Hay una partición sin secuencias válidas.")
    manifest = {"format_version": 1, "signature": alice_ctc_signature(cfg), "dataset": catalog["dataset"],
                "version": catalog["version"], "subject": subject, "sources_sha256": hashes,
                "channels": [names[i] for i in columns], "excluded_channels": [n for i, n in enumerate(names) if i not in columns],
                "sample_rate": rate, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
                "phones": data["fonemas"], "blank_id": len(data["fonemas"]), "blocks": block_info,
                "arrays": arrays, "records": records, "dropped_boundary_chunks": dropped,
                "counts": {k: len(v) for k, v in records.items()},
                "notes": ["EEG de escucha; fonemas generados por g2p-en, no anotación fonética manual.",
                          "Filtrado causal por segmento y remuestreo offline; no se mide latencia real.",
                          "Referencias usan tiempos de palabras para delimitar fragmentos; sin alineación de fonemas."]}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    return manifest


class AliceCTCDataset(Dataset[SpeechSample]):
    """Expone EEG Alice con los mismos SpeechSample/SpeechBatch del paper T12."""

    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        """Comprueba firma, integridad y compatibilidad de todas las secuencias CTC."""
        self.root = Path(cfg["datos"]["preparados"])
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["signature"] != alice_ctc_signature(cfg) or split not in {"train", "val", "test"}:
            raise ValueError("Preparación Alice o partición incompatible con el YAML.")
        self.rows = self.manifest["records"][split]
        self._arrays: dict[str, np.ndarray] = {}
        for name in {row["path"] for row in self.rows}:
            if file_sha256(self.root / name) != self.manifest["arrays"][name]["sha256"]:
                raise ValueError(f"Señal preparada modificada: {name}")
            array = np.load(self.root / name, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != self.manifest["arrays"][name]["shape"] or array.shape[1] != cfg["modelo"]["entrada"]:
                raise ValueError(f"Forma o precisión EEG no válida: {name}")
        for row in self.rows:
            minimum = len(row["ids"]) + sum(a == b for a, b in zip(row["ids"], row["ids"][1:]))
            steps = (row["stop"] - row["start"] - cfg["modelo"]["kernel"]) // cfg["modelo"]["stride"] + 1
            if (steps < minimum or minimum != row["minimum_ctc_steps"] or not row["ids"]
                    or any(i < 0 or i >= self.manifest["blank_id"] for i in row["ids"])
                    or not 0 <= row["start"] < row["stop"] <= self.manifest["arrays"][row["path"]]["shape"][0]):
                raise ValueError("Una referencia o longitud no es compatible con CTC.")

    def __len__(self) -> int:
        """Devuelve el número de fragmentos de esta partición."""
        return len(self.rows)

    def __getitem__(self, index: int) -> SpeechSample:
        """Entrega solo EEG como entrada y fonemas como objetivo separado."""
        row = self.rows[index]
        if row["path"] not in self._arrays:
            self._arrays[row["path"]] = np.load(self.root / row["path"], mmap_mode="r", allow_pickle=False)
        features = self._arrays[row["path"]][row["start"]:row["stop"]].copy()
        return SpeechSample(torch.from_numpy(features), torch.tensor(row["ids"], dtype=torch.int64),
                            0, row["reference"], index, row["segment"])
