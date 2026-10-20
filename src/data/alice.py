"""EEG BrainVision de Alice, ventanas por palabra y partición por segmentos."""

from __future__ import annotations

import configparser
import csv
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from src.data.alice_catalog import load_catalog, verify_catalog


@dataclass
class AliceSplit:
    """Ventanas EEG, objetivo acústico normalizado y referencias solo para auditar."""

    features: np.ndarray
    targets: np.ndarray
    events: list[dict[str, str]]


@dataclass
class AlicePrepared:
    """Particiones separadas y estadísticas ajustadas exclusivamente en train."""

    splits: dict[str, AliceSplit]
    normalization: dict[str, Any]
    report: dict[str, Any]


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def prepare_alice(cfg: dict[str, Any], normalization: dict[str, Any] | None = None) -> AlicePrepared:
    """Separa antes de filtrar; ajusta con train o restaura estadísticas guardadas."""
    data_cfg, pre = cfg["datos"], cfg["preprocesamiento"]
    root = Path(data_cfg["raiz"])
    catalog = load_catalog(Path(data_cfg["catalogo"]))
    hashes = verify_catalog(root, catalog)
    subject = data_cfg["sujeto"]
    participant = next(row for row in _tsv(root / "participants.tsv") if row["participant_id"] == subject)
    if participant["analyzed"] != "yes":
        raise ValueError("El piloto exige un sujeto incluido en el análisis original.")
    base = root / subject / "eeg" / f"{subject}_task-alicelistening"
    header_text = base.with_name(base.name + "_eeg.vhdr").read_text(encoding="utf-8-sig")
    header = configparser.ConfigParser(interpolation=None)
    header.read_string(header_text[header_text.index("[Common Infos]"):])
    info = header["Common Infos"]
    if (info["DataFormat"] != "BINARY" or info["DataOrientation"] != "MULTIPLEXED"
            or header["Binary Infos"]["BinaryFormat"] != "IEEE_FLOAT_32"):
        raise ValueError("Este lector exige BrainVision multiplexado float32 little-endian.")
    count = int(info["NumberOfChannels"])
    sample_rate = 1e6 / float(info["SamplingInterval"])
    rows = _tsv(base.with_name(base.name + "_channels.tsv"))
    names = [header["Channel Infos"][f"Ch{i + 1}"].split(",")[0] for i in range(count)]
    if names != [row["name"] for row in rows]:
        raise ValueError("El orden del TSV no coincide con el binario EEG.")
    columns = [i for i, row in enumerate(rows) if row["type"] == "EEG" and row["status"] == "good"]
    if len(columns) != cfg["modelo"]["entrada"]:
        raise ValueError("El número de canales EEG válidos no coincide con la configuración.")
    filename = Path(info["DataFile"])
    if filename.name != str(filename):
        raise ValueError("DataFile debe ser un nombre local sin subcarpetas.")
    raw = np.memmap(base.parent / filename, mode="r", dtype="<f4")
    if raw.size % count:
        raise ValueError("El tamaño del binario no es múltiplo del número de canales.")
    raw = raw.reshape(-1, count)
    resolutions = np.array([float(header["Channel Infos"][f"Ch{i + 1}"].split(",")[2]) for i in columns])
    events = _tsv(base.with_name(base.name + "_events.tsv"))
    groups = {name: list(data_cfg[f"{name}_segmentos"]) for name in ("train", "val", "test")}
    all_groups = sum(groups.values(), [])
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("Un segmento aparece en varias particiones.")
    segments = sorted({int(e["segment"]) for e in events})
    if set(all_groups) != set(segments):
        raise ValueError("La configuración debe asignar cada segmento exactamente una vez.")
    target_rate = pre["frecuencia_salida"]
    if int(sample_rate) != sample_rate or target_rate <= 2 * max(pre["banda_hz"]):
        raise ValueError("Frecuencia de remuestreo o banda no válida.")
    a, z = [round(t * target_rate) for t in pre["ventana_s"]]
    b, c = [round(t * target_rate) for t in pre["baseline_s"]]
    if not (b < c <= 0 <= a < z) or pre["clip_z"] <= 0 or pre["epsilon"] <= 0:
        raise ValueError("Ventanas de baseline/respuesta o normalización no válidas.")
    sos = signal.butter(pre["orden_filtro"], pre["banda_hz"], btype="bandpass", fs=sample_rate, output="sos")
    divisor = gcd(int(sample_rate), int(target_rate))
    stored: dict[str, list] = {name: [] for name in groups}
    segment_report = []
    starts = {seg: min(float(e["onset"]) for e in events if int(e["segment"]) == seg) for seg in segments}
    for index, seg in enumerate(segments):
        split = next(name for name, ids in groups.items() if seg in ids)
        start = int(np.ceil(starts[seg] * sample_rate))
        end = int(np.floor(starts[segments[index + 1]] * sample_rate)) if index + 1 < len(segments) else len(raw)
        if not 0 <= start < end <= len(raw):
            raise ValueError("Las marcas de eventos quedan fuera del EEG.")
        block = np.asarray(raw[start:end, columns], dtype=np.float64) * resolutions
        if not np.isfinite(block).all():
            raise ValueError("El EEG contiene valores no finitos.")
        block -= block.mean(axis=1, keepdims=True)
        block = signal.sosfilt(sos, block, axis=0)
        block = signal.resample_poly(block, target_rate // divisor, int(sample_rate) // divisor, axis=0)
        kept = 0
        candidates = [e for e in events if int(e["segment"]) == seg]
        for event in candidates:
            onset = float(event["onset"])
            try:
                power = float(event[data_cfg["objetivo"]])
            except ValueError:
                continue
            if not np.isfinite(power) or power <= 0:
                continue
            if (onset + pre["baseline_s"][0] < start / sample_rate + pre["margen_segmento_s"]
                    or onset + pre["ventana_s"][1] > end / sample_rate - pre["margen_segmento_s"]):
                continue
            position = round((onset - start / sample_rate) * target_rate)
            if position + b < 0 or position + z > len(block):
                raise ValueError("Una ventana queda fuera del segmento ya separado.")
            baseline = block[position + b:position + c].mean(axis=0)
            features = (block[position + a:position + z] - baseline).astype(np.float32)
            stored[split].append((features, np.log10(power), event))
            kept += 1
        segment_report.append({"segmento": seg, "split": split, "inicio_s": start / sample_rate,
                               "fin_s": end / sample_rate, "eventos": len(candidates), "usados": kept})
    if any(not values for values in stored.values()):
        raise ValueError("Una partición no tiene ejemplos válidos.")
    splits = {name: AliceSplit(np.stack([v[0] for v in values]),
                              np.array([v[1] for v in values], dtype=np.float32), [v[2] for v in values])
              for name, values in stored.items()}
    if normalization is None:
        xmean = splits["train"].features.mean(axis=(0, 1))
        xstd = np.maximum(splits["train"].features.std(axis=(0, 1)), pre["epsilon"])
        ymean = float(splits["train"].targets.mean())
        ystd = float(splits["train"].targets.std())
    else:
        xmean = np.asarray(normalization["xmean"], dtype=np.float32)
        xstd = np.asarray(normalization["xstd"], dtype=np.float32)
        ymean, ystd = normalization["ymean"], normalization["ystd"]
    if (xmean.shape != (len(columns),) or xstd.shape != xmean.shape
            or not np.isfinite(xmean).all() or not np.isfinite(xstd).all()
            or np.any(xstd < pre["epsilon"]) or not np.isfinite([ymean, ystd]).all()
            or ystd <= pre["epsilon"]):
        raise ValueError("Las estadísticas de normalización no son válidas.")
    for split in splits.values():
        split.features = np.clip((split.features - xmean) / xstd, -pre["clip_z"], pre["clip_z"]).astype(np.float32)
        split.targets = ((split.targets - ymean) / ystd).astype(np.float32)
    normalization = {"xmean": xmean.tolist(), "xstd": xstd.tolist(), "ymean": ymean, "ystd": ystd}
    report = {"dataset": catalog["dataset"], "version": catalog["version"], "sujeto": subject,
              "analyzed_original": participant["analyzed"], "archivos_sha256": hashes,
              "frecuencia_original": sample_rate, "frecuencia_modelo": target_rate,
              "canales_entrada": [names[i] for i in columns],
              "canales_excluidos": [name for i, name in enumerate(names) if i not in columns],
              "objetivo": "log10(sound_power)", "segmentos": segment_report,
              "ejemplos": {name: len(split.targets) for name, split in splits.items()},
              "forma_ventana": list(splits["train"].features.shape[1:]),
              "normalizacion": "estadísticas solo de train" if normalization is None else "restaurada del checkpoint",
              "filtrado": "causal por segmento; remuestreo offline"}
    return AlicePrepared(splits, normalization, report)
