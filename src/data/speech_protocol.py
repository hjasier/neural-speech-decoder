"""Partición T12 fija por bloques, independiente de las semillas del modelo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import loadmat


def split_training_blocks(blocks: Sequence[int], fraction: float) -> tuple[list[int], list[int]]:
    """Reserva un sufijo de bloques completos, lo más próximo a la fracción.

    Conserva al menos un bloque en cada parte. Si dos cortes quedan a igual
    distancia, reserva menos bloques. No usa señales, textos ni resultados.
    """
    ordered = sorted(set(blocks))
    if not 0 < fraction < 1 or len(ordered) < 2:
        raise ValueError("Se necesitan al menos dos bloques y una fracción entre cero y uno.")
    cut = min(range(1, len(ordered)), key=lambda index: (
        abs(sum(block in ordered[index:] for block in blocks) - len(blocks) * fraction), -index,
    ))
    validation = set(ordered[cut:])
    return ([i for i, block in enumerate(blocks) if block not in validation],
            [i for i, block in enumerate(blocks) if block in validation])


def train_fold_spec(cfg: dict[str, Any]) -> dict[str, int] | None:
    """Lee `datos.pliegue`: qué parte de train se aparta para el ajuste cruzado.

    Sin la clave no se aparta nada y train queda igual que en la partición.
    """
    spec = cfg["datos"].get("pliegue")
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) != {"pliegues", "excluido"}
            or any(type(spec[key]) is not int for key in spec)
            or spec["pliegues"] < 2 or not 0 <= spec["excluido"] < spec["pliegues"]):
        raise ValueError("datos.pliegue requiere los enteros pliegues >= 2 y 0 <= excluido < pliegues.")
    return spec


def fold_blocks(train_blocks: Sequence[int], folds: int, fold: int, rotation: int = 0) -> list[int]:
    """Bloques de train que forman un pliegue: tramos contiguos casi iguales.

    Cada bloque cae en un solo pliegue. Si el reparto no es exacto, los
    primeros tramos tienen un bloque más; `rotation` (la posición de la
    sesión) cambia qué pliegue recibe cada tramo, para que los bloques
    sobrantes no se acumulen siempre en los mismos pliegues. Con menos
    bloques que pliegues, alguno queda vacío en esa sesión. Apartar bloques
    enteros conserva su normalización.
    """
    ordered = sorted(set(train_blocks))
    base, extra = divmod(len(ordered), folds)
    group = (fold + rotation) % folds
    start = group * base + min(group, extra)
    held = ordered[start:start + base + (group < extra)]
    if len(held) == len(ordered):
        raise ValueError("El pliegue apartaría todos los bloques de train de una sesión.")
    return held


def build_speech_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    """Construye la partición desde blockIdx y fija los hashes de los .mat."""
    spec = cfg["protocolo"]
    if spec["metodo"] != "ultimos_bloques_por_sesion" or spec["seleccion_checkpoint"] != "per":
        raise ValueError("Este protocolo requiere bloques finales y selección por PER de validación.")
    sessions = []
    totals = {"train": 0, "val": 0, "test": 0}
    for name in cfg["datos"]["sesiones"]:
        sources, blocks = {}, {}
        for split in ("train", "test"):
            path = Path(cfg["datos"]["raiz"]) / split / f"{name}.mat"
            values = loadmat(path, variable_names=["blockIdx"])["blockIdx"].ravel()
            if not len(values) or not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
                raise ValueError(f"Bloques inválidos: {path}")
            blocks[split] = values.astype(int).tolist()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources[split] = {"path": f"{split}/{name}.mat", "sha256": digest, "trials": len(values)}
        train, val = split_training_blocks(blocks["train"], spec["fraccion_validacion"])
        entry = {"session": name, "sources": sources,
                 "train": train, "val": val, "test": list(range(len(blocks["test"]))),
                 "train_blocks": sorted({blocks["train"][i] for i in train}),
                 "val_blocks": sorted({blocks["train"][i] for i in val})}
        sessions.append(entry)
        for split in totals:
            totals[split] += len(entry[split])
    return {"format_version": 1, "name": spec["nombre"], "method": spec["metodo"],
            "validation_fraction_target": spec["fraccion_validacion"],
            "checkpoint_selection": "minimum validation PER; first checkpoint wins ties",
            "sessions": sessions, "totals": totals}


def read_speech_protocol(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Lee el manifiesto fijado por SHA-256 y exige cobertura y separación."""
    spec = cfg.get("protocolo")
    if spec is None:
        return None
    path = Path(spec["manifiesto"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError("La partición T12 ha cambiado: no coincide su SHA-256 fijado.")
    state = json.loads(payload)
    if (state["format_version"] != 1 or state["name"] != spec["nombre"]
            or state["method"] != spec["metodo"] or spec["metodo"] != "ultimos_bloques_por_sesion"
            or state["validation_fraction_target"] != spec["fraccion_validacion"]
            or spec["seleccion_checkpoint"] != "per"
            or state["checkpoint_selection"] != "minimum validation PER; first checkpoint wins ties"
            or [entry["session"] for entry in state["sessions"]] != cfg["datos"]["sesiones"]):
        raise ValueError("El protocolo no corresponde a esta configuración.")
    totals = {"train": 0, "val": 0, "test": 0}
    for entry in state["sessions"]:
        for split in totals:
            indices = entry[split]
            if (not indices or any(type(i) is not int for i in indices)
                    or indices != sorted(set(indices))):
                raise ValueError("La partición contiene índices vacíos, repetidos o inválidos.")
            totals[split] += len(indices)
        if (set(entry["train"]) & set(entry["val"])
                or sorted(entry["train"] + entry["val"]) != list(range(entry["sources"]["train"]["trials"]))
                or entry["test"] != list(range(entry["sources"]["test"]["trials"]))
                or not entry["train_blocks"] or not entry["val_blocks"]
                or max(entry["train_blocks"]) >= min(entry["val_blocks"])):
            raise ValueError("Train/val deben cubrir el train oficial sin solaparse; test debe quedar intacto.")
    if totals != state["totals"]:
        raise ValueError("Los recuentos no corresponden a la partición.")
    return state
