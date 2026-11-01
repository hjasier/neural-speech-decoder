"""Cabecera OpenFst: inspección acotada sin cargar estados ni arcos."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any


def read_fst_header(path: Path) -> dict[str, Any]:
    """Lee exclusivamente la cabecera binaria little endian de OpenFst 1.6.5."""
    with path.open("rb") as stream:
        if stream.read(4) != struct.pack("<i", 2125659606):
            raise ValueError("El recurso no tiene una cabecera OpenFst compatible.")
        names = []
        for _ in range(2):
            size_data = stream.read(4)
            if len(size_data) != 4:
                raise ValueError("Cabecera OpenFst truncada.")
            size = struct.unpack("<i", size_data)[0]
            if not 1 <= size <= 64:
                raise ValueError("Tipo OpenFst no reconocido.")
            value = stream.read(size)
            if len(value) != size:
                raise ValueError("Cabecera OpenFst truncada.")
            names.append(value.decode("ascii"))
        tail = stream.read(40)
        if len(tail) != 40:
            raise ValueError("Cabecera OpenFst truncada.")
    values = struct.unpack("<iiQqqq", tail)
    header = dict(zip(("version", "flags", "properties", "start", "states", "arcs"), values))
    if header["states"] < 0 or header["arcs"] < 0:
        raise ValueError("La cabecera no declara los tamaños del grafo.")
    return {"fst_type": names[0], "arc_type": names[1], **header}


def require_mappable_fst(path: Path) -> dict[str, Any]:
    """Exige ConstFst estándar alineado; un VectorFst no admite esta lectura."""
    header = read_fst_header(path)
    if (header["fst_type"] != "const" or header["arc_type"] != "standard"
            or not (header["version"] == 1 or header["flags"] & 4)):
        raise ValueError("La lectura map requiere un ConstFst estándar alineado.")
    return header
