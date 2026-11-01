"""Control Linux de páginas residentes del grafo compartido de solo lectura.

MADV_DONTNEED conserva el mapeo: el siguiente acceso recupera el mismo archivo.
El kernel puede conservar páginas en su caché recuperable de archivos.
"""

from __future__ import annotations

import ctypes
import math
import mmap
from pathlib import Path
import re
from time import perf_counter
from typing import Any


def mapped_regions(path: Path) -> list[tuple[int, int]]:
    """Localiza únicamente los mapeos compartidos y de solo lectura del grafo."""
    regions = []
    filename_expected = str(path.resolve())
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        filename = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[5])
        if filename != filename_expected:
            continue
        if fields[1] != "r--s":
            raise ValueError("El grafo debe estar mapeado como archivo compartido de solo lectura.")
        start, end = fields[0].split("-")
        regions.append((int(start, 16), int(end, 16) - int(start, 16)))
    return regions


def mapped_bytes(path: Path) -> int:
    """Suma los bytes virtuales de los mapeos reales del grafo."""
    return sum(size for _, size in mapped_regions(path))


def worker_memory() -> dict[str, int]:
    """Distingue memoria residente anónima y respaldada por archivos en Linux."""
    names = {"VmRSS:": "rss_bytes", "RssAnon:": "anonymous_bytes", "RssFile:": "file_bytes"}
    values = dict.fromkeys(names.values(), 0)
    for line in Path("/proc/self/status").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in names:
            values[names[fields[0]]] = int(fields[1]) * 1024
    return values


class MappedGraphCache:
    """Descarta páginas residentes del grafo; sus direcciones y datos siguen válidos."""

    def __init__(self, path: Path, policy: dict[str, Any]) -> None:
        """Aplica el control solo a rangos de archivo compartidos y de solo lectura."""
        threshold = policy["rss_archivos_umbral_gib"]
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or threshold <= 0
                or type(policy["liberar_al_final_frase"]) is not bool):
            raise ValueError("Política de caché del mapeo no válida.")
        self.threshold = threshold * (1 << 30)
        self.release_on_finish = policy["liberar_al_final_frase"]
        self.regions = mapped_regions(path)
        if not self.regions:
            raise ValueError("No se han encontrado los mapeos del grafo para controlar su caché.")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        self._libc.madvise.restype = ctypes.c_int
        self.stats: dict[str, Any] = {"evictions": 0, "eviction_ms": 0.0,
            "peak_observed_file_bytes": 0, "policy": policy,
            "scope": "solo mapeos de TLG.fst; la caché de archivos del kernel puede seguir siendo recuperable"}

    def maintain(self, *, finish: bool = False) -> dict[str, Any]:
        """Libera páginas entre llamadas nativas, por umbral o al finalizar una frase."""
        before = worker_memory()
        self.stats["peak_observed_file_bytes"] = max(self.stats["peak_observed_file_bytes"], before["file_bytes"])
        if before["file_bytes"] >= self.threshold or (finish and self.release_on_finish):
            started = perf_counter()
            for address, size in self.regions:
                # En MAP_SHARED de solo lectura, el siguiente acceso relee el mismo archivo.
                # No aplicar este consejo a buffers anónimos ni a copias privadas modificadas.
                if self._libc.madvise(address, size, mmap.MADV_DONTNEED) != 0:
                    raise OSError(ctypes.get_errno(), "No se pudo liberar la caché residente del grafo.")
            self.stats["evictions"] += 1
            self.stats["eviction_ms"] += (perf_counter() - started) * 1000
            self.stats["last_before"] = before
            self.stats["last_after"] = worker_memory()
        return self.stats
