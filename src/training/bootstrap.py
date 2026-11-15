"""Intervalos de confianza por bootstrap para tasas de error agregadas.

Sigue el suplemento 4.1 del paper: se remuestrean frases con reemplazo y en
cada remuestreo se recalcula la tasa agregada (errores totales entre símbolos
totales), no la media de las tasas por frase. Método del percentil.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.utils.seed import crear_generador_numpy


def bootstrap_rate_interval(
    errors: Sequence[int], totals: Sequence[int],
    *, resamples: int, confidence: float, seed: int,
) -> dict[str, float | int]:
    """Devuelve la tasa agregada y su intervalo; cada posición es una frase."""
    error_counts = np.asarray(errors, dtype=np.int64)
    total_counts = np.asarray(totals, dtype=np.int64)
    if (error_counts.ndim != 1 or error_counts.shape != total_counts.shape or not len(error_counts)
            or (error_counts < 0).any() or (total_counts <= 0).any()):
        raise ValueError("Se necesita una frase o más, con errores no negativos y totales positivos.")
    if type(resamples) is not int or resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("remuestreos debe ser un entero positivo y la confianza estar entre cero y uno.")
    rng = crear_generador_numpy(seed)
    rates = np.empty(resamples, dtype=np.float64)
    # Por tandas: 10.000 remuestreos de miles de frases no caben holgados de una vez.
    for start in range(0, resamples, 500):
        stop = min(start + 500, resamples)
        chosen = rng.integers(len(error_counts), size=(stop - start, len(error_counts)))
        rates[start:stop] = error_counts[chosen].sum(axis=1) / total_counts[chosen].sum(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(rates, [tail, 1 - tail])
    return {"rate": float(error_counts.sum() / total_counts.sum()), "low": float(low), "high": float(high),
            "errors": int(error_counts.sum()), "total": int(total_counts.sum()), "trials": len(error_counts),
            "confidence": confidence, "resamples": resamples}
