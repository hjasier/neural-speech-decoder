"""Modelo de errores de fonemas aprendido de pares reales, para pares sintéticos.

Alinea cada secuencia predicha con su referencia por distancia de edición y
cuenta, por fonema de referencia, aciertos, sustituciones y borrados, y por
posición las inserciones. Al aplicarlo, cada frase toma además un factor de
dificultad de la distribución real de PER por frase, para que haya frases
casi perfectas y frases muy degradadas como en la GRU.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

START = "^"  # contexto de una inserción antes del primer fonema


def align_phones(reference: Sequence[str], hypothesis: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """Alineamiento de distancia de edición mínima; None marca inserción o borrado.

    Desempate fijo: acierto o sustitución, luego borrado, luego inserción.
    """
    rows, cols = len(reference), len(hypothesis)
    cost = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cost[:, 0] = np.arange(rows + 1)
    cost[0, :] = np.arange(cols + 1)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                             cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    pairs: list[tuple[str | None, str | None]] = []
    i, j = rows, cols
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j - 1]))
            j -= 1
    return pairs[::-1]


def fit_error_model(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> dict[str, Any]:
    """Cuenta errores de pares (referencia, predicción) y devuelve un modelo JSON.

    `sustituciones[r][h]` y `borrados[r]` cuentan sobre las apariciones de r;
    `inserciones[c][h]`, los fonemas insertados tras el contexto c (o START).
    """
    occurrences: dict[str, int] = {}
    substitutions: dict[str, dict[str, int]] = {}
    deletions: dict[str, int] = {}
    insertions: dict[str, dict[str, int]] = {}
    sentence_rates: list[float] = []
    errors = length = 0
    for reference, hypothesis in pairs:
        if not reference:
            raise ValueError("Una referencia de fonemas está vacía.")
        context, sentence_errors = START, 0
        for ref, hyp in align_phones(reference, hypothesis):
            if ref is None:
                bucket = insertions.setdefault(context, {})
                bucket[hyp] = bucket.get(hyp, 0) + 1
                sentence_errors += 1
                continue
            occurrences[ref] = occurrences.get(ref, 0) + 1
            if hyp is None:
                deletions[ref] = deletions.get(ref, 0) + 1
                sentence_errors += 1
            elif hyp != ref:
                bucket = substitutions.setdefault(ref, {})
                bucket[hyp] = bucket.get(hyp, 0) + 1
                sentence_errors += 1
            context = ref
        sentence_rates.append(sentence_errors / len(reference))
        errors += sentence_errors
        length += len(reference)
    if not length:
        raise ValueError("No hay pares para ajustar el modelo de errores.")
    occurrences[START] = len(sentence_rates)
    return {"apariciones": occurrences, "sustituciones": substitutions, "borrados": deletions,
            "inserciones": insertions, "per_frases": sentence_rates, "errores": errors, "longitud": length,
            "per": errors / length}


class PhonemeNoise:
    """Aplica un modelo de `fit_error_model` a secuencias limpias de fonemas."""

    def __init__(self, model: dict[str, Any]) -> None:
        """Precalcula probabilidades por fonema y los factores de dificultad por frase."""
        self.per = model["per"]
        if self.per <= 0:
            raise ValueError("El modelo de errores no tiene errores que aplicar.")
        self.factors = np.asarray(model["per_frases"], dtype=float) / self.per
        self.table: dict[str, tuple[float, float, list[str], np.ndarray, float, list[str], np.ndarray]] = {}
        for phone, count in model["apariciones"].items():
            subs = model["sustituciones"].get(phone, {})
            inserted = model["inserciones"].get(phone, {})
            n_sub, n_ins = sum(subs.values()), sum(inserted.values())
            self.table[phone] = (n_sub / count, model["borrados"].get(phone, 0) / count,
                                 sorted(subs), np.array([subs[key] for key in sorted(subs)], dtype=float),
                                 n_ins / count, sorted(inserted),
                                 np.array([inserted[key] for key in sorted(inserted)], dtype=float))

    def _insert(self, context: str, scale: float, rng: np.random.Generator) -> list[str]:
        entry = self.table.get(context)
        if entry is None or not entry[5]:
            return []
        # Inserciones geométricas con media `escala · tasa`, como en los datos reales.
        mean = scale * entry[4]
        stop = 1.0 / (1.0 + mean)
        count = rng.geometric(stop) - 1
        weights = entry[6] / entry[6].sum()
        return [entry[5][index] for index in rng.choice(len(entry[5]), size=count, p=weights)]

    def apply(self, phones: Sequence[str], rng: np.random.Generator) -> list[str]:
        """Devuelve una versión ruidosa; la dificultad de la frase se sortea primero."""
        scale = float(self.factors[rng.integers(len(self.factors))])
        noisy = self._insert(START, scale, rng)
        for phone in phones:
            entry = self.table.get(phone)
            if entry is None:
                noisy.append(phone)
                continue
            p_sub, p_del = scale * entry[0], scale * entry[1]
            total = p_sub + p_del
            if total > 1.0:
                p_sub, p_del = p_sub / total, p_del / total
            draw = rng.random()
            if draw < p_del:
                pass
            elif draw < p_del + p_sub and entry[2]:
                noisy.append(entry[2][rng.choice(len(entry[2]), p=entry[3] / entry[3].sum())])
            else:
                noisy.append(phone)
            noisy.extend(self._insert(phone, scale, rng))
        return noisy


def phone_error_rate(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> float:
    """PER agregado de pares (referencia, predicción): errores entre longitud total."""
    errors = length = 0
    for reference, hypothesis in pairs:
        errors += sum(ref != hyp for ref, hyp in align_phones(reference, hypothesis))
        length += len(reference)
    return errors / length
