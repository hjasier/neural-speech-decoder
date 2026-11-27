"""Frases de texto externo para los pares sintéticos de la Línea A.

Parte documentos en frases y se queda con las que se parecen a las de T12:
empiezan en mayúscula, acaban en punto, interrogación o exclamación, solo usan
letras y la puntuación de las frases de T12, y su número de palabras cae en el
rango configurado. Las frases con cifras se descartan, porque la limpieza de
las etiquetas borraría los números y el texto no coincidiría con los fonemas.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import re
from typing import Any

from src.training.speech_trigram_evaluation import word_tokens

# Final de frase seguido de espacio y una mayúscula (o comillas) que abre la siguiente.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"']?\s+(?=[\"']?[A-Z])")
# Mismo juego de caracteres que las frases de T12; el punto solo al final.
_SENTENCE = re.compile(r"[\"']?[A-Z][A-Za-z ,'\";:?!-]*[.?!][\"']?")


def split_sentences(document: str) -> Iterator[str]:
    """Frases candidatas de un documento, por párrafos y fronteras de puntuación."""
    for paragraph in document.split("\n"):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            yield from (piece.strip() for piece in _BOUNDARY.split(paragraph) if piece.strip())


def normalized_text(text: str) -> str:
    """Clave de comparación: la misma limpieza que el WER y las etiquetas."""
    return " ".join(word_tokens(text))


def rejection_reason(sentence: str, spec: dict[str, Any]) -> str | None:
    """Motivo por el que una frase no sirve, o None si se conserva."""
    if not _SENTENCE.fullmatch(sentence) or "--" in sentence:
        return "caracteres"
    words = len(word_tokens(sentence))
    if not spec["min_palabras"] <= words <= spec["max_palabras"]:
        return "longitud"
    return None


def candidate_sentences(documents: Iterable[str], spec: dict[str, Any], excluded: set[str],
                        counts: dict[str, int]) -> dict[str, str]:
    """Frases únicas por texto normalizado, en orden de aparición, sin las de T12.

    `counts` acumula documentos, frases vistas y descartes por motivo.
    """
    unique: dict[str, str] = {}
    for document in documents:
        counts["documentos"] = counts.get("documentos", 0) + 1
        for sentence in split_sentences(document):
            counts["frases"] = counts.get("frases", 0) + 1
            reason = rejection_reason(sentence, spec)
            key = normalized_text(sentence)
            if reason is None and key in excluded:
                reason = "texto_t12"
            elif reason is None and key in unique:
                reason = "repetida"
            if reason is not None:
                counts[f"descartada_{reason}"] = counts.get(f"descartada_{reason}", 0) + 1
                continue
            unique[key] = sentence
    return unique
