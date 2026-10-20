"""Transcripciones a etiquetas de fonemas, siguiendo el vocabulario de speechBCI.

Los recursos de g2p-en se preparan explícitamente antes de construir el
codificador. Este módulo no descarga recursos al importarse ni al codificar.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile


def verify_phoneme_resources(cfg: dict[str, Any]) -> Path:
    """Verifica los ZIP fijados y sus archivos extraídos; devuelve su carpeta."""
    root = Path(cfg["etiquetas"]["recursos"])
    for resource in cfg["etiquetas"]["recursos_nltk"]:
        archive = root / resource["archivo"]
        if not archive.is_file():
            raise FileNotFoundError(
                f"Falta {archive}. Ejecuta scripts/get_phoneme_resources.py con esta configuración."
            )
        if hashlib.sha256(archive.read_bytes()).hexdigest() != resource["sha256"]:
            raise ValueError(f"El SHA-256 no coincide: {archive}")
        with ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                path = archive.parent / member.filename
                if not path.is_file() or path.read_bytes() != bundle.read(member):
                    raise ValueError(f"Recurso sin extraer o modificado: {path}")
    return root


@dataclass(frozen=True)
class PhonemeTarget:
    """Texto limpio, símbolos y sus IDs; sin blank ni padding en las etiquetas."""

    text: str
    phones: tuple[str, ...]
    ids: tuple[int, ...]

    @property
    def minimum_ctc_steps(self) -> int:
        """Pasos mínimos de CTC: longitud más separaciones entre IDs repetidos."""
        repeats = sum(left == right for left, right in zip(self.ids, self.ids[1:]))
        return len(self.ids) + repeats


class PhonemeEncoder:
    """Convierte referencias a los 40 símbolos del vocabulario configurado.

    Conserva IDs 0–39, igual que las etiquetas efectivas de TensorFlow
    después de retirar su desplazamiento de almacenamiento. El blank
    ocupa el siguiente ID y nunca aparece en una secuencia objetivo.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Carga g2p-en con recursos locales verificados y el vocabulario YAML."""
        phones = tuple(cfg["datos"]["fonemas"])
        if not phones or len(set(phones)) != len(phones):
            raise ValueError("El vocabulario debe ser no vacío y no contener duplicados.")
        self.silence = cfg["etiquetas"]["silencio"]
        self.max_length = cfg["etiquetas"]["max_fonemas"]
        if self.silence not in phones or self.max_length <= 0:
            raise ValueError("El símbolo de silencio o la longitud máxima no son válidos.")
        self.phone_to_id = {phone: index for index, phone in enumerate(phones)}
        self.blank_id = len(phones)
        root = verify_phoneme_resources(cfg)

        # g2p-en intenta descargar al importarse si faltan recursos. Los
        # comprobamos antes y aislamos la búsqueda de otras copias del equipo.
        import nltk

        nltk.data.path[:] = [str(root.resolve())]
        nltk.data.clear_cache()
        for required in (
            "corpora/cmudict.zip",
            "taggers/averaged_perceptron_tagger.zip",
            "taggers/averaged_perceptron_tagger_eng/",
        ):
            nltk.data.find(required)
        from g2p_en import G2p

        self._g2p = G2p()

    def encode(self, text: str | None) -> PhonemeTarget:
        """Limpia texto, obtiene fonemas sin estrés e inserta SIL por palabra.

        None y HELD OUT no se convierten en etiquetas. Un texto que queda
        vacío tras retirar puntuación produce SIL, como en el original.
        No se truncan secuencias ni se eliminan ejemplos con errores.
        """
        if text is None or text.strip().upper() == "HELD OUT":
            raise ValueError("La transcripción está oculta; no se puede crear un objetivo.")
        cleaned = re.sub(r"[^a-zA-Z\- ']", "", text.strip()).replace("--", "").lower()
        phones = []
        if cleaned:
            for phone in self._g2p(cleaned):
                if phone == " ":
                    phones.append(self.silence)
                phone = re.sub(r"[0-9]", "", phone)
                if re.match(r"[A-Z]+", phone):
                    phones.append(phone)
        phones.append(self.silence)

        if len(phones) > self.max_length:
            raise ValueError(f"La secuencia supera los {self.max_length} fonemas configurados.")
        unknown = set(phones) - self.phone_to_id.keys()
        if unknown:
            raise ValueError(f"Fonemas fuera del vocabulario: {sorted(unknown)}")
        return PhonemeTarget(
            text=cleaned,
            phones=tuple(phones),
            ids=tuple(self.phone_to_id[phone] for phone in phones),
        )
