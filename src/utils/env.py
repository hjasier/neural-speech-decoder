"""Carga privada de la credencial W&B, sin ejecutar ni expandir el archivo."""

from __future__ import annotations

import os
from pathlib import Path
import re


def cargar_clave_wandb(*archivos: Path) -> bool:
    """Lee WANDB_API_KEY de .env si falta en el entorno; devuelve si hay clave.

    Sin rutas explícitas usa el .env de la raíz del proyecto. Admite una
    asignación por línea, comillas simples/dobles y comentarios. Solo carga
    WANDB_API_KEY; no interpreta escapes, interpolaciones ni comandos.
    """
    if os.environ.get("WANDB_API_KEY", "").strip():
        return True
    for archivo in archivos or (Path(__file__).resolve().parents[2] / ".env",):
        if not archivo.is_file():
            continue
        for linea in archivo.read_text(encoding="utf-8-sig").splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?WANDB_API_KEY\s*=(.*)", linea)
            if match is None:
                continue
            valor = match[1].strip()
            if valor.startswith(("'", '"')):
                quoted = re.fullmatch(r"(['\"])(.*?)\1\s*(?:#.*)?", valor)
                if quoted is None:
                    raise ValueError("WANDB_API_KEY en .env tiene comillas sin cerrar o texto sobrante.")
                valor = quoted[2]
            else:
                valor = "" if valor.startswith("#") else re.split(r"\s+#", valor, maxsplit=1)[0].strip()
            if not valor:
                continue
            if any(character.isspace() for character in valor):
                raise ValueError("WANDB_API_KEY en .env debe ser una clave de una sola línea, sin espacios.")
            os.environ["WANDB_API_KEY"] = valor
            return True
    return False
