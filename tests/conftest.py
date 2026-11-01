"""Configuración pequeña de la GRU real para comprobaciones CPU."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def cfg_habla_pequena() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/speech_salidas_t4.yaml").read_text())
    cfg["modelo"].update(entrada=4, capa_dia=4, unidades=8, kernel=3, stride=2,
                         dropout_entrada=0.0, dropout_gru=0.0)
    cfg["datos"]["sesiones"] = ["sesion_a", "sesion_b"]
    return cfg
