"""Configuración pequeña de la GRU real para comprobaciones CPU."""

from pathlib import Path

import pytest
import yaml

from src.data.speech import AREA_6V_CHANNELS


@pytest.fixture
def cfg_habla_pequena() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/speech_salidas_t4.yaml").read_text())
    cfg["modelo"].update(entrada=4, capa_dia=4, unidades=8, kernel=3, stride=2,
                         dropout_entrada=0.0, dropout_gru=0.0)
    cfg["datos"]["sesiones"] = ["sesion_a", "sesion_b"]
    return cfg


@pytest.fixture
def cfg_habla_online_pequena(cfg_habla_pequena: dict) -> dict:
    """Conserva las 256 características y los tiempos online con una GRU pequeña."""
    root = Path(__file__).resolve().parents[1]
    temporal = yaml.safe_load((root / "configs/entrenos_online/speech_online_temporal.yaml").read_text())
    cfg_habla_pequena["preprocesamiento"] = temporal["preprocesamiento"]
    cfg_habla_pequena["modelo"].update(
        temporal["modelo"], entrada=2 * AREA_6V_CHANNELS, capa_dia=2 * AREA_6V_CHANNELS,
    )
    return cfg_habla_pequena
