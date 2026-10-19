"""Construcción del modelo elegido en la configuración."""

from __future__ import annotations

import torch.nn as nn


class ModeloPlaceholder(nn.Module):
    """Sustituye esto por la arquitectura del paper."""

    def __init__(self, dim_entrada: int = 32, clases: int = 10):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.ReLU(),
            nn.Linear(64, clases),
        )

    def forward(self, x):
        return self.red(x)


def construir_modelo(cfg: dict) -> nn.Module:
    """Construye la GRU de habla o la plantilla durante la migración del pipeline."""
    nombre = cfg["modelo"]["nombre"]
    if nombre == "speech_gru":
        from src.models.speech_gru import SpeechGRU

        mcfg = cfg["modelo"]
        sessions = cfg["datos"]["sesiones"]
        phones = cfg["datos"]["fonemas"]
        if not sessions or len(set(sessions)) != len(sessions):
            raise ValueError("Las sesiones deben ser una lista no vacía sin duplicados.")
        if not phones or len(set(phones)) != len(phones):
            raise ValueError("El vocabulario de fonemas está vacío o tiene duplicados.")
        if mcfg["bidireccional"]:
            raise ValueError("El baseline TensorFlow replicado es unidireccional.")
        return SpeechGRU(
            input_features=mcfg["entrada"],
            input_size=mcfg["capa_dia"],
            hidden_size=mcfg["unidades"],
            num_layers=mcfg["capas"],
            num_sessions=len(sessions),
            num_classes=len(phones) + 1,
            kernel_size=mcfg["kernel"],
            stride=mcfg["stride"],
            input_dropout=mcfg["dropout_entrada"],
            gru_dropout=mcfg["dropout_gru"],
        )
    if nombre == "mi_modelo":
        return ModeloPlaceholder()
    raise ValueError(f"modelo desconocido: {nombre}")
