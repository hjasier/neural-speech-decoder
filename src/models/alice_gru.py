"""Regresor temporal pequeño inspirado en la familia GRU del baseline de habla."""

from __future__ import annotations

import torch
from torch import nn


class AliceGRU(nn.Module):
    """Recibe exclusivamente EEG y estima log-potencia acústica normalizada."""

    def __init__(self, cfg: dict) -> None:
        """Construye proyección, GRU unidireccional y salida escalar desde YAML."""
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(cfg["entrada"], cfg["proyeccion"]), nn.Tanh())
        self.gru = nn.GRU(cfg["proyeccion"], cfg["unidades"], cfg["capas"],
                          batch_first=True, dropout=cfg["dropout"] if cfg["capas"] > 1 else 0.0)
        self.output_layer = nn.Linear(cfg["unidades"], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Devuelve una estimación por ventana, sin usar palabra ni referencia."""
        states, _ = self.gru(self.input_layer(features))
        return self.output_layer(states[:, -1]).squeeze(-1)
