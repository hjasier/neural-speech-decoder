"""Bucle de entrenamiento y evaluacion."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.tracking import registrar

OPTIMIZADORES = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def entrenar(
    modelo: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    run=None,
) -> nn.Module:
    """Entrena el clasificador sintético durante los pasos del smoke."""
    tcfg = cfg["entrenamiento"]
    opt = OPTIMIZADORES[tcfg["optimizador"]](
        modelo.parameters(), lr=tcfg["learning_rate"]
    )
    criterio = nn.CrossEntropyLoss()
    max_steps = tcfg.get("max_steps")

    paso = 0
    for epoca in range(tcfg["epochs"]):
        modelo.train()
        for x, y in train_loader:
            opt.zero_grad()
            perdida = criterio(modelo(x), y)
            perdida.backward()
            opt.step()

            paso += 1
            # Registrar escalares evita conservar el grafo de cada paso.
            registrar(run, {"train/loss": perdida.item()}, paso)
            if max_steps and paso >= max_steps:
                return modelo

        metricas = evaluar(modelo, val_loader, cfg)
        registrar(run, {f"val/{k}": v for k, v in metricas.items()}, paso)
        print(
            f"epoca {epoca + 1}/{tcfg['epochs']}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metricas.items())
        )
    return modelo


@torch.no_grad()
def evaluar(modelo: nn.Module, loader: DataLoader, cfg: dict) -> dict[str, float]:
    """Calcula pérdida y aciertos sin construir gradientes."""
    modelo.eval()
    criterio = nn.CrossEntropyLoss()
    perdida_total, aciertos, total = 0.0, 0, 0

    for x, y in loader:
        logits = modelo(x)
        perdida_total += criterio(logits, y).item() * y.size(0)
        aciertos += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return {"loss": perdida_total / max(total, 1), "accuracy": aciertos / max(total, 1)}
