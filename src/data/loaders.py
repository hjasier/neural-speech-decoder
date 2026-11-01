"""Carga de datos y splits.

REGLAS QUE NO SE ROMPEN (y que el revisor del PR comprueba):
  1. El split se hace ANTES de cualquier transformacion ajustada a los datos.
  2. La normalizacion usa estadisticas de ENTRENAMIENTO, nunca del conjunto
     completo. Lo contrario es una fuga: no da error e infla tu metrica.
  3. El data augmentation se aplica a train, jamas a validacion ni a test.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.seed import crear_generador


class DatosSmoke(Dataset):
    """Ejemplos sintéticos para comprobar el lanzador sin descargar datos."""

    def __init__(self, n: int = 512, dim: int = 32, clases: int = 10) -> None:
        """Genera señales y clases sintéticas con una semilla fija."""
        g = crear_generador(0)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, clases, (n,), generator=g)

    def __len__(self) -> int:
        """Devuelve el número de ejemplos sintéticos."""
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve un ejemplo y su clase."""
        return self.x[i], self.y[i]


def construir_loaders(
    cfg: dict, *, split_evaluacion: str | None = None
) -> tuple[DataLoader, DataLoader]:
    """Devuelve train/evaluación; T12 con protocolo usa val y reserva test."""
    dcfg = cfg["datos"]

    if cfg["modelo"]["nombre"] == "speech_gru":
        from src.data.speech_dataset import (
            SpeechDataset,
            SessionBatchSampler,
            collate_speech,
        )

        train_ds = SpeechDataset(cfg, "train")
        evaluation = split_evaluacion or ("val" if cfg.get("protocolo") else "test")
        evaluation_ds = SpeechDataset(cfg, evaluation)
        sampler = SessionBatchSampler(
            train_ds.session_indices,
            dcfg["batch_size"],
            cfg["entrenamiento"]["pasos"],
            crear_generador(cfg["experimento"]["seed"]),
            resumable=bool(cfg.get("reanudacion")),
        )
        common = dict(
            num_workers=dcfg["num_workers"],
            pin_memory=dcfg["pin_memory"],
            collate_fn=collate_speech,
        )
        return (
            DataLoader(
                train_ds,
                batch_sampler=sampler,
                generator=crear_generador(cfg["experimento"]["seed"]),
                **common,
            ),
            DataLoader(
                evaluation_ds,
                batch_size=dcfg["batch_size"],
                shuffle=False,
                generator=crear_generador(cfg["experimento"]["seed"]),
                **common,
            ),
        )

    train_ds: Dataset = DatosSmoke()
    val_ds: Dataset = DatosSmoke(n=128)

    # El recorte afecta solo a train; validación conserva todos sus ejemplos.
    if dcfg.get("subset"):
        n = int(dcfg["subset"])
        train_ds = Subset(train_ds, range(min(n, len(train_ds))))

    comun = dict(batch_size=dcfg["batch_size"], num_workers=dcfg.get("num_workers", 0))
    return (
        DataLoader(train_ds, shuffle=True, **comun),
        DataLoader(val_ds, shuffle=False, **comun),
    )
