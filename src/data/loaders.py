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


class DatasetPlaceholder(Dataset):
    """Sustituye esto por el dataset de tu paper.

    Se deja aqui un dataset sintetico para que `python run.py --smoke` funcione
    desde el minuto cero, antes de que hayas escrito nada. Es lo que permite que
    tengas el vertical slice en verde el primer dia.
    """

    def __init__(self, n: int = 512, dim: int = 32, clases: int = 10):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, dim, generator=g)
        self.y = torch.randint(0, clases, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def construir_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Devuelve train/evaluación; para habla, evaluación es el test publicado."""
    dcfg = cfg["datos"]

    if cfg["modelo"]["nombre"] == "speech_gru":
        from src.data.speech_dataset import SpeechDataset, SessionBatchSampler, collate_speech
        from src.utils.seed import crear_generador

        train_ds = SpeechDataset(cfg, "train")
        test_ds = SpeechDataset(cfg, "test")
        sampler = SessionBatchSampler(
            train_ds.session_indices, dcfg["batch_size"], cfg["entrenamiento"]["pasos"],
            crear_generador(cfg["experimento"]["seed"]),
        )
        common = dict(
            num_workers=dcfg["num_workers"], pin_memory=dcfg["pin_memory"], collate_fn=collate_speech,
        )
        return (
            DataLoader(
                train_ds, batch_sampler=sampler,
                generator=crear_generador(cfg["experimento"]["seed"]), **common,
            ),
            DataLoader(
                test_ds, batch_size=dcfg["batch_size"], shuffle=False,
                generator=crear_generador(cfg["experimento"]["seed"]), **common,
            ),
        )

    train_ds: Dataset = DatasetPlaceholder()
    val_ds: Dataset = DatasetPlaceholder(n=128)

    # Regla 5/20: el subset se aplica DESPUES del split, para no cambiar la
    # distribucion de validacion.
    if dcfg.get("subset"):
        n = int(dcfg["subset"])
        train_ds = Subset(train_ds, range(min(n, len(train_ds))))

    comun = dict(batch_size=dcfg["batch_size"],
                 num_workers=dcfg.get("num_workers", 0))
    return (DataLoader(train_ds, shuffle=True, **comun),
            DataLoader(val_ds, shuffle=False, **comun))
