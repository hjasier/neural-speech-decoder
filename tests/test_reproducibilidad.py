"""Inicialización reproducible de la GRU y sus cabezas intermedias."""

import torch

from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed


# Comprueba que al crear dos modelos con la misma semilla sus pesos iniciales son exactamente iguales.
def test_la_misma_seed_da_los_mismos_pesos(cfg_habla_pequena: dict) -> None:
    """La misma semilla reproduce todas las claves y pesos del modelo de habla."""
    fijar_seed(42)
    a = construir_modelo(cfg_habla_pequena).state_dict()
    fijar_seed(42)
    b = construir_modelo(cfg_habla_pequena).state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[nombre], b[nombre]) for nombre in a)
