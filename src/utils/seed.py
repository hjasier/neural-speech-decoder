"""Semillas aleatorias, fijadas en UN SOLO SITIO.

Si te encuentras llamando a manual_seed en tres ficheros distintos, es que
esta funcion no se esta usando. El checker avisa de eso.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def crear_generador(seed: int) -> torch.Generator:
    """Crea un RNG CPU independiente para muestreo o workers, con semilla YAML."""
    return torch.Generator().manual_seed(seed)


def crear_generador_numpy(seed: int) -> np.random.Generator:
    """Crea un RNG independiente para controles estadísticos con semilla YAML."""
    return np.random.default_rng(seed)


def fijar_seed(seed: int, determinista: bool = True) -> None:
    """Fija las semillas de las tres fuentes de aleatoriedad del proyecto.

    Args:
        seed: la semilla. Va en configs/, no aqui.
        determinista: fuerza los kernels deterministas de cuDNN. Cuesta algo de
            velocidad; a la escala de la regla 5/20 no se nota y a cambio dos
            ejecuciones dan el mismo numero.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def capturar_azar(device: torch.device) -> dict[str, Any]:
    """Captura RNG en tipos compatibles con torch.load(weights_only=True)."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restaurar_azar(state: dict[str, Any], device: torch.device) -> None:
    """Restaura los RNG después de construir modelo, optimizador y loaders."""
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("La reanudación requiere el mismo número de dispositivos CUDA.")
        torch.cuda.set_rng_state_all(state["cuda"])
    elif state["cuda"]:
        raise ValueError("Este estado requiere CUDA; no se reanuda en CPU.")
