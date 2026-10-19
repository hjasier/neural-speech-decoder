"""Adam con la convención de epsilon del TensorFlow 2.7 de speechBCI."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn


class TensorFlowAdam(torch.optim.Optimizer):
    """Conserva epsilon-hat y la corrección por paso global del original.

    En PyTorch Adam epsilon se suma después de corregir el segundo momento.
    Con el epsilon grande del baseline, sustituirlo directamente cambia la
    actualización. El contador global avanza una vez por lote.
    """

    def __init__(
        self, params: Iterable[nn.Parameter], *, lr: float,
        betas: tuple[float, float], eps: float,
    ) -> None:
        """Recibe todos los hiperparámetros de la configuración del experimento."""
        if lr < 0 or eps <= 0 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("Parámetros de Adam no válidos.")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, iterations=0))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Aplica m/(sqrt(v)+epsilon) con el escalado del paso global de TF."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["iterations"] += 1
            iteration = group["iterations"]
            beta1, beta2 = group["betas"]
            alpha = group["lr"] * math.sqrt(1 - beta2 ** iteration) / (1 - beta1 ** iteration)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("Este Adam solo admite gradientes densos.")
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.addcdiv_(mean, variance.sqrt().add_(group["eps"]), value=-alpha)
        return loss
