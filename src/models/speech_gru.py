"""GRU del baseline offline de speechBCI, implementada en PyTorch.

La entrada contiene características de 6v ya normalizadas y suavizadas,
en el orden tx1 seguido de spikePow. La salida son logits para CTC.
La preparación de señales y la conversión de texto a fonemas pertenecen
al pipeline de datos, no a este módulo.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpeechGRU(nn.Module):
    """Transformación por sesión, ventanas temporales, GRU y salida de fonemas.

    Reproduce la arquitectura unidireccional de models.GRU y las redes de
    entrada de NeuralSequenceDecoder en speechBCI. El estado inicial de
    la primera GRU es entrenable; los de las demás capas empiezan en cero.
    El dropout de las GRU se aplica a las entradas de cada capa, con una
    máscara compartida en el tiempo, como en Keras 2.7.
    """

    def __init__(
        self,
        *,
        input_features: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_sessions: int,
        num_classes: int,
        kernel_size: int,
        stride: int,
        input_dropout: float,
        gru_dropout: float,
        intermediate_layers: Sequence[int] = (),
    ) -> None:
        """Construye el modelo con los tamaños y probabilidades del YAML."""
        super().__init__()
        dimensions = (
            input_features, input_size, hidden_size, num_layers,
            num_sessions, num_classes, kernel_size, stride,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Los tamaños del modelo deben ser positivos.")
        if not 0 <= input_dropout < 1 or not 0 <= gru_dropout < 1:
            raise ValueError("Las probabilidades de dropout deben estar en [0, 1).")
        intermediate_layers = tuple(intermediate_layers)
        if (any(type(depth) is not int or not 0 < depth < num_layers for depth in intermediate_layers)
                or intermediate_layers != tuple(sorted(set(intermediate_layers)))):
            raise ValueError("Las salidas intermedias deben ser capas anteriores a la final, únicas y ordenadas.")

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_sessions = num_sessions
        self.kernel_size = kernel_size
        self.stride = stride
        self.gru_dropout = gru_dropout
        self.available_exits = intermediate_layers + (num_layers,)

        self.input_layers = nn.ModuleList(
            nn.Linear(input_features, input_size) for _ in range(num_sessions)
        )
        self.input_activation = nn.Softsign()
        self.input_dropout = nn.Dropout(input_dropout)
        self.gru_layers = nn.ModuleList(
            nn.GRU(
                input_size=kernel_size * input_size if index == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            for index in range(num_layers)
        )
        self.initial_state = nn.Parameter(torch.empty(1, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self._initialize_parameters()
        # Crear las cabezas después conserva la inicialización del tronco
        # y las claves originales cuando no se solicitan salidas adicionales.
        self.intermediate_heads = nn.ModuleDict({
            str(depth): nn.Linear(hidden_size, num_classes) for depth in intermediate_layers
        })
        for head in self.intermediate_heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _initialize_parameters(self) -> None:
        """Usa identidad, Glorot, ortogonal y sesgos cero como TensorFlow."""
        for layer in self.input_layers:
            if layer.in_features == layer.out_features:
                nn.init.eye_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in self.gru_layers:
            nn.init.xavier_uniform_(layer.weight_ih_l0)
            # Keras inicializa recurrent_kernel con forma H × 3H. Se
            # inicializa la transpuesta para conservar esa orientación.
            nn.init.orthogonal_(layer.weight_hh_l0.T)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)

        nn.init.xavier_uniform_(self.initial_state)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def output_lengths(self, input_lengths: Tensor) -> Tensor:
        """Calcula floor((T − ventana) / avance) + 1, sin incluir padding."""
        if input_lengths.ndim != 1 or input_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Las longitudes deben ser un vector de enteros.")
        if bool((input_lengths < self.kernel_size).any()):
            raise ValueError("Una secuencia es más corta que la ventana de entrada.")
        return (input_lengths - self.kernel_size) // self.stride + 1

    def _input_features(self, features: Tensor, session_ids: Tensor) -> Tensor:
        """Valida, adapta la sesión y forma ventanas sin ejecutar GRU."""
        if features.ndim != 3 or features.shape[-1] != self.input_features:
            raise ValueError("La entrada debe tener forma batch × tiempo × características.")
        if features.shape[0] == 0 or features.shape[1] < self.kernel_size:
            raise ValueError("El lote está vacío o la señal es más corta que la ventana.")
        if session_ids.shape != (features.shape[0],) or session_ids.dtype != torch.int64:
            raise ValueError("Se necesita un índice int64 de sesión por ejemplo.")
        if session_ids.device != features.device:
            raise ValueError("Las señales y los índices de sesión deben estar en el mismo dispositivo.")
        if bool(((session_ids < 0) | (session_ids >= self.num_sessions)).any()):
            raise ValueError("El índice de sesión no pertenece a este modelo.")

        weights = torch.stack([layer.weight for layer in self.input_layers])
        biases = torch.stack([layer.bias for layer in self.input_layers])
        weights = weights.index_select(0, session_ids)
        biases = biases.index_select(0, session_ids)
        x = torch.bmm(features, weights.transpose(1, 2)) + biases[:, None, :]
        x = self.input_dropout(self.input_activation(x))

        # TensorFlow extract_patches aplana primero el tiempo y después
        # las características: [t0_f0, t0_f1, ..., t1_f0, ...].
        x = x.unfold(1, self.kernel_size, self.stride)
        x = x.permute(0, 1, 3, 2).flatten(start_dim=2)
        return x

    def forward_exits(
        self, features: Tensor, session_ids: Tensor, *, layers: Sequence[int] | None = None,
    ) -> dict[int, Tensor]:
        """Calcula las salidas pedidas en una pasada hasta su mayor profundidad.

        Los índices son desde uno. La última salida usa output_layer; las
        intermedias usan sus propias proyecciones. No ejecuta GRU posteriores
        ni cabezas que no se hayan solicitado. Las longitudes CTC son comunes.
        """
        requested = self.available_exits if layers is None else tuple(layers)
        if (not requested or any(type(depth) is not int or depth not in self.available_exits for depth in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError(f"Selecciona salidas distintas entre {self.available_exits}.")
        maximum = max(requested)
        x = self._input_features(features, session_ids)
        outputs: dict[int, Tensor] = {}

        for depth, layer in enumerate(self.gru_layers, start=1):
            if depth > maximum:
                break
            if self.training and self.gru_dropout > 0:
                mask = F.dropout(torch.ones_like(x[:, :1, :]), p=self.gru_dropout)
                x = x * mask
            state = None
            if depth == 1:
                state = self.initial_state.unsqueeze(0).expand(1, x.shape[0], -1).contiguous()
            x, _ = layer(x, state)
            if depth in requested:
                head = self.output_layer if depth == self.num_layers else self.intermediate_heads[str(depth)]
                outputs[depth] = head(x)
        return outputs

    def forward(self, features: Tensor, session_ids: Tensor, *, exit_layer: int | None = None) -> Tensor:
        """Devuelve logits B × T' × C y detiene el cálculo en la salida elegida.

        Sin exit_layer mantiene la salida final del baseline. Para lotes con
        padding, output_lengths indica los pasos válidos de cualquier salida.
        """
        depth = self.num_layers if exit_layer is None else exit_layer
        return self.forward_exits(features, session_ids, layers=(depth,))[depth]
