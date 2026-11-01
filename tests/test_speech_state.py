"""Continuidad de la memoria recurrente e independencia entre flujos."""

import pytest
import torch

from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed


# Comprueba que dividir las ventanas no pierde ni reinicia la memoria de ninguna capa.
@pytest.mark.parametrize("layers", [(2,), (3,), (2, 3, 5)])
def test_window_groups_preserve_recurrent_state(
    cfg_habla_pequena: dict, layers: tuple[int, ...],
) -> None:
    """Las salidas coinciden con la frase completa aunque un bloque tenga una ventana."""
    fijar_seed(0)
    model = construir_modelo(cfg_habla_pequena).eval()
    features = torch.randn(2, 37, 4)
    sessions = torch.tensor([0, 1])
    with torch.inference_mode():
        expected = model.forward_exits(features, sessions, layers=layers)
        windows = model.prepare_windows(features, sessions)
        states = None
        pieces = {depth: [] for depth in layers}
        for block in torch.tensor_split(windows, [1, 4, 9], dim=1):
            outputs, states = model.forward_windows(block, states, layers=layers)
            for depth in layers:
                pieces[depth].append(outputs[depth])
    assert len(states) == max(layers)
    assert all(state.shape == (1, 2, model.hidden_size) for state in states)
    for depth in layers:
        torch.testing.assert_close(torch.cat(pieces[depth], dim=1), expected[depth],
                                   atol=1e-6, rtol=1e-5)


# Comprueba que intercalar dos frases no mezcla sus memorias ni deja estado oculto al iniciar otra.
def test_interleaved_streams_keep_independent_states(cfg_habla_pequena: dict) -> None:
    """Cada flujo conserva su contexto aunque otro use el mismo modelo entre bloques."""
    fijar_seed(0)
    model = construir_modelo(cfg_habla_pequena).eval()
    features = torch.randn(2, 27, 4)
    sessions = torch.tensor([0, 1])
    with torch.inference_mode():
        # Los dos adaptadores deben ser distintos para detectar mezclas de sesión.
        model.input_layers[1].weight.mul_(0.5)
        model.input_layers[1].bias.fill_(0.2)
        expected = model.forward_exits(features, sessions)
        windows = model.prepare_windows(features, sessions)
        first_a, state_a = model.forward_windows(windows[:1, :3])
        first_b, state_b = model.forward_windows(windows[1:, :5])
        rest_a, _ = model.forward_windows(windows[:1, 3:], state_a)
        rest_b, _ = model.forward_windows(windows[1:, 5:], state_b)
        restarted, _ = model.forward_windows(windows)
    for depth in model.available_exits:
        actual = torch.cat((torch.cat((first_a[depth], rest_a[depth]), dim=1),
                            torch.cat((first_b[depth], rest_b[depth]), dim=1)), dim=0)
        torch.testing.assert_close(actual, expected[depth], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(restarted[depth], expected[depth], atol=1e-6, rtol=1e-5)
