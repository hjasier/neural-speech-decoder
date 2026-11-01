"""Formas CTC e independencia del lote en la GRU del proyecto."""

import torch

from src.models.factory import construir_modelo


# Comprueba que las tres salidas tienen el tamaño esperado y que las longitudes para CTC no cuentan el relleno.
def test_forward_devuelve_la_forma_esperada(cfg_habla_pequena: dict) -> None:
    """Cada salida conserva batch, ventanas temporales y 40 fonemas más blank."""
    modelo = construir_modelo(cfg_habla_pequena).eval()
    with torch.no_grad():
        salidas = modelo.forward_exits(torch.randn(4, 14, 4), torch.tensor([0, 1, 0, 1]))
    assert set(salidas) == {2, 3, 5}
    assert all(salida.shape == (4, 6, 41) for salida in salidas.values())
    # Excluir padding y bins sin ventana completa al calcular longitudes CTC.
    assert modelo.output_lengths(torch.tensor([14, 11])).tolist() == [6, 5]


# Comprueba que una señal da el mismo resultado sola que dentro de un lote, sin mezclarse con las demás.
def test_el_batch_no_se_mezcla(cfg_habla_pequena: dict) -> None:
    """Las tres salidas coinciden al procesar juntas o separadas ambas sesiones."""
    modelo = construir_modelo(cfg_habla_pequena).eval()
    x = torch.randn(4, 14, 4)
    sesiones = torch.tensor([0, 1, 0, 1])
    with torch.no_grad():
        # Distinguir adaptadores que inicialmente son iguales.
        modelo.input_layers[1].weight.mul_(0.5)
        modelo.input_layers[1].bias.fill_(0.2)
        juntas = modelo.forward_exits(x, sesiones)
        sueltas = [modelo.forward_exits(x[i:i + 1], sesiones[i:i + 1]) for i in range(4)]
    for capa, salida in juntas.items():
        torch.testing.assert_close(salida, torch.cat([s[capa] for s in sueltas]), atol=1e-6, rtol=1e-5)
