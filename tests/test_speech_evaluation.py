"""PER global por recuentos, sin promediar porcentajes de frases o sesiones."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.speech_dataset import SpeechSample, collate_speech
from src.training.speech import evaluar_salidas
from src.utils.seed import crear_generador


class ConstantPhone(nn.Module):
    """Predice siempre un único fonema 0 para conocer los errores de antemano."""

    available_exits = (2, 3, 5)

    def output_lengths(self, lengths):
        return lengths

    def forward_exits(self, features, _sessions, *, layers=None):
        logits = torch.full((*features.shape[:2], 41), -4.0)
        logits[:, :, 0] = 4
        return {depth: logits for depth in self.available_exits}


# Comprueba que el PER suma errores y fonemas de todo el conjunto, sin promediar los porcentajes de cada frase, lote o sesión.
def test_global_per_uses_phone_counts():
    rows = [SpeechSample(torch.zeros(length, 4), torch.tensor(ids), day, "referencia", index, 1)
            for index, (length, ids, day) in enumerate(((14, [0], 0), (12, [1, 2, 3, 4], 1), (10, [0], 0)))]
    cfg = {"datos": {"fonemas": [str(i) for i in range(40)], "pin_memory": False},
           "evaluacion": {"max_lotes": None, "micro_batch_size": 2}}
    loader = DataLoader(rows, batch_size=2, collate_fn=collate_speech, generator=crear_generador(0))
    metrics, _ = evaluar_salidas(ConstantPhone(), loader, cfg, nn.Identity(), torch.device("cpu"))
    for result in metrics.values():
        assert result["errores"] == 4 and result["fonemas"] == 6 and result["frases"] == 3
        assert result["per"] == 4 / 6
