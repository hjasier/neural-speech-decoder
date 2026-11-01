"""Un orden de clases incorrecto puede producir palabras sin lanzar errores."""

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.inference.trigram import TrigramStream, kaldi_columns


def test_trigram_receives_all_logits_with_blank_and_sil_first() -> None:
    """Conserva todas las alternativas y respeta los símbolos del checkpoint."""
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/entrenos_online/speech_trigrama.yaml").read_text())
    neural = yaml.safe_load((root / "configs/entrenos_online/speech_online.yaml").read_text())
    phones = list(reversed(neural["datos"]["fonemas"]))
    columns = kaldi_columns(phones, len(phones), spec["fonemas_grafo"])
    requests = []

    def request(payload: dict) -> dict:
        requests.append(payload)
        return {"candidates": []}

    stream = TrigramStream(SimpleNamespace(columns=columns, _open_flow=lambda: 1, _request=request))
    logits = torch.arange(-41, 41).reshape(2, 41).float()
    stream.push(logits)
    actual = np.frombuffer(base64.b64decode(requests[0]["logits"]), dtype="<f4").reshape(requests[0]["shape"])
    # En este checkpoint invertido: blank=40, SIL=0, AA=39, ..., ZH=1.
    expected = logits[:, [40, 0, *range(39, 0, -1)]].numpy()
    np.testing.assert_array_equal(actual, expected)
