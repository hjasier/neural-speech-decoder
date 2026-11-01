"""Orden de canales y correspondencia entre señales, fonemas y longitudes."""

import numpy as np
import torch

from src.data.preprocessing import extract_model_features
from src.data.speech import SpeechTrial
from src.data.speech_dataset import SpeechSample, collate_speech


# Comprueba que usamos los canales correctos y colocamos primero tx1 y después spikePow, sin cambiar su orden.
def test_features_keep_tx1_before_spike_power_and_preserve_channel_order() -> None:
    """Intercambiar modalidades o usar los canales de otra área no cambia la forma."""
    tx1 = np.arange(2 * 256, dtype=np.float32).reshape(2, 256)
    power = tx1 + 10000
    trial = SpeechTrial(0, 1, "referencia", power, tx1)
    features = extract_model_features(trial)
    assert features.shape == (2, 256)
    np.testing.assert_array_equal(features[:, :128], tx1[:, :128])
    np.testing.assert_array_equal(features[:, 128:], power[:, :128])


# Comprueba que al juntar frases de distinta duración cada señal conserva sus fonemas y su longitud, sin confundir el fonema 0 con el relleno.
def test_collate_preserves_trials_targets_zero_phone_and_padding_lengths() -> None:
    """El padding de señales no añade fonemas ni cambia la correspondencia del lote."""
    a = SpeechSample(torch.arange(16).reshape(4, 4).float(), torch.tensor([0, 2]), 1, "a", 7, 3)
    b = SpeechSample(torch.arange(8).reshape(2, 4).float() + 100, torch.tensor([3]), 0, "b", 8, 4)
    batch = collate_speech([a, b])
    assert batch.features.shape == (2, 4, 4)
    torch.testing.assert_close(batch.features[0], a.features)
    torch.testing.assert_close(batch.features[1, :2], b.features)
    assert torch.count_nonzero(batch.features[1, 2:]) == 0
    assert batch.targets.tolist() == [0, 2, 3]
    assert batch.input_lengths.tolist() == [4, 2]
    assert batch.target_lengths.tolist() == [2, 1]
    assert batch.session_ids.tolist() == [1, 0]
    assert batch.references == ("a", "b") and batch.trial_indices == (7, 8)
    assert batch.blocks == (3, 4)
