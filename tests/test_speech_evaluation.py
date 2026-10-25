"""Controles emparejados y PER agregado por fonemas, sin interpretar calidad."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.speech_dataset import SpeechSample, collate_speech
from src.training.speech import evaluar_salidas
from src.training.speech_evaluation import evaluate_signal_controls, signal_control
from src.utils.seed import crear_generador


class Rows(Dataset):
    split = "val"

    def __init__(self):
        self.rows = [SpeechSample(torch.arange(length * 4).reshape(length, 4).float() + index * 1000,
                                  torch.tensor(ids), day, f"reference {index}", index, day + 1)
                     for index, (length, ids, day) in enumerate(((14, [0], 0), (12, [1, 2, 3, 4], 1), (10, [0], 0)))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ConstantPhone(nn.Module):
    def __init__(self, exits):
        super().__init__()
        self.available_exits = exits

    def output_lengths(self, lengths):
        return lengths

    def forward_exits(self, features, _sessions, *, layers=None):
        logits = torch.full((*features.shape[:2], 41), -4.0, device=features.device)
        logits[:, :, 0] = 4
        return {depth: logits for depth in (self.available_exits if layers is None else layers)}


def _cfg():
    return {"datos": {"sesiones": ["day_a", "day_b"], "fonemas": [str(i) for i in range(40)], "pin_memory": False},
            "evaluacion": {"max_lotes": None, "micro_batch_size": 2}}


def _spec():
    return {"split": "val", "salidas": None, "controles": ["original", "cero", "permutada"], "semilla_control": 314159}


def test_controls_preserve_labels_padding_channels_and_independent_trial_permutations():
    rows = Rows()
    batch = collate_speech(rows.rows)
    original = batch.features.clone()
    assert signal_control(batch, "original", 1, "val") is batch
    zero = signal_control(batch, "cero", 1, "val")
    assert torch.count_nonzero(zero.features) == 0 and zero.targets is batch.targets
    permuted = signal_control(batch, "permutada", 314159, "val")
    repeat = signal_control(batch, "permutada", 314159, "val")
    torch.testing.assert_close(permuted.features, repeat.features, rtol=0, atol=0)
    torch.testing.assert_close(batch.features, original, rtol=0, atol=0)
    assert permuted.targets is batch.targets and permuted.references == batch.references
    for i, length in enumerate(batch.input_lengths.tolist()):
        assert sorted(map(tuple, permuted.features[i, :length].tolist())) == sorted(map(tuple, original[i, :length].tolist()))
        assert torch.equal(permuted.features[i, length:], original[i, length:])
    assert not torch.equal(permuted.features, original)
    for i, row in enumerate(rows.rows):
        single = signal_control(collate_speech([row]), "permutada", 314159, "val")
        assert torch.equal(single.features[0], permuted.features[i, :len(row.features)])
    reordered = signal_control(collate_speech([rows.rows[2], rows.rows[0], rows.rows[1]]), "permutada", 314159, "val")
    for i, previous in enumerate((2, 0, 1)):
        length = len(rows.rows[previous].features)
        assert torch.equal(reordered.features[i, :length], permuted.features[previous, :length])


@pytest.mark.parametrize("exits", [(2,), (3,), (5,), (2, 3, 5)])
def test_global_per_uses_phone_counts_and_matches_independent_aggregation(exits):
    cfg, spec = _cfg(), _spec()
    loader = DataLoader(Rows(), batch_size=3, collate_fn=collate_speech, generator=crear_generador(0))
    model = ConstantPhone(exits)
    predictions = []
    result = evaluate_signal_controls(model, loader, cfg, spec, nn.Identity(), torch.device("cpu"),
                                      prediction_sink=predictions.append)
    assert len(predictions) == 3 and result["test_evaluated"] is False
    existing, _ = evaluar_salidas(model, loader, cfg, nn.Identity(), torch.device("cpu"))
    for control in spec["controles"]:
        for depth in exits:
            total = result["global"][control][depth]
            sessions = result["by_session"][control][depth]
            assert total["errores"] == sum(value["errores"] for value in sessions.values()) == 4
            assert total["fonemas"] == sum(value["fonemas"] for value in sessions.values()) == 6
            assert total["frases"] == sum(value["frases"] for value in sessions.values()) == 3
            assert total["per"] == 4 / 6
            assert sessions["day_a"]["per"] == 0 and sessions["day_b"]["per"] == 1
            assert total["per"] != sum(value["per"] for value in sessions.values()) / len(sessions)
            assert total["ctc_loss"] == pytest.approx(existing[depth]["ctc_loss"], rel=1e-6)
            for row in predictions:
                assert row["predictions"][control][depth]["ids"] == [0]


def test_partial_or_wrong_split_cannot_be_reported_as_complete():
    cfg, spec = _cfg(), _spec()
    model = ConstantPhone((5,))
    loader = DataLoader(Rows(), batch_size=2, drop_last=True, collate_fn=collate_speech, generator=crear_generador(0))
    with pytest.raises(ValueError, match="completa"):
        evaluate_signal_controls(model, loader, cfg, spec, nn.Identity(), torch.device("cpu"))
    wrong = copy.deepcopy(spec)
    wrong["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        evaluate_signal_controls(model, loader, cfg, wrong, nn.Identity(), torch.device("cpu"))
