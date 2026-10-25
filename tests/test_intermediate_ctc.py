"""Contrato de CTC conjunta, acumulación y selección de checkpoint por capa final."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.data.speech_dataset import SpeechSample, collate_speech
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.speech import (
    _backward_microbatch, _criterion, _ctc_weights, entrenar_habla, evaluar_habla, evaluar_salidas,
)
from src.utils.seed import fijar_seed, crear_generador


def _fixture():
    cfg = yaml.safe_load(Path("configs/speech_salidas_t4.yaml").read_text())
    cfg["modelo"].update(entrada=4, capa_dia=4, unidades=8, kernel=2, stride=2,
                         dropout_entrada=0.0, dropout_gru=0.0)
    cfg["datos"]["sesiones"] = ["day_a", "day_b"]
    cfg["datos"]["batch_size"] = 3
    cfg["entrenamiento"]["micro_batch_size"] = 2
    cfg["entrenamiento"]["ruido"] = {"blanco_sd": 0.0, "offset_sd": 0.0}
    cfg["evaluacion"]["micro_batch_size"] = 2
    rows = [SpeechSample(torch.linspace(-1, 1, length * 4).reshape(length, 4),
                         torch.tensor(ids), index % 2, "reference", index, 1)
            for index, (length, ids) in enumerate(((14, [0, 1]), (12, [1, 1]), (10, [2])))]
    return cfg, rows, collate_speech(rows)


@pytest.mark.parametrize("weights", [
    {"2": 0.5, "5": 0.5}, {"2": 0.25, "3": 0.25, "5": 0.6},
    {"2": -0.25, "3": 0.75, "5": 0.5}, {"2": 0.25, "3": float("nan"), "5": 0.5},
])
def test_invalid_ctc_weights_are_rejected(weights):
    cfg, _, _ = _fixture()
    cfg["entrenamiento"]["ctc_pesos"] = weights
    with pytest.raises(ValueError):
        _ctc_weights(cfg)


def test_joint_loss_and_gradients_match_explicit_weighted_ctc():
    cfg, _, batch = _fixture()
    fijar_seed(0)
    model = construir_modelo(cfg).train()
    reference = copy.deepcopy(model)
    logged = {}
    loss = _backward_microbatch(model, batch, cfg, nn.Identity(), _criterion(cfg),
                               torch.device("cpu"), 3, component_losses=logged)
    outputs = reference.forward_exits(batch.features, batch.session_ids)
    criterion = nn.CTCLoss(blank=40, reduction="none", zero_infinity=False)
    weights = _ctc_weights(cfg)
    parts = {depth: criterion(logits.log_softmax(-1).transpose(0, 1).contiguous(), batch.targets,
                              reference.output_lengths(batch.input_lengths), batch.target_lengths).mean()
             for depth, logits in outputs.items()}
    expected = sum(weights[depth] * value for depth, value in parts.items())
    expected.backward()
    assert loss == pytest.approx(expected.item(), rel=1e-6)
    for depth, value in parts.items():
        assert logged[depth] == pytest.approx(value.item(), rel=1e-6)
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=1e-6)
    assert all(head.weight.grad.abs().sum() > 0 for head in model.intermediate_heads.values())


def test_unequal_microbatches_preserve_joint_loss_and_gradients():
    cfg, _, batch = _fixture()
    fijar_seed(0)
    full = construir_modelo(cfg).train()
    split = copy.deepcopy(full)
    full_loss = _backward_microbatch(full, batch, cfg, nn.Identity(), _criterion(cfg), torch.device("cpu"), 3)
    components = {}
    split_loss = sum(_backward_microbatch(split, small, cfg, nn.Identity(), _criterion(cfg),
                                         torch.device("cpu"), 3, component_losses=components)
                     for small in batch.microbatches(2))
    assert split_loss == pytest.approx(full_loss, rel=1e-6)
    assert split_loss == pytest.approx(sum(_ctc_weights(cfg)[depth] * value for depth, value in components.items()), rel=1e-6)
    for (name, parameter), (_, other) in zip(full.named_parameters(), split.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=1e-6, msg=name)


def test_shared_evaluation_matches_each_independent_exit():
    cfg, rows, _ = _fixture()
    model = construir_modelo(cfg).eval()
    loader = DataLoader(rows, batch_size=2, collate_fn=collate_speech, generator=crear_generador(0))
    together, _ = evaluar_salidas(model, loader, cfg, nn.Identity(), torch.device("cpu"))

    class Profile(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.model = model
            self.depth = depth

        def forward(self, features, sessions):
            return self.model(features, sessions, exit_layer=self.depth)

        def output_lengths(self, lengths):
            return self.model.output_lengths(lengths)

    for depth, metrics in together.items():
        separate = evaluar_habla(Profile(depth), loader, cfg, nn.Identity(), torch.device("cpu"))
        for key, value in metrics.items():
            assert value == pytest.approx(separate[key], rel=1e-6)
        assert metrics["fonemas"] == 5 and metrics["frases"] == 3


def test_checkpoint_selection_uses_final_exit_and_logs_each_head(tmp_path, monkeypatch):
    cfg, _, batch = _fixture()
    cfg["entrenamiento"]["pasos"] = 3
    cfg["evaluacion"]["cada_pasos"] = 1
    model = construir_modelo(cfg).train()
    captured = []
    run = SimpleNamespace(id="joint_contract", summary={}, log=lambda values, step=None: captured.append((step, dict(values))))

    class Loader:
        dataset = SimpleNamespace(normalization_states={}, sessions=cfg["datos"]["sesiones"])

        def __iter__(self):
            return iter([batch, batch, batch])

    values = iter([(0.10, 0.20, 0.50), (0.80, 0.60, 0.40), (0.05, 0.10, 0.45)])
    validation_loader = object()

    def controlled_validation(_model, loader, *_args):
        assert loader is validation_loader
        per = next(values)
        return ({depth: {"per": value, "ctc_loss": 10.0, "frases": 3, "errores": round(value * 20),
                         "fonemas": 20, "completo": 1}
                 for depth, value in zip((2, 3, 5), per, strict=True)}, 0.0)

    monkeypatch.setattr("src.training.speech.evaluar_salidas", controlled_validation)
    monkeypatch.setattr("src.training.checkpoints.preparation_signature", lambda config: config["protocolo"]["sha256"])
    result = entrenar_habla(model, Loader(), validation_loader, cfg, run, nn.Identity(), torch.device("cpu"), tmp_path)
    state = torch.load(tmp_path / "mejor_val.pt", weights_only=True)
    assert state["step"] == 2 and state["selection_exit"] == 5
    assert state["exit_metrics"][2]["per"] == 0.80 and result["per"] == 0.40
    assert state["ctc_weights"] == cfg["entrenamiento"]["ctc_pesos"]
    restored = construir_modelo(cfg)
    load_checkpoint(tmp_path / "mejor_val.pt", restored, cfg, Loader.dataset)
    report = json.loads((tmp_path / "resultado.json").read_text())
    assert report["best_val_by_exit"]["2"]["per"] == 0.80
    assert report["test_evaluated"] is False
    for depth in (2, 3, 5):
        assert any(f"train/capa_{depth}/ctc_loss" in values for _, values in captured)
        assert any(f"val/capa_{depth}/per" in values for _, values in captured)
    for _, values in captured:
        if "train/ctc_loss" in values:
            assert values["train/ctc_loss"] == pytest.approx(sum(
                _ctc_weights(cfg)[depth] * values[f"train/capa_{depth}/ctc_loss"] for depth in (2, 3, 5)), rel=1e-6)
    subprocess.run([sys.executable, "scripts/report_speech_training.py", "--run-dir", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "resumen_salidas.json").read_text())
    assert summary["selection_exit"] == 5 and summary["best_val_by_exit"]["2"]["per"] == 0.80
