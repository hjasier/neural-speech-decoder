"""Contrato de CTC conjunta, acumulación y selección de checkpoint por capa final."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from src.data.speech_dataset import SpeechSample, collate_speech
from src.models.factory import construir_modelo
from src.training.checkpoints import load_checkpoint
from src.training.speech import (
    _backward_microbatch, _criterion, _ctc_weights, entrenar_habla,
)
from src.utils.seed import fijar_seed


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


# Comprueba que la pérdida conjunta y sus gradientes coinciden con calcular la CTC de cada salida y aplicar sus pesos.
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
    weights = {int(depth): weight for depth, weight in cfg["entrenamiento"]["ctc_pesos"].items()}
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


# Comprueba que dividir un lote en microlotes, aunque el último sea más pequeño, conserva la pérdida y los gradientes del lote completo.
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


# Comprueba que elegimos los pesos por el PER de la salida final y conservamos el primer mínimo si hay empate, aunque otra salida mejore después.
def test_joint_checkpoint_is_selected_by_final_exit(tmp_path, monkeypatch):
    cfg, _, batch = _fixture()
    cfg["entrenamiento"]["pasos"] = 4
    cfg["evaluacion"]["cada_pasos"] = 1
    model = construir_modelo(cfg).train()
    run = SimpleNamespace(id="joint_contract", summary={}, log=lambda *_args, **_kwargs: None)

    class Loader:
        dataset = SimpleNamespace(normalization_states={}, sessions=cfg["datos"]["sesiones"])

        def __iter__(self):
            return iter([batch] * 4)

    values = iter([(0.10, 0.20, 0.50), (0.80, 0.60, 0.40), (0.05, 0.10, 0.45), (0.01, 0.02, 0.40)])
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
    assert all(torch.equal(restored.state_dict()[name], value) for name, value in state["model_state"].items())
    report = json.loads((tmp_path / "resultado.json").read_text())
    assert report["best_val_by_exit"]["2"]["per"] == 0.80
    assert report["test_evaluated"] is False
