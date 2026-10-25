"""Regresiones de salida temprana real y compatibilidad con el baseline."""

from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
import yaml

from src.models.factory import construir_modelo
from src.utils.seed import fijar_seed


def _config(intermediate: bool = True) -> dict:
    cfg = yaml.safe_load(Path("configs/speech_salidas_t4.yaml").read_text())
    cfg["modelo"].update(entrada=4, capa_dia=4, unidades=8, kernel=2, stride=2)
    cfg["datos"]["sesiones"] = ["day_a", "day_b"]
    if not intermediate:
        cfg["modelo"].pop("salidas_intermedias")
    return cfg


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.linspace(-1, 1, 112).reshape(2, 14, 4), torch.tensor([0, 1])


@pytest.mark.parametrize("depth", [2, 3, 5])
def test_single_exit_executes_only_prefix_and_selected_head(depth):
    model = construir_modelo(_config()).eval()
    recurrent, heads, handles = [], [], []
    for number, layer in enumerate(model.gru_layers, start=1):
        handles.append(layer.register_forward_pre_hook(lambda _m, _a, n=number: recurrent.append(n)))
    for name, head in [*model.intermediate_heads.items(), ("5", model.output_layer)]:
        handles.append(head.register_forward_pre_hook(lambda _m, _a, n=int(name): heads.append(n)))
    with torch.no_grad():
        result = model(*_inputs(), exit_layer=depth)
    for handle in handles:
        handle.remove()
    assert recurrent == list(range(1, depth + 1))
    assert heads == [depth]
    assert result.shape == (2, 7, 41)


def test_all_exits_share_one_recurrent_pass_and_match_individual_profiles():
    model = construir_modelo(_config()).eval()
    called = []
    handles = [layer.register_forward_pre_hook(lambda _m, _a, n=i: called.append(n))
               for i, layer in enumerate(model.gru_layers, start=1)]
    with torch.no_grad():
        together = model.forward_exits(*_inputs())
    for handle in handles:
        handle.remove()
    assert called == [1, 2, 3, 4, 5]
    assert list(together) == [2, 3, 5]
    with torch.no_grad():
        for depth, result in together.items():
            torch.testing.assert_close(result, model(*_inputs(), exit_layer=depth), rtol=0, atol=0)
        torch.testing.assert_close(model(*_inputs()), together[5], rtol=0, atol=0)


def test_early_loss_reaches_prefix_and_does_not_touch_later_layers():
    fijar_seed(0)
    model = construir_modelo(_config()).train()
    model(*_inputs(), exit_layer=2).square().mean().backward()
    assert model.intermediate_heads["2"].weight.grad.abs().sum() > 0
    assert model.gru_layers[0].weight_ih_l0.grad.abs().sum() > 0
    assert model.gru_layers[1].weight_ih_l0.grad.abs().sum() > 0
    assert all(layer.weight_ih_l0.grad is None for layer in model.gru_layers[2:])
    assert model.output_layer.weight.grad is None
    assert model.intermediate_heads["3"].weight.grad is None


def test_adding_heads_preserves_initialization_of_shared_parameters():
    fijar_seed(0)
    original = construir_modelo(_config(False))
    fijar_seed(0)
    expanded = construir_modelo(_config())
    for name, value in original.state_dict().items():
        assert torch.equal(value, expanded.state_dict()[name]), name


def test_baseline_matches_class_frozen_in_training_notebook():
    notebook = json.loads(Path("notebooks/07_baseline_t12_colab.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if "source_bundle" in cell["metadata"].get("tags", []))
    constants = {}
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SOURCE_B64":
                    constants[target.id] = ast.literal_eval(node.value)
    with ZipFile(io.BytesIO(base64.b64decode(constants["SOURCE_B64"]))) as archive:
        source = archive.read("src/models/speech_gru.py")
    namespace = {"__name__": "frozen_baseline"}
    exec(compile(source, "frozen_speech_gru.py", "exec"), namespace)
    cfg = _config(False)
    m = cfg["modelo"]
    fijar_seed(0)
    frozen = namespace["SpeechGRU"](
        input_features=m["entrada"], input_size=m["capa_dia"], hidden_size=m["unidades"],
        num_layers=m["capas"], num_sessions=2, num_classes=41, kernel_size=m["kernel"],
        stride=m["stride"], input_dropout=m["dropout_entrada"], gru_dropout=m["dropout_gru"],
    ).eval()
    fijar_seed(0)
    current = construir_modelo(cfg).eval()
    assert set(frozen.state_dict()) == set(current.state_dict())
    for name, value in frozen.state_dict().items():
        assert torch.equal(value, current.state_dict()[name]), name
    current.load_state_dict(frozen.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(current(*_inputs()), frozen(*_inputs()), rtol=0, atol=0)


def test_baseline_rejects_an_exit_without_a_head():
    model = construir_modelo(_config(False)).eval()
    with pytest.raises(ValueError, match="salidas"):
        model(*_inputs(), exit_layer=2)


def test_final_layer_cannot_be_redeclared_as_intermediate():
    cfg = _config()
    cfg["modelo"]["salidas_intermedias"] = [2, 5]
    with pytest.raises(ValueError, match="anteriores a la final"):
        construir_modelo(cfg)
