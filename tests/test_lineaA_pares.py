"""Línea A: pares de train por ajuste cruzado, sin frases vistas ni test."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from scripts.export_phoneme_pairs import check_train_coverage, pairs_spec, render_markdown, source_jobs
from scripts.run_queue import queue_jobs
from src.data.speech_dataset import SpeechDataset, SpeechSample, collate_speech
from src.data.speech_protocol import fold_blocks, train_fold_spec
from src.models.speech_gru import SpeechGRU
from src.training.speech_pairs import predict_phonemes

ROOT = Path(__file__).resolve().parents[1]
CAUSAL = yaml.safe_load((ROOT / "configs/final/t12_causal.yaml").read_text())
PROTOCOL = json.loads((ROOT / CAUSAL["protocolo"]["manifiesto"]).read_text())


# Un bloque en dos pliegues daría pares de frases que el modelo sí vio.
def test_folds_split_each_session_into_disjoint_contiguous_spans() -> None:
    blocks = [9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    for rotation in range(7):
        parts = [fold_blocks(blocks, 5, fold, rotation=rotation) for fold in range(5)]
        assert sorted(sum(parts, [])) == sorted(blocks)
        assert all(part == list(range(part[0], part[0] + len(part))) for part in parts if part)
        assert sorted(len(part) for part in parts) == [2, 2, 2, 2, 3]
    # La rotación cambia qué pliegue recibe el tramo largo.
    assert {len(fold_blocks(blocks, 5, 0, rotation=rotation)) for rotation in range(5)} == {2, 3}
    with pytest.raises(ValueError, match="todos los bloques"):
        fold_blocks([4], 2, 0)


def test_fold_spec_is_optional_and_strict() -> None:
    assert train_fold_spec(CAUSAL) is None
    assert train_fold_spec({"datos": {"pliegue": {"pliegues": 5, "excluido": 4}}}) == {"pliegues": 5, "excluido": 4}
    for invalid in ({"pliegues": 1, "excluido": 0}, {"pliegues": 5, "excluido": 5}, {"pliegues": 5},
                    {"pliegues": 5, "excluido": 0, "semilla": 1}, {"pliegues": 5.0, "excluido": 0}, [5, 0]):
        with pytest.raises(ValueError, match="datos.pliegue"):
            train_fold_spec({"datos": {"pliegue": invalid}})


# Ningún modelo puede quedarse sin train de un día: su capa de sesión no aprendería nada.
def test_partition_v2_keeps_train_blocks_in_every_session_and_fold() -> None:
    for position, entry in enumerate(PROTOCOL["sessions"]):
        parts = [fold_blocks(entry["train_blocks"], 5, fold, rotation=position) for fold in range(5)]
        assert sorted(sum(parts, [])) == entry["train_blocks"]
        assert all(len(part) < len(entry["train_blocks"]) for part in parts)


PREPARED = ROOT / CAUSAL["datos"]["preparados"] / "manifest.json"


@pytest.mark.skipif(not PREPARED.is_file(), reason="faltan las señales preparadas de la partición v2")
def test_dataset_fold_removes_exactly_the_held_out_blocks_from_train() -> None:
    cfg = copy.deepcopy(CAUSAL)
    cfg["datos"]["pliegue"] = {"pliegues": 5, "excluido": 2}
    keys = lambda dataset: {(row["session_id"], row["trial_index"]) for row in dataset._rows}  # noqa: E731
    train, held, full = SpeechDataset(cfg, "train"), SpeechDataset(cfg, "pliegue"), SpeechDataset(CAUSAL, "train")
    assert not keys(train) & keys(held) and keys(train) | keys(held) == keys(full)
    assert all(len(indices) for indices in train.session_indices)
    assert len(SpeechDataset(cfg, "val")) == PROTOCOL["totals"]["val"]
    fixed = copy.deepcopy(cfg)
    fixed["preprocesamiento"]["normalizacion"] = {"modo": "train_fija"}
    with pytest.raises(ValueError, match="normalización"):
        SpeechDataset(fixed, "train")


# Si los modelos de pliegues usasen otra receta, su ruido no sería el de la referencia.
def test_fold_plan_matches_the_reference_recipe(tmp_path: Path) -> None:
    jobs = queue_jobs(ROOT / "configs/final/cola_pliegues.yaml")
    assert [train_fold_spec(job.config)["excluido"] for job in jobs] == [0, 1, 2, 3, 4]
    assert len({job.config["salida"]["directorio"] for job in jobs}) == 5
    spec = pairs_spec(ROOT / "configs/final/pares_ajuste_cruzado.yaml")
    folds, reference = source_jobs(spec)
    assert len(folds) == 5 and [job.seed for job in reference] == [0, 1, 2, 3, 4]
    plan = yaml.safe_load((ROOT / "configs/final/cola_pliegues.yaml").read_text())
    del plan["variantes"]["t12_causal_pliegue3"]
    plan["variantes"]["t12_causal_pliegue4"]["cambios"]["entrenamiento.pasos"] = 20001
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="exactamente una vez"):
        source_jobs({**spec, "pliegues": str(tmp_path / "plan.yaml")})
    plan["variantes"]["t12_causal_pliegue3"] = {"config": "configs/final/t12_causal.yaml",
                                                "cambios": {"datos.pliegue": {"pliegues": 5, "excluido": 3}}}
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError, match="receta"):
        source_jobs({**spec, "pliegues": str(tmp_path / "plan.yaml")})


def test_train_coverage_rejects_missing_or_repeated_sentences() -> None:
    keys = [(entry["session"], index) for entry in PROTOCOL["sessions"] for index in entry["train"]]
    check_train_coverage(CAUSAL, keys)
    for broken in (keys[1:], keys + keys[:1]):
        with pytest.raises(ValueError, match="exactamente una vez"):
            check_train_coverage(CAUSAL, broken)


class _Rows(list):
    """Lista de ensayos con el atributo split que tiene SpeechDataset."""

    def __init__(self, rows: list[SpeechSample], split: str) -> None:
        super().__init__(rows)
        self.split = split


# Los pares deben salir de la misma pasada causal y voraz que mide el PER de val.
def test_predictions_are_greedy_ctc_of_the_final_exit() -> None:
    torch.manual_seed(0)
    model = SpeechGRU(input_features=4, input_size=4, hidden_size=8, num_layers=3, num_sessions=2, num_classes=5,
                      kernel_size=3, stride=2, input_dropout=0.0, gru_dropout=0.0, intermediate_layers=(2,))
    cfg = {"datos": {"fonemas": ["AA", "B", "K", "SIL"], "sesiones": ["dia_a", "dia_b"]},
           "evaluacion": {"micro_batch_size": 2}}
    rows = [SpeechSample(torch.randn(length, 4), torch.tensor(ids), index % 2, f"frase {index}", 10 + index, 3)
            for index, (length, ids) in enumerate(((15, [0, 3, 1]), (9, [2]), (12, [1, 1])))]
    loader = DataLoader(_Rows(rows, "pliegue"), batch_size=3, collate_fn=collate_speech)
    pairs = list(predict_phonemes(model, loader, cfg, nn.Identity(), torch.device("cpu")))
    assert [pair["split"] for pair in pairs] == ["train"] * 3
    assert [pair["source_trial_index"] for pair in pairs] == [10, 11, 12]
    assert pairs[0]["reference_phones"] == "AA SIL B" and pairs[1]["session"] == "dia_b"
    with torch.no_grad():
        logits = model.forward_exits(rows[0].features[None], torch.tensor([0]), layers=(3,))[3]
    from src.inference.ctc import decode_ctc
    expected = decode_ctc(logits, model.output_lengths(torch.tensor([15])), 4)[0]
    assert pairs[0]["phones"] == " ".join(cfg["datos"]["fonemas"][phone] for phone in expected)
    with pytest.raises(ValueError, match="test"):
        next(predict_phonemes(model, DataLoader(_Rows(rows, "test"), collate_fn=collate_speech), cfg,
                              nn.Identity(), torch.device("cpu")))


def test_markdown_lists_every_source_and_split() -> None:
    source = {"job": "t12_causal_pliegue0_seed0", "fold": 0, "split": "train", "rows": 1400,
              "per": 0.3, "val_per_training": 0.25}
    manifest = {"protocol": "t12_bloques_v2", "sources": [source, {**source, "job": "t12_causal_seed0", "fold": None,
                                                                    "split": "val", "rows": 1900}],
                "totals": {"train": {"rows": 1400, "per": 0.3}, "val": {"rows": 1900, "per": 0.3}}}
    markdown = render_markdown(manifest)
    assert "| t12_causal_pliegue0_seed0 | 0 | train | 1400 | 30,00 | 25,00 |" in markdown
    assert "| t12_causal_seed0 | — | val | 1900 |" in markdown
