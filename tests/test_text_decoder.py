"""Línea A: el decodificador de texto se entrena con train, se elige por val y mide WER como el trigrama."""

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

from scripts.train_text_decoder import load_config, render_markdown  # noqa: E402
from src.data.speech_dataset import file_sha256  # noqa: E402
from src.training.speech_trigram_evaluation import word_error_counts  # noqa: E402
from src.training.text_decoder import (  # noqa: E402
    collate_pairs, encode_pairs, final_report, generate_texts, learning_rate_factor, read_pairs, score_texts,
    target_text, train_text_decoder,
)

ROOT = Path(__file__).resolve().parents[1]
PHRASES = [("HH AH L OW SIL", "Hello."), ("W ER L D SIL", "world"), ("HH AH L OW SIL W ER L D SIL", "Hello, world!"),
           ("G UH D SIL", "good"), ("G UH D SIL W ER L D SIL", "Good world.")]


def _tokenizer() -> transformers.PreTrainedTokenizerFast:
    """Tokenizador de palabras completo en memoria: no descarga nada."""
    words = sorted({token for phones, text in PHRASES for token in (phones.split() + target_text(text).split())})
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "fonemas": 3, "a": 4, "texto:": 5,
             **{word: 6 + index for index, word in enumerate(words)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    core.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", eos_token="</s>",
                                                unk_token="<unk>")


def _model(tokenizer: transformers.PreTrainedTokenizerFast) -> transformers.T5ForConditionalGeneration:
    torch.manual_seed(0)
    config = transformers.T5Config(vocab_size=len(tokenizer), d_model=32, d_ff=64, d_kv=16, num_layers=1,
                                   num_decoder_layers=1, num_heads=2, dropout_rate=0.0, pad_token_id=0,
                                   eos_token_id=1, decoder_start_token_id=0)
    return transformers.T5ForConditionalGeneration(config)


def _rows(split: str, model: str = "t12_causal_pliegue0_seed0", seed: int = 0) -> list[dict]:
    return [{"split": split, "session": "dia", "block": 3, "source_trial_index": index, "phones": phones,
             "reference_text": text, "model": model, "seed": seed, "fold": 0}
            for index, (phones, text) in enumerate(PHRASES)]


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config(ROOT / "configs/final/decodificador_t5.yaml")
    cfg["datos"].update(pares=str(tmp_path), max_tokens_entrada=16, max_tokens_salida=8)
    cfg["entrenamiento"].update(pasos=40, batch_size=2, learning_rate=1e-2, calentamiento_pasos=4, imprimir_cada=10)
    cfg["evaluacion"].update(cada_pasos=10, batch_size=4, num_beams_final=2)
    cfg["bootstrap"]["remuestreos"] = 200
    cfg["ejecucion"]["dispositivo"] = "cpu"
    return cfg


def _write_pairs(root: Path, train: list[dict], val: list[dict]) -> None:
    for split, rows in (("train", train), ("val", val)):
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    files = {name: file_sha256(root / name) for name in ("train.jsonl", "val.jsonl")}
    (root / "manifest.json").write_text(json.dumps({"files": files}))


# El objetivo debe limpiarse como el WER del trigrama; si no, la comparación no es justa.
def test_target_and_score_use_the_trigram_word_cleanup() -> None:
    assert target_text("Hello, World!  It's done.") == "hello world it's done"
    rows = _rows("val") + _rows("val", model="t12_causal_seed1", seed=1)
    hypotheses = ["hello", "word", "hello world", "good", "good"] * 2
    score = score_texts(rows, hypotheses)
    expected = [word_error_counts(row["reference_text"], text) for row, text in zip(rows, hypotheses)]
    assert score["conteos"] == expected
    assert score["errores"] == sum(item[0] for item in expected) and score["palabras"] == 14
    assert set(score["por_modelo"]) == {"t12_causal_pliegue0_seed0", "t12_causal_seed1"}
    assert score["por_modelo"]["t12_causal_seed1"]["palabras"] == 7


def test_pairs_are_checked_against_their_manifest(tmp_path: Path) -> None:
    _write_pairs(tmp_path, _rows("train"), _rows("val"))
    pairs, _ = read_pairs({"datos": {"pares": str(tmp_path)}})
    assert [len(pairs[split]) for split in ("train", "val")] == [5, 5]
    (tmp_path / "val.jsonl").write_text((tmp_path / "val.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="han cambiado"):
        read_pairs({"datos": {"pares": str(tmp_path)}})
    _write_pairs(tmp_path, _rows("train"), _rows("test"))
    with pytest.raises(ValueError, match="solo frases de val"):
        read_pairs({"datos": {"pares": str(tmp_path)}})


# Truncar en silencio haría que el modelo viese media frase y el WER no lo diría.
def test_encoding_rejects_overlong_sentences_and_pads_labels_with_ignore(tmp_path: Path) -> None:
    tokenizer, cfg = _tokenizer(), _cfg(tmp_path)
    encoded = encode_pairs(tokenizer, _rows("train"), cfg)
    assert encoded[0]["labels"] == tokenizer("hello")["input_ids"]
    batch = collate_pairs(encoded[:2], tokenizer.pad_token_id)
    assert batch["labels"].eq(-100).sum() == 0 and batch["attention_mask"][0].sum() == len(encoded[0]["input_ids"])
    batch = collate_pairs([encoded[0], encoded[2]], tokenizer.pad_token_id)
    assert batch["labels"][0, len(encoded[0]["labels"]):].eq(-100).all()
    cfg["datos"]["max_tokens_entrada"] = 9
    with pytest.raises(ValueError, match="max_tokens_entrada"):
        encode_pairs(tokenizer, _rows("train"), cfg)


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    assert [learning_rate_factor(step, 4, 10) for step in (0, 3, 4, 7, 10)] == [0.25, 1.0, 1.0, 0.5, 0.0]


# El modelo que queda cargado debe ser el del paso con menor WER de val, no el último.
def test_training_keeps_the_best_validation_step(tmp_path: Path) -> None:
    tokenizer, cfg = _tokenizer(), _cfg(tmp_path)
    model = _model(tokenizer)
    pairs = {"train": _rows("train"), "val": _rows("val") + _rows("val", model="t12_causal_seed1", seed=1)}
    lines = []
    result = train_text_decoder(model, tokenizer, pairs, cfg, None, torch.device("cpu"), log=lines.append)
    assert result["completed_steps"] == 40 and [item["step"] for item in result["history"]] == [10, 20, 30, 40]
    best = min(result["history"], key=lambda item: item["wer"])
    assert result["best_step"] == best["step"] and result["best_val_wer_seleccion"] == best["wer"]
    again = score_texts(pairs["val"], generate_texts(model, tokenizer, encode_pairs(tokenizer, pairs["val"], cfg),
                                                     cfg, torch.device("cpu"), 1))
    assert again["wer"] == best["wer"]
    assert any(line.startswith("val: WER=") for line in lines)

    report, predictions = final_report(model, tokenizer, pairs, cfg, torch.device("cpu"))
    assert report["split"] == "val" and report["num_beams"] == 2 and len(predictions) == 10
    assert report["frases_val_con_texto_en_train"] == 5
    for value in report["por_modelo"].values():
        assert value["intervalo"]["trials"] == 5 and value["intervalo"]["low"] <= value["wer"] <= value["intervalo"]["high"]
    markdown = render_markdown({"val": report, "modelo": {"base": "t5-diminuto", "commit": None},
                                "best_step": result["best_step"], "completed_steps": 40,
                                "bootstrap": cfg["bootstrap"]})
    assert "| t12_causal_seed1 |" in markdown and "Media entre semillas" in markdown


def test_repository_profile_is_valid() -> None:
    cfg = load_config(ROOT / "configs/final/decodificador_t5.yaml")
    assert cfg["modelo"]["base"] == "google/flan-t5-base" and cfg["wandb"]["proyecto"] == "tiiac-2026"
    assert cfg["datos"]["pares"] == "data/lineaA/pares_ajuste_cruzado_v1"


# Los sintéticos solo entran en train: val sigue siendo la de los fonemas reales.
def test_synthetic_pairs_are_read_checked_and_mixed_into_training(tmp_path: Path) -> None:
    real, synthetic = tmp_path / "reales", tmp_path / "sinteticos"
    real.mkdir()
    _write_pairs(real, _rows("train"), _rows("val"))
    synthetic.mkdir()
    rows = [{**row, "model": "sintetico", "seed": None, "fold": None} for row in _rows("train")]
    (synthetic / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (synthetic / "manifest.json").write_text(json.dumps({"files": {"train.jsonl": file_sha256(synthetic / "train.jsonl")}}))
    cfg = _cfg(real)
    cfg["datos"]["sinteticos"] = {"pares": str(synthetic), "proporcion": 0.5}
    pairs, _ = read_pairs(cfg)
    assert [len(pairs[key]) for key in ("train", "val", "sinteticos")] == [5, 5, 5]
    tokenizer = _tokenizer()
    result = train_text_decoder(_model(tokenizer), tokenizer, pairs, cfg, None, torch.device("cpu"), log=lambda _: None)
    assert result["completed_steps"] == 40 and len(result["history"]) == 4

    cfg["datos"]["sinteticos"]["proporcion"] = 1.0
    with pytest.raises(ValueError, match="proporcion"):
        read_pairs(cfg)
    cfg["datos"]["sinteticos"]["proporcion"] = 0.5
    (synthetic / "train.jsonl").write_text((synthetic / "train.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="han cambiado"):
        read_pairs(cfg)
