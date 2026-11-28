"""Línea A: el LLM causal con LoRA se entrena solo en sus adaptadores y genera sin ver el objetivo."""

from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from scripts.train_text_decoder import load_config, render_markdown  # noqa: E402
from src.training.text_decoder import (  # noqa: E402
    collate_pairs, collate_prompts, encode_pairs, final_report, generate_texts, score_texts, target_text,
    train_text_decoder, wrap_lora,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/final/decodificador_llm_lora.yaml"
PHRASES = [("HH AH L OW SIL", "Hello."), ("W ER L D SIL", "world"), ("HH AH L OW SIL W ER L D SIL", "Hello, world!"),
           ("G UH D SIL", "good"), ("G UH D SIL W ER L D SIL", "Good world.")]


def _tokenizer() -> transformers.PreTrainedTokenizerFast:
    """Tokenizador de palabras sin eos automático y con pad igual a eos, como en Qwen."""
    words = sorted({token for phones, text in PHRASES for token in (phones.split() + target_text(text).split())})
    vocab = {"</s>": 0, "<unk>": 1, "fonemas:": 2, "texto:": 3, **{word: 4 + index for index, word in enumerate(words)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=core, pad_token="</s>", eos_token="</s>",
                                                unk_token="<unk>")


def _model(tokenizer: transformers.PreTrainedTokenizerFast) -> transformers.Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = transformers.Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                      max_position_embeddings=128, tie_word_embeddings=True,
                                      pad_token_id=0, eos_token_id=0, bos_token_id=0)
    return transformers.Qwen2ForCausalLM(config)


def _rows(split: str, model: str = "t12_causal_seed0") -> list[dict]:
    return [{"split": split, "session": "dia", "block": 3, "source_trial_index": index, "phones": phones,
             "reference_text": text, "model": model, "seed": 0, "fold": None}
            for index, (phones, text) in enumerate(PHRASES)]


def _cfg() -> dict:
    cfg = load_config(PROFILE)
    cfg["datos"].update(max_tokens_entrada=16, max_tokens_salida=8)
    cfg["datos"].pop("sinteticos")
    cfg["modelo"]["lora"].update(r=4, alpha=8, dropout=0.0)
    cfg["entrenamiento"].update(pasos=40, batch_size=2, learning_rate=5e-2, calentamiento_pasos=4, imprimir_cada=10)
    cfg["evaluacion"].update(cada_pasos=10, batch_size=4, num_beams_final=2)
    cfg["bootstrap"]["remuestreos"] = 200
    cfg["ejecucion"]["dispositivo"] = "cpu"
    return cfg


# La pérdida solo puede caer en el texto: si cubriese los fonemas, el modelo
# aprendería a repetir la entrada y no a traducirla.
def test_causal_encoding_masks_the_prompt_and_pads_by_length() -> None:
    tokenizer, cfg = _tokenizer(), _cfg()
    encoded = encode_pairs(tokenizer, _rows("train"), cfg)
    first = encoded[0]
    prompt = tokenizer("fonemas: HH AH L OW SIL\ntexto:", add_special_tokens=False)["input_ids"]
    assert first["prompt_ids"] == prompt and first["input_ids"][:len(prompt)] == prompt
    assert first["labels"] == [-100] * len(prompt) + tokenizer("hello")["input_ids"] + [tokenizer.eos_token_id]
    batch = collate_pairs([encoded[0], encoded[2]], tokenizer.pad_token_id)
    assert batch["attention_mask"][0].sum() == len(first["input_ids"])  # el eos final no queda enmascarado
    assert batch["labels"].shape == batch["input_ids"].shape
    prompts = collate_prompts([encoded[0], encoded[2]], tokenizer.pad_token_id)
    assert prompts["input_ids"][0, -len(prompt):].tolist() == prompt
    assert prompts["attention_mask"][0].sum() == len(prompt) and prompts["attention_mask"][1].all()


def test_lora_trains_only_adapters_and_keeps_the_best_step() -> None:
    tokenizer, cfg = _tokenizer(), _cfg()
    model = wrap_lora(_model(tokenizer), cfg)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    assert frozen and all("lora_" in name for name, value in model.named_parameters() if value.requires_grad)
    pairs = {"train": _rows("train"), "val": _rows("val") + _rows("val", model="t12_causal_seed1")}
    result = train_text_decoder(model, tokenizer, pairs, cfg, None, torch.device("cpu"), log=lambda _: None)
    assert all(torch.equal(value, frozen[name]) for name, value in model.named_parameters() if name in frozen)
    best = min(result["history"], key=lambda item: item["wer"])
    assert result["best_step"] == best["step"]
    again = score_texts(pairs["val"], generate_texts(model, tokenizer, encode_pairs(tokenizer, pairs["val"], cfg),
                                                     cfg, torch.device("cpu"), 1))
    assert again["wer"] == best["wer"]
    report, predictions = final_report(model, tokenizer, pairs, cfg, torch.device("cpu"))
    assert report["num_beams"] == 2 and len(predictions) == 10
    markdown = render_markdown({"val": report, "modelo": {"base": "qwen-diminuto", "commit": None},
                                "best_step": result["best_step"], "completed_steps": 40,
                                "bootstrap": cfg["bootstrap"], "config": cfg})
    assert markdown.startswith("# Línea A: decodificador LLM con LoRA")


def test_repository_llm_profile_is_valid_and_complete(tmp_path: Path) -> None:
    cfg = load_config(PROFILE)
    assert cfg["modelo"]["base"] == "Qwen/Qwen2.5-3B" and len(cfg["modelo"]["revision"]) == 40
    synthetic = load_config(ROOT / "configs/final/decodificador_t5_sinteticos.yaml")
    assert cfg["datos"]["pares"] == synthetic["datos"]["pares"]
    assert cfg["datos"]["sinteticos"] == synthetic["datos"]["sinteticos"]
    assert cfg["evaluacion"]["num_beams_final"] == synthetic["evaluacion"]["num_beams_final"]
    broken = tmp_path / "perfil.yaml"
    broken.write_text(PROFILE.read_text(encoding="utf-8").replace("  dtype: bfloat16\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="modelo.dtype"):
        load_config(broken)
