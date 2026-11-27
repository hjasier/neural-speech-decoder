"""Línea A: pares sintéticos con errores aprendidos de los pares reales de train."""

import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts.build_synthetic_pairs import synthetic_spec
from src.data.phoneme_noise import PhonemeNoise, align_phones, fit_error_model, phone_error_rate
from src.data.speech_dataset import file_sha256
from src.data.text_corpus import candidate_sentences, rejection_reason, split_sentences
from src.utils.seed import crear_generador_numpy

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"min_palabras": 3, "max_palabras": 6}


def test_alignment_counts_each_kind_of_error() -> None:
    reference, hypothesis = "K AE T SIL".split(), "K AH T T".split()
    pairs = align_phones(reference, hypothesis)
    assert [ref for ref, _ in pairs if ref is not None] == reference
    assert [hyp for _, hyp in pairs if hyp is not None] == hypothesis
    assert sum(ref != hyp for ref, hyp in pairs) == 2
    assert phone_error_rate([(reference, hypothesis), (reference, reference)]) == 2 / 8

    model = fit_error_model([("A B".split(), "X A B".split()), ("A B C".split(), "A C".split()),
                             ("A B".split(), "A D".split())])
    assert model["inserciones"] == {"^": {"X": 1}}
    assert model["borrados"] == {"B": 1} and model["sustituciones"] == {"B": {"D": 1}}
    assert model["apariciones"] == {"A": 3, "B": 3, "C": 1, "^": 3}
    assert model["errores"] == 3 and model["per"] == 3 / 7
    assert model["per_frases"] == [1 / 2, 1 / 3, 1 / 2]


# El ruido debe reproducir el PER real; si no, el T5 se entrenaría con fonemas
# mucho más limpios o más sucios que los de la GRU.
def test_noise_reproduces_the_real_error_rate() -> None:
    rng = crear_generador_numpy(0)
    phones = "AA AE AH B D EH F G SIL".split()
    real = []
    for _ in range(400):
        reference = list(rng.choice(phones, size=12))
        hypothesis = [phone if rng.random() > 0.2 else str(rng.choice(phones)) for phone in reference]
        if rng.random() < 0.3:
            hypothesis.insert(int(rng.integers(len(hypothesis))), "SIL")
        if rng.random() < 0.3:
            hypothesis.pop(int(rng.integers(len(hypothesis))))
        real.append((reference, hypothesis))
    model = fit_error_model(real)
    noise = PhonemeNoise(model)
    rng = crear_generador_numpy(1)
    noisy = phone_error_rate((reference, noise.apply(reference, rng)) for reference, _ in real * 5)
    assert noisy == pytest.approx(model["per"], rel=0.1)


def test_sentences_are_filtered_like_t12_and_exclude_its_texts() -> None:
    document = ('Port-au-Prince (CNN) -- Victims waited. "Not yet!" she laughs. It costs 5 dollars.\n'
                "The U.N. said so. We can do this together. Short one. We can do this together!")
    assert list(split_sentences("One two three. Four five six?\n\nSeven eight nine!")) == [
        "One two three.", "Four five six?", "Seven eight nine!"]
    assert rejection_reason("It costs 5 dollars.", SPEC) == "caracteres"
    assert rejection_reason("Short one.", SPEC) == "longitud"
    assert rejection_reason('"Not yet!" she laughs.', SPEC) is None
    counts: dict[str, int] = {}
    kept = candidate_sentences([document], SPEC, {"not yet she laughs"}, counts)
    assert list(kept.values()) == ["We can do this together."]
    assert counts["descartada_texto_t12"] == 1 and counts["descartada_repetida"] == 1


def test_repository_profiles_are_valid() -> None:
    spec = synthetic_spec(ROOT / "configs/final/pares_sinteticos.yaml")
    assert spec["errores"] == "data/lineaA/pares_ajuste_cruzado_v1"
    assert spec["frases"]["max_palabras"] == 18
    from scripts.train_text_decoder import load_config

    cfg = load_config(ROOT / "configs/final/decodificador_t5_sinteticos.yaml")
    base = load_config(ROOT / "configs/final/decodificador_t5.yaml")
    assert cfg["datos"]["sinteticos"]["pares"] == spec["salida"]
    assert cfg["evaluacion"] == {**base["evaluacion"], "cada_pasos": cfg["evaluacion"]["cada_pasos"]}
    assert cfg["datos"]["pares"] == base["datos"]["pares"]


def _write_real_pairs(root: Path) -> None:
    root.mkdir(parents=True)
    rows = [{"split": "train", "phones": "HH AH L OW SIL", "reference_phones": "HH AH L OW SIL",
             "reference_text": "Hello."},
            {"split": "train", "phones": "W ER D SIL", "reference_phones": "W ER L D SIL", "reference_text": "World."}]
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "val.jsonl").write_text("ilegible\n")
    (root / "manifest.json").write_text(json.dumps({"files": {"train.jsonl": file_sha256(root / "train.jsonl"),
                                                              "val.jsonl": "no se lee"}}))


# De punta a punta con un parquet diminuto: excluye T12, no lee val y deja manifiesto.
def test_builder_writes_pairs_without_reading_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    labels = yaml.safe_load((ROOT / "configs/final/t12_causal.yaml").read_text(encoding="utf-8"))
    if not (ROOT / labels["etiquetas"]["recursos"]).is_dir():
        pytest.skip("Faltan los recursos de g2p-en preparados.")
    from scripts import build_synthetic_pairs, get_text_corpus

    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"text": ["Nuclear rockets can destroy airfields with ease. The cat sat on the mat.",
                                      "Dogs like to run in the park. It costs 5 dollars."]}), corpus)
    excluded = tmp_path / "t12" / "train"
    excluded.mkdir(parents=True)
    (excluded / "dia.json").write_text(json.dumps({"targets": [
        {"reference": "Nuclear rockets can destroy airfields with ease."}]}))
    _write_real_pairs(tmp_path / "reales")
    spec = synthetic_spec(ROOT / "configs/final/pares_sinteticos.yaml")
    spec = copy.deepcopy(spec)
    spec["corpus"].update(archivo=str(corpus), sha256=get_text_corpus.sha256_of(corpus))
    spec.update(etiquetas=str(ROOT / "configs/final/t12_causal.yaml"), excluir=str(tmp_path / "t12"),
                errores=str(tmp_path / "reales"), procesos=1, salida=str(tmp_path / "sinteticos"),
                tabla=str(tmp_path / "tabla"))
    profile = tmp_path / "perfil.yaml"
    profile.write_text(yaml.safe_dump(spec))
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["build", "--config", str(profile)])
    assert build_synthetic_pairs.main() == 0

    rows = [json.loads(line) for line in (tmp_path / "sinteticos" / "train.jsonl").read_text().splitlines()]
    assert sorted(row["reference_text"] for row in rows) == ["Dogs like to run in the park.", "The cat sat on the mat."]
    assert all(row["split"] == "train" and row["model"] == "sintetico" for row in rows)
    assert rows[0]["reference_phones"].endswith("SIL")
    manifest = json.loads((tmp_path / "sinteticos" / "manifest.json").read_text())
    assert manifest["files"]["train.jsonl"] == file_sha256(tmp_path / "sinteticos" / "train.jsonl")
    assert manifest["conteos"]["descartada_texto_t12"] == 1 and manifest["conteos"]["pares"] == 2
    assert manifest["comprobacion"]["per_real"] == pytest.approx(1 / 10)
    assert (tmp_path / "tabla.md").read_text().startswith("# Línea A: pares sintéticos")
    assert build_synthetic_pairs.main() == 1  # la carpeta ya existe: no se sobrescribe
