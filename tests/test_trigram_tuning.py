"""El ajuste del trigrama compara búsquedas sobre los mismos logits y agrega como el WER."""

import importlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.tune_speech_trigram import render_markdown, search_grid, summarize, tuning_spec
from src.inference.speech import SpeechChunk
from src.inference.trigram import TrigramDecoder
from src.training.speech_online_evaluation import OnlineTrial
from src.training.speech_trigram_evaluation import collect_final_logits, search_word_errors

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load((ROOT / "configs/entrenos_online/speech_trigrama.yaml").read_text())["busqueda"]


# La búsqueda de partida va primera: su WER debe reproducir el ya medido antes de seguir con el resto.
def test_grid_always_contains_the_starting_search_once() -> None:
    grid = search_grid(BASE, {"acoustic_scale": [0.5, 1.0], "blank_penalty": [0.0]})
    assert grid[0] == {"acoustic_scale": BASE["acoustic_scale"], "blank_penalty": BASE["blank_penalty"]}
    assert len(grid) == 3
    same = search_grid(BASE, {"acoustic_scale": [0.5, BASE["acoustic_scale"]]})
    assert same == [{"acoustic_scale": BASE["acoustic_scale"]}, {"acoustic_scale": 0.5}]
    for invalid in ({"nbest": [1, 2]}, {"temperatura": [1.0]}, {"beam": [18.0, 18.0]}, {"beam": []}, {}):
        with pytest.raises(ValueError):
            search_grid(BASE, invalid)


# El perfil del repositorio debe ser válido y no puede ajustar sobre test.
def test_repository_profile_is_valid_and_rejects_test(tmp_path: Path) -> None:
    spec = tuning_spec(ROOT / "configs/final/ajuste_trigrama.yaml")
    assert len(search_grid(BASE, spec["rejilla"])) == 20
    path = tmp_path / "ajuste.yaml"
    path.write_text(yaml.safe_dump({**spec, "split": "test"}))
    with pytest.raises(ValueError, match="solo en val"):
        tuning_spec(path)


def _row(job: str, seed: int, parameters: dict, errors: int, words: int) -> dict:
    return {"trabajo": job, "seed": seed, "parametros": parameters, "errores": errors, "palabras": words,
            "segundos": 1.0, "errores_por_frase": [errors]}


# Promediar el WER de cada semilla daría otro número cuando las semillas tienen distintas palabras.
def test_summary_pools_errors_across_seeds_and_keeps_first_on_ties() -> None:
    start, other, tied = {"beam": 18.0}, {"beam": 12.0}, {"beam": 24.0}
    rows = [_row("a", 0, start, 30, 100), _row("b", 1, start, 10, 300),
            _row("a", 0, other, 10, 100), _row("b", 1, other, 20, 300),
            _row("a", 0, tied, 20, 100), _row("b", 1, tied, 10, 300)]
    summary = summarize(rows, [start, other, tied], ["a", "b"], start)
    assert [entry["wer"] for entry in summary["combinaciones"]] == [0.1, 0.075, 0.075]
    assert summary["elegida"] == other and summary["wer_partida"] == 0.1
    assert [entry["elegida"] for entry in summary["combinaciones"]] == [False, True, False]
    assert "| 12 | 7,50 | 6,67–10,00 | 2 | elegida |" in render_markdown(summary)
    with pytest.raises(ValueError, match="Faltan combinaciones"):
        summarize(rows[:-1], [start, other, tied], ["a", "b"], start)


# Comparar búsquedas exige que todas reciban exactamente los logits de la evaluación por bloques.
def test_logits_are_collected_per_block_and_decoded_once_per_sentence() -> None:
    spec = yaml.safe_load((ROOT / "configs/entrenos_online/speech_online_evaluacion.yaml").read_text())
    model = SimpleNamespace(available_exits=(5,), initial_state=torch.zeros(1))
    blocks = []

    def neural_start(*_args: object, **_kwargs: object) -> SimpleNamespace:
        def push(raw: np.ndarray) -> SpeechChunk:
            blocks.append(len(raw))
            return SpeechChunk({5: torch.full((1, 41), float(len(blocks)))}, {5: []})

        return SimpleNamespace(push_with_logits=push, finish=lambda: {5: []})

    decoder = SimpleNamespace(checkpoint=SimpleNamespace(model=model), start=neural_start)
    trials = [OnlineTrial(np.zeros((10, 256), dtype=np.float32), (0,), "a", 0, 1, "uno dos"),
              OnlineTrial(np.zeros((4, 256), dtype=np.float32), (0,), "a", 1, 1, "tres")]
    items = collect_final_logits(decoder, trials, spec, expected_trials=2)
    assert blocks == [4, 4, 2, 4] and [len(item["logits"]) for item in items] == [3, 1]
    assert items[0]["logits"][:, 0].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="split completo"):
        collect_final_logits(decoder, trials, spec, expected_trials=3)

    pushed = []
    language = SimpleNamespace(start=lambda: SimpleNamespace(
        push=lambda logits: pushed.append(len(logits)), finish=lambda: "uno"))
    assert search_word_errors(language, items) == ([1, 1], [2, 1]) and pushed == [3, 1]


class _FakeBinding:
    """Sustituye al lm_decoder compilado: registra con qué opciones nace cada frase."""

    def __init__(self) -> None:
        self.penalties: list[float] = []

    def DecodeOptions(self, *values: float) -> tuple:  # noqa: N802
        return values

    def DecodeResource(self, *paths: str) -> tuple:  # noqa: N802
        return paths

    def BrainSpeechDecoder(self, resource: tuple, options: tuple) -> SimpleNamespace:  # noqa: N802
        return SimpleNamespace(options=options, FinishDecoding=lambda: None,
                               result=lambda: [SimpleNamespace(sentence=f"beam {options[2]}", ac_score=0.0, lm_score=0.0)])

    def DecodeNumpy(self, decoder: SimpleNamespace, values: np.ndarray, priors: np.ndarray, penalty: float) -> None:  # noqa: N802
        self.penalties.append(penalty)


# Si el trabajador conservase la búsqueda inicial, toda la rejilla daría el mismo WER sin avisar.
def test_worker_changes_search_between_sentences_without_reloading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    binding = _FakeBinding()
    monkeypatch.setitem(sys.modules, "lm_decoder", binding)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.delitem(sys.modules, "trigram_worker", raising=False)
    worker = importlib.import_module("trigram_worker")
    logits = {"shape": [1, 41], "logits": __import__("base64").b64encode(np.zeros((1, 41), "<f4").tobytes()).decode()}
    tuned = {**BASE, "beam": 12.0, "blank_penalty": 1.5}
    requests = [{"op": "init", "model": "grafo", "options": BASE, "threads": 2},
                {"op": "start", "flow": 1}, {"op": "options", "options": tuned},
                {"op": "push", "flow": 1, **logits}, {"op": "finish", "flow": 1},
                {"op": "options", "options": tuned}, {"op": "start", "flow": 2},
                {"op": "push", "flow": 2, **logits}, {"op": "finish", "flow": 2}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(request) + "\n" for request in requests)))
    threads = torch.get_num_threads()
    try:
        assert worker.main() == 0
    finally:
        torch.set_num_threads(threads)
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()][1:]
    # Con una frase abierta se rechaza: esa frase seguiría con la búsqueda anterior.
    assert [reply["ok"] for reply in replies] == [True, True, False, True, True, True, True, True, True]
    assert replies[4]["candidates"][0]["sentence"] == f"beam {BASE['beam']}"
    assert replies[8]["candidates"][0]["sentence"] == "beam 12.0"
    assert binding.penalties == [BASE["blank_penalty"], 1.5]


# El cliente debe enviar la búsqueda completa: el binding no admite cambios parciales.
def test_client_sends_the_whole_search_when_reconfiguring() -> None:
    sent = []
    client = SimpleNamespace(spec={"busqueda": BASE}, _request=sent.append)
    TrigramDecoder.configure(client, {**BASE, "beam": 12.0})
    assert sent == [{"op": "options", "options": {**BASE, "beam": 12.0}}]
    assert client.spec["busqueda"]["beam"] == 12.0
