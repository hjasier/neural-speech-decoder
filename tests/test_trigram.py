"""Orden de clases y agregación WER: fallos que pueden pasar inadvertidos."""

import base64
import hashlib
import mmap
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.inference.speech import SpeechChunk
from src.inference.trigram import TrigramStream, kaldi_columns
from src.training.speech_online_evaluation import OnlineTrial
from src.training.speech_trigram_evaluation import evaluate_trigram
from scripts.trigram_mmap_cache import MappedGraphCache


def test_releasing_graph_pages_preserves_graph_and_other_memory(tmp_path: Path) -> None:
    """Descartar páginas de solo lectura debe conservar pesos, archivo y buffers ajenos."""
    graph = tmp_path / "grafo con espacios.fst"
    data = bytes(range(256)) * (32 * 1024)
    graph.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()

    def resident_bytes() -> int:
        total, selected = 0, False
        for line in Path("/proc/self/smaps").read_text().splitlines():
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                selected = line.endswith(str(graph))
            elif selected and line.startswith("Rss:"):
                total += int(line.split()[1]) * 1024
        return total

    with graph.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped, mmap.mmap(-1, 4096) as other:
        other[:8] = b"preserva"
        assert hashlib.sha256(mapped).hexdigest() == expected
        assert resident_bytes() >= len(data)
        cache = MappedGraphCache(graph, {"rss_archivos_umbral_gib": 1.5, "liberar_al_final_frase": True})
        cache.maintain(finish=True)
        assert resident_bytes() == 0
        assert other[:8] == b"preserva"
        assert hashlib.sha256(mapped).hexdigest() == expected
    assert hashlib.sha256(graph.read_bytes()).hexdigest() == expected


def test_trigram_receives_all_logits_with_blank_and_sil_first() -> None:
    """Conserva todas las alternativas y respeta los símbolos del checkpoint."""
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/entrenos_online/speech_trigrama.yaml").read_text())
    neural = yaml.safe_load((root / "configs/entrenos_online/speech_online.yaml").read_text())
    phones = list(reversed(neural["datos"]["fonemas"]))
    columns = kaldi_columns(phones, len(phones), spec["fonemas_grafo"])
    requests = []

    def request(payload: dict) -> dict:
        requests.append(payload)
        return {"candidates": []}

    stream = TrigramStream(SimpleNamespace(columns=columns, _open_flow=lambda: 1, _request=request))
    logits = torch.arange(-41, 41).reshape(2, 41).float()
    stream.push(logits)
    actual = np.frombuffer(base64.b64decode(requests[0]["logits"]), dtype="<f4").reshape(requests[0]["shape"])
    # En este checkpoint invertido: blank=40, SIL=0, AA=39, ..., ZH=1.
    expected = logits[:, [40, 0, *range(39, 0, -1)]].numpy()
    np.testing.assert_array_equal(actual, expected)


def test_trigram_wer_sums_errors_and_reference_words() -> None:
    """Una frase correcta y otra con tres borrados producen 3/5, sin promediar WER."""
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/entrenos_online/speech_online_evaluacion.yaml").read_text())
    spec.update(salidas=[5], calentamiento=0)
    model = SimpleNamespace(available_exits=(5,), initial_state=torch.zeros(1))

    def neural_start(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(hypotheses={5: [0]}, finish=lambda: {5: []},
                               push_with_logits=lambda raw: SpeechChunk({5: torch.zeros((1, 41))}, {5: []}))

    decoder = SimpleNamespace(checkpoint=SimpleNamespace(model=model), start=neural_start)
    language = SimpleNamespace(start=lambda: SimpleNamespace(push=lambda logits: "a", finish=lambda: "a"))
    trials = [OnlineTrial(np.zeros((19, 256), dtype=np.float32), (0,), session, index, 1, reference)
              for index, (session, reference) in enumerate((("a", "A"), ("b", "A B C D!")))]
    rows = []
    result = evaluate_trigram(decoder, language, trials, spec, expected_trials=2, prediction_sink=rows.append)
    assert result["global"][5]["wer"] == 3 / 5
    assert result["global"][5]["palabras"] == 5
    assert result["global"][5]["frases"] == 2
    assert result["by_session"][5]["a"]["wer"] == 0
    assert result["by_session"][5]["b"]["wer"] == 3 / 4
    assert rows[1]["profiles"][5]["errores_palabras"] == 3
