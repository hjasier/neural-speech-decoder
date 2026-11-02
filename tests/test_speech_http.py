"""Transporte HTTP: pérdidas/reintentos y reloj de llegada independiente del cálculo."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Callable, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import yaml

from src.app.speech_http import DemoController, DemoHTTPServer
from src.app.speech_replay import ReplaySample


class RecordingDecoder:
    """Registra los bloques de cada frase sin ejecutar entrenamiento ni pesos grandes."""

    def __init__(self) -> None:
        self.checkpoint = SimpleNamespace(model=SimpleNamespace(num_layers=5, input_features=256))
        self.received: list[list[np.ndarray]] = []

    def start(self, session: str, *, layers: tuple[int, ...]) -> SimpleNamespace:
        blocks: list[np.ndarray] = []
        self.received.append(blocks)

        def push(values: np.ndarray) -> SimpleNamespace:
            blocks.append(values.copy())
            return SimpleNamespace(logits={5: values[:, :1]})

        return SimpleNamespace(push_with_logits=push, phonemes={5: []}, finish=lambda: None)


class RecordingLanguage:
    """Puede retener un bloque mientras llegan más peticiones HTTP."""

    def __init__(self, blocked: bool) -> None:
        self.entered, self.release = threading.Event(), threading.Event()
        self.finished = self.aborted = 0
        if not blocked:
            self.release.set()

    def start(self) -> SimpleNamespace:
        """Abre un acumulador nuevo y registra cierres distintos de cancelaciones."""
        values: list[int] = []
        stream = SimpleNamespace(text="", closed=False)

        def push(logits: np.ndarray) -> str:
            self.entered.set()
            assert self.release.wait(timeout=5)
            values.extend(int(value) for value in logits[:, 0])
            stream.text = " ".join(map(str, values))
            return stream.text

        def finish() -> str:
            if not stream.closed:
                self.finished += 1
                stream.closed = True
            return stream.text

        def abort() -> None:
            if not stream.closed:
                self.aborted += 1
                stream.closed = True

        stream.push, stream.finish, stream.abort = push, finish, abort
        return stream

    def close(self) -> None:
        self.release.set()


@contextmanager
def http_demo(tmp_path: Path, *, blocked: bool = False) -> Iterator[tuple]:
    """Abre el servidor real con motores mínimos para aislar el transporte."""
    raw = np.arange(9 * 256, dtype=np.float32).reshape(9, 256)
    sample = ReplaySample("sample", "session", 12, raw, "solo para la pantalla", np.zeros(256), np.ones(256))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/entrenos_online/speech_demo_http.yaml").read_text())
    cfg.update(salida=str(tmp_path), hilos_cpu=2)
    cfg["transmision"].update(bloque_bins=2, max_bloques_pendientes=8)
    decoder, language = RecordingDecoder(), RecordingLanguage(blocked)
    controller = DemoController(decoder, language, SimpleNamespace(load=lambda: sample), cfg)
    server = DemoHTTPServer(("127.0.0.1", 0), controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/"

    def request(operation: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(url + operation, data=body, headers={"Content-Type": "application/json"}), timeout=3) as response:
            return json.load(response)

    try:
        yield request, sample, decoder, language
    finally:
        language.release.set()
        server.shutdown()
        controller.close()
        server.server_close()
        thread.join(timeout=2)


def wait_state(request: Callable, status: str) -> dict:
    """Espera únicamente al trabajo pequeño de esta comprobación CPU."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        flow = request("state")["stream"]
        if flow["status"] == status:
            return flow
        time.sleep(0.01)
    raise AssertionError(f"La transmisión no llegó a {status}")


def test_http_retries_order_and_new_phrase_preserve_input(tmp_path: Path) -> None:
    """Un reintento no duplica señal; ni un paquete viejo ni otra frase comparten estado."""
    with http_demo(tmp_path) as (request, sample, decoder, _):
        request("sample", {})
        old_id = None
        expected = " ".join(str(int(value)) for value in sample.raw[:, 0])
        for repetition in range(2):
            flow = request("start", {"sample_id": sample.id})
            stream_id = flow["id"]
            if old_id:
                with pytest.raises(HTTPError) as error:
                    request("block", {"stream_id": old_id, "sequence": 0, "features": sample.raw[:2].tolist()})
                assert error.value.code == 409
            first = {"stream_id": stream_id, "sequence": 0, "features": sample.raw[:2].tolist()}
            assert request("block", first)["accepted"] is True
            assert request("block", first)["accepted"] is False
            with pytest.raises(HTTPError) as error:
                request("block", {"stream_id": stream_id, "sequence": 2, "features": sample.raw[4:6].tolist()})
            assert error.value.code == 409
            for sequence, start in enumerate(range(2, len(sample.raw), 2), 1):
                request("block", {"stream_id": stream_id, "sequence": sequence, "features": sample.raw[start:start + 2].tolist()})
            request("finish", {"stream_id": stream_id})
            final = wait_state(request, "finished")
            assert final["text"] == expected
            assert final["received_bins"] == final["processed_bins"] == len(sample.raw)
            assert final["input_sha256"] == sample.sha256
            np.testing.assert_array_equal(np.concatenate(decoder.received[repetition]), sample.raw)
            old_id = stream_id


def test_http_accepts_arrivals_while_inference_waits_and_can_cancel(tmp_path: Path) -> None:
    """Bloquear el decodificador no debe parar HTTP ni ocultar la cola de señal."""
    with http_demo(tmp_path, blocked=True) as (request, sample, decoder, language):
        request("sample", {})
        stream_id = request("start", {"sample_id": sample.id})["id"]
        request("block", {"stream_id": stream_id, "sequence": 0, "features": sample.raw[:2].tolist()})
        assert language.entered.wait(timeout=1)
        for sequence, start in enumerate(range(2, len(sample.raw), 2), 1):
            request("block", {"stream_id": stream_id, "sequence": sequence, "features": sample.raw[start:start + 2].tolist()})
        state = request("state")["stream"]
        assert state["received_bins"] == len(sample.raw)
        assert state["processed_bins"] == 0
        assert state["pending_ms"] == len(sample.raw) * 20
        request("cancel", {"stream_id": stream_id})
        language.release.set()
        final = wait_state(request, "cancelled")
        assert final["processed_bins"] == 2
        assert len(decoder.received[0]) == 1

        # Cancelar antes del primer bloque tampoco debe finalizar una búsqueda vacía.
        empty_id = request("start", {"sample_id": sample.id})["id"]
        request("cancel", {"stream_id": empty_id})
        empty = wait_state(request, "cancelled")
        assert empty["processed_bins"] == 0 and not decoder.received[1]
        assert language.aborted == 2 and language.finished == 0
