"""Turnos, voz incremental y sesión compartida; CPU pequeña, sin pesos ni red RTC."""

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import yaml

from src.app.speech_calls import StableWords, VoiceActivity, load_livekit_env
from src.app.speech_http import DemoController, DemoHTTPServer, RequestError
from src.app.speech_replay import ReplaySample
from test_speech_http import RecordingDecoder, RecordingLanguage


def test_vad_requires_voice_and_resets_between_turns() -> None:
    """Silencio y golpes cortos no disparan; dos intervenciones generan dos turnos."""
    vad = VoiceActivity({"umbral_rms": .01, "voz_minima_ms": 200, "silencio_ms": 600, "turno_maximo_ms": 2000})
    assert not vad.push(0, 1000)
    assert not vad.push(.1, 20)
    assert not vad.push(0, 600)
    assert not vad.started
    for _ in range(2):
        for _ in range(10):
            assert not vad.push(.1, 20)
        assert vad.started
        assert not vad.push(0, 580)
        assert vad.push(0, 20)
        assert not vad.push(0, 600)
    assert not vad.push(.1, 200)
    assert vad.push(.1, 1800)


def test_partial_words_wait_for_stability_and_never_repeat_a_revision() -> None:
    """Retiene la última palabra, reinicia prefijos cambiados y vacía el final."""
    words = StableWords(.3)
    assert words.update("hello wor", 0) == ""
    assert words.update("hello world", .2) == ""
    assert words.update("hello world", .4) == "hello"
    assert words.update("hello there friend", .5) == ""
    assert words.update("hello there friend", .9) == "there"
    assert words.update("hello their friend", 1.3) == ""
    assert words.revised
    assert words.update("hello their friend", 1.4, final=True) == "friend"
    assert words.update("hello their friend", 1.5, final=True) == ""
    assert words.spoken == ["hello", "there", "friend"]


def test_env_is_data_whitelisted_and_environment_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No evalúa sustituciones ni lee otras claves; respeta entorno y primer archivo."""
    first, second = tmp_path / "a.env", tmp_path / "b.env"
    first.write_text('export LIVEKIT_URL="wss://example.invalid"\nLIVEKIT_API_KEY=from-file\nOTHER_SECRET=ignored\n')
    second.write_text("LIVEKIT_API_KEY=second-file\nLIVEKIT_API_SECRET='$(do-not-execute)'\n")
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_API_KEY", "from-environment")
    assert load_livekit_env([str(first), str(second)]) == {
        "LIVEKIT_URL": "wss://example.invalid", "LIVEKIT_API_KEY": "from-environment",
        "LIVEKIT_API_SECRET": "$(do-not-execute)"}


@contextmanager
def shared_demo(tmp_path: Path) -> Iterator[tuple]:
    """Abre un receptor y dos clientes HTTP sobre un único consumidor pequeño."""
    raw = np.arange(8 * 256, dtype=np.float32).reshape(8, 256)
    sample = ReplaySample("shared", "t12", 7, raw, "reference must not be spoken", np.zeros(256), np.ones(256))
    cfg = {"salida": str(tmp_path), "hilos_cpu": 2,
           "transmision": {"bloque_bins": 2, "max_bloques_pendientes": 8}}
    decoder, language = RecordingDecoder(), RecordingLanguage(False)
    controller = DemoController(decoder, language, SimpleNamespace(load=lambda: sample), cfg)
    server = DemoHTTPServer(("127.0.0.1", 0), controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield controller, sample, base, decoder
    finally:
        server.shutdown()
        controller.close()
        server.server_close()
        thread.join(timeout=2)


def read_json(url: str) -> dict:
    """Consulta el servidor de prueba desde una conexión independiente."""
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def test_call_reserves_engine_and_two_viewers_only_see_received_bins(tmp_path: Path) -> None:
    """Los observadores se incorporan a mitad de frase y no pueden secuestrar la llamada."""
    with shared_demo(tmp_path) as (controller, sample, base, decoder):
        controller.claim_call("call-owner")
        with pytest.raises(RequestError):
            controller.claim_call("second-call")
        with pytest.raises(RequestError):
            controller.load_sample()
        controller.load_sample("call-owner")
        initial = read_json(base + "/api/view?sample_id=shared&from=0")
        assert initial["sample"]["bins"] == 8
        assert initial["features"] == []
        flow = controller.start(sample.id, "call-owner")
        with pytest.raises(RequestError):
            controller.command("cancel", {"stream_id": flow["id"]})
        for sequence, offset in enumerate(range(0, 4, 2)):
            controller.command("block", {"stream_id": flow["id"], "sequence": sequence,
                                         "features": sample.raw[offset:offset + 2]}, "call-owner")
        first = read_json(base + "/api/view?sample_id=shared&from=0")
        second = read_json(base + "/api/view?sample_id=shared&from=2")
        np.testing.assert_array_equal(first["features"], sample.raw[:4])
        np.testing.assert_array_equal(second["features"], sample.raw[2:4])
        assert read_json(base + "/api/state")["call_active"]
        with pytest.raises(HTTPError) as error:
            read_json(base + "/api/view?sample_id=shared&from=6")
        assert error.value.code == 409
        controller.command("cancel", {"stream_id": flow["id"]}, "call-owner")
        deadline = time.monotonic() + 2
        while controller.state()["stream"]["status"] != "cancelled" and time.monotonic() < deadline:
            time.sleep(.01)
        controller.release_call("other-owner")
        assert controller.state()["call_active"]
        controller.release_call("call-owner")
        assert not controller.state()["call_active"]
        with pytest.raises(RequestError):
            controller.load_sample("call-owner")
        controller.load_sample()
        assert len(decoder.received) == 1


def test_call_page_and_origin_validation(tmp_path: Path) -> None:
    """Sirve la segunda web y rechaza controles procedentes de otra página."""
    with shared_demo(tmp_path) as (_, _, base, _):
        with urlopen(base + "/llamada", timeout=3) as response:
            assert response.status == 200
            assert b"Voz T12" in response.read()
        request = Request(base + "/api/sample", data=b"{}", headers={
            "Content-Type": "application/json", "Origin": "https://other.invalid"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 403


def test_replay_and_synthesis_use_predictions_and_cancel_cleanly(tmp_path: Path) -> None:
    """El puente reproduce señal real del emisor; la voz jamás usa la referencia."""
    pytest.importorskip("livekit.rtc")
    from src.app.speech_calls import CallManager, Call

    async def exercise(controller: DemoController, spec: dict) -> None:
        # Construir sin abrir red; la prueba usa el mismo turno y reloj de producción.
        manager = object.__new__(CallManager)
        manager.controller, manager.spec, manager.lock = controller, spec, threading.RLock()
        utterances: list[str] = []

        async def synthesize(text: str) -> bytes:
            utterances.append(text)
            return b"\0" * 960

        async def capture(_call: Call, _audio: bytes) -> None:
            await asyncio.sleep(0)

        async def playout() -> None:
            await asyncio.sleep(0)

        manager.voice = SimpleNamespace(synthesize=synthesize)
        manager._publish = capture
        call = Call()
        call.stop, call.source = asyncio.Event(), SimpleNamespace(wait_for_playout=playout)
        controller.claim_call(call.id)
        await manager._respond(call)
        assert call.turns == 1 and call.status == "listening"
        assert utterances and "reference" not in " ".join(utterances)
        assert " ".join(utterances) == controller.state()["stream"]["text"]
        assert controller.state()["stream"]["input_sha256"] == controller.sample.sha256
        call.stop.set()
        await manager._respond(call)
        assert call.turns == 1
        controller.release_call(call.id)

    with shared_demo(tmp_path) as (controller, _, _, _):
        spec = yaml.safe_load(Path("configs/entrenos_online/speech_demo_http.yaml").read_text())["llamadas"]
        asyncio.run(exercise(controller, spec))
