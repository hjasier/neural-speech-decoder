"""Llamadas LiveKit: micrófono, turnos simulados T12 y voz incremental local.

El audio entrante solo dispara turnos; nunca sustituye la señal neuronal.
Las credenciales se mantienen fuera de la configuración y los registros.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from time import monotonic
import traceback
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from src.app.speech_http import DemoController, RequestError


def load_livekit_env(paths: list[str]) -> dict[str, str]:
    """Lee solo las tres variables LiveKit, sin ejecutar .env ni imprimir valores."""
    names = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    values: dict[str, str] = {}
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        for line in source.read_text().splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() in names and key.strip() not in values:
                value = value.strip()
                if value.startswith(('"', "'")) and value.endswith(value[0]):
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
                values[key.strip()] = value
    for name in names:
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


class VoiceActivity:
    """Detector de energía con duración mínima, silencio final y turno máximo."""

    def __init__(self, spec: dict[str, Any]) -> None:
        """Conserva los umbrales del perfil, sin cargar un modelo adicional."""
        self.spec = spec
        self.voice_ms = self.silence_ms = self.total_ms = 0.0
        self.started = False

    def push(self, level: float, milliseconds: float) -> bool:
        """Indica una vez cuándo acaba una intervención con voz suficiente."""
        if level >= self.spec["umbral_rms"]:
            self.voice_ms += milliseconds
            self.silence_ms = 0.0
            self.started = self.voice_ms >= self.spec["voz_minima_ms"]
        elif self.voice_ms:
            self.silence_ms += milliseconds
            if not self.started and self.silence_ms >= self.spec["silencio_ms"]:
                self.voice_ms = self.silence_ms = self.total_ms = 0.0
        if self.voice_ms:
            self.total_ms += milliseconds
        ended = self.started and (self.silence_ms >= self.spec["silencio_ms"]
                                  or self.total_ms >= self.spec["turno_maximo_ms"])
        if ended:
            self.voice_ms = self.silence_ms = self.total_ms = 0.0
            self.started = False
        return ended


class StableWords:
    """Pronuncia cada posición una vez, esperando estabilidad del prefijo parcial."""

    def __init__(self, seconds: float) -> None:
        """Retiene la última palabra parcial y espera el intervalo configurado."""
        self.seconds = seconds
        self.previous: list[str] = []
        self.since: list[float] = []
        self.spoken: list[str] = []
        self.revised = False

    def update(self, text: str, now: float, *, final: bool = False) -> str:
        """Devuelve solo palabras nuevas; una corrección no vuelve a leer la frase."""
        words = text.split()
        common = 0
        for old, new in zip(self.previous, words):
            if old != new:
                break
            common += 1
        self.since = self.since[:common] + [now] * (len(words) - common)
        self.previous = words
        if words[:len(self.spoken)] != self.spoken:
            self.revised = True
            if not final:
                return ""
        end = len(words) if final else max(0, len(words) - 1)
        if not final:
            end = min(end, next((i for i in range(len(self.spoken), end)
                                 if now - self.since[i] < self.seconds), end))
        extra = words[len(self.spoken):end]
        self.spoken.extend(extra)
        return " ".join(extra)


class LocalVoice:
    """Síntesis Flite en CPU: texto por archivo y PCM mono, sin shell ni red."""

    def __init__(self, spec: dict[str, Any]) -> None:
        """Comprueba la voz y localiza el FFmpeg del equipo."""
        self.spec = spec
        self.executable = shutil.which("ffmpeg")
        if not self.executable or not re.fullmatch(r"[a-zA-Z0-9_]+", spec["voz"]):
            raise ValueError("Instala FFmpeg con el filtro flite y configura una voz válida.")
        result = subprocess.run([self.executable, "-hide_banner", "-filters"],
                                capture_output=True, timeout=5, check=True)
        if not re.search(rb"\bflite\b", result.stdout):
            raise ValueError("El FFmpeg instalado no incluye el sintetizador flite.")

    async def synthesize(self, text: str) -> bytes:
        """Genera voz del texto predicho y termina el subproceso al cancelar."""
        if not text.strip():
            return b""
        with tempfile.TemporaryDirectory(prefix="speech-call-") as directory:
            Path(directory, "speech.txt").write_text(text, encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                self.executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                "-filter_threads", "1", "-f", "lavfi", "-i", f"flite=textfile=speech.txt:voice={self.spec['voz']}",
                "-ar", str(self.spec["sample_rate"]), "-ac", "1", "-f", "s16le", "pipe:1",
                cwd=directory, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                audio, _ = await asyncio.wait_for(process.communicate(), self.spec["timeout_segundos"])
                if process.returncode or not audio:
                    raise RuntimeError("No se pudo sintetizar la voz local.")
                return audio
            finally:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()


@dataclass
class Call:
    """Estado de una llamada; el control secreto nunca entra en el estado público."""

    id: str = field(default_factory=lambda: uuid4().hex)
    control: str = field(default_factory=lambda: uuid4().hex)
    status: str = "connecting"
    error: str | None = None
    level: float = 0.0
    received_audio_ms: float = 0.0
    turns: int = 0
    stream_id: str | None = None
    spoken_text: str = ""
    revised: bool = False
    created: float = field(default_factory=monotonic)
    heartbeat: float = field(default_factory=monotonic)
    cooldown_until: float = 0.0
    stop: asyncio.Event | None = None
    turn: asyncio.Task | None = None
    source: Any = None
    joined: bool = False
    end_requested: bool = False

    @property
    def room(self) -> str:
        """Asigna una sala impredecible y aislada para esta llamada."""
        return "speech-demo-" + self.id

    @property
    def identity(self) -> str:
        """Identifica únicamente el micrófono autorizado para esta llamada."""
        return "mobile-" + self.id

    def snapshot(self) -> dict[str, Any]:
        """Expone progreso sin tokens, claves ni audio del micrófono."""
        return {"id": self.id, "status": self.status, "error": self.error, "level": self.level,
                "received_audio_ms": self.received_audio_ms, "turns": self.turns,
                "stream_id": self.stream_id, "spoken_text": self.spoken_text, "revised": self.revised}


class CallManager:
    """Un puente RTC en su propio bucle; comparte el único motor del visor."""

    def __init__(self, controller: DemoController, spec: dict[str, Any]) -> None:
        """Habilita llamadas si hay claves, SDK y síntesis; conserva la demo sin ellos."""
        self.controller, self.spec = controller, spec
        self.lock = threading.RLock()
        self.active: Call | None = None
        self.error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._future = None
        self.credentials = load_livekit_env(spec["env_archivos"])
        try:
            if not all(self.credentials.get(name) for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")):
                raise ValueError("Faltan LIVEKIT_URL, LIVEKIT_API_KEY o LIVEKIT_API_SECRET en .env.")
            parsed = urlsplit(self.credentials["LIVEKIT_URL"])
            if parsed.scheme not in ("wss", "ws") or not parsed.hostname:
                raise ValueError("LIVEKIT_URL debe ser una dirección ws:// o wss:// válida.")
            from livekit import api, rtc
            self.api, self.rtc = api, rtc
            self.voice = LocalVoice(spec["tts"])
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, name="livekit-calls", daemon=True)
            self._thread.start()
        except ImportError:
            self.error = "Instala las dependencias de requirements-llamadas.txt y reinicia el servidor."
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            self.error = str(error)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.run_until_complete(self._loop.shutdown_default_executor())
        self._loop.close()

    def state(self) -> dict[str, Any]:
        """Permite al visor y al móvil observar la misma llamada sin controlarla."""
        with self.lock:
            return {"available": self.error is None, "error": self.error,
                    "active": self.active.snapshot() if self.active else None}

    def _token(self, call: Call, *, bot: bool = False) -> str:
        return (self.api.AccessToken(self.credentials["LIVEKIT_API_KEY"], self.credentials["LIVEKIT_API_SECRET"])
                .with_identity("decoder-" + call.id if bot else call.identity)
                .with_name("Voz T12" if bot else "Móvil")
                .with_ttl(timedelta(seconds=self.spec["token_segundos"]))
                .with_grants(self.api.VideoGrants(room_join=True, room=call.room,
                                                 can_publish=True, can_subscribe=True, can_publish_data=False,
                                                 can_publish_sources=None if bot else ["microphone"]))
                .to_jwt())

    def command(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Crea una llamada o aplica un control autenticado de su propietario."""
        with self.lock:
            if self.error:
                raise RequestError(self.error, 503)
            if operation == "start":
                if self.active and self.active.status not in ("ended", "error"):
                    raise RequestError("Ya hay una llamada activa.", 409)
                if self._future is not None and not self._future.done():
                    raise RequestError("La llamada anterior se está cerrando.", 409)
                call = Call()
                self.controller.claim_call(call.id)
                try:
                    token = self._token(call)
                    self.active = call
                    self._future = asyncio.run_coroutine_threadsafe(self._run(call), self._loop)
                except Exception:
                    self.controller.release_call(call.id)
                    raise
                return {"id": call.id, "control": call.control, "url": self.credentials["LIVEKIT_URL"], "token": token}
            call = self.active
            control = payload.get("control")
            if (call is None or payload.get("id") != call.id or not isinstance(control, str)
                    or not hmac.compare_digest(control, call.control)):
                raise RequestError("Esta petición no controla la llamada activa.", 403)
            call.heartbeat = monotonic()
            if operation == "end":
                call.end_requested = True
                self._loop.call_soon_threadsafe(self._stop, call)
            elif operation == "respond":
                if call.status not in ("listening", "user_speaking"):
                    raise RequestError("Espera a que termine la respuesta actual.", 409)
                # Reservar el turno antes de devolver HTTP evita dos disparos simultáneos.
                call.status = "decoding"
                self._loop.call_soon_threadsafe(self._trigger, call)
            elif operation != "ping":
                raise RequestError("Control de llamada desconocido.", 404)
            return call.snapshot()

    def _stop(self, call: Call) -> None:
        if call.stop:
            call.stop.set()

    def _trigger(self, call: Call) -> None:
        if call.stop is None or call.end_requested or call.stop.is_set() or call.turn and not call.turn.done():
            return
        self._set(call, status="decoding")
        call.turn = asyncio.create_task(self._respond(call))

    def _set(self, call: Call, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(call, key, value)

    async def _run(self, call: Call) -> None:
        rtc = self.rtc
        room = rtc.Room()
        call.stop = asyncio.Event()
        if call.end_requested:
            call.stop.set()
        listeners: set[asyncio.Task] = set()

        @room.on("track_subscribed")
        def subscribed(track: Any, publication: Any, participant: Any) -> None:
            if (participant.identity == call.identity and track.kind == rtc.TrackKind.KIND_AUDIO
                    and not listeners):
                self._set(call, joined=True)
                task = asyncio.create_task(self._listen(call, track))
                listeners.add(task)

        @room.on("participant_disconnected")
        def left(participant: Any) -> None:
            if participant.identity == call.identity:
                call.stop.set()

        @room.on("disconnected")
        def disconnected(*_args: Any) -> None:
            call.stop.set()

        try:
            await asyncio.wait_for(room.connect(self.credentials["LIVEKIT_URL"], self._token(call, bot=True)),
                                   self.spec["conexion_timeout_segundos"])
            if call.stop.is_set():
                return
            call.source = rtc.AudioSource(self.spec["tts"]["sample_rate"], 1,
                                          queue_size_ms=self.spec["tts"]["cola_ms"])
            track = rtc.LocalAudioTrack.create_audio_track("voz-t12", call.source)
            await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            self._set(call, status="listening")
            while not call.stop.is_set():
                now = monotonic()
                if (now - call.heartbeat > self.spec["heartbeat_timeout_segundos"]
                        or now - call.created > self.spec["duracion_maxima_segundos"]
                        or not call.joined and now - call.created > self.spec["conexion_timeout_segundos"]):
                    call.stop.set()
                    break
                for listener in listeners:
                    if listener.done():
                        listener.result()
                        call.stop.set()
                if call.turn and call.turn.done():
                    call.turn.result()
                await asyncio.sleep(self.spec["sondeo_segundos"])
        except Exception:
            # No volcar excepciones del SDK: pueden contener URLs o tokens de conexión.
            self._set(call, status="error", error="La llamada se interrumpió. Comprueba LiveKit, la red y la síntesis local.")
        finally:
            call.stop.set()
            tasks = list(listeners) + ([call.turn] if call.turn else [])
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if call.stream_id:
                with suppress(RequestError):
                    self.controller.command("cancel", {"stream_id": call.stream_id}, call.id)
            try:
                if call.source:
                    with suppress(Exception):
                        call.source.clear_queue()
                        await call.source.aclose()
                with suppress(Exception):
                    await room.disconnect()
            finally:
                self.controller.release_call(call.id)
                self._set(call, status="error" if call.error else "ended", level=0.0)
                self._record(call)

    async def _listen(self, call: Call, track: Any) -> None:
        audio = self.rtc.AudioStream(track, sample_rate=self.spec["entrada_sample_rate"], num_channels=1,
                                    capacity=10, frame_size_ms=20)
        detector = VoiceActivity(self.spec["vad"])
        try:
            async for event in audio:
                if call.stop.is_set():
                    break
                frame = event.frame
                values = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                level = float(np.sqrt(np.mean(values * values))) if len(values) else 0.0
                milliseconds = frame.samples_per_channel * 1000 / frame.sample_rate
                self._set(call, level=level, received_audio_ms=call.received_audio_ms + milliseconds)
                if call.status not in ("listening", "user_speaking") or monotonic() < call.cooldown_until:
                    detector = VoiceActivity(self.spec["vad"])
                    continue
                ended = detector.push(level, milliseconds)
                if detector.started:
                    self._set(call, status="user_speaking")
                if ended:
                    self._trigger(call)
        finally:
            await audio.aclose()

    async def _replay(self, call: Call, sample: Any) -> None:
        spec = self.controller.cfg["transmision"]
        started = monotonic()
        for sequence, offset in enumerate(range(0, len(sample.raw), spec["bloque_bins"])):
            end = min(offset + spec["bloque_bins"], len(sample.raw))
            await asyncio.sleep(max(0, started + end * self.controller.state()["bin_ms"] / 1000 - monotonic()))
            payload = {"stream_id": call.stream_id, "sequence": sequence, "features": sample.raw[offset:end]}
            await self._deliver(call, "block", payload)
        await self._deliver(call, "finish", {"stream_id": call.stream_id})

    async def _deliver(self, call: Call, operation: str, payload: dict[str, Any]) -> None:
        while not call.stop.is_set():
            try:
                self.controller.command(operation, payload, call.id)
                return
            except RequestError as error:
                if error.status != 429:
                    raise
                await asyncio.sleep(self.spec["reintento_segundos"])
        raise asyncio.CancelledError

    async def _respond(self, call: Call) -> None:
        replay = None
        # La lectura .mat no se puede cancelar: esperar su cierre antes de liberar el motor.
        loading = asyncio.create_task(asyncio.to_thread(self.controller.load_sample, call.id))
        try:
            try:
                await asyncio.shield(loading)
            except asyncio.CancelledError:
                await loading
                raise
            if call.stop.is_set():
                return
            with self.controller.lock:
                sample = self.controller.sample
            flow = self.controller.start(sample.id, call.id)
            self._set(call, stream_id=flow["id"], spoken_text="", revised=False)
            replay = asyncio.create_task(self._replay(call, sample))
            words = StableWords(self.spec["tts"]["estabilidad_segundos"])
            while not call.stop.is_set():
                if replay.done():
                    replay.result()
                flow = self.controller.state()["stream"]
                if flow["status"] in ("cancelled", "error"):
                    raise RuntimeError("Se interrumpió la inferencia de la llamada.")
                final = flow["status"] == "finished"
                text = words.update(flow["text"], monotonic(), final=final)
                if text:
                    self._set(call, status="speaking")
                    # Fragmentos pequeños evitan sintetizar toda una frase de una vez.
                    tokens = text.split()
                    width = self.spec["tts"]["palabras_por_fragmento"]
                    for offset in range(0, len(tokens), width):
                        fragment = " ".join(tokens[offset:offset + width])
                        audio = await self.voice.synthesize(fragment)
                        await self._publish(call, audio)
                        spoken = (call.spoken_text + " " + fragment).strip()
                        self._set(call, spoken_text=spoken, revised=words.revised)
                if final:
                    break
                await asyncio.sleep(self.spec["sondeo_segundos"])
            if replay:
                await replay
            await call.source.wait_for_playout()
            self._set(call, status="listening", turns=call.turns + 1, revised=words.revised,
                      cooldown_until=monotonic() + self.spec["vad"]["pausa_eco_ms"] / 1000)
        finally:
            if replay:
                replay.cancel()
                await asyncio.gather(replay, return_exceptions=True)

    async def _publish(self, call: Call, audio: bytes) -> None:
        rate = self.spec["tts"]["sample_rate"]
        samples = rate * self.spec["tts"]["frame_ms"] // 1000
        size = samples * 2
        for offset in range(0, len(audio), size):
            if call.stop.is_set():
                raise asyncio.CancelledError
            data = audio[offset:offset + size].ljust(size, b"\0")
            await call.source.capture_frame(self.rtc.AudioFrame(data, rate, 1, samples))

    def _record(self, call: Call) -> None:
        try:
            path = Path(self.controller.cfg["salida"]) / ("llamada_" + call.id + ".json")
            path.write_text(json.dumps(call.snapshot(), indent=2, ensure_ascii=False) + "\n")
        except OSError:
            traceback.print_exc()

    def close(self) -> None:
        """Cuelga, descarta audio y cierra el bucle antes de cerrar la inferencia."""
        if self._loop is None:
            return
        if self.active:
            self._loop.call_soon_threadsafe(self._stop, self.active)
        if self._future:
            with suppress(Exception):
                self._future.result(timeout=self.spec["conexion_timeout_segundos"] + 5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
