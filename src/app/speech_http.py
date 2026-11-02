"""Receptor HTTP local: cola de bloques y una GRU + WFST persistentes en CPU."""

from __future__ import annotations

from contextlib import suppress
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Full, Queue, SimpleQueue
import threading
from time import perf_counter
import traceback
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from src.app.speech_replay import DemoSamples, ReplaySample
from src.data.speech import BIN_SECONDS
from src.inference.speech import OnlineSpeechDecoder
from src.inference.trigram import TrigramDecoder

STATIC = Path(__file__).with_name("static")
TERMINAL = ("finished", "cancelled", "error")


class RequestError(ValueError):
    """Error de protocolo que puede corregir el emisor, sin modificar el modelo."""

    def __init__(self, message: str, status: int = 400) -> None:
        """Asocia al mensaje el estado HTTP de la respuesta."""
        super().__init__(message)
        self.status = status


class StreamInbox:
    """Una frase: recepción ordenada, reintentos idempotentes y cola acotada."""

    def __init__(self, sample: ReplaySample, block_bins: int, capacity: int) -> None:
        """Conserva metadatos y hash esperado; no guarda señal futura ni referencia."""
        self.id = uuid4().hex
        self.session = sample.session
        self.trial_index = sample.trial_index
        self.total_bins = len(sample.raw)
        self.expected_hash = sample.sha256
        self.block_bins = block_bins
        self.queue: Queue[np.ndarray | None] = Queue(maxsize=capacity)
        self.cancelled = threading.Event()
        self.status = "receiving"
        self.received_bins = self.processed_bins = 0
        self.packet_hashes: list[str] = []
        self.input_hash = hashlib.sha256()
        self.text = ""
        self.phones: list[str] = []
        self.processing_ms = self.packet_ms = 0.0
        self.started = perf_counter()
        self.elapsed_ms = 0.0
        self.error: str | None = None

    def accept(self, sequence: int, values: Any, feature_count: int) -> bool:
        """Encola una vez; rechaza pérdidas, cambios y paquetes fuera de orden."""
        if type(sequence) is not int or sequence < 0:
            raise RequestError("El número de bloque debe ser un entero no negativo.")
        features = np.asarray(values, dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != feature_count
                or not 0 < len(features) <= self.block_bins or not np.isfinite(features).all()):
            raise RequestError("El bloque debe contener bins crudos finitos con las 256 columnas del modelo.")
        raw = features.astype('<f4', copy=False).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if sequence < len(self.packet_hashes):
            if digest != self.packet_hashes[sequence]:
                raise RequestError("Ese bloque ya se recibió con otros valores.", 409)
            return False
        if self.status != "receiving" or sequence != len(self.packet_hashes):
            raise RequestError("El bloque no es el siguiente de esta transmisión.", 409)
        remaining = self.total_bins - self.received_bins
        if len(features) != min(self.block_bins, remaining):
            raise RequestError("El tamaño del bloque no corresponde a los bins pendientes.")
        try:
            self.queue.put_nowait(features.copy())
        except Full as error:
            raise RequestError("La cola está llena; reintenta el mismo bloque.", 429) from error
        self.packet_hashes.append(digest)
        self.input_hash.update(raw)
        self.received_bins += len(features)
        return True

    def finish(self) -> None:
        """Cierra la entrada solo cuando se han recibido todos los bins correctos."""
        if self.status in TERMINAL or self.status == "finishing":
            return
        if self.status != "receiving":
            raise RequestError("La transmisión se está deteniendo.", 409)
        if self.received_bins != self.total_bins or self.input_hash.hexdigest() != self.expected_hash:
            raise RequestError("Faltan bins o la señal recibida no coincide con la muestra.", 409)
        try:
            self.queue.put_nowait(None)
        except Full as error:
            raise RequestError("La cola está llena; reintenta el cierre.", 429) from error
        self.status = "finishing"

    def cancel(self) -> None:
        """Detiene la frase y despierta al consumidor si espera más señal."""
        if self.status not in TERMINAL:
            self.status = "cancelling"
            self.cancelled.set()
            with suppress(Full):
                self.queue.put_nowait(None)

    def snapshot(self) -> dict[str, Any]:
        """Copia el progreso; los parciales sustituyen al texto anterior."""
        return {"id": self.id, "session": self.session, "trial_index": self.trial_index,
                "status": self.status, "total_bins": self.total_bins, "received_bins": self.received_bins,
                "processed_bins": self.processed_bins, "packets": len(self.packet_hashes),
                "pending_ms": (self.received_bins - self.processed_bins) * BIN_SECONDS * 1000 if self.status not in TERMINAL else 0,
                "discarded_bins": self.received_bins - self.processed_bins if self.status == "cancelled" else 0,
                "text": self.text, "phones": list(self.phones), "processing_ms": self.processing_ms,
                "packet_ms": self.packet_ms, "elapsed_ms": self.elapsed_ms, "error": self.error,
                "input_sha256": self.input_hash.hexdigest(), "expected_sha256": self.expected_hash}


class DemoController:
    """Un único consumidor de inferencia; los handlers HTTP solo reciben y consultan."""

    def __init__(self, decoder: OnlineSpeechDecoder, language: TrigramDecoder,
                 samples: DemoSamples, cfg: dict[str, Any]) -> None:
        """Comparte pesos y grafo; crea estados nuevos para cada frase."""
        self.decoder, self.language, self.samples, self.cfg = decoder, language, samples, cfg
        self.lock = threading.RLock()
        self.sample: ReplaySample | None = None
        self.active: StreamInbox | None = None
        self.loading_sample = False
        self.fatal_error: str | None = None
        self._jobs: SimpleQueue[StreamInbox | None] = SimpleQueue()
        self._worker = threading.Thread(target=self._work, name="speech-inference", daemon=True)
        self._worker.start()

    def state(self) -> dict[str, Any]:
        """Devuelve el estado sin bloquear por el cálculo de la GRU o del WFST."""
        with self.lock:
            return {"ready": self.fatal_error is None, "error": self.fatal_error,
                    "loading_sample": self.loading_sample, "cpu_threads": self.cfg["hilos_cpu"],
                    "block_bins": self.cfg["transmision"]["bloque_bins"], "bin_ms": BIN_SECONDS * 1000,
                    "stream": self.active.snapshot() if self.active else None}

    def load_sample(self) -> dict[str, Any]:
        """Prepara una nueva muestra del emisor cuando no existe una frase activa."""
        with self.lock:
            if self.loading_sample or self.active and self.active.status not in TERMINAL:
                raise RequestError("Detén la transmisión antes de elegir otra muestra.", 409)
            self.loading_sample = True
        try:
            sample = self.samples.load()
            payload = sample.payload()
            with self.lock:
                self.sample = sample
                self.active = None
            return payload
        finally:
            with self.lock:
                self.loading_sample = False

    def start(self, sample_id: str) -> dict[str, Any]:
        """Reserva una frase; solo entrega sesión y metadatos al consumidor."""
        with self.lock:
            if self.fatal_error:
                raise RequestError(self.fatal_error, 503)
            if self.loading_sample or self.active and self.active.status not in TERMINAL:
                raise RequestError("Ya hay una transmisión activa.", 409)
            if self.sample is None or sample_id != self.sample.id:
                raise RequestError("Carga primero una muestra válida.", 409)
            spec = self.cfg["transmision"]
            self.active = StreamInbox(self.sample, spec["bloque_bins"], spec["max_bloques_pendientes"])
            self._jobs.put(self.active)
            return self.active.snapshot()

    def command(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Valida cada petición antes de modificar la recepción de su frase."""
        with self.lock:
            flow = self.active
            if flow is None or payload.get("stream_id") != flow.id:
                raise RequestError("La petición no corresponde a la transmisión activa.", 409)
            if operation == "block":
                accepted = flow.accept(payload.get("sequence"), payload.get("features"),
                                       self.decoder.checkpoint.model.input_features)
                return {"accepted": accepted, "received_bins": flow.received_bins}
            if operation == "finish":
                flow.finish()
            elif operation == "cancel":
                flow.cancel()
            else:
                raise RequestError("Operación desconocida.", 404)
            return flow.snapshot()

    def _work(self) -> None:
        while (flow := self._jobs.get()) is not None:
            neural = words = None
            try:
                depth = self.decoder.checkpoint.model.num_layers
                neural = self.decoder.start(flow.session, layers=(depth,))
                words = self.language.start()
                while not flow.cancelled.is_set():
                    packet = flow.queue.get()
                    if packet is None or flow.cancelled.is_set():
                        break
                    start = perf_counter()
                    chunk = neural.push_with_logits(packet)
                    text = words.push(chunk.logits[depth])
                    milliseconds = (perf_counter() - start) * 1000
                    with self.lock:
                        flow.processed_bins += len(packet)
                        flow.text, flow.phones = text, neural.phonemes[depth]
                        flow.packet_ms = milliseconds
                        flow.processing_ms += milliseconds
                start = perf_counter()
                neural.finish()
                if flow.cancelled.is_set():
                    words.abort()
                    text = words.text
                else:
                    text = words.finish()
                if not flow.cancelled.is_set() and flow.processed_bins != flow.total_bins:
                    raise RuntimeError("El consumidor terminó sin procesar todos los bins recibidos.")
                with self.lock:
                    flow.processing_ms += (perf_counter() - start) * 1000
                    flow.elapsed_ms = (perf_counter() - flow.started) * 1000
                    flow.text, flow.phones = text, neural.phonemes[depth]
                    flow.status = "cancelled" if flow.cancelled.is_set() else "finished"
            except Exception:
                traceback.print_exc()
                with self.lock:
                    flow.status = "error"
                    flow.error = "La decodificación se ha detenido. Consulta la terminal del servidor."
                    self.fatal_error = flow.error
            finally:
                if words is not None:
                    with suppress(Exception):
                        words.abort()
                with self.lock:
                    record = flow.snapshot()
                try:
                    directory = Path(self.cfg["salida"])
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / (flow.id + ".json")).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                except OSError:
                    traceback.print_exc()
                    with self.lock:
                        self.fatal_error = "No se pudo guardar el registro. Revisa la terminal y el espacio en disco."

    def close(self) -> None:
        """Cancela la frase y cierra el trabajador nativo antes de salir."""
        with self.lock:
            if self.active:
                self.active.cancel()
        self._jobs.put(None)
        self._worker.join(timeout=5)
        self.language.close()
        self._worker.join(timeout=5)


class DemoHTTPServer(ThreadingHTTPServer):
    """Servidor de la página y de la recepción HTTP en la interfaz local."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: DemoController | None) -> None:
        """Asocia las conexiones HTTP con un controlador de inferencia."""
        self.controller = controller
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    """API JSON y archivos estáticos explícitos, sin resolver rutas del cliente."""

    protocol_version = "HTTP/1.1"

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._respond(status, json.dumps(payload, allow_nan=False, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        """Sirve la vista y el progreso sin ejecutar inferencia en el handler."""
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._json(200, self.server.controller.state())
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in assets:
            self._json(404, {"error": "Ruta no disponible."})
            return
        name, mime = assets[path]
        self._respond(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self) -> None:
        """Recibe señal cruda y devuelve el acuse antes de procesarla."""
        try:
            if self.headers.get_content_type() != "application/json":
                raise RequestError("La petición debe usar application/json.", 415)
            size = int(self.headers.get("Content-Length", "0"))
            controller = self.server.controller
            limit = controller.cfg["transmision"]["bloque_bins"] * controller.decoder.checkpoint.model.input_features * 40 + 4096
            if not 0 < size <= limit:
                self.close_connection = True
                raise RequestError("El cuerpo de la petición no tiene un tamaño válido.", 413)
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise RequestError("Se esperaba un objeto JSON.")
            operation = urlsplit(self.path).path.removeprefix("/api/")
            if operation == "sample":
                self._json(200, controller.load_sample())
            elif operation == "start":
                self._json(201, controller.start(payload.get("sample_id")))
            elif operation in ("block", "finish", "cancel"):
                self._json(202, controller.command(operation, payload))
            else:
                raise RequestError("Ruta no disponible.", 404)
        except RequestError as error:
            self._json(error.status, {"error": str(error)})
        except (ValueError, TypeError, OverflowError) as error:
            self._json(400, {"error": str(error)})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "No se pudo preparar la muestra. Consulta la terminal del servidor."})

    def log_message(self, format: str, *args: Any) -> None:
        """Mantiene la terminal legible durante las consultas y el envío de bloques."""
        return
