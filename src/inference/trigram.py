"""Cliente incremental del WFST oficial, ejecutado en su propio Python CPU.

Cada frase abre un TrigramStream con start(). Recibe chunk.logits[depth]
de SpeechStream.push_with_logits() y se cierra con finish() al terminar.
El texto parcial sustituye al anterior; no se concatena. Para otra frase
se abre un flujo nuevo o se llama a reset(). El consumidor cierra el grafo
con TrigramDecoder.close() cuando termina de usarlo.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
from typing import Any, Sequence

import numpy as np
from torch import Tensor

from src.inference.fst import require_mappable_fst


@dataclass(frozen=True)
class WordHypothesis:
    """Una frase candidata y los dos log-scores originales del decodificador."""

    sentence: str
    acoustic_score: float
    language_score: float


def kaldi_columns(phones: Sequence[str], blank_id: int, graph_phones: Sequence[str]) -> tuple[int, ...]:
    """Ordena como los autores: blank, SIL y los 39 fonemas ARPABET del grafo.

    Las columnas originales pertenecen al checkpoint. Un vocabulario distinto
    debe rechazarse aunque el número de clases coincida.
    """
    if (blank_id != len(phones) or len(set(phones)) != len(phones)
            or len(set(graph_phones)) != len(graph_phones)
            or set(phones) != set(graph_phones) or "SIL" not in graph_phones):
        raise ValueError("El vocabulario del checkpoint no corresponde al grafo de fonemas.")
    return (blank_id, phones.index("SIL"), *(phones.index(phone) for phone in graph_phones if phone != "SIL"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TrigramDecoder:
    """Comparte un grafo CPU entre flujos independientes, con llamadas secuenciales."""

    def __init__(
        self, spec: dict[str, Any], phones: Sequence[str], blank_id: int, *, log_path: Path,
    ) -> None:
        """Verifica recursos y abre el trabajador; la GRU conserva su entorno actual."""
        self.columns = kaldi_columns(phones, blank_id, spec["fonemas_grafo"])
        self.spec = spec
        model = Path(spec["modelo"])
        read_mode = spec.get("lectura_grafo", "read")
        if read_mode not in ("read", "map"):
            raise ValueError("lectura_grafo debe ser read o map.")
        if read_mode == "map":
            require_mappable_fst(model / "TLG.fst")
        manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["archive_sha256"] != spec["archivo_sha256"]
                or manifest["graph_phones"] != spec["fonemas_grafo"]):
            raise ValueError("El trigrama preparado no procede del archivo oficial configurado.")
        for name in ("TLG.fst", "words.txt"):
            if _digest(model / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"El recurso de lenguaje ha cambiado: {name}")
        self.resources = manifest
        self._log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(spec["hilos_cpu"]),
                       "MKL_NUM_THREADS": str(spec["hilos_cpu"]), "OPENBLAS_NUM_THREADS": str(spec["hilos_cpu"])}
        worker = Path(__file__).resolve().parents[2] / "scripts/trigram_worker.py"
        try:
            self._process = subprocess.Popen([str(Path(spec["python_runtime"]).absolute()), "-u", str(worker)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                             text=True, encoding="utf-8", bufsize=1, env=environment)
        except BaseException:
            self._log.close()
            raise
        self._next_id = 0
        self.graph_cache: dict[str, Any] | None = None
        try:
            self.runtime = self._read_reply()
            if self.runtime["torch"] != spec["torch_runtime"]:
                raise ValueError("El trabajador no usa la versión CPU de PyTorch declarada.")
            if read_mode == "map" and not self.runtime.get("mapped_graphs"):
                raise ValueError("El runtime necesita compilarse con soporte para lectura map.")
            self._request({"op": "init", "model": str(model.resolve()), "options": spec["busqueda"],
                           "threads": spec["hilos_cpu"], "read_mode": read_mode,
                           "mapped_cache": spec.get("cache_mapeo")})
        except BaseException:
            self.close()
            raise

    def _process_failure(self) -> RuntimeError:
        try:
            returncode = self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            returncode = None
        status = f"código {returncode}" if returncode is not None else "salida cerrada; proceso todavía activo"
        if returncode is not None and returncode < 0:
            try:
                status += f", {signal.Signals(-returncode).name}"
            except ValueError:
                status += f", señal {-returncode}"
        try:
            with self._log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 8192))
                tail = stream.read().decode("utf-8", errors="replace").strip()
        except OSError:
            tail = "No se pudo leer el log del trabajador."
        return RuntimeError(f"El trabajador del trigrama terminó ({status}); revisar {self._log_path}\n{tail}")

    def _read_reply(self) -> dict[str, Any]:
        if not select.select([self._process.stdout], [], [], self.spec["timeout_segundos"])[0]:
            raise RuntimeError(f"El trabajador del trigrama no responde; revisar {self._log_path}")
        line = self._process.stdout.readline()
        if not line:
            raise self._process_failure()
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"Error del trigrama: {response.get('error')}; revisar {self._log_path}")
        return response

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as error:
            raise self._process_failure() from error
        response = self._read_reply()
        if "mapped_cache" in response:
            self.graph_cache = response["mapped_cache"]
        return response

    def _open_flow(self) -> int:
        self._next_id += 1
        self._request({"op": "start", "flow": self._next_id})
        return self._next_id

    def start(self) -> TrigramStream:
        """Inicia una frase con estado WFST propio y el grafo compartido."""
        return TrigramStream(self)

    def close(self) -> None:
        """Libera el proceso y su grafo, también al interrumpir una evaluación."""
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process.stdout.close()
        self._log.close()


class TrigramStream:
    """Conserva el beam de una frase; los parciales pueden cambiar al llegar otro bloque."""

    def __init__(self, decoder: TrigramDecoder) -> None:
        """Abre un flujo sin compartir sus hipótesis con otras frases."""
        self._decoder = decoder
        self._flow = decoder._open_flow()
        self._closed = False
        self.candidates: tuple[WordHypothesis, ...] = ()

    @property
    def text(self) -> str:
        """Devuelve la mejor hipótesis actual, que puede ser todavía vacía."""
        return self.candidates[0].sentence if self.candidates else ""

    def _update(self, response: dict[str, Any]) -> str:
        self.candidates = tuple(WordHypothesis(**candidate) for candidate in response["candidates"])
        return self.text

    def push(self, logits: Tensor) -> str:
        """Consume logits T × clases, sin argmax ni colapso CTC previo ni cierre del beam."""
        if self._closed:
            raise ValueError("La frase de palabras está cerrada; usa reset().")
        if logits.ndim != 2 or logits.shape[1] != len(self._decoder.columns):
            raise ValueError("Los logits no conservan las clases del checkpoint.")
        if not logits.shape[0]:
            return self.text
        values = logits.detach().float().cpu().numpy()[:, self._decoder.columns]
        values = np.ascontiguousarray(values, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("Los logits contienen NaN o infinito.")
        return self._update(self._decoder._request({
            "op": "push", "flow": self._flow, "shape": list(values.shape),
            "logits": base64.b64encode(values.tobytes()).decode("ascii"),
        }))

    def finish(self) -> str:
        """Finaliza la búsqueda una sola vez y conserva las candidatas resultantes."""
        if not self._closed:
            self._update(self._decoder._request({"op": "finish", "flow": self._flow}))
            self._closed = True
        return self.text

    def reset(self) -> None:
        """Borra las hipótesis; una frase cerrada abre un estado nuevo en el mismo grafo."""
        if self._closed:
            self._flow = self._decoder._open_flow()
        else:
            self._decoder._request({"op": "reset", "flow": self._flow})
        self._closed = False
        self.candidates = ()
