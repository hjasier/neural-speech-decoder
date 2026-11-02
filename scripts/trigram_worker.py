"""Trabajador JSON para el lm_decoder oficial; ejecutar con su Python 3.10 CPU.

No importa el proyecto ni PyTorch 2.5. Las matrices viajan como float32 little
endian y el propio binding oficial aplica log_softmax y la penalización blank.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import platform
import resource as process_resource
import sys
from typing import Any

import numpy as np
import torch
import lm_decoder

from trigram_mmap_cache import MappedGraphCache, mapped_bytes


def respond(payload: dict[str, Any]) -> None:
    """Emite una única respuesta; los mensajes nativos del decoder van a stderr."""
    print(json.dumps({"ok": True, **payload}, allow_nan=False), flush=True)


def main() -> int:
    """Comparte recursos del grafo, con un BrainSpeechDecoder por frase activa."""
    respond({"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
             "mapped_graphs": hasattr(lm_decoder, "SetFstReadMode")})
    flows = {}
    resource = options = None
    graph_cache = None
    blank_penalty = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "init":
                torch.set_num_threads(request["threads"])
                model = Path(request["model"])
                cfg = request["options"]
                read_mode = request.get("read_mode", "read")
                cache_policy = request.get("mapped_cache")
                if read_mode not in ("read", "map"):
                    raise ValueError("Modo de lectura del grafo no reconocido.")
                if read_mode == "map" and not hasattr(lm_decoder, "SetFstReadMode"):
                    raise ValueError("El runtime no tiene soporte de lectura map.")
                if cache_policy is not None and read_mode != "map":
                    raise ValueError("El control de caché requiere lectura map.")
                if hasattr(lm_decoder, "SetFstReadMode"):
                    lm_decoder.SetFstReadMode(read_mode)
                blank_penalty = cfg["blank_penalty"]
                options = lm_decoder.DecodeOptions(
                    cfg["max_active"], cfg["min_active"], cfg["beam"], cfg["lattice_beam"],
                    cfg["acoustic_scale"], cfg["blank_skip_threshold"], cfg["length_penalty"], cfg["nbest"],
                )
                resource = lm_decoder.DecodeResource(str(model / "TLG.fst"), "", "", str(model / "words.txt"), "")
                mapped = mapped_bytes(model / "TLG.fst") if read_mode == "map" else 0
                if read_mode == "map" and (not mapped or mapped < (model / "TLG.fst").stat().st_size - 4096):
                    raise RuntimeError("OpenFst no ha mapeado el grafo completo; se rechaza el fallback a RAM.")
                graph_cache = MappedGraphCache(model / "TLG.fst", cache_policy) if cache_policy is not None else None
                respond({"read_mode": read_mode, "mapped_graph_bytes": mapped,
                         "peak_rss_bytes": process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss * 1024})
                continue
            identifier = request["flow"]
            if operation == "start":
                if resource is None or identifier in flows:
                    raise ValueError("Grafo sin inicializar o identificador de frase repetido.")
                flows[identifier] = lm_decoder.BrainSpeechDecoder(resource, options)
                respond({})
                continue
            decoder = flows[identifier]
            if operation == "abort":
                del flows[identifier]
                decoder = None
                cache = graph_cache.maintain(finish=True) if graph_cache is not None else None
                respond({"mapped_cache": cache} if cache is not None else {})
                continue
            if operation == "reset":
                decoder.Reset()
                respond({})
                continue
            if operation == "push":
                shape = request["shape"]
                if len(shape) != 2 or min(shape) <= 0:
                    raise ValueError("Forma de logits no válida.")
                values = np.frombuffer(base64.b64decode(request["logits"], validate=True), dtype="<f4").reshape(shape).copy()
                if not np.isfinite(values).all():
                    raise ValueError("Logits no finitos.")
                lm_decoder.DecodeNumpy(decoder, values, np.zeros((1, shape[1]), dtype=np.float32), blank_penalty)
            elif operation == "finish":
                decoder.FinishDecoding()
                del flows[identifier]
            else:
                raise ValueError(f"Operación desconocida: {operation}")
            candidates = [{"sentence": result.sentence.strip(),
                                     "acoustic_score": float(result.ac_score),
                                     "language_score": float(result.lm_score)} for result in decoder.result()]
            cache_stats = graph_cache.maintain(finish=operation == "finish") if graph_cache is not None else None
            respond({"candidates": candidates, "mapped_cache": cache_stats})
        except Exception as error:
            print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
