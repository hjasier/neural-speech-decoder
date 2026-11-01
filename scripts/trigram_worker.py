"""Trabajador JSON para el lm_decoder oficial; ejecutar con su Python 3.10 CPU.

No importa el proyecto ni PyTorch 2.5. Las matrices viajan como float32 little
endian y el propio binding oficial aplica log_softmax y la penalización blank.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import torch
import lm_decoder


def respond(payload: dict[str, Any]) -> None:
    """Emite una única respuesta; los mensajes nativos del decoder van a stderr."""
    print(json.dumps({"ok": True, **payload}, allow_nan=False), flush=True)


def main() -> int:
    """Comparte recursos del grafo, con un BrainSpeechDecoder por frase activa."""
    respond({"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__})
    flows = {}
    resource = options = None
    blank_penalty = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "init":
                torch.set_num_threads(request["threads"])
                model = Path(request["model"])
                cfg = request["options"]
                blank_penalty = cfg["blank_penalty"]
                options = lm_decoder.DecodeOptions(
                    cfg["max_active"], cfg["min_active"], cfg["beam"], cfg["lattice_beam"],
                    cfg["acoustic_scale"], cfg["blank_skip_threshold"], cfg["length_penalty"], cfg["nbest"],
                )
                resource = lm_decoder.DecodeResource(str(model / "TLG.fst"), "", "", str(model / "words.txt"), "")
                respond({})
                continue
            identifier = request["flow"]
            if operation == "start":
                if resource is None or identifier in flows:
                    raise ValueError("Grafo sin inicializar o identificador de frase repetido.")
                flows[identifier] = lm_decoder.BrainSpeechDecoder(resource, options)
                respond({})
                continue
            decoder = flows[identifier]
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
            respond({"candidates": [{"sentence": result.sentence.strip(),
                                     "acoustic_score": float(result.ac_score),
                                     "language_score": float(result.lm_score)} for result in decoder.result()]})
        except Exception as error:
            print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
