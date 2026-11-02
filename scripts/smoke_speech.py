"""Smoke del pipeline de habla con T12 local o entradas sintéticas en CI."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from math import isclose, isfinite
from pathlib import Path
import tempfile
from typing import Any

import torch

from scripts.prepare_speech_data import prepare_signals
from scripts.smoke_speech_data import comprobar_datos_smoke, preparar_subconjunto_smoke
from scripts.smoke_speech_synthetic import preparar_datos_sinteticos_smoke
from scripts.smoke_trigram import preparar_trigrama_smoke
from src.data.preprocessing import extract_model_features
from src.data.speech import load_session
from src.data.speech_dataset import SpeechDataset, collate_speech, file_sha256
from src.data.speech_protocol import build_speech_protocol, read_speech_protocol
from src.inference.ctc import decode_ctc
from src.inference.speech import OnlineSpeechDecoder
from src.inference.trigram import TrigramDecoder
from src.training.speech import evaluar_habla
from src.training.speech_trigram_evaluation import word_error_counts


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return file_sha256(path)


def preparar_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Verifica T12 en local; en GitHub Actions usa entradas sintéticas declaradas."""
    cfg = deepcopy(cfg)
    spec = cfg["smoke"]
    mode = os.environ.get("TIIAC_SMOKE_DATOS", "sinteticos" if os.environ.get("GITHUB_ACTIONS") == "true" else "t12")
    if mode not in ("t12", "sinteticos"):
        raise ValueError("TIIAC_SMOKE_DATOS debe ser t12 o sinteticos.")
    if mode == "t12":
        source, protocol, data_report = comprobar_datos_smoke(cfg)
    else:
        data_report = {"archivos_originales_verificados": 0, "etiquetas_verificadas": 0,
                       "archivo_descarga_verificado": False}
    parent = Path(spec["directorio"])
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="habla_", dir=parent)).resolve().relative_to(Path.cwd())
    cfg["datos"].update(raiz=str(root / "originales"), preparados=str(root / "preparados"))
    cfg["etiquetas"]["salida"] = str(root / "fonemas")
    cfg["salida"] = {"directorio": str(root / "checkpoints")}
    cfg["wandb"]["directorio"] = str(root / "tracking")
    cfg["protocolo"]["manifiesto"] = str(root / "particion.json")
    if mode == "t12":
        provenance = preparar_subconjunto_smoke(cfg, source, protocol, root)
    else:
        preparar_datos_sinteticos_smoke(cfg)
        provenance = {}
        print("Smoke CI: DATOS SINTÉTICOS. Esta ejecución no comprueba la descarga de T12.", flush=True)
    cfg["smoke"]["origen_datos"] = mode
    cfg["smoke"]["datos_verificados"] = data_report
    cfg["smoke"]["procedencia"] = provenance
    cfg["smoke"]["trigrama"] = preparar_trigrama_smoke(cfg, root)
    cfg["protocolo"]["sha256"] = _write_json(
        Path(cfg["protocolo"]["manifiesto"]), build_speech_protocol(cfg),
    )
    prepare_signals(cfg)
    print(f"Smoke: datos {mode}, GRU reducida y WFST con un trigrama de tres palabras.", flush=True)
    return cfg


@torch.inference_mode()
def verificar_smoke(cfg: dict[str, Any], run: Any, metrics: dict[str, float | int]) -> None:
    """Recarga los pesos y exige igualdad de métricas, logits y fonemas por bloques."""
    output = Path(cfg["salida"]["directorio"]) / run.id
    result = json.loads((output / "resultado.json").read_text(encoding="utf-8"))
    if result["completed_steps"] != cfg["entrenamiento"]["pasos"]:
        raise RuntimeError("El smoke no ha completado sus actualizaciones.")
    decoder = OnlineSpeechDecoder.from_checkpoint(output / "mejor_val.pt", device="cpu")
    checkpoint = decoder.checkpoint
    dataset = SpeechDataset(cfg, "val")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["datos"]["batch_size"], collate_fn=collate_speech,
    )
    reloaded = evaluar_habla(checkpoint.model, loader, cfg, checkpoint.smoother, torch.device("cpu"))
    for key in ("ctc_loss", "per", "errores", "fonemas", "frases", "completo"):
        if not isfinite(reloaded[key]) or not isclose(reloaded[key], metrics[key], rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f"La métrica {key} cambia al recargar el checkpoint.")
    protocol = read_speech_protocol(cfg)
    checked, max_error = 0, 0.0
    language = TrigramDecoder(cfg["smoke"]["trigrama"], checkpoint.phones, checkpoint.blank_id,
                              log_path=output / "trigrama.log")
    report_data: dict[str, Any] = {}
    try:
        _verificar_palabras(cfg, decoder, language, protocol, report_data)
    finally:
        language.close()
    for session_id, session in enumerate(cfg["datos"]["sesiones"]):
        raw_session = load_session(Path(cfg["datos"]["raiz"]) / "train" / f"{session}.mat")
        for index in protocol["sessions"][session_id]["val"]:
            raw = extract_model_features(raw_session.trials[index])
            normalized = checkpoint.normalization[session].transform_features(raw, session)
            features = torch.from_numpy(normalized).unsqueeze(0)
            expected = checkpoint.model(checkpoint.smoother(features), torch.tensor([session_id]))
            stream = decoder.start(session)
            parts = []
            for start in range(0, len(raw), cfg["smoke"]["bloque_bins"]):
                chunk = stream.push_with_logits(raw[start:start + cfg["smoke"]["bloque_bins"]])
                parts.append(chunk.logits[checkpoint.model.num_layers])
            stream.finish()
            observed = torch.cat(parts).unsqueeze(0)
            torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
            expected_ids = decode_ctc(expected, torch.tensor([expected.shape[1]]), checkpoint.blank_id)[0]
            if stream.hypotheses[checkpoint.model.num_layers] != expected_ids:
                raise RuntimeError("Los fonemas cambian al procesar la frase por bloques.")
            max_error = max(max_error, float((observed - expected).abs().max()))
            checked += 1
    report = {"ok": True, "datos": cfg["smoke"]["origen_datos"], "dispositivo": "cpu", "trigrama": True,
              "datos_reales_comprobados": cfg["smoke"]["origen_datos"] == "t12",
              "pasos": result["completed_steps"], "frases_por_bloques": checked,
              "max_error_logits": max_error, "metricas_recargadas": reloaded,
              "checkpoint": str(output / "mejor_val.pt"), "datos_verificados": cfg["smoke"]["datos_verificados"],
              "procedencia": cfg["smoke"]["procedencia"], **report_data}
    report_path = output / "smoke.json"
    _write_json(report_path, report)
    run.summary.update({"smoke/ok": True, "smoke/datos": cfg["smoke"]["origen_datos"],
                        "smoke/frases_por_bloques": checked, "smoke/max_error_logits": max_error})
    print(f"Smoke OK: señal a texto, recarga y bloques comprobados en {checked} frases. Informe: {report_path}", flush=True)


def _verificar_palabras(
    cfg: dict[str, Any], decoder: OnlineSpeechDecoder, language: TrigramDecoder,
    protocol: dict[str, Any], report: dict[str, Any],
) -> None:
    checkpoint = decoder.checkpoint
    options = cfg["smoke"]["lenguaje"]
    for phrase in cfg["smoke"]["frases_control"]:
        ids = [checkpoint.phones.index(phone) for phone in phrase["fonemas"]]
        path = [checkpoint.blank_id]
        for phone in ids:
            path.extend([phone, phone, checkpoint.blank_id])
        logits = torch.full((len(path), len(checkpoint.phones) + 1), options["logit_bajo"])
        logits[torch.arange(len(path)), torch.tensor(path)] = options["logit_alto"]
        stream = language.start()
        for part in logits.split(cfg["smoke"]["bloque_bins"]):
            stream.push(part)
        if stream.finish() != phrase["texto"]:
            raise RuntimeError(f"El WFST no recupera la frase conocida {phrase['texto']!r}.")
    errors, words, predictions = 0, 0, []
    for session_id, session in enumerate(checkpoint.sessions):
        raw_session = load_session(Path(cfg["datos"]["raiz"]) / "train" / f"{session}.mat")
        for index in protocol["sessions"][session_id]["val"]:
            trial = raw_session.trials[index]
            raw = extract_model_features(trial)
            neural, incremental = decoder.start(session), language.start()
            chunks = []
            for start in range(0, len(raw), cfg["smoke"]["bloque_bins"]):
                logits = neural.push_with_logits(raw[start:start + cfg["smoke"]["bloque_bins"]]).logits[checkpoint.model.num_layers]
                incremental.push(logits)
                chunks.append(logits)
            neural.finish()
            complete = language.start()
            complete.push(torch.cat(chunks))
            text = incremental.finish()
            if text != complete.finish():
                raise RuntimeError("El texto cambia al agrupar los logits del WFST.")
            count, total = word_error_counts(trial.text, text)
            errors, words = errors + count, words + total
            predictions.append({"session": session, "index": index, "reference": trial.text, "text": text})
    report.update(wer=errors / words, errores_palabras=errors, palabras=words, predicciones=predictions,
                  vocabulario_smoke=list(options["lexico"]), frases_conocidas_verificadas=len(cfg["smoke"]["frases_control"]))
    print(f"WFST OK: señal a texto en {len(predictions)} frases; WER del smoke={report['wer']:.2%}.", flush=True)
