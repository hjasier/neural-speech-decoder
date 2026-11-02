"""Verificación de T12 y preparación de un subconjunto real para el smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat
import yaml

from src.data.speech import load_session
from src.data.speech_dataset import file_sha256, read_label_manifest
from src.data.speech_protocol import read_speech_protocol


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return file_sha256(path)


def comprobar_datos_smoke(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Exige originales y etiquetas verificables antes de crear o entrenar nada.

    Comprueba los archivos de las 19 sesiones del perfil de referencia,
    aunque el entrenamiento corto solo vaya a utilizar dos de ellas.
    El archivo comprimido también se verifica si se ha conservado.
    """
    source_path = Path(cfg["smoke"]["config_datos"])
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    protocol = read_speech_protocol(source)
    if protocol is None:
        raise ValueError("El smoke necesita la partición fija de T12.")
    if source["datos"]["fonemas"] != cfg["datos"]["fonemas"]:
        raise ValueError("El vocabulario del smoke no coincide con el de los datos T12.")
    selected = set(cfg["datos"]["sesiones"])
    if not selected or not selected <= set(source["datos"]["sesiones"]):
        raise ValueError("Las sesiones del smoke deben pertenecer al perfil T12 de referencia.")
    originals = [(Path(source["datos"]["raiz"]) / entry["path"], entry["sha256"])
                 for session in protocol["sessions"] for entry in session["sources"].values()]
    missing = [str(path) for path, _ in originals if not path.is_file()]
    download_help = (
        "Descarga y extrae los datos con:\n"
        "  python scripts/get_data.py --dataset speech_2023 --files competitionData.tar.gz"
    )
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} archivos originales de T12: "
                                + ", ".join(missing[:5]) + "\n" + download_help)
    labels_help = (
        f"Prepara las etiquetas con:\n"
        f"  python scripts/get_phoneme_resources.py --config {source_path}\n"
        f"  python scripts/prepare_phoneme_labels.py --config {source_path}"
    )
    try:
        labels = read_label_manifest(source)
        expected_labels = {f"{split}/{session['session']}.json": session["sources"][split]["trials"]
                           for session in protocol["sessions"] for split in ("train", "test")}
        actual_labels = {entry["path"]: entry for entry in labels["files"]}
        if set(actual_labels) != set(expected_labels):
            raise ValueError("Las etiquetas no cubren las sesiones y particiones de T12.")
        for relative, trials in expected_labels.items():
            path = Path(source["etiquetas"]["salida"]) / relative
            entry = actual_labels[relative]
            if not path.is_file() or entry["trials"] != trials or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Faltan etiquetas o han cambiado: {path}")
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"No se pueden usar las etiquetas T12: {error}\n{labels_help}") from error
    print(f"Comprobando SHA-256 de {len(originals)} archivos originales de T12…", flush=True)
    for path, expected in originals:
        if file_sha256(path) != expected:
            raise ValueError(f"El SHA-256 no coincide: {path}. Los datos están incompletos o alterados.\n{download_help}")
    archive = Path(source["datos"]["raiz"]).parent / "competitionData.tar.gz"
    if archive.is_file():
        catalog = json.loads(Path(cfg["smoke"]["manifiesto_descarga"]).read_text(encoding="utf-8"))
        entry = next(item for item in catalog["files"] if item["name"] == archive.name)
        print(f"Comprobando tamaño y SHA-256 de {archive.name}…", flush=True)
        if archive.stat().st_size != entry["size_bytes"] or file_sha256(archive) != entry["sha256"]:
            raise ValueError(f"La descarga no supera la verificación: {archive}.\n{download_help}")
    report = {"archivos_originales_verificados": len(originals), "etiquetas_verificadas": len(expected_labels),
              "archivo_descarga_verificado": archive.is_file(), "protocolo_origen_sha256": source["protocolo"]["sha256"]}
    print(f"Datos T12 OK: {len(originals)} originales y {len(expected_labels)} archivos de etiquetas.", flush=True)
    return source, protocol, report


def preparar_subconjunto_smoke(
    cfg: dict[str, Any], source: dict[str, Any], protocol: dict[str, Any], root: Path,
) -> dict[str, Any]:
    """Copia unas pocas frases reales conservando su separación train/val/test.

    Solo escribe en la carpeta de esta ejecución. Los índices y bloques
    originales se conservan en la procedencia; los bloques del subconjunto
    agrupan train y val para reutilizar la preparación habitual sin fugas.
    """
    partitions = {entry["session"]: entry for entry in protocol["sessions"]}
    counts = cfg["smoke"]["frases_por_sesion"]
    files, totals, provenance = [], {"train": 0, "test": 0}, {}
    for session in cfg["datos"]["sesiones"]:
        part = partitions[session]
        selection = {split: part[split][:count] for split, count in counts.items()}
        if any(len(selection[split]) != count for split, count in counts.items()):
            raise ValueError(f"No hay suficientes frases en {session} para el smoke configurado.")
        provenance[session] = selection
        for split, indices in (("train", selection["train"] + selection["val"]), ("test", selection["test"])):
            raw = load_session(Path(source["datos"]["raiz"]) / split / f"{session}.mat")
            labels = json.loads((Path(source["etiquetas"]["salida"]) / split / f"{session}.json").read_text())
            if len(raw.trials) != part["sources"][split]["trials"] or len(labels["targets"]) != len(raw.trials):
                raise ValueError(f"No coinciden los ensayos y las etiquetas originales: {split}/{session}")
            crossings, powers = np.empty((1, len(indices)), dtype=object), np.empty((1, len(indices)), dtype=object)
            targets, texts, blocks = [], [], []
            for index, original_index in enumerate(indices):
                trial, target = raw.trials[original_index], labels["targets"][original_index]
                if target["trial_index"] != original_index or target["reference"] != trial.text:
                    raise ValueError(f"Referencia desalineada: {split}/{session}, ensayo {original_index}")
                crossings[0, index], powers[0, index] = trial.threshold_crossings, trial.spike_power
                texts.append(trial.text)
                blocks.append(1 if split == "test" or index < counts["train"] else 2)
                targets.append({**target, "trial_index": index, "source_trial_index": original_index,
                                "source_block": trial.block})
            destination = Path(cfg["datos"]["raiz"]) / split / f"{session}.mat"
            destination.parent.mkdir(parents=True, exist_ok=True)
            savemat(destination, {"tx1": crossings, "spikePow": powers, "sentenceText": np.asarray(texts), "blockIdx": blocks})
            relative = f"{split}/{session}.json"
            digest = _write_json(Path(cfg["etiquetas"]["salida"]) / relative, {
                "session": session, "split": split, "targets": targets,
                "transcriptions_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
            })
            files.append({"path": relative, "sha256": digest, "trials": len(targets)})
            totals[split] += len(targets)
            del raw, crossings, powers, trial
    _write_json(Path(cfg["etiquetas"]["salida"]) / "manifest.json", {
        "format_version": 1, "sessions": cfg["datos"]["sesiones"], "phones": cfg["datos"]["fonemas"],
        "blank_id": len(cfg["datos"]["fonemas"]), "label_config": cfg["etiquetas"], "totals": totals, "files": files,
    })
    _write_json(root / "procedencia.json", {"source_protocol_sha256": source["protocolo"]["sha256"], "sessions": provenance})
    return provenance
