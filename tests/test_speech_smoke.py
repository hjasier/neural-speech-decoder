"""El smoke exige datos descargados íntegros y recorre el pipeline de habla."""

import json
import hashlib
import os
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import pytest
import numpy as np
from scipy.io import savemat
import yaml

from scripts.smoke_speech_data import comprobar_datos_smoke
from src.data.speech_dataset import file_sha256
from src.data.speech_protocol import build_speech_protocol
from src.models.factory import construir_modelo
from src.utils.seed import capturar_azar, fijar_seed, restaurar_azar


def _clean_copy(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "run.py", tmp_path / "run.py")
    shutil.copytree(root / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__", "static"))
    (tmp_path / "scripts").mkdir()
    for name in ("smoke_speech.py", "smoke_speech_data.py", "smoke_speech_synthetic.py", "smoke_trigram.py", "prepare_speech_data.py", "trigram_worker.py", "trigram_mmap_cache.py"):
        shutil.copy2(root / "scripts" / name, tmp_path / "scripts" / name)
    (tmp_path / "configs").mkdir()
    shutil.copy2(root / "configs/smoke.yaml", tmp_path / "configs/smoke.yaml")
    (tmp_path / "configs/entrenos_online").mkdir()
    shutil.copy2(root / "configs/entrenos_online/speech_trigrama_compacto.yaml",
                 tmp_path / "configs/entrenos_online/speech_trigrama_compacto.yaml")
    shutil.copy2(root / "configs/entrenos_online/speech_online.yaml",
                 tmp_path / "configs/entrenos_online/speech_online.yaml")
    return root


def _json_file(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_sha256(path)


def _download_fixture(tmp_path: Path) -> dict:
    """Archivos mínimos de prueba; el comando público no genera estos datos."""
    cfg = yaml.safe_load((tmp_path / "configs/smoke.yaml").read_text())
    source_path = tmp_path / cfg["smoke"]["config_datos"]
    source = yaml.safe_load(source_path.read_text())
    source["datos"]["sesiones"] = cfg["datos"]["sesiones"] + ["t12.fixture_extra"]
    files, totals = [], {"train": 0, "test": 0}
    for session in source["datos"]["sesiones"]:
        for split, count in (("train", 6), ("test", 2)):
            crossings, powers = np.empty((1, count), dtype=object), np.empty((1, count), dtype=object)
            texts, targets = [], []
            for index in range(count):
                phrase = cfg["smoke"]["frases_control"][index % 3]
                texts.append(phrase["texto"])
                ids = [cfg["datos"]["fonemas"].index(phone) for phone in phrase["fonemas"]]
                targets.append({"trial_index": index, "reference": phrase["texto"], "ids": ids,
                                "minimum_ctc_steps": len(ids)})
                crossings[0, index] = np.full((54 + index * 4, 256), index + 1, dtype=np.float32)
                powers[0, index] = crossings[0, index] * 2
            path = tmp_path / source["datos"]["raiz"] / split / f"{session}.mat"
            path.parent.mkdir(parents=True, exist_ok=True)
            savemat(path, {"tx1": crossings, "spikePow": powers, "sentenceText": np.asarray(texts),
                           "blockIdx": np.arange(count) // 2 + 1})
            relative = f"{split}/{session}.json"
            digest = _json_file(tmp_path / source["etiquetas"]["salida"] / relative, {
                "session": session, "split": split, "targets": targets,
                "transcriptions_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
            })
            files.append({"path": relative, "trials": count, "sha256": digest})
            totals[split] += count
    _json_file(tmp_path / source["etiquetas"]["salida"] / "manifest.json", {
        "format_version": 1, "sessions": source["datos"]["sesiones"], "phones": source["datos"]["fonemas"],
        "blank_id": len(source["datos"]["fonemas"]), "label_config": source["etiquetas"], "files": files, "totals": totals,
    })
    source_path.write_text(yaml.safe_dump(source))
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        protocol = build_speech_protocol(source)
    finally:
        os.chdir(previous)
    source["protocolo"]["sha256"] = _json_file(tmp_path / source["protocolo"]["manifiesto"], protocol)
    source_path.write_text(yaml.safe_dump(source))
    archive = tmp_path / "data/speech_2023/competitionData.tar.gz"
    archive.write_bytes(b"verified download fixture")
    _json_file(tmp_path / cfg["smoke"]["manifiesto_descarga"], {"files": [{
        "name": archive.name, "size_bytes": archive.stat().st_size, "sha256": file_sha256(archive),
    }]})
    return cfg


def _run_smoke(tmp_path: Path, *, mode: str | None = "t12", github_actions: bool = False) -> subprocess.CompletedProcess:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("WANDB_", "LIVEKIT_"))}
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    environment["GITHUB_ACTIONS"] = "true" if github_actions else "false"
    if mode is None:
        environment.pop("TIIAC_SMOKE_DATOS", None)
    else:
        environment["TIIAC_SMOKE_DATOS"] = mode
    return subprocess.run(
        [sys.executable, "run.py", "--smoke"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=90, check=False,
    )


def test_smoke_verifies_downloads_then_trains_reloads_and_decodes_words(tmp_path: Path) -> None:
    """Con entradas verificadas recorre señal a texto sin tocar los originales."""
    root = _clean_copy(tmp_path)
    _download_fixture(tmp_path)
    before = {path: file_sha256(path) for path in (tmp_path / "data/speech_2023").rglob("*.mat")}
    runtime = root / "data/language_models/runtime"
    if not (runtime / "bin/python").is_file():
        pytest.skip("La integración WFST necesita el runtime CPU del notebook 13.")
    (tmp_path / "data/language_models").mkdir(parents=True)
    (tmp_path / "data/language_models/runtime").symlink_to(runtime, target_is_directory=True)
    completed = _run_smoke(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Smoke OK" in completed.stdout
    assert {path: file_sha256(path) for path in before} == before
    assert not (tmp_path / "data/language_models/speech_trigram_compacto").exists()
    assert not (tmp_path / "checkpoints").exists()
    reports = list((tmp_path / "runs/smoke").glob("*/checkpoints/*/smoke.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["ok"] and report["datos"] == "t12"
    assert report["datos_reales_comprobados"] is True
    assert report["datos_verificados"]["archivos_originales_verificados"] == 6
    assert report["datos_verificados"]["etiquetas_verificadas"] == 6
    assert report["datos_verificados"]["archivo_descarga_verificado"] is True
    for selection in report["procedencia"].values():
        assert selection == {"train": [0, 1, 2, 3], "val": [4, 5], "test": [0, 1]}
    assert report["dispositivo"] == "cpu" and report["trigrama"] is True
    assert report["vocabulario_smoke"] == ["ah", "bee", "see"]
    assert report["frases_conocidas_verificadas"] == 3
    assert len(report["predicciones"]) == 4 and isfinite(report["wer"])
    assert report["pasos"] == 3 and report["frases_por_bloques"] == 4
    assert report["max_error_logits"] < 1e-6
    result = json.loads((reports[0].parent / "resultado.json").read_text())
    assert result["test_evaluated"] is False
    assert result["config"]["wandb"]["modo"] == "offline"
    state = torch.load(tmp_path / report["checkpoint"], map_location="cpu", weights_only=True)
    previous = capturar_azar(torch.device("cpu"))
    try:
        fijar_seed(state["config"]["experimento"]["seed"])
        initial = construir_modelo(state["config"])
    finally:
        restaurar_azar(previous, torch.device("cpu"))
    assert not torch.equal(initial.output_layer.weight, state["model_state"]["output_layer.weight"])
    assert all(torch.isfinite(value).all() for value in state["model_state"].values())
    assert state["val_metrics"]["per"] == report["metricas_recargadas"]["per"]


def test_smoke_fails_instead_of_skipping_the_word_decoder(tmp_path: Path) -> None:
    """Sin runtime no puede anunciar que el pipeline entero está comprobado."""
    _clean_copy(tmp_path)
    _download_fixture(tmp_path)
    completed = _run_smoke(tmp_path)
    assert completed.returncode != 0
    assert "Falta el runtime CPU del trigrama" in completed.stderr
    assert "Smoke OK" not in completed.stdout
    assert not list((tmp_path / "runs").rglob("smoke.json"))


def test_smoke_missing_download_fails_with_download_and_extraction_command(tmp_path: Path) -> None:
    """No inventa señales ni empieza a entrenar cuando falta competitionData."""
    _clean_copy(tmp_path)
    # La partición es una fuente del repositorio, incluso antes de descargar.
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "docs/datasets/speech_2023"
    destination.mkdir(parents=True)
    shutil.copy2(root / "docs/datasets/speech_2023/particion_t12_v1.json", destination)
    completed = _run_smoke(tmp_path)
    assert completed.returncode == 2
    assert "Faltan 38 archivos originales de T12" in completed.stderr
    assert "python scripts/get_data.py" in completed.stderr and "tar -xzf" not in completed.stderr
    assert "Smoke OK" not in completed.stdout and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("damaged", ["original", "etiquetas", "descarga"])
def test_smoke_rejects_changed_downloads_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damaged: str,
) -> None:
    """Detecta corrupción, también en sesiones que no usa el entrenamiento corto."""
    _clean_copy(tmp_path)
    cfg = _download_fixture(tmp_path)
    paths = {
        "original": "data/speech_2023/competitionData/train/t12.fixture_extra.mat",
        "etiquetas": "data/speech_2023/fonemas_original/train/t12.fixture_extra.json",
        "descarga": "data/speech_2023/competitionData.tar.gz",
    }
    path = tmp_path / paths[damaged]
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        comprobar_datos_smoke(cfg)
    assert not (tmp_path / "runs").exists()


def test_smoke_requires_labels_and_explains_how_to_prepare_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tener las señales descargadas no basta si faltan sus etiquetas."""
    _clean_copy(tmp_path)
    cfg = _download_fixture(tmp_path)
    (tmp_path / "data/speech_2023/fonemas_original/manifest.json").unlink()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="python scripts/prepare_phoneme_labels.py"):
        comprobar_datos_smoke(cfg)
    assert not (tmp_path / "runs").exists()


def test_smoke_accepts_verified_extracted_files_without_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eliminar el tar tras extraerlo no invalida unos originales correctos."""
    _clean_copy(tmp_path)
    cfg = _download_fixture(tmp_path)
    (tmp_path / "data/speech_2023/competitionData.tar.gz").unlink()
    monkeypatch.chdir(tmp_path)
    _, _, report = comprobar_datos_smoke(cfg)
    assert report["archivos_originales_verificados"] == 6
    assert report["archivo_descarga_verificado"] is False


def test_smoke_in_github_actions_uses_synthetic_inputs_without_t12(tmp_path: Path) -> None:
    """CI ejecuta la GRU y el WFST reales, pero declara que no ha verificado T12."""
    root = _clean_copy(tmp_path)
    runtime = root / "data/language_models/runtime"
    if not (runtime / "bin/python").is_file():
        pytest.skip("La integración WFST necesita el runtime CPU del notebook 13.")
    (tmp_path / "data/language_models").mkdir(parents=True)
    (tmp_path / "data/language_models/runtime").symlink_to(runtime, target_is_directory=True)
    completed = _run_smoke(tmp_path, mode=None, github_actions=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DATOS SINTÉTICOS" in completed.stdout and "Smoke OK" in completed.stdout
    assert not (tmp_path / "data/speech_2023").exists()
    reports = list((tmp_path / "runs/smoke").glob("*/checkpoints/*/smoke.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["datos"] == "sinteticos" and report["datos_reales_comprobados"] is False
    assert report["datos_verificados"]["archivos_originales_verificados"] == 0
    assert report["trigrama"] is True and report["frases_por_bloques"] == 4


def test_github_actions_can_require_real_data_explicitly(tmp_path: Path) -> None:
    """El modo explícito t12 conserva la comprobación de descargas incluso en CI."""
    _clean_copy(tmp_path)
    completed = _run_smoke(tmp_path, mode="t12", github_actions=True)
    assert completed.returncode == 2 and "Smoke OK" not in completed.stdout
    assert "DATOS SINTÉTICOS" not in completed.stdout
