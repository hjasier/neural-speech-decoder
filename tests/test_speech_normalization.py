"""Normalización por bloque y rodante: causalidad, pesos y coherencia con la caché."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat
import yaml

from scripts.prepare_speech_data import prepare_signals
from src.data.preprocessing import (
    TrialMoments, extract_model_features, normalization_spec, pooled_statistics,
    rolling_statistics, session_statistics,
)
from src.data.speech import load_session
from src.data.speech_dataset import SpeechDataset, file_sha256, preparation_signature
from src.data.speech_protocol import build_speech_protocol
from src.training.speech_online_evaluation import iter_online_trials
from src.utils.seed import crear_generador_numpy

ROLLING = {"modo": "rodante", "frases_mezcla": 10, "ventana_frases": 20, "desviacion_minima_relativa": 0.1}


def _moments(values: list[float], block: int = 1) -> list[TrialMoments]:
    """Frases de dos bins y un canal cuya media es el valor indicado."""
    return [TrialMoments(block, 2, np.array([2 * value]), np.array([2 * value ** 2])) for value in values]


# Sin la clave, las configuraciones y cachés anteriores siguen usando solo train.
def test_missing_normalization_key_keeps_the_fixed_train_mode() -> None:
    assert normalization_spec({"preprocesamiento": {}}) == {"modo": "train_fija"}
    with pytest.raises(ValueError):
        normalization_spec({"preprocesamiento": {"normalizacion": {"modo": "rodante"}}})
    with pytest.raises(ValueError):
        normalization_spec({"preprocesamiento": {"normalizacion": {"modo": "sesion"}}})


# Si la frase i usara su propia señal o la posterior, cambiarlas alteraría sus estadísticas.
def test_rolling_statistics_never_use_the_current_or_later_sentences() -> None:
    previous = (np.array([5.0]), np.array([1.0]))
    values = [float(index) for index in range(30)]
    reference = rolling_statistics(_moments(values), previous, blend=10, window=20)
    for position in range(30):
        changed = values[:position] + [1e6] * (30 - position)
        altered = rolling_statistics(_moments(changed), previous, blend=10, window=20)
        for before, after in zip(reference[:position + 1], altered[:position + 1], strict=True):
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])


# Ecuación 2: peso (11 - i)/10 al bloque anterior, (i - 1)/10 al actual y, después, 20 frases.
def test_rolling_statistics_follow_the_paper_weights_and_window() -> None:
    previous = (np.array([100.0]), np.array([4.0]))
    values = [float(index) for index in range(40)]
    rolled = rolling_statistics(_moments(values), previous, blend=10, window=20)
    np.testing.assert_allclose(rolled[0][0], [100.0])
    np.testing.assert_allclose(rolled[0][1], [4.0])
    np.testing.assert_allclose(rolled[1][0], [0.9 * 100.0 + 0.1 * 0.0])
    np.testing.assert_allclose(rolled[9][0], [0.1 * 100.0 + 0.9 * np.mean(values[:9])])
    np.testing.assert_allclose(rolled[9][1], [0.1 * 4.0 + 0.9 * np.std(values[:9])])
    np.testing.assert_allclose(rolled[10][0], [np.mean(values[:10])])
    np.testing.assert_allclose(rolled[35][0], [np.mean(values[15:35])])
    np.testing.assert_allclose(rolled[35][1], [np.std(values[15:35])])


# Train se normaliza por bloque; val y test, de forma causal desde el bloque anterior por número.
def test_session_statistics_assign_blocks_to_train_and_rolling_to_evaluation() -> None:
    train = _moments([1.0, 3.0], block=1) + _moments([10.0, 20.0], block=2)
    test = _moments([7.0, 9.0], block=3)
    splits = {"train": train, "test": test}
    by_block = session_statistics(splits, {"modo": "bloque"}, train_indices=[0, 1])
    np.testing.assert_allclose([item[0][0] for item in by_block["train"]], [2.0, 2.0, 15.0, 15.0])
    np.testing.assert_allclose([item[0][0] for item in by_block["test"]], [8.0, 8.0])

    rolled = session_statistics(splits, ROLLING, train_indices=[0, 1])
    np.testing.assert_allclose([item[0][0] for item in rolled["train"][:2]], [2.0, 2.0])
    np.testing.assert_allclose(rolled["train"][2][0], [2.0])
    np.testing.assert_allclose(rolled["train"][3][0], [0.9 * 2.0 + 0.1 * 10.0])
    np.testing.assert_allclose(rolled["test"][0][0], pooled_statistics(train[2:])[0])
    with pytest.raises(ValueError, match="frases de train"):
        session_statistics(splits, ROLLING, train_indices=[])
    with pytest.raises(ValueError, match="bloque anterior"):
        # El bloque 2 es de evaluación y no hay ninguno con número menor en el día.
        late_train = _moments([1.0, 3.0], block=2) + _moments([5.0, 7.0], block=3)
        session_statistics({"train": late_train, "test": _moments([7.0, 9.0], block=4)},
                           ROLLING, train_indices=[2, 3])


# Caso real (sesión 06.16): un canal callado en las frases previas dividía por epsilon y daba 1e8.
def test_rolling_deviation_has_a_floor_and_silences_channels_without_scale() -> None:
    def trial(block: int, first: list[float], second: list[float]) -> TrialMoments:
        values = np.array([first, second], dtype=np.float64).T
        return TrialMoments(block, 2, values.sum(axis=0), np.square(values).sum(axis=0))

    # Canal 0: varía en train (desviación 2) pero no en el bloque anterior ni en val.
    # Canal 1: no varía nunca, así que no tiene una escala que se pueda usar.
    train = [trial(1, [1.0, 5.0], [0.0, 0.0]), trial(2, [3.0, 3.0], [0.0, 0.0])]
    validation = [trial(3, [3.0, 3.0], [0.0, 0.0]), trial(3, [9.0, 3.0], [4.0, 0.0])]
    rolled = session_statistics({"train": train + validation, "test": [trial(4, [3.0, 3.0], [0.0, 0.0])]},
                                ROLLING, train_indices=[0])
    for mean, scale in rolled["train"][2:]:
        np.testing.assert_allclose(mean[0], 3.0)
        np.testing.assert_allclose(scale[0], 0.1 * 2.0)
        assert np.isinf(scale[1])
    # El valor 9 queda a (9 - 3) / 0,2 = 30 desviaciones, no a 6e8; el canal sin escala vale cero.
    normalized = (np.array([9.0, 4.0]) - rolled["train"][3][0]) / (rolled["train"][3][1] + 1e-8)
    np.testing.assert_allclose(normalized, [30.0, 0.0], rtol=1e-6)


def _fixture(tmp_path: Path, mode: dict) -> dict:
    """Un día con cinco bloques de train y uno de test, etiquetas y partición propias."""
    cfg = yaml.safe_load(Path("configs/entrenos_online/speech_online.yaml").read_text())
    name = "t12.fixture"
    cfg["datos"].update(raiz=str(tmp_path / "crudo"), preparados=str(tmp_path / f"preparados_{mode['modo']}"),
                        sesiones=[name])
    cfg["etiquetas"]["salida"] = str(tmp_path / "etiquetas")
    cfg["preprocesamiento"]["normalizacion"] = mode
    rng = crear_generador_numpy(0)
    files, totals = [], {"train": 0, "test": 0}
    for split, blocks in (("train", [1, 2, 3, 4, 5]), ("test", [6])):
        count = 4 * len(blocks)
        crossings, powers = np.empty((1, count), dtype=object), np.empty((1, count), dtype=object)
        texts, targets = [], []
        for index in range(count):
            # Cada bloque deriva: sin normalización adaptada, su media se desplaza.
            drift = blocks[index // 4]
            crossings[0, index] = rng.poisson(drift, size=(60 + index, 256)).astype(np.float32)
            powers[0, index] = rng.gamma(2.0, 50.0 * drift, size=(60 + index, 256)).astype(np.float32)
            texts.append("a b")
            targets.append({"trial_index": index, "reference": "a b", "ids": [0, 39, 6, 39],
                            "minimum_ctc_steps": 4})
        path = Path(cfg["datos"]["raiz"]) / split / f"{name}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        savemat(path, {"tx1": crossings, "spikePow": powers, "sentenceText": np.asarray(texts),
                       "blockIdx": np.repeat(blocks, 4)})
        label = Path(cfg["etiquetas"]["salida"]) / split / f"{name}.json"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(json.dumps({
            "session": name, "split": split, "targets": targets,
            "transcriptions_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        }))
        files.append({"path": f"{split}/{name}.json", "trials": count, "sha256": file_sha256(label)})
        totals[split] = count
    (Path(cfg["etiquetas"]["salida"]) / "manifest.json").write_text(json.dumps({
        "format_version": 1, "sessions": [name], "phones": cfg["datos"]["fonemas"],
        "blank_id": len(cfg["datos"]["fonemas"]), "label_config": cfg["etiquetas"], "files": files, "totals": totals,
    }))
    manifest = tmp_path / "particion.json"
    payload = json.dumps(build_speech_protocol(cfg), indent=2) + "\n"
    manifest.write_text(payload)
    cfg["protocolo"].update(manifiesto=str(manifest), sha256=hashlib.sha256(payload.encode()).hexdigest())
    return cfg


# La evaluación por bloques debe reconstruir bit a bit la caché desde la señal cruda de cada frase.
@pytest.mark.parametrize("mode", [{"modo": "bloque"}, ROLLING])
def test_prepared_cache_matches_per_trial_statistics_from_raw_signal(tmp_path: Path, mode: dict) -> None:
    cfg = _fixture(tmp_path, mode)
    assert prepare_signals(cfg) == {"train": 16, "val": 4, "test": 4}
    for split in ("val", "test"):
        dataset = SpeechDataset(cfg, split)
        decoder = SimpleNamespace(checkpoint=SimpleNamespace(
            config=cfg, sessions=dataset.sessions, data_signature=preparation_signature(cfg), normalization={},
        ))
        trials = list(iter_online_trials(dataset, decoder))
        assert len(trials) == 4 and all(trial.normalization is not None for trial in trials)
        normalized = np.concatenate([
            trial.normalization.transform_features(trial.raw, trial.session) for trial in trials
        ])
        # La deriva del bloque queda compensada sin mirar etiquetas.
        assert abs(float(normalized.mean())) < (0.05 if mode["modo"] == "bloque" else 0.5)


# Con normalización rodante, alterar frases posteriores de val no puede cambiar las anteriores.
def test_rolling_cache_of_a_sentence_ignores_later_validation_signal(tmp_path: Path) -> None:
    cfg = _fixture(tmp_path, ROLLING)
    prepare_signals(cfg)
    before = SpeechDataset(cfg, "val")[0].features.numpy().copy()
    path = Path(cfg["datos"]["raiz"]) / "train" / "t12.fixture.mat"
    session = load_session(path)
    crossings, powers = np.empty((1, 20), dtype=object), np.empty((1, 20), dtype=object)
    for trial in session.trials:
        scale = 1000.0 if trial.index > 16 else 1.0
        crossings[0, trial.index] = trial.threshold_crossings * scale
        powers[0, trial.index] = trial.spike_power * scale
    savemat(path, {"tx1": crossings, "spikePow": powers, "sentenceText": np.asarray(["a b"] * 20),
                   "blockIdx": np.repeat([1, 2, 3, 4, 5], 4)})
    payload = json.dumps(build_speech_protocol(cfg), indent=2) + "\n"
    Path(cfg["protocolo"]["manifiesto"]).write_text(payload)
    cfg["protocolo"]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    cfg["datos"]["preparados"] += "_alterado"
    prepare_signals(cfg)
    after = SpeechDataset(cfg, "val")
    np.testing.assert_array_equal(before, after[0].features.numpy())
    assert not np.array_equal(
        extract_model_features(session.trials[17]), extract_model_features(load_session(path).trials[17]),
    )
