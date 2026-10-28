"""Reanudación en CPU y procesos nuevos: ruido, dropout, muestreo y Adam reales."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from test_checkpoint_verification import ROOT, _prepared_fixture
from src.data.speech_dataset import SessionBatchSampler
from src.training.speech_resume import flush_metrics, read_training_checkpoint
from src.utils.seed import capturar_azar, crear_generador, fijar_seed, restaurar_azar
from src.utils.tracking import abrir_run


def test_sampler_restores_partly_consumed_days_and_rng():
    def sampler():
        return SessionBatchSampler([range(0, 5), range(5, 12)], 4, 11,
                                   crear_generador(13), resumable=True)
    full = list(sampler())
    first = sampler()
    iterator = iter(first)
    prefix = [next(iterator) for _ in range(4)]
    saved = first.state_dict()
    resumed = sampler()
    resumed.load_state_dict(saved)
    assert prefix + list(resumed) == full
    invalid = copy.deepcopy(saved)
    invalid['cursors'][next(iter(invalid['cursors']))] = 100
    with pytest.raises(ValueError, match='cursor'):
        sampler().load_state_dict(invalid)


def test_all_rng_states_round_trip_without_pickle_allowlisting():
    fijar_seed(17)
    state = capturar_azar(torch.device('cpu'))
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restaurar_azar(state, torch.device('cpu'))
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def test_tracking_online_requires_original_run_and_replay_is_monotonic(monkeypatch):
    calls = []
    run = SimpleNamespace(id='original', resumed=True, step=3, summary={}, finish=lambda **_: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: (calls.append(kw), run)[1]))
    cfg = {'experimento': {'nombre': 'fixture'}, 'modelo': {'nombre': 'speech_gru'},
           'wandb': {'modo': 'online', 'proyecto': 'fixture'}, 'reanudacion': {'cada_pasos': 2}}
    saved = {'logical_id': 'original', 'last_segment_id': 'original', 'entity': 'owner'}
    assert abrir_run(cfg, resume=saved) is run
    assert calls[0]['id'] == 'original' and calls[0]['resume'] == 'must'
    assert calls[0]['entity'] == 'owner'
    published = []
    def log(values, *, step, commit):
        assert commit and step >= run.step
        published.append(step)
        run.step = step + 1
    run.log = log
    rows = [{'step': i, 'values': {'train/loss': i}} for i in range(1, 6)]
    flush_metrics(run, rows, 5)
    flush_metrics(run, rows, 5)
    assert published == [3, 4, 5]
    run.step = 8
    with pytest.raises(ValueError, match='adelantado'):
        flush_metrics(run, rows, 5)


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('variant', ['baseline_5', 'conjunta_235', 'control_2', 'control_3'])
def test_continuous_equals_resumed_in_fresh_cpu_processes(tmp_path, monkeypatch, variant):
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    cfg = _prepared_fixture(fixture, monkeypatch)
    if variant != 'conjunta_235':
        cfg['modelo'].pop('salidas_intermedias')
        cfg['entrenamiento'].pop('ctc_pesos')
        cfg['modelo']['capas'] = {'baseline_5': 5, 'control_2': 2, 'control_3': 3}[variant]
    cfg['modelo'].update(dropout_entrada=0.2, dropout_gru=0.3)
    cfg['entrenamiento'].update(pasos=8, ruido={'blanco_sd': 0.1, 'offset_sd': 0.05})
    cfg['evaluacion']['cada_pasos'] = 3
    cfg['reanudacion'] = {'cada_pasos': 3, 'max_pasos_sesion': None}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}

    def directory(name):
        dest = tmp_path / name
        dest.mkdir()
        for folder in ['labels', 'prepared']:
            shutil.copytree(fixture / folder, dest / folder)
        shutil.copyfile(fixture / 'split.json', dest / 'split.json')
        return dest

    def execute(dest, limit, checkpoint=None, crash=False):
        current = copy.deepcopy(cfg)
        current['reanudacion']['max_pasos_sesion'] = limit
        (dest / 'config.yaml').write_text(yaml.safe_dump(current))
        args = [sys.executable, str(ROOT / 'run.py'), '--config', 'config.yaml']
        if checkpoint is not None:
            args += ['--reanudar', str(checkpoint.relative_to(dest))]
        if crash:
            code = f'''import os,runpy,sys
sys.path.insert(0,{str(ROOT)!r})
from src.training.optim import TensorFlowAdam
original=TensorFlowAdam.step
def interrupted(self, closure=None):
    result=original(self, closure)
    if self.param_groups[0]['iterations']==6: os._exit(91)
    return result
TensorFlowAdam.step=interrupted
sys.argv=['run.py','--config','config.yaml']
runpy.run_path({str(ROOT / 'run.py')!r},run_name='__main__')
'''
            args = [sys.executable, '-c', code]
        result = subprocess.run(args, cwd=dest, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == (91 if crash else 0), result.stdout + result.stderr
        files = list((dest / 'checkpoints/fixture').glob('*/entrenamiento.pt'))
        assert len(files) == 1
        return files[0], torch.load(files[0], weights_only=True)

    continuous = directory('continuous')
    _, full = execute(continuous, None)
    split = directory('split')
    checkpoint, state = execute(split, 2)
    assert state['step'] == 2
    assert [r['step'] for r in state['progress']['history']] == [1]  # Sin evaluación extra por pausa.
    logical_id = state['tracking']['logical_id']
    checkpoint, state = execute(split, 3, checkpoint)
    assert state['step'] == 5 and state['tracking']['logical_id'] == logical_id
    checkpoint, state = execute(split, None, checkpoint)

    def equivalent(actual):
        for key in ['model_state', 'optimizer_state', 'sampler', 'loaders']:
            _assert_tree_equal(full[key], actual[key])
        assert torch.equal(full['rng']['torch'], actual['rng']['torch'])
        assert actual['step'] == 8
        assert [p['step'] for p in actual['progress']['history']] == [1, 4, 7, 8]
        assert actual['progress']['best_step'] == full['progress']['best_step']
        _assert_tree_equal(full['best_checkpoint']['model_state'], actual['best_checkpoint']['model_state'])
        for a, b in zip(full['progress']['metric_history'], actual['progress']['metric_history'], strict=True):
            assert a['step'] == b['step']
            for key, value in a['values'].items():
                if 'segundos' not in key:
                    assert value == b['values'][key], (key, a['step'])
    equivalent(state)
    report = json.loads((checkpoint.parent / 'resultado.json').read_text())
    assert report['training_complete'] and report['completed_steps'] == 8
    assert report['run_id'] == logical_id and report['test_evaluated'] is False
    assert report['tracking_segment_id'] != logical_id  # Offline tiene segmentos separados.
    # Repetir una reanudación ya terminada no añade updates ni evaluaciones.
    _, completed = execute(split, None, checkpoint)
    equivalent(completed)
    if variant == 'conjunta_235':
        abrupt = directory('abrupt')
        lost, state = execute(abrupt, None, crash=True)
        assert state['step'] == 4  # Los pasos 5/6 no alcanzaron el commit.
        _, recovered = execute(abrupt, None, lost)
        equivalent(recovered)

    legacy = split / 'legacy.pt'
    torch.save({'format_version': 2, 'model_state': state['model_state']}, legacy)
    monkeypatch.chdir(split)
    current = yaml.safe_load(Path('config.yaml').read_text())
    with pytest.raises(ValueError, match='antiguos'):
        read_training_checkpoint(legacy, current)
    altered = copy.deepcopy(current)
    altered['entrenamiento']['learning_rate']['inicio'] *= 2
    with pytest.raises(ValueError, match='configuración'):
        read_training_checkpoint(checkpoint, altered)
