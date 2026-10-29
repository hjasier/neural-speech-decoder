"""Snapshot reanudable y transferencia de la ruta correcta al proceso Colab."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import pytest
import yaml

from scripts.export_colab import export_notebook


def test_resume_notebook_extracts_verified_sources_without_touching_older_ones(tmp_path, monkeypatch):
    nb = nbformat.read('notebooks/11_entrenamiento_reanudable_colab.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        # Conservar las salidas de la ejecución real del autor del notebook 11.
        for node in ast.walk(ast.parse(cell.source)):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and any(
                        isinstance(target, ast.Name) and target.id in {'config_code', 'probe', 'verification', 'extract', 'verify'}
                        for target in node.targets)):
                compile(node.value.value, 'notebook:embedded', 'exec')
    bundle = next(c for c in nb.cells if 'source_bundle' in c.metadata.get('tags', []))
    namespace = {'PROJECT': tmp_path / 'project'}
    exec(compile(bundle.source, 'source_bundle', 'exec'), namespace)
    assert namespace['SOURCE_MANIFEST']['files'] == nb.metadata.source_snapshot.files
    assert not any(Path(name).name.startswith('.env') for name in namespace['SOURCE_MANIFEST']['files'])
    project = namespace['PROJECT']
    assert (project / 'src/training/speech_resume.py').is_file()
    assert (project / 'requirements-e1.lock').is_file()
    result = subprocess.run([sys.executable, 'scripts/configure_speech_resume.py', '--variant', 'conjunta_235',
                             '--seed-index', '2', '--output', 'configs/active.yaml'], cwd=project,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load((project / 'configs/active.yaml').read_text())
    original = yaml.safe_load((project / 'configs/speech_salidas_t4.yaml').read_text())
    assert cfg['modelo'] == original['modelo']
    assert cfg['entrenamiento'] == original['entrenamiento']
    assert cfg['protocolo'] == original['protocolo']
    assert cfg['experimento']['seed'] == 2
    assert cfg['reanudacion'] == {'cada_pasos': 250, 'max_pasos_sesion': None}
    for name in Path('notebooks').glob('*.ipynb'):
        if name.name.startswith('11_'):
            continue
        with pytest.raises(ValueError, match='conservar'):
            export_notebook(tmp_path / name.name, perfil='speech_reanudable')
        assert not (tmp_path / name.name).exists()


@pytest.mark.parametrize('resume', [False, True])
def test_colab_launcher_detects_fresh_and_resumed_run_folders(tmp_path, monkeypatch, resume):
    nb = nbformat.read('notebooks/11_entrenamiento_reanudable_colab.ipynb', as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == 'code' and c.source.startswith('#@title 9.'))
    project, results, drive = (tmp_path / name for name in ['project', 'results', 'drive'])
    for p in (project, results, drive): p.mkdir()
    (project / 'config.yaml').write_text('fixture: true\n')
    folder = drive / 'fixture' / 'id_fixture'
    def create_files():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'entrenamiento.pt').write_bytes(b'fixture')
        (folder / 'mejor_val.pt').write_bytes(b'fixture')
    if resume: create_files()
    commands = []
    class Process:
        returncode = 0
        stdout = ['fixture output\n']
        def wait(self, **_): return 0
        def poll(self): return 0
    def popen(command, **_):
        commands.append(command)
        create_files()
        return Process()
    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setenv('WANDB_API_KEY', 'fixture-not-a-credential')
    relative = 'checkpoints/fixture/id_fixture/entrenamiento.pt'
    scope = {'PROJECT': project, 'RESULTS': results, 'CHECKPOINTS': drive,
             'CONFIG': 'config.yaml', 'cfg': {'salida': {'directorio': 'checkpoints/fixture'}},
             'PYTHON': Path(sys.executable), 'Path': Path, 'os': os, 'json': json,
             'shutil': __import__('shutil'), 'subprocess': subprocess,
             'RESUME_FILE': folder / 'entrenamiento.pt' if resume else None,
             'RESUME_ARGUMENT': relative if resume else None}
    exec(compile(cell.source, 'notebook:launcher', 'exec'), scope)
    assert scope['LAST_CHECKPOINT'] == folder / 'mejor_val.pt'
    assert scope['LAST_STATUS']['entrenamiento'] == 'fixture/id_fixture/entrenamiento.pt'
    assert scope['LAST_STATUS']['completo'] is True
    if resume:
        assert commands[0][-2:] == ['--reanudar', relative]
    else:
        assert '--reanudar' not in commands[0]
