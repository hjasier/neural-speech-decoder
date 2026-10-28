"""Credencial privada: prioridad, archivo local/Drive y ausencia de filtrado."""

import os
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
import pytest

from src.utils.env import cargar_clave_wandb


@pytest.mark.parametrize("line", [
    "WANDB_API_KEY=fixture-private-value",
    " export WANDB_API_KEY = 'fixture-private-value' # comentario",
    'WANDB_API_KEY="fixture-private-value"',
    "WANDB_API_KEY=fixture-private-value # comentario",
])
def test_env_loads_only_key_without_output(tmp_path, monkeypatch, capsys, line):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_FIXTURE", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("\ufeff# privado\nUNRELATED_FIXTURE=ignored\n" + line + "\n")
    assert cargar_clave_wandb(dotenv)
    assert os.environ["WANDB_API_KEY"] == "fixture-private-value"
    assert "UNRELATED_FIXTURE" not in os.environ
    assert capsys.readouterr() == ("", "")


def test_env_priority_missing_empty_and_default_root(tmp_path, monkeypatch):
    import src.utils.env as env_module

    monkeypatch.setenv("WANDB_API_KEY", "environment-wins")
    dotenv = tmp_path / ".env"
    dotenv.write_text("WANDB_API_KEY='malformed-private-value\n")
    assert cargar_clave_wandb(dotenv)
    assert os.environ["WANDB_API_KEY"] == "environment-wins"
    monkeypatch.delenv("WANDB_API_KEY")
    dotenv.write_text("WANDB_API_KEY= # empty\n")
    assert not cargar_clave_wandb(tmp_path / "missing", dotenv)
    dotenv.write_text("WANDB_API_KEY=default-root-value\n")
    monkeypatch.setattr(env_module, "__file__", str(tmp_path / "src/utils/env.py"))
    assert cargar_clave_wandb()
    assert os.environ["WANDB_API_KEY"] == "default-root-value"


def test_malformed_key_error_does_not_echo_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("WANDB_API_KEY='private-value-without-closing-quote\n")
    with pytest.raises(ValueError) as error:
        cargar_clave_wandb(dotenv)
    assert "private-value" not in str(error.value)
    assert capsys.readouterr() == ("", "")


def test_env_never_executes_or_expands_values(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("WANDB_API_KEY='${DO_NOT_EXPAND}$(literal)`literal`'\n")
    assert cargar_clave_wandb(dotenv)
    assert os.environ["WANDB_API_KEY"] == '${DO_NOT_EXPAND}$(literal)`literal`'


def test_tracking_loads_dotenv_before_init_without_adding_key_to_config(tmp_path, monkeypatch):
    import src.utils.env as env_module
    from src.utils.tracking import abrir_run

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(env_module, "__file__", str(tmp_path / "src/utils/env.py"))
    (tmp_path / ".env").write_text("WANDB_API_KEY=fixture-from-project\n")
    cfg = {"experimento": {"nombre": "fixture"}, "wandb": {"modo": "online", "proyecto": "tiiac-2026"}}
    original = copy.deepcopy(cfg)
    def init(**options):
        assert os.environ["WANDB_API_KEY"] == "fixture-from-project"
        assert "fixture-from-project" not in repr(options)
        return SimpleNamespace(id="fixture")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    assert abrir_run(cfg).id == "fixture"
    assert cfg == original


@pytest.mark.parametrize("source", ["environment", "project", "drive", "secrets", "prompt"])
def test_colab_auth_reuses_private_key_without_echo(tmp_path, monkeypatch, capsys, source):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    project, drive = tmp_path / "project", tmp_path / "drive"
    project.mkdir(); drive.mkdir()
    private = "fixture-credential-not-real"
    if source == "environment":
        monkeypatch.setenv("WANDB_API_KEY", private)
    elif source in {"project", "drive"}:
        (project if source == "project" else drive).joinpath(".env").write_text("WANDB_API_KEY=" + private)
    calls = []
    def get_secret(_):
        calls.append("secrets")
        if source == "secrets":
            return private
        raise LookupError("fixture missing")
    def prompt(_):
        calls.append("prompt")
        return private
    userdata = SimpleNamespace(get=get_secret, SecretNotFoundError=LookupError,
                               NotebookAccessError=PermissionError, TimeoutException=TimeoutError)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(userdata=userdata))
    monkeypatch.setattr("getpass.getpass", prompt)
    monkeypatch.syspath_prepend(str(Path.cwd()))  # Restaura sys.path al terminar.
    nb = nbformat.read("notebooks/11_entrenamiento_reanudable_colab.ipynb", as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and c.source.startswith("#@title 8."))
    exec(compile(cell.source, "notebook:auth", "exec"),
         {"PROJECT": project, "DRIVE_ROOT": drive, "os": os, "sys": sys})
    assert os.environ["WANDB_API_KEY"] == private
    assert calls == ({"secrets": ["secrets"], "prompt": ["secrets", "prompt"]}.get(source, []))
    assert private not in "".join(capsys.readouterr())
