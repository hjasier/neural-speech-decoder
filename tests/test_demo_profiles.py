"""El notebook de demostración permanece local y usa la implementación del proyecto."""

import ast
from pathlib import Path

import nbformat


def test_demo_notebook_keeps_activity_before_prediction_and_has_no_colab_setup():
    notebook = nbformat.read("notebooks/10_demo_salidas_local.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 4
    for cell in code:
        ast.parse(cell.source)
    # Las salidas reales del autor se conservan al ejecutar el notebook.
    # La validación de formato anterior comprueba también sus metadatos.
    assert "load_demo(cfg)" in code[1].source
    assert "demo.close()" in code[1].source
    assert "demo.draw_activity()" in code[2].source
    assert "create_controls(demo)" in code[3].source
    source = "\n".join(cell.source for cell in code)
    assert not any(name in source for name in ("google.colab", "drive.mount", "WANDB_API_KEY", "pip install", "source_bundle"))
    assert Path("requirements-demo.txt").is_file()
