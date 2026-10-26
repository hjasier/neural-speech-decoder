"""Actualiza el código empaquetado en el notebook de entrenamiento Colab.

    python scripts/export_colab.py

Solo incluye fuentes, configuración, dependencias y el manifiesto público
de Dryad. No empaqueta datos, pesos, credenciales, Git ni el entorno local.
No instala paquetes ni ejecuta el modelo.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import textwrap
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path("notebooks/04_entrenamiento_colab.ipynb")


def build_source_bundle(perfil: str = "speech") -> tuple[bytes, dict[str, Any]]:
    """Empaqueta una lista explícita de archivos y registra sus SHA-256."""
    if perfil not in {"speech", "alice", "speech_comparacion", "speech_salidas", "speech_comparativa"}:
        raise ValueError(f"Perfil desconocido: {perfil}")
    paths = [
        Path("run.py"), Path("requirements.txt"), Path("configs/speech_original.yaml"),
        Path("configs/speech_2h.yaml"),
        Path("docs/datasets/speech_2023/manifest.json"),
        *(Path("scripts") / name for name in (
            "get_data.py", "get_phoneme_resources.py",
            "prepare_phoneme_labels.py", "prepare_speech_data.py",
        )),
        *(path.relative_to(ROOT) for path in sorted((ROOT / "src").rglob("*.py"))),
    ]
    if perfil == "alice":
        paths = [Path("run.py"), Path("requirements-alice.txt"),
                 Path("configs/alice_ctc.yaml"), Path("configs/alice_ctc_cpu.yaml"),
                 Path("configs/alice_ctc_t4.yaml"),
                 Path("docs/datasets/alice_eeg/catalogo_ctc.json"),
                 Path("docs/datasets/alice_eeg/catalogo_ctc_t4.json"),
                 *(Path("scripts") / name for name in ("get_alice.py", "get_phoneme_resources.py", "prepare_alice_ctc.py", "report_alice_ctc.py")),
                 *(path.relative_to(ROOT) for path in sorted((ROOT / "src").rglob("*.py")))]
    elif perfil in {"speech_comparacion", "speech_salidas", "speech_comparativa"}:
        paths = [
            Path("run.py"), Path("requirements.txt"),
            Path("configs/speech_comparacion.yaml"), Path("configs/speech_comparacion_t4.yaml"),
            Path("docs/datasets/speech_2023/manifest.json"),
            Path("docs/datasets/speech_2023/particion_t12_v1.json"),
            *(Path("scripts") / name for name in (
                "get_data.py", "get_phoneme_resources.py", "prepare_phoneme_labels.py",
                "prepare_speech_data.py", "report_speech_training.py",
            )),
            *(path.relative_to(ROOT) for path in sorted((ROOT / "src").rglob("*.py"))),
        ]
        if perfil in {"speech_salidas", "speech_comparativa"}:
            paths += [Path("configs/speech_salidas_t4.yaml"), Path("configs/speech_salidas_e1.yaml"),
                      Path("configs/verificacion_checkpoint.yaml"), Path("scripts/verify_speech_checkpoint.py")]
        if perfil == "speech_comparativa":
            paths += [Path("configs/speech_control_2_t4.yaml"), Path("configs/speech_control_3_t4.yaml"),
                      Path("configs/speech_comparativa.yaml"), Path("scripts/configure_speech_comparison.py"),
                      Path("configs/speech_evaluacion.yaml"), Path("scripts/evaluate_speech.py"),
                      Path("configs/speech_benchmark.yaml"), Path("scripts/benchmark_speech.py")]
    files = {path.as_posix(): (ROOT / path).read_bytes() for path in sorted(paths)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "format_version": 1,
        "git_commit_base": revision,
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    files["source_snapshot.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in files.items():
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED)
    return buffer.getvalue(), manifest


def export_notebook(path: Path, perfil: str = "speech") -> dict[str, Any]:
    """Refresca la celda marcada source_bundle y elimina salidas del notebook."""
    if perfil in {"speech_comparacion", "speech_salidas", "speech_comparativa"} and path.name in {
        "04_entrenamiento_colab.ipynb", "05_alice_colab.ipynb", "06_demo_ahpsn.ipynb",
    }:
        raise ValueError("El perfil de comparación debe usar otro notebook; se conservan los históricos.")
    if perfil in {"speech_salidas", "speech_comparativa"} and path.name == "07_baseline_t12_colab.ipynb":
        raise ValueError("El perfil de salidas debe conservar el notebook 07 del baseline.")
    if perfil == "speech_comparativa" and path.name == "08_salidas_t12_colab.ipynb":
        raise ValueError("El perfil comparativo debe conservar el notebook 08 de la demo.")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if "source_bundle" in cell["metadata"].get("tags", [])]
    if len(cells) != 1:
        raise ValueError("El notebook debe contener una celda marcada source_bundle.")
    payload, manifest = build_source_bundle(perfil)
    digest = hashlib.sha256(payload).hexdigest()
    chunks = "\n".join(f"    {chunk!r}" for chunk in textwrap.wrap(base64.b64encode(payload).decode(), 120))
    header = (
        "#@title 2. Copiar el código del proyecto (snapshot incluido)\n"
        "import base64\nimport hashlib\nimport io\nimport json\n"
        "from pathlib import PurePosixPath\nfrom zipfile import ZipFile\n\n"
        f"SOURCE_SHA256 = {digest!r}\nSOURCE_B64 = (\n{chunks}\n)\n"
    )
    extraction = '''
SOURCE_BYTES = base64.b64decode(SOURCE_B64)
if hashlib.sha256(SOURCE_BYTES).hexdigest() != SOURCE_SHA256:
    raise ValueError("El paquete de código no supera su SHA-256.")
PROJECT.mkdir(parents=True, exist_ok=True)
with ZipFile(io.BytesIO(SOURCE_BYTES)) as archive:
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or ".." in relative.parts or member.is_dir():
            raise ValueError(f"Ruta inesperada en el paquete: {member.filename}")
        destination = PROJECT.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
SOURCE_MANIFEST = json.loads((PROJECT / "source_snapshot.json").read_text())
for name, expected in SOURCE_MANIFEST["files"].items():
    if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Código distinto del snapshot: {name}")
print("Código preparado:", len(SOURCE_MANIFEST["files"]), "archivos")
print("Commit base:", SOURCE_MANIFEST["git_commit_base"])
print("SHA-256 del paquete:", SOURCE_SHA256)
'''
    cells[0]["source"] = (header + extraction).splitlines(keepends=True)
    notebook["metadata"]["source_snapshot"] = {"sha256": digest, **manifest}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"files": len(manifest["files"]), "bytes": len(payload), "sha256": digest}


def main() -> int:
    """Genera el snapshot dentro del notebook indicado, sin entrenar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--perfil", choices=("speech", "alice", "speech_comparacion", "speech_salidas", "speech_comparativa"), default="speech")
    args = parser.parse_args()
    notebooks = {"speech": NOTEBOOK, "alice": Path("notebooks/05_alice_colab.ipynb"),
                 "speech_comparacion": Path("notebooks/07_baseline_t12_colab.ipynb"),
                 "speech_salidas": Path("notebooks/08_salidas_t12_colab.ipynb"),
                 "speech_comparativa": Path("notebooks/09_comparativa_t12_colab.ipynb")}
    relative = args.notebook or notebooks[args.perfil]
    path = relative if relative.is_absolute() else ROOT / relative
    print(json.dumps({"notebook": str(relative), **export_notebook(path, args.perfil)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
