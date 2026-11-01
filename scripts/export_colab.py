"""Actualiza el código empaquetado en un lanzador Colab sin ejecutar.

    python scripts/export_colab.py --perfil speech_trigrama

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
NOTEBOOKS = {
    "speech_reanudable": Path("notebooks/11_entrenamiento_reanudable_colab.ipynb"),
    "speech_online": Path("notebooks/12_entrenamiento_online_colab.ipynb"),
    "speech_trigrama": Path("notebooks/13_trigrama_colab.ipynb"),
}


def build_source_bundle(perfil: str = "speech_reanudable") -> tuple[bytes, dict[str, Any]]:
    """Empaqueta el flujo T12 vigente y registra los SHA-256 de sus fuentes."""
    if perfil not in NOTEBOOKS:
        raise ValueError("Perfil Colab no reconocido.")
    configs = [Path("configs") / name for name in (
        "speech_comparacion_t4.yaml", "speech_salidas_t4.yaml",
        "speech_control_2_t4.yaml", "speech_control_3_t4.yaml",
        "speech_comparativa.yaml", "speech_reanudacion.yaml",
        "speech_evaluacion.yaml", "speech_benchmark.yaml", "verificacion_checkpoint.yaml",
    )]
    evaluation_scripts = ("evaluate_speech.py", "benchmark_speech.py")
    if perfil in ("speech_online", "speech_trigrama"):
        configs = [Path("configs/speech_reanudacion.yaml"), Path("configs/verificacion_checkpoint.yaml"),
                   *(path.relative_to(ROOT) for path in sorted(
                       (ROOT / "configs/entrenos_online").glob("speech_online*.yaml")))]
        evaluation_scripts = ("evaluate_speech_online.py",)
        if perfil == "speech_trigrama":
            configs.append(Path("configs/entrenos_online/speech_trigrama.yaml"))
            configs.append(Path("configs/entrenos_online/speech_trigrama_compacto.yaml"))
            evaluation_scripts += ("prepare_speech_trigram.py", "evaluate_speech_trigram.py", "trigram_worker.py",
                                   "trigram_mmap_cache.py", "compact_speech_trigram.py", "check_speech_trigram_cpu.py")
    paths = [
        Path("run.py"), Path("requirements.txt"),
        Path("docs/datasets/speech_2023/manifest.json"),
        Path("docs/datasets/speech_2023/particion_t12_v1.json"),
        *configs,
        *(Path("scripts") / name for name in (
            "get_data.py", "get_phoneme_resources.py", "prepare_phoneme_labels.py",
            "prepare_speech_data.py", "report_speech_training.py",
            "configure_speech_comparison.py", "configure_speech_resume.py",
            "verify_speech_checkpoint.py", *evaluation_scripts,
        )),
        *(path.relative_to(ROOT) for path in sorted((ROOT / "src").rglob("*.py"))),
    ]
    if perfil == "speech_trigrama":
        paths.append(Path("requirements-trigram-runtime.txt"))
    files = {path.as_posix(): (ROOT / path).read_bytes() for path in sorted(paths)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "format_version": 1,
        "git_commit_base": revision,
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    if perfil in ("speech_online", "speech_trigrama"):
        manifest["profile"] = perfil
    files["source_snapshot.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in files.items():
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED)
    return buffer.getvalue(), manifest


def export_notebook(path: Path, perfil: str = "speech_reanudable") -> dict[str, Any]:
    """Refresca la celda source_bundle sin reemplazar notebooks ya ejecutados."""
    historical = {
        "01_exploracion_t12.ipynb", "02_test_fonemas_t12.ipynb", "03_pruebas_fonem_bueno_t12.ipynb",
        "04_entrenamiento_colab.ipynb", "05_alice_colab.ipynb", "06_demo_ahpsn.ipynb",
        "07_baseline_t12_colab.ipynb", "08_salidas_t12_colab.ipynb",
        "09_comparativa_t12_colab.ipynb", "10_demo_salidas_local.ipynb",
    }
    if path.name in historical:
        raise ValueError("Conserva los notebooks históricos y sus snapshots; exporta a un lanzador nuevo.")
    for expected, notebook_path in NOTEBOOKS.items():
        if path.name == notebook_path.name and perfil != expected:
            raise ValueError(f"{path.name} requiere --perfil {expected}.")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if any(cell.get("outputs") or cell.get("execution_count") is not None
           for cell in notebook["cells"] if cell["cell_type"] == "code"):
        raise ValueError("El notebook ya tiene ejecuciones: conserva sus salidas y su snapshot original.")
    cells = [cell for cell in notebook["cells"] if "source_bundle" in cell["metadata"].get("tags", [])]
    if len(cells) != 1:
        raise ValueError("El notebook debe contener una celda marcada source_bundle.")
    # Solo se modifica la copia que se exporta; no los snapshots históricos.
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and source.startswith("#@title 3."):
            start = source.index('subprocess.run([UV, "pip", "install"')
            end = source.index('subprocess.run([UV, "pip", "check"', start)
            source = (source[:start]
                      + 'subprocess.run([UV, "pip", "install", "--python", str(PYTHON), '
                        '"-r", "requirements.txt"], cwd=PROJECT, check=True)\n'
                      + source[end:])
        source = source.replace("requirements-e1.lock", "requirements.txt")
        cell["source"] = source.splitlines(keepends=True)
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
    parser.add_argument("--perfil", choices=tuple(NOTEBOOKS), default="speech_online")
    args = parser.parse_args()
    relative = args.notebook or NOTEBOOKS[args.perfil]
    path = relative if relative.is_absolute() else ROOT / relative
    print(json.dumps({"notebook": str(relative), **export_notebook(path, args.perfil)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
