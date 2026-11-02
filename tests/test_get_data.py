"""Descargas verificadas y reanudables, probadas con un servidor HTTP local."""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import sys
import tarfile
import threading
from types import SimpleNamespace

from filelock import FileLock
import pytest

from scripts import get_data


@pytest.fixture
def download_server() -> SimpleNamespace:
    """Sirve bytes conocidos y simula bloqueos, cortes y respuestas Range incorrectas."""
    payload = bytes(range(256)) * 4096
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=0) as archive:
        member = tarfile.TarInfo("fixture/sample.bin")
        member.size = len(payload)
        archive.addfile(member, BytesIO(payload))
    data = buffer.getvalue()
    state = SimpleNamespace(data=data, payload=payload, requests=[], cut=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            state.requests.append((self.path, self.headers.get("Range")))
            if self.path == "/blocked":
                self.send_error(403)
                return
            start = 0
            if self.headers.get("Range") and self.path != "/ignore-range":
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
            payload = data[start:]
            if self.path == "/html":
                # Mismo tamaño: solo el SHA-256 puede detectar este falso archivo.
                payload = b"<html>blocked</html>".ljust(len(data), b" ")
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Type", "text/html" if self.path == "/html" else "application/octet-stream")
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            self.end_headers()
            if self.path == "/cut" and state.cut:
                state.cut = False
                payload = payload[:600000]
                self.close_connection = True
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # gdown cierra la consulta inicial antes de pedir el Range.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _entry(server: SimpleNamespace, path: str = "/file") -> dict:
    return {"name": "fixture.tar.gz", "size_bytes": len(server.data),
            "sha256": hashlib.sha256(server.data).hexdigest(), "url": server.url + "/file",
            "mirrors": [{"provider": "google_drive", "url": server.url + path}]}


def _download(entry: dict, root: Path, **kwargs: object) -> Path:
    return get_data.descargar(entry, root, reserva=0, reintentos=0, timeout=3, **kwargs)


def test_download_prefers_mirror_and_publishes_only_verified_bytes(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server, "/mirror")
    output = _download(entry, tmp_path)
    assert output.read_bytes() == download_server.data
    assert download_server.requests == [("/mirror", None)]
    get_data.verificar(output, entry)


def test_drive_failure_reports_cause_without_trying_another_source(
    download_server: SimpleNamespace, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="403"):
        _download(_entry(download_server, "/blocked"), tmp_path)
    assert download_server.requests == [("/blocked", None)]


def test_missing_drive_link_fails_without_network(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = {**_entry(download_server), "mirrors": []}
    with pytest.raises(ValueError, match="No hay enlace de Drive"):
        _download(entry, tmp_path)
    assert download_server.requests == []


def test_blocked_download_shows_both_manual_links_and_destination(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    entry = _entry(download_server)
    other = {**_entry(download_server, "/other"), "name": "other.tar.gz"}
    monkeypatch.setattr(get_data, "leer_manifiesto", lambda _: {"files": [entry, other]})
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(tmp_path), "--files", "all", "--reserve-gib", "0"])

    def fail_download(**_kwargs: object) -> None:
        raise get_data.gdown.FileURLRetrievalError("error original de gdown")

    monkeypatch.setattr(get_data.gdown, "download", fail_download)
    assert get_data.main() == 1
    error = capsys.readouterr().err
    assert "Drive bloquea la descarga" in error
    assert "estos dos archivos" in error and str(tmp_path) in error
    assert "vuelve a ejecutar" in error and "descomprimirlos" in error
    for item in (entry, other):
        assert item["name"] in error
        assert item["mirrors"][0]["url"] in error
    assert "error original de gdown" not in error
    assert download_server.requests == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_manual_archive_is_verified_before_preparing_without_network(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool,
) -> None:
    entry = _entry(download_server)
    original = tmp_path / entry["name"]
    data = b"incompleto" if corrupt else download_server.data
    original.write_bytes(data)
    monkeypatch.setattr(get_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    folder = tmp_path / "speech_2023"
    if corrupt:
        with pytest.raises(ValueError, match="tamaño incorrecto"):
            _download(entry, folder, archivo_local=original)
        assert not (folder / entry["name"]).exists()
    else:
        assert _download(entry, folder, archivo_local=original).read_bytes() == data
    assert original.read_bytes() == data
    assert download_server.requests == []


def test_cli_accepts_both_manual_archives_and_extracts_without_network(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(download_server)
    other = {**entry, "name": "other.tar.gz"}
    for item in (entry, other):
        (tmp_path / item["name"]).write_bytes(download_server.data)
    monkeypatch.setattr(get_data, "leer_manifiesto", lambda _: {"files": [entry, other]})
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(tmp_path), "--files", "all", "--reserve-gib", "0"])
    assert get_data.main() == 0
    for item in (entry, other):
        assert (tmp_path / "speech_2023" / item["name"]).read_bytes() == download_server.data
        assert (tmp_path / item["name"]).is_file()
    assert (tmp_path / "speech_2023/fixture/sample.bin").read_bytes() == download_server.payload
    assert download_server.requests == []


def test_cli_downloads_verifies_and_extracts_archive(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(download_server)
    monkeypatch.setattr(get_data, "leer_manifiesto", lambda _: {"files": [entry]})
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(tmp_path), "--files", "all", "--reserve-gib", "0"])
    assert get_data.main() == 0
    assert (tmp_path / "speech_2023" / entry["name"]).read_bytes() == download_server.data
    assert (tmp_path / "speech_2023/fixture/sample.bin").read_bytes() == download_server.payload
    assert download_server.requests == [("/file", None)]


def _tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    return path


def test_extraction_reuses_existing_files_and_completes_missing_files(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "archive.tar.gz", {"sample/a.bin": b"existing", "sample/b.bin": b"new"})
    original_archive = archive.read_bytes()
    folder = tmp_path / "output"
    existing = folder / "sample/a.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    stat = existing.stat()
    get_data._extraer_tar(archive, folder, 0)
    assert (folder / "sample/b.bin").read_bytes() == b"new"
    (folder / "sample/b.bin").unlink()
    get_data._extraer_tar(archive, folder, 0)
    assert existing.stat().st_ino == stat.st_ino
    assert existing.stat().st_mtime_ns == stat.st_mtime_ns
    assert (folder / "sample/b.bin").read_bytes() == b"new"
    assert archive.read_bytes() == original_archive


def test_conflicting_existing_file_is_preserved_without_publishing_new_files(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "archive.tar.gz", {"sample/new.bin": b"new", "sample/existing.bin": b"good"})
    folder = tmp_path / "output"
    existing = folder / "sample/existing.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"evil")  # Mismo tamaño: hay que comprobar el contenido.
    with pytest.raises(ValueError, match="se conserva"):
        get_data._extraer_tar(archive, folder, 0)
    assert existing.read_bytes() == b"evil"
    assert not (folder / "sample/new.bin").exists()
    assert not list(folder.glob(".extraccion_*"))


@pytest.mark.parametrize("name", ["../escape", "/escape", "nested\\escape"])
def test_extraction_rejects_unsafe_paths(tmp_path: Path, name: str) -> None:
    archive = _tar(tmp_path / "archive.tar.gz", {name: b"data"})
    folder = tmp_path / "output"
    folder.mkdir()
    with pytest.raises(ValueError, match="Ruta no permitida"):
        get_data._extraer_tar(archive, folder, 0)
    assert list(folder.iterdir()) == []
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_extraction_rejects_archive_links(tmp_path: Path, kind: bytes) -> None:
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("sample/link")
        member.type = kind
        member.linkname = "../../outside"
        stream.addfile(member)
    folder = tmp_path / "output"
    folder.mkdir()
    with pytest.raises(ValueError, match="Entrada no permitida"):
        get_data._extraer_tar(archive, folder, 0)
    assert list(folder.iterdir()) == []


def test_extraction_checks_disk_space_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _tar(tmp_path / "archive.tar.gz", {"sample/data.bin": b"data"})
    folder = tmp_path / "output"
    folder.mkdir()
    monkeypatch.setattr(get_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Espacio insuficiente para descomprimir"):
        get_data._extraer_tar(archive, folder, 0)
    assert list(folder.iterdir()) == []
    assert archive.is_file()


def test_interrupted_transfer_is_preserved_and_resumed(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server, "/cut")
    with pytest.raises(ValueError, match="Se conservan los parciales"):
        _download(entry, tmp_path)
    assert not (tmp_path / entry["name"]).exists()
    partial = get_data._parcial(tmp_path, entry["name"])
    size = partial.stat().st_size
    assert 0 < size < len(download_server.data)
    assert partial.read_bytes() == download_server.data[:size]
    output = _download(entry, tmp_path)
    assert output.read_bytes() == download_server.data
    assert ("/cut", f"bytes={size}-") in download_server.requests


def test_legacy_curl_partial_is_resumed(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server)
    (tmp_path / (entry["name"] + ".part")).write_bytes(download_server.data[:73])
    output = _download(entry, tmp_path)
    assert output.read_bytes() == download_server.data
    assert ("/file", "bytes=73-") in download_server.requests


def test_ignored_range_preserves_partial_and_restart_recovers(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server, "/ignore-range")
    (tmp_path / (entry["name"] + ".part")).write_bytes(download_server.data[:73])
    with pytest.raises(ValueError, match="Se conservan los parciales"):
        _download(entry, tmp_path)
    assert get_data._parcial(tmp_path, entry["name"]).read_bytes() == download_server.data[:73]
    output = _download(entry, tmp_path, reiniciar=True)
    assert output.read_bytes() == download_server.data


def test_html_with_expected_length_is_rejected_by_hash(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server, "/html")
    with pytest.raises(ValueError, match="SHA-256"):
        _download(entry, tmp_path)
    assert not (tmp_path / entry["name"]).exists()
    assert get_data._parcial(tmp_path, entry["name"]).is_file()


def test_existing_valid_file_needs_no_network_or_disk_reserve(download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry(download_server)
    output = tmp_path / entry["name"]
    output.write_bytes(download_server.data)
    monkeypatch.setattr(get_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert _download(entry, tmp_path, reiniciar=True) == output
    assert download_server.requests == []


def test_existing_corrupt_file_is_preserved(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server)
    output = tmp_path / entry["name"]
    output.write_bytes(b"original incompleto")
    with pytest.raises(ValueError, match="tamaño incorrecto"):
        _download(entry, tmp_path, reiniciar=True)
    assert output.read_bytes() == b"original incompleto"
    assert download_server.requests == []


def test_completed_partial_is_verified_without_network(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server)
    (tmp_path / (entry["name"] + ".part")).write_bytes(download_server.data)
    assert _download(entry, tmp_path).read_bytes() == download_server.data
    assert download_server.requests == []


def test_simultaneous_download_is_rejected(download_server: SimpleNamespace, tmp_path: Path) -> None:
    entry = _entry(download_server)
    with FileLock(str(tmp_path / f".{entry['name']}.lock")):
        with pytest.raises(ValueError, match="otra descarga"):
            _download(entry, tmp_path)
    assert download_server.requests == []


def test_disk_space_is_checked_before_download(download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="Espacio insuficiente"):
        _download(_entry(download_server), tmp_path)
    assert download_server.requests == []


@pytest.mark.parametrize("name", ["../escape", "..", "nested/file", "nested\\file"])
def test_download_rejects_paths_outside_destination(download_server: SimpleNamespace, tmp_path: Path, name: str) -> None:
    entry = {**_entry(download_server), "name": name}
    with pytest.raises(ValueError, match="nombre no permitido"):
        _download(entry, tmp_path)
    assert download_server.requests == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_verify_only_checks_integrity_without_network(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool,
) -> None:
    entry = _entry(download_server)
    folder = tmp_path / "speech_2023"
    folder.mkdir()
    (folder / entry["name"]).write_bytes(b"corrupt" if corrupt else download_server.data)
    monkeypatch.setattr(get_data, "leer_manifiesto", lambda _: {"files": [entry]})
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(tmp_path),
                                     "--files", entry["name"], "--verify-only"])
    assert get_data.main() == (1 if corrupt else 0)
    assert download_server.requests == []


def test_space_for_whole_selection_is_checked_before_first_transfer(
    download_server: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(download_server)
    other = {**entry, "name": "other.tar.gz"}
    monkeypatch.setattr(get_data, "leer_manifiesto", lambda _: {"files": [entry, other]})
    monkeypatch.setattr(get_data.shutil, "disk_usage", lambda _: SimpleNamespace(free=len(download_server.data)))
    monkeypatch.setattr(sys, "argv", ["get_data.py", "--dest", str(tmp_path), "--files", "all", "--reserve-gib", "0"])
    assert get_data.main() == 1
    assert download_server.requests == []


@pytest.mark.parametrize("option,value", [("--timeout", "nan"), ("--reserve-gib", "inf")])
def test_nonfinite_options_are_rejected(monkeypatch: pytest.MonkeyPatch, option: str, value: str) -> None:
    monkeypatch.setattr(sys, "argv", ["get_data.py", option, value])
    with pytest.raises(SystemExit) as error:
        get_data.main()
    assert error.value.code == 2
