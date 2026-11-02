"""Los ajustes de compilación no deben dejar modificadas las fuentes oficiales."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts.prepare_speech_trigram import mapped_binding, runtime_build_threads


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_build_restores_sources_after_success_or_failure(tmp_path: Path, mapped: bool, failed: bool) -> None:
    """Conserva ambos archivos incluso si falla setup.py con los parches activos."""
    project = tmp_path / "LanguageModelDecoder/runtime/server/x86"
    (project / "python").mkdir(parents=True)
    cmake = project / "CMakeLists.txt"
    binding = project / "python/lm_decoder.cc"
    original_cmake = b"# upstream\r\nBUILD_COMMAND     make -j 4\r\n"
    original_binding = b'#include "utils/log.h"\r\n    m.def("DecodeNumpy", &DecodeNumpy);\r\n'
    cmake.write_bytes(original_cmake)
    binding.write_bytes(original_binding)

    with pytest.raises(RuntimeError, match="build failed") if failed else nullcontext():
        with runtime_build_threads(tmp_path, 2), mapped_binding(tmp_path, mapped):
            assert "make -j 2" in cmake.read_text()
            assert ("SetFstReadMode" in binding.read_text()) is mapped
            if failed:
                raise RuntimeError("build failed")

    assert cmake.read_bytes() == original_cmake
    assert binding.read_bytes() == original_binding


def test_unexpected_upstream_cmake_is_preserved(tmp_path: Path) -> None:
    """Rechaza una fuente distinta antes de escribir cambios o iniciar la compilación."""
    project = tmp_path / "LanguageModelDecoder/runtime/server/x86"
    project.mkdir(parents=True)
    cmake = project / "CMakeLists.txt"
    original = b"BUILD_COMMAND another-build-tool\n"
    cmake.write_bytes(original)

    with pytest.raises(ValueError, match="CMake oficial ha cambiado"):
        with runtime_build_threads(tmp_path, 2):
            pytest.fail("No debe compilar una fuente inesperada")

    assert cmake.read_bytes() == original
