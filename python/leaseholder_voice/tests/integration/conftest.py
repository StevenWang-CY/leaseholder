"""Fixtures of the voice-plane integration test (Linux; needs livekit-server)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.mocks import OpenAIMocks

from .namespace import TOOLS
from .plane import VoicePlane

_MEASUREMENTS: list[str] = []


@pytest.fixture(scope="session")
def voice_plane(tmp_path_factory: pytest.TempPathFactory) -> Iterator[VoicePlane]:
    if sys.platform != "linux":
        pytest.skip("needs Linux user and network namespaces")
    if missing := [tool for tool in TOOLS if shutil.which(tool) is None]:
        pytest.skip(f"missing tools: {', '.join(missing)}")
    livekit_server = os.environ.get("LIVEKIT_SERVER_BIN") or shutil.which("livekit-server")
    if not livekit_server:
        pytest.skip("livekit-server not found: put it on PATH or set LIVEKIT_SERVER_BIN")
    with (
        tempfile.TemporaryDirectory(prefix="lh-voice-") as sockets,  # short socket paths
        OpenAIMocks(Path(sockets)) as mocks,
        VoicePlane(tmp_path_factory.mktemp("voice-plane"), mocks, livekit_server) as plane,
    ):
        yield plane


@pytest.fixture
def report() -> Callable[[str], None]:
    """Add a line to the measurements printed at the end of the run."""
    return _MEASUREMENTS.append


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _MEASUREMENTS:
        terminalreporter.section("voice-plane measurements")
        for line in _MEASUREMENTS:
            terminalreporter.write_line(line)
