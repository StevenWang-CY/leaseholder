"""A call's GPU node: the worker asks a fake control plane on a Unix socket, and the call's STT and TTS sockets
follow the node it answers."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from leaseholder_voice import placement
from leaseholder_voice.worker import placement_nodes

NODES = ("phastform", "gray")
ROOM = "lh-bot-1f2e3d4c-0"


class FakeGateway:
    """``POST /v1/sessions`` on a Unix socket: keeps each request's body and answers with ``status`` and
    ``answer``, or not at all while ``hang``."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []
        self.status = 200
        self.answer: Any = {"session_id": "voice-RM_a1", "class": "realtime", "node": "gray"}
        self.hang = False
        self.released = asyncio.Event()

    async def _post(self, request: web.Request) -> web.Response:
        self.asked.append(await request.json())
        if self.hang:
            await self.released.wait()
        return web.json_response(self.answer, status=self.status)


@pytest.fixture
def sockets() -> Iterator[Path]:
    # a short directory: socket paths are limited to 104 bytes on macOS, 108 on Linux
    with tempfile.TemporaryDirectory(prefix="lh-voice-") as directory:
        yield Path(directory)


@asynccontextmanager
async def gateway(sockets: Path) -> AsyncIterator[FakeGateway]:
    fake = FakeGateway()
    app = web.Application()
    app.add_routes([web.post("/v1/sessions", fake._post)])
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.UnixSite(runner, str(sockets / "gateway.sock")).start()
    try:
        yield fake
    finally:
        fake.released.set()
        await runner.cleanup()


def ask(
    sockets: Path, nodes: tuple[str, ...] = NODES, **options: Any
) -> Coroutine[Any, Any, placement.Sockets]:
    """The placement of the call ``voice-RM_a1``, by the control plane on ``gateway.sock`` in ``sockets``, whose
    default STT and TTS are Phastform's."""
    return placement.place(
        "voice-RM_a1",
        ROOM,
        nodes=nodes,
        control_plane=str(sockets / "gateway.sock"),
        stt=str(sockets / "phastform-stt.sock"),
        tts=str(sockets / "phastform-tts.sock"),
        **options,
    )


def test_a_calls_stt_and_tts_follow_its_node(
    sockets: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="leaseholder_voice.placement")

    async def scenario() -> tuple[FakeGateway, placement.Sockets]:
        async with gateway(sockets) as fake:
            return fake, await ask(sockets)

    fake, placed = asyncio.run(scenario())
    assert fake.asked == [{"session_id": "voice-RM_a1", "class": "realtime"}]
    assert placed == placement.Sockets(
        str(sockets / "gray-stt.sock"), str(sockets / "gray-tts.sock"), "gray"
    )
    [line] = [r for r in caplog.records if r.name == "leaseholder_voice.placement"]
    assert line.getMessage() == "session placed"
    assert (line.session_id, line.room, line.node) == ("voice-RM_a1", ROOM, "gray")


def test_without_nodes_no_placement_is_asked_for(sockets: Path) -> None:
    async def scenario() -> tuple[FakeGateway, placement.Sockets]:
        async with gateway(sockets) as fake:
            return fake, await ask(sockets, nodes=())

    fake, placed = asyncio.run(scenario())
    assert fake.asked == []
    assert placed == placement.Sockets(
        str(sockets / "phastform-stt.sock"), str(sockets / "phastform-tts.sock")
    )


@pytest.mark.parametrize(
    ("status", "answer", "hang", "cause"),
    [
        (503, {"error": {"code": "no_ready_node"}}, False, "HTTP 503"),
        (200, {"node": "cyan"}, False, "HTTP 200"),  # a node the worker has no sockets for
        (200, ["gray"], False, "HTTP 200"),
        (200, {"node": "gray"}, True, "ReadTimeout"),
    ],
)
def test_a_failed_placement_keeps_the_default_sockets(
    sockets: Path,
    caplog: pytest.LogCaptureFixture,
    status: int,
    answer: Any,
    hang: bool,
    cause: str,
) -> None:
    async def scenario() -> placement.Sockets:
        async with gateway(sockets) as fake:
            fake.status, fake.answer, fake.hang = status, answer, hang
            return await ask(sockets, timeout=0.2)

    placed = asyncio.run(scenario())
    assert placed == placement.Sockets(
        str(sockets / "phastform-stt.sock"), str(sockets / "phastform-tts.sock")
    )
    [warning] = [r for r in caplog.records if r.name == "leaseholder_voice.placement"]
    assert warning.levelno == logging.WARNING and cause in warning.getMessage()


def test_without_a_control_plane_the_call_goes_on(sockets: Path) -> None:
    placed = asyncio.run(ask(sockets))  # nothing listens on the socket
    assert placed.node is None and placed.stt.endswith("phastform-stt.sock")


def test_lh_nodes_names_two_nodes_or_more(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LH_LLM_UDS", "/run/user/1007/lh/gateway.sock")
    monkeypatch.delenv("LH_NODES", raising=False)
    assert placement_nodes() == ()
    monkeypatch.setenv("LH_NODES", "phastform, gray")
    assert placement_nodes() == NODES
    for setting in ("phastform", "phastform,,gray", "phastform,phastform", "phast form,gray"):
        monkeypatch.setenv("LH_NODES", setting)
        with pytest.raises(ValueError, match="LH_NODES"):
            placement_nodes()
    # The bypass: the LLM goes straight to an engine, and nothing places calls.
    monkeypatch.setenv("LH_NODES", "phastform,gray")
    monkeypatch.setenv("LH_LLM_UDS", "/run/user/1007/lh/phastform-engine.sock")
    assert placement_nodes() == ()
