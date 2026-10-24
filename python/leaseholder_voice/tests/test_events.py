"""Lifecycle events and turn lines: LiveKit's signals driven through the hooks of a stand-in session,
and the delivery to a fake control plane on a Unix socket."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from livekit import rtc
from livekit.agents import (
    AgentHandoff,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
    llm,
)
from livekit.agents.voice import io

from leaseholder_voice import events

SCHEMA = Path(__file__).resolve().parents[3] / "crates/lh-proto/schema/lifecycle_event.schema.json"

# LiveKit's metrics of one turn, in seconds: the caller's message, then the reply.
USER = {
    "transcription_delay": 0.2123,
    "end_of_turn_delay": 0.6,
    "on_user_turn_completed_delay": 0.0,
}
REPLY = {
    "llm_node_ttft": 0.061,
    "llm_node_ttfs": 0.3015,
    "tts_node_ttfb": 0.1762,
    "playback_latency": 0.002,
    "e2e_latency": 1.1,
}


class FakeSession(rtc.EventEmitter[str]):
    """What a Lifecycle uses of an AgentSession: its events, and its audio output's."""

    def __init__(self) -> None:
        super().__init__()
        self.output = SimpleNamespace(audio=rtc.EventEmitter[str]())

    def user(self, state: str) -> None:
        self.emit("user_state_changed", UserStateChangedEvent(old_state="away", new_state=state))

    def message(self, role: str, metrics: dict[str, float], interrupted: bool = False) -> None:
        item = llm.ChatMessage(role=role, content=["..."], interrupted=interrupted, metrics=metrics)
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))

    def playback(self, played_s: float, *, interrupted: bool) -> None:
        self.output.audio.emit("playback_started", io.PlaybackStartedEvent(created_at=time.time()))
        finished = io.PlaybackFinishedEvent(playback_position=played_s, interrupted=interrupted)
        self.output.audio.emit("playback_finished", finished)


class Recorder:
    """A sender that keeps what it is given."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event_type: str, **fields: int) -> None:
        self.events.append({"type": event_type, **fields})


def watched() -> tuple[FakeSession, Recorder, events.Lifecycle]:
    session, recorder = FakeSession(), Recorder()
    lifecycle = events.Lifecycle("voice-RM_a1B2c3", recorder)  # type: ignore[arg-type]
    lifecycle.attach(session)  # type: ignore[arg-type]
    return session, recorder, lifecycle


def test_livekit_signals_map_to_the_protocols_events() -> None:
    session, recorder, _ = watched()
    session.user("speaking")
    session.user("listening")  # VAD's end of speech: not an event
    session.message("user", USER)
    session.playback(2.0414, interrupted=True)  # a barge-in
    session.message("assistant", REPLY, interrupted=True)
    session.user("away")
    session.message("user", USER)
    session.playback(3.2, interrupted=False)
    item = AgentHandoff(new_agent_id="agent")
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=item))
    session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
    assert recorder.events == [
        {"type": "user_speech_started"},
        {"type": "user_turn_ended"},
        {"type": "playback_started"},
        {"type": "interrupted", "played_ms": 2041},
        {"type": "user_turn_ended"},
        {"type": "playback_started"},
        {"type": "playback_ended", "played_ms": 3200},
        {"type": "session_closed"},
    ]
    # Each is a variant of lh-proto's LifecycleEvent, with exactly that variant's fields.
    variants = {
        v["properties"]["type"]["const"]: v for v in json.loads(SCHEMA.read_text())["oneOf"]
    }
    for event in recorder.events:
        assert set(variants[event["type"]]["required"]) == set(event), event


def test_a_turn_line_holds_its_stage_times(caplog: pytest.LogCaptureFixture) -> None:
    session, _, lifecycle = watched()
    caplog.set_level(logging.INFO, logger="leaseholder_voice.turns")
    session.message("user", USER)
    session.message("assistant", REPLY, interrupted=True)
    session.message("assistant", {"llm_node_ttft": 0.05})  # a reply to no message of the caller's
    first, second = (r for r in caplog.records if r.name == "leaseholder_voice.turns")
    assert first.getMessage() == "turn"
    assert {k: getattr(first, k) for k in ("session_id", "turn", "interrupted")} == {
        "session_id": "voice-RM_a1B2c3",
        "turn": 0,
        "interrupted": True,
    }
    assert {stage: getattr(first, stage) for stage in events.STAGES} == {
        "transcript_ms": 212.3,
        "end_of_turn_ms": 600.0,
        "llm_ttft_ms": 61.0,
        "llm_first_sentence_ms": 301.5,
        # from the end of the turn to the first sentence: 1100 - 2 - 176.2 - 600 ms
        "turn_to_first_sentence_ms": 321.8,
        "tts_first_audio_ms": 176.2,
        "e2e_ms": 1100.0,
    }
    assert (second.turn, second.llm_ttft_ms) == (1, 50.0)
    assert not any(hasattr(second, stage) for stage in events.STAGES if stage != "llm_ttft_ms")
    assert lifecycle.turns == 2


def test_a_failing_hook_never_reaches_livekit(caplog: pytest.LogCaptureFixture) -> None:
    session, recorder, _ = watched()
    # LiveKit's emitters pass a handler's TypeError on to the code that emitted the event: here
    # the audio output, whose playback this event would end.
    finished = io.PlaybackFinishedEvent(playback_position=None, interrupted=False)  # type: ignore[arg-type]
    session.output.audio.emit("playback_finished", finished)
    assert "a lifecycle hook failed" in caplog.text
    session.playback(1.0, interrupted=False)
    assert recorder.events == [
        {"type": "playback_started"},
        {"type": "playback_ended", "played_ms": 1000},
    ]


# Delivery


class FakeGateway:
    """``POST /v1/events`` on a Unix socket. Keeps each batch; ``answers`` sets how each request is
    answered in turn (a status, or ``"hang"``), and 202 follows."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.batches: list[list[dict[str, Any]]] = []
        self.answers: list[int | str] = []
        self.released = asyncio.Event()

    async def _post(self, request: web.Request) -> web.Response:
        self.batches.append(await request.json())
        answer = self.answers.pop(0) if self.answers else 202
        if answer == "hang":
            await self.released.wait()
            answer = 202
        return web.json_response(
            {"accepted": len(self.batches[-1]), "duplicates": 0}, status=answer
        )


@pytest.fixture
def socket_path() -> Iterator[str]:
    # a short directory: socket paths are limited to 104 bytes on macOS, 108 on Linux
    with tempfile.TemporaryDirectory(prefix="lh-voice-") as directory:
        yield str(Path(directory) / "gateway.sock")


@asynccontextmanager
async def gateway(path: str) -> AsyncIterator[FakeGateway]:
    fake = FakeGateway(path)
    app = web.Application()
    app.add_routes([web.post("/v1/events", fake._post)])
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    await web.UnixSite(runner, path).start()
    try:
        yield fake
    finally:
        fake.released.set()
        await runner.cleanup()


async def until(condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)


def test_events_go_out_in_order_and_what_waits_goes_together(socket_path: str) -> None:
    async def scenario() -> tuple[FakeGateway, events.EventSender, float, float]:
        async with gateway(socket_path) as fake:
            fake.answers = ["hang"]
            sender = events.EventSender("voice-RM_x", uds=socket_path)
            started = time.time()
            sender.emit("user_speech_started")
            await until(lambda: fake.batches)
            # queued while the first POST waits for its answer: they go out together, in order
            sender.emit("user_turn_ended")
            sender.emit("playback_started")
            sender.emit("playback_ended", played_ms=1840)
            fake.released.set()
            await until(lambda: len(fake.batches) == 2)
            await sender.aclose()
            return fake, sender, started, time.time()

    fake, sender, started, finished = asyncio.run(scenario())
    assert [[e["type"] for e in batch] for batch in fake.batches] == [
        ["user_speech_started"],
        ["user_turn_ended", "playback_started", "playback_ended"],
    ]
    delivered = [event for batch in fake.batches for event in batch]
    # The session, the sender's nonce and a sequence number.
    nonce = delivered[0]["event_id"].split("-")[2]
    assert re.fullmatch(r"[0-9a-f]{8}", nonce)
    assert [e["event_id"] for e in delivered] == [f"voice-RM_x-{nonce}-{n}" for n in range(1, 5)]
    assert {(e["session_id"], e["source"]) for e in delivered} == {("voice-RM_x", "voice-worker")}
    assert all(started * 1000 - 1 <= e["at_ms"] <= finished * 1000 for e in delivered)
    assert delivered[-1]["played_ms"] == 1840
    schema = json.loads(SCHEMA.read_text())
    assert all(set(schema["required"]) <= set(e) for e in delivered)
    assert events.SOURCE in [v["const"] for v in schema["$defs"]["EventSource"]["oneOf"]]
    assert (sender.sent, sender.dropped, sender.failed) == (4, 0, 0)


def test_a_new_sender_for_the_same_session_repeats_no_id(socket_path: str) -> None:
    # A worker that rejoins a room builds a new sender for its session: within the control plane's
    # deduplication window (10 minutes), a repeated id would make its events repeats, and drop them.
    async def scenario() -> list[str]:
        async with gateway(socket_path) as fake:
            for _ in range(2):
                sender = events.EventSender("voice-RM_x", uds=socket_path)
                sender.emit("user_speech_started")
                await sender.aclose()
            return [e["event_id"] for batch in fake.batches for e in batch]

    first, second = asyncio.run(scenario())
    assert first != second
    assert all(re.fullmatch(r"voice-RM_x-[0-9a-f]{8}-1", event_id) for event_id in (first, second))


def test_a_slow_or_refusing_control_plane_loses_batches_and_nothing_else(
    socket_path: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="leaseholder_voice.events")

    async def scenario() -> tuple[FakeGateway, events.EventSender]:
        async with gateway(socket_path) as fake:
            fake.answers = ["hang", 400]
            sender = events.EventSender("s", uds=socket_path, timeout=0.2)
            sender.emit("user_speech_started")  # no answer within the timeout
            await until(lambda: sender.failed == 1)
            sender.emit("user_turn_ended")  # refused
            await until(lambda: sender.failed == 2)
            sender.emit("playback_started")  # answered again
            await until(lambda: sender.sent == 1)
            await sender.aclose()
            return fake, sender

    fake, sender = asyncio.run(scenario())
    assert [batch[0]["type"] for batch in fake.batches] == [
        "user_speech_started",
        "user_turn_ended",
        "playback_started",
    ]
    assert (sender.sent, sender.dropped, sender.failed) == (1, 0, 2)
    # One warning per outage, naming the first failure, and a line when events flow again.
    messages = [r.getMessage() for r in caplog.records if r.name == "leaseholder_voice.events"]
    assert len(messages) == 2
    assert messages[0].startswith("lifecycle events do not reach the control plane: ReadTimeout")
    assert messages[1] == "lifecycle events reach the control plane again"


def test_a_full_queue_drops_its_oldest_events(socket_path: str) -> None:
    async def scenario() -> tuple[FakeGateway, events.EventSender]:
        async with gateway(socket_path) as fake:
            fake.answers = ["hang"]
            sender = events.EventSender("s", uds=socket_path, limit=3)
            sender.emit("user_speech_started")
            await until(lambda: fake.batches)
            for _ in range(5):  # while the first POST waits: two more than the queue holds
                sender.emit("playback_started")
            fake.released.set()
            await until(lambda: len(fake.batches) == 2)
            await sender.aclose()
            return fake, sender

    fake, sender = asyncio.run(scenario())
    assert [e["event_id"].rsplit("-", 1)[1] for e in fake.batches[1]] == ["4", "5", "6"]
    assert (sender.sent, sender.dropped, sender.failed) == (4, 2, 0)


def test_without_a_control_plane_nothing_raises(socket_path: str) -> None:
    async def scenario() -> events.EventSender:
        sender = events.EventSender("s", uds=socket_path)  # nothing listens there
        for _ in range(3):
            sender.emit("user_speech_started")
            await asyncio.sleep(0.01)
        await sender.aclose()
        return sender

    sender = asyncio.run(scenario())
    assert (sender.sent, sender.dropped, sender.failed) == (0, 0, 3)


def test_a_session_end_sends_its_last_events_and_logs_its_counts(
    socket_path: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="leaseholder_voice")

    async def scenario(uds: str | None) -> list[list[dict[str, Any]]]:
        async with gateway(socket_path) as fake:
            session, callbacks = FakeSession(), []
            ctx = SimpleNamespace(add_shutdown_callback=callbacks.append)
            events.observe(ctx, session, "s", uds=uds)  # type: ignore[arg-type]
            session.user("speaking")
            session.message("user", USER)
            session.playback(1.5, interrupted=False)
            session.message("assistant", REPLY)
            session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
            [shutdown] = callbacks
            await shutdown()  # the job's shutdown, right after the close
            return fake.batches

    batches = asyncio.run(scenario(socket_path))
    assert [e["type"] for batch in batches for e in batch] == [
        "user_speech_started",
        "user_turn_ended",
        "playback_started",
        "playback_ended",
        "session_closed",
    ]
    [ended] = [r for r in caplog.records if r.getMessage() == "session ended"]
    assert (ended.session_id, ended.turns, ended.events_sent) == ("s", 1, 5)
    assert (ended.events_dropped, ended.events_failed) == (0, 0)

    caplog.clear()
    assert asyncio.run(scenario(None)) == []  # LH_EVENTS=off: the turn lines alone
    [ended] = [r for r in caplog.records if r.getMessage() == "session ended"]
    assert ended.turns == 1 and not hasattr(ended, "events_sent")
    assert [r.name for r in caplog.records].count("leaseholder_voice.turns") == 1
