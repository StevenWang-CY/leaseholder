"""A barge-in closes the LLM stream, and every LLM request carries the session headers.

The bot speaks a turn; the agent answers from a mock LLM that streams one token per
100 ms for 20 s; one second into the agent's reply audio the bot barges in. The mock
LLM records the request headers and the moment the client disconnects, and takes the
agent's lifecycle events as the control plane does. While the barge-in plays, the socket
audit checks that nothing listens outside loopback.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from leaseholder_voice.protocol import (
    CLASS_HEADER,
    REALTIME_CLASS,
    SESSION_ID_HEADER,
    room_session_id,
)

from .plane import SocketAudit, VoicePlane

CASES: dict[str, dict[str, Any]] = {
    # Stock LiveKit Agents: the barge-in pauses the reply audio, and the reply (with its
    # LLM stream) is interrupted once the barge-in turn has a final transcript.
    "pause-then-interrupt": {"resume_false_interruption": True},
    # The reply is interrupted after 0.5 s (min_duration) of barge-in speech.
    "interrupt-on-speech": {"resume_false_interruption": False},
}

MIN_INTERRUPTION = 0.5  # interruption.min_duration, the LiveKit default

# The call's lifecycle events, in order, around whatever else VAD makes of the audio: the
# turn, the reply, the barge-in that interrupts it, its turn and the reply to it, the end.
SKELETON = (
    "user_speech_started",
    "user_turn_ended",
    "playback_started",
    "user_speech_started",
    "interrupted",
    "user_turn_ended",
    "playback_started",
    "session_closed",
)

# Seconds from the barge-in's speech onset to the mock LLM seeing the disconnect. Measured
# on cyan, 10 runs (2026-10-22): pause-then-interrupt 2.17-2.20 s, interrupt-on-speech
# 0.57-0.62 s.
BOUND = {
    # 1.31 s of barge-in speech, then VAD end of speech (0.55 s of silence) and the STT
    # call: closed 0.86-0.89 s after the speech ended. 0.8 s of headroom for a busy host.
    "pause-then-interrupt": 3.0,
    # MIN_INTERRUPTION plus transport, VAD and cancellation (0.07-0.12 s), with 0.38 s of
    # headroom; the stock pause path would land above 2 s.
    "interrupt-on-speech": 1.0,
}


def _first(events: list[dict[str, Any]], after: float, **match: Any) -> float | None:
    for event in events:
        if event["t"] > after and all(event.get(k) == v for k, v in match.items()):
            return event["t"]
    return None


def _in_order(expected: tuple[str, ...], seen: list[str]) -> bool:
    remaining = iter(seen)
    return all(name in remaining for name in expected)


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES)
def test_barge_in_closes_llm_stream(
    voice_plane: VoicePlane, case: str, report: Callable[[str], None]
) -> None:
    room = f"barge-in-{case}-{uuid.uuid4().hex[:8]}"
    audits: list[SocketAudit] = []

    def on_event(event: dict[str, Any]) -> None:
        if event["event"] == "speech_onset" and event["clip"] == "barge_in":
            audits.append(voice_plane.audit())

    events = voice_plane.run_bot(room, CASES[case], on_event)
    connected = next(e for e in events if e["event"] == "connected")
    barge_in = next(e for e in events if e.get("clip") == "barge_in" and "onset" in e)
    onset = barge_in["onset"]
    streams = [s for s in voice_plane.mocks.chat_streams() if s.opened_at > connected["t"]]
    agent = voice_plane.agent_events(room)
    session_id = room_session_id(connected["room_sid"])
    lifecycle = voice_plane.lifecycle_events(session_id)
    turns = voice_plane.agent_turns(session_id)
    record = voice_plane.run_dir / f"{room}.json"
    record.write_text(
        json.dumps(
            {
                "bot": events,
                "agent": agent,
                "llm_streams": [dataclasses.asdict(s) for s in streams],
                "socket_audits": [dataclasses.asdict(a) for a in audits],
                "lifecycle_events": lifecycle,
                "turns": turns,
            },
            indent=1,
        )
    )

    assert len(audits) == 1, f"the socket audit did not run once; see {record}"
    assert not audits[0].problems, f"sockets outside loopback:\n{audits[0]}"

    assert streams, f"the agent sent no LLM request; see {record}"
    for stream in streams:
        assert stream.headers.get(SESSION_ID_HEADER.lower()) == session_id
        assert stream.headers.get(CLASS_HEADER.lower()) == REALTIME_CLASS

    replying = [s for s in streams if s.open_at(onset)]
    assert len(replying) == 1, "expected exactly one LLM stream open at the barge-in"
    [reply] = replying
    assert reply.completed_at is None, "the reply's LLM stream ran to completion"
    assert reply.disconnected_at is not None, "the client never closed the LLM stream"
    close = reply.disconnected_at - onset
    if case == "interrupt-on-speech":  # not before min_duration of barge-in speech
        assert close >= MIN_INTERRUPTION, f"closed too early; see {record}"
    else:  # not before the end of the barge-in turn
        assert reply.disconnected_at > barge_in["speech_end"], f"closed too early; see {record}"
    assert close <= BOUND[case], f"closed {close:.3f} s after the barge-in; see {record}"

    # Every lifecycle event reached the control plane, in order: one sender's ids, the session, the
    # sender's nonce and a sequence number.
    types = [e["type"] for e in lifecycle]
    assert lifecycle, f"no lifecycle event reached the control plane; see {record}"
    nonce = lifecycle[0]["event_id"].removeprefix(f"{session_id}-").split("-")[0]
    ids = [f"{session_id}-{nonce}-{n}" for n in range(1, len(lifecycle) + 1)]
    assert [e["event_id"] for e in lifecycle] == ids, f"events lost or reordered; see {record}"
    assert _in_order(SKELETON, types), f"lifecycle events {types}; see {record}"
    assert types.count("session_closed") == 1 and types[-1] == "session_closed", types
    # The barge-in's interruption comes with the reply's stream closing, and says how much of
    # the reply the bot heard.
    spans = next(e for e in events if e["event"] == "agent_audio")["spans"]
    cut = next(e for e in lifecycle if e["type"] == "interrupted" and e["received_at"] > onset)
    assert abs(cut["received_at"] - reply.disconnected_at) <= 1.0, f"see {record}"
    heard = next((end - start for start, end in spans if start < onset <= end), None)
    assert heard is not None, f"no reply audio under way at the barge-in; see {record}"
    assert abs(cut["played_ms"] - heard * 1000) <= 250, f"played vs heard; see {record}"
    assert turns and {"llm_ttft_ms", "e2e_ms"} <= turns[0].keys(), f"turn lines; see {record}"

    marks = {
        "agent VAD onset": _first(agent, onset, event="user_state", state="speaking"),
        "agent audio stopped": next((end for start, end in spans if start < onset <= end), None),
        "LLM node cancelled": _first(agent, onset, event="llm_node_finished"),
        "interrupted event": cut["received_at"],
        "barge-in speech ended": barge_in["speech_end"],
    }
    timeline = ", ".join(
        f"{name} +{(t - onset) * 1000:.0f} ms" if t is not None else f"{name} n/a"
        for name, t in marks.items()
    )
    report(f"{case}: barge-in onset -> LLM stream closed {close * 1000:.0f} ms ({timeline})")
