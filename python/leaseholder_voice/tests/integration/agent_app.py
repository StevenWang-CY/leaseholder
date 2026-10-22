"""Agent worker of the integration test; runs inside the voice-plane namespace.

The ``voice_plane`` fixture starts it as
``python -m livekit.agents start tests/integration/agent_app.py``. The environment gives
the LiveKit URL and keys, the Unix sockets of the mock endpoints and a directory for
per-room event logs; the dispatch metadata selects the interruption handling.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    llm,
    room_io,
)
from livekit.plugins import silero

from leaseholder_voice import plugins
from leaseholder_voice.protocol import room_session_id

AGENT_NAME = "leaseholder-voice-test"
BASE_URL = "http://lh.invalid/v1"  # requests go over the Unix sockets; the host never resolves


class EventLog:
    """Session events as JSON lines, timed with ``time.monotonic()``."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("a", buffering=1)

    def write(self, event: str, **fields: Any) -> None:
        self._file.write(json.dumps({"event": event, "t": time.monotonic(), **fields}) + "\n")


class StoryAgent(Agent):
    """The stock agent; its LLM node is wrapped only to log when the framework ends it."""

    def __init__(self, log: EventLog) -> None:
        super().__init__(instructions="You are a voice assistant.")
        self._log = log

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[Any]:
        self._log.write("llm_node_started")
        outcome = "completed"
        try:
            node = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
            async with contextlib.aclosing(node) as chunks:
                async for chunk in chunks:
                    yield chunk
        except BaseException as exc:  # CancelledError on interruption
            outcome = type(exc).__name__
            raise
        finally:
            self._log.write("llm_node_finished", outcome=outcome)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(
    host="127.0.0.1",  # the default "" puts the health server on every interface
    port=0,
    num_idle_processes=1,  # the production default is one per CPU (48 on cyan)
    load_fnc=lambda: 0.0,  # other tenants' CPU load must not mark the worker unavailable
    drain_timeout=10,
    setup_fnc=prewarm,
)


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    options = json.loads(ctx.job.metadata)
    log = EventLog(Path(os.environ["LH_AGENT_EVENTS"]) / f"{ctx.job.room.name}.jsonl")
    session_id = room_session_id(ctx.job.room.sid)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=plugins.make_stt(BASE_URL, uds=os.environ["LH_STT_UDS"]),
        llm=plugins.make_llm(BASE_URL, session_id, uds=os.environ["LH_LLM_UDS"]),
        tts=plugins.make_tts(BASE_URL, uds=os.environ["LH_TTS_UDS"]),
        turn_handling={
            "turn_detection": "vad",
            "interruption": {
                "mode": "vad",
                "resume_false_interruption": options["resume_false_interruption"],
            },
        },
        aec_warmup_duration=None,  # the bot plays files: there is no echo to wait out
        user_away_timeout=None,
    )
    session.on("user_state_changed", lambda ev: log.write("user_state", state=ev.new_state))
    session.on("agent_state_changed", lambda ev: log.write("agent_state", state=ev.new_state))
    log.write("session_starting", room_sid=ctx.job.room.sid, session_id=session_id, **options)
    # one bot per room: when it leaves, the session closes and deleting the room ends the job
    room_options = room_io.RoomOptions(delete_room_on_close=True)
    await session.start(
        agent=StoryAgent(log), room=ctx.room, room_options=room_options, record=False
    )
