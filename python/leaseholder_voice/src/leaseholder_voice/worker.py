"""The Leaseholder voice agent: a stock LiveKit Agents worker for realtime voice sessions.

``lh-voice-worker start`` registers the agent ``leaseholder-voice`` with the LiveKit server and
serves every dispatched room with one ``AgentSession``: Silero VAD and LiveKit's audio
end-of-turn model (``turn-detector-v1-mini``, run locally) on the CPU, then STT, the LLM and TTS
on the GPU node. Turn handling (endpointing, interruptions, preemptive generation) keeps LiveKit's
defaults.

Experiments use ``LH_TURN_DETECTION=vad`` instead: the end of a turn is then a fixed delay after
the end of speech, the same for every arm, so voice-to-voice latency moves only with serving. The
end-of-turn model waits up to the endpointing maximum (3 s) whenever it judges a line unfinished,
which on synthesized callers happened on about one turn in six (W1) and would hide serving effects
in the tail.

The worker runs inside the voice-plane network namespace, where the only route to the servers is
a Unix socket each. The environment, set by ``lh-voice-worker.service``, configures it:

| Variable | Meaning |
|---|---|
| ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET`` | The LiveKit server |
| ``LH_LLM_UDS`` | The control plane's socket (the LLM path), or the engine's tunnel |
| ``LH_STT_UDS``, ``LH_TTS_UDS`` | The tunnels to the STT and TTS servers |
| ``LH_IDLE_PROCESSES`` | Pre-started job processes (default 2) |
| ``LH_LLM_TIMEOUT`` | Seconds before an LLM request fails (default 30: a cold prefill exceeds 10) |
| ``LH_TURN_DETECTION`` | ``model`` (default: the end-of-turn model) or ``vad`` (fixed endpointing, for experiments) |
"""

from __future__ import annotations

import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero

from . import plugins
from .protocol import room_session_id

AGENT_NAME = "leaseholder-voice"
BASE_URL = "http://lh.invalid/v1"
"""Requests go over the Unix sockets; the host name never resolves."""

INSTRUCTIONS = (
    "You are Ava, the phone assistant of a small neighborhood restaurant. You take reservations, "
    "answer questions about the menu, opening hours and directions, and keep every answer to one "
    "or two short, natural sentences. You are speaking, so never use lists, markdown or emoji."
)


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per job process, before a room is assigned."""
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(
    host="127.0.0.1",  # the default "" puts the health server on every interface
    port=0,
    num_idle_processes=int(os.environ.get("LH_IDLE_PROCESSES", "2")),  # default: one per CPU
    load_fnc=lambda: 0.0,  # other tenants' CPU load on a shared host must not drain the worker
    setup_fnc=prewarm,
)


def turn_detection() -> inference.TurnDetector | str:
    mode = os.environ.get("LH_TURN_DETECTION", "model")
    if mode == "vad":
        return "vad"
    if mode != "model":
        raise ValueError(f"LH_TURN_DETECTION must be model or vad, got {mode!r}")
    # The local model: the cloud version needs LiveKit Cloud, and the namespace has no route out.
    return inference.TurnDetector(version="v1-mini")


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    session_id = room_session_id(ctx.job.room.sid)
    llm_timeout = float(os.environ.get("LH_LLM_TIMEOUT", "30"))
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=plugins.make_stt(BASE_URL, uds=os.environ["LH_STT_UDS"]),
        llm=plugins.make_llm(BASE_URL, session_id, uds=os.environ["LH_LLM_UDS"]),
        tts=plugins.make_tts(BASE_URL, uds=os.environ["LH_TTS_UDS"]),
        turn_handling={"turn_detection": turn_detection()},
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=1, timeout=llm_timeout)
        ),
    )
    # One caller per room: when the caller hangs up, the session closes and the room is deleted.
    room_options = room_io.RoomOptions(delete_room_on_close=True)
    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
        room_options=room_options,
        record=False,
    )


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
