"""User bot of the integration test; runs inside the voice-plane namespace.

It speaks a user turn, waits for the agent's reply audio, then barges in with a second
utterance while the agent is still talking. Every observation is one JSON line on
stdout, timed with ``time.monotonic()``, the clock that every process on the host
shares. The LiveKit token comes from ``LH_BOT_TOKEN``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

from livekit import rtc

from leaseholder_voice.audio import AGENT_STATE, AgentAudio, Clip, Microphone, agents, until


def emit(event: str, **fields: Any) -> None:
    """One JSON line on stdout, stamped with ``time.monotonic()``."""
    print(json.dumps({"event": event, "t": time.monotonic(), **fields}), flush=True)


async def scenario(args: argparse.Namespace) -> None:
    room = rtc.Room()
    agent_audio = AgentAudio()
    tasks: set[asyncio.Task[None]] = set()
    subscribed = asyncio.Event()

    @room.on("track_subscribed")
    def _on_track(track: rtc.Track, _: Any, participant: rtc.RemoteParticipant) -> None:
        is_agent = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        if is_agent and track.kind == rtc.TrackKind.KIND_AUDIO:
            tasks.add(asyncio.create_task(agent_audio.consume(track)))

    @room.on("participant_attributes_changed")
    def _on_attributes(changed: dict[str, str], participant: rtc.Participant) -> None:
        if AGENT_STATE in changed:
            emit("agent_state", state=changed[AGENT_STATE])

    @room.on("local_track_subscribed")
    def _on_local_track_subscribed(_: rtc.Track) -> None:
        subscribed.set()

    await room.connect(args.url, os.environ["LH_BOT_TOKEN"])
    emit("connected", room=room.name, room_sid=await room.sid)
    microphone = Microphone(on_onset=lambda clip: emit("speech_onset", clip=clip.name))
    track = rtc.LocalAudioTrack.create_audio_track("microphone", microphone.source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)
    tasks.add(asyncio.create_task(microphone.run()))
    try:
        await asyncio.wait_for(subscribed.wait(), args.timeout)
        await until(
            lambda: any(p.attributes.get(AGENT_STATE) == "listening" for p in agents(room)),
            args.timeout,
            "the agent never listened",
        )
        emit("agent_listening")
        await asyncio.sleep(0.5)  # let the agent's audio input run on silence first

        user_turn = await microphone.play(Clip.load("user_turn", args.user_turn))
        emit("clip_played", clip="user_turn", **user_turn)
        reply = await agent_audio.first_onset_after(user_turn["speech_end"], args.timeout)
        emit("agent_reply_started", at=reply)

        await asyncio.sleep(max(0.0, reply + args.barge_in_delay - time.monotonic()))
        barge_in = await microphone.play(Clip.load("barge_in", args.barge_in))
        emit("clip_played", clip="barge_in", **barge_in)
        await asyncio.sleep(args.listen_after)
        emit("agent_audio", spans=agent_audio.spans)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await microphone.source.aclose()
        await room.disconnect()
    emit("done")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="LiveKit signalling URL")
    parser.add_argument("--user-turn", required=True, help="16 kHz mono WAV of the user turn")
    parser.add_argument("--barge-in", required=True, help="16 kHz mono WAV of the barge-in")
    parser.add_argument(
        "--barge-in-delay",
        type=float,
        default=1.0,
        help="seconds from the agent's first reply audio to the barge-in",
    )
    parser.add_argument(
        "--listen-after",
        type=float,
        default=5.0,
        help="seconds to stay in the room after the barge-in clip",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="per-step timeout")
    args = parser.parse_args()
    try:
        asyncio.run(scenario(args))
    except Exception as exc:
        emit("error", error=repr(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
