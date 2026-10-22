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
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
from livekit import rtc

SAMPLE_RATE = 16_000
FRAME = SAMPLE_RATE // 100  # 10 ms
AGENT_RATE = 24_000
AGENT_SPEECH_RMS = 0.02  # of full scale; the mock TTS tone is at 0.25
AGENT_STATE = "lk.agent.state"


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "t": time.monotonic(), **fields}), flush=True)


@dataclass
class Clip:
    name: str
    samples: np.ndarray
    onset_frame: int  # first 10 ms frame within 20 dB of the loudest
    offset_frame: int  # first frame after the last such frame

    @classmethod
    def load(cls, name: str, path: str) -> Clip:
        with wave.open(path) as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"{path}: expected 16 kHz mono 16-bit PCM")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        frames = samples[: len(samples) // FRAME * FRAME].reshape(-1, FRAME).astype(np.float64)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        speech = np.flatnonzero(rms >= 0.1 * rms.max())
        return cls(name, samples, int(speech[0]), int(speech[-1]) + 1)


class Microphone:
    """A continuous 16 kHz track paced in real time: silence, or the clip being played."""

    def __init__(self) -> None:
        # a short queue keeps a frame's capture time within 50 ms of when it is sent
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=50)
        self._clips: asyncio.Queue[tuple[Clip, asyncio.Future[dict[str, float]]]] = asyncio.Queue()

    def play(self, clip: Clip) -> asyncio.Future[dict[str, float]]:
        """Queue ``clip``; the future resolves to its start, onset, speech_end and end."""
        done = asyncio.get_running_loop().create_future()
        self._clips.put_nowait((clip, done))
        return done

    async def run(self) -> None:
        silence = np.zeros(FRAME, dtype=np.int16)
        clip: Clip | None = None
        done: asyncio.Future[dict[str, float]] | None = None
        times: dict[str, float] = {}
        index = 0
        deadline = time.monotonic()
        while True:
            if clip is None and not self._clips.empty():
                clip, done = self._clips.get_nowait()
                times, index = {}, 0
            samples = silence
            if clip is not None:
                samples = clip.samples[index * FRAME : (index + 1) * FRAME]
                samples = np.pad(samples, (0, FRAME - len(samples)))
            frame = rtc.AudioFrame(samples.tobytes(), SAMPLE_RATE, 1, FRAME)
            await self.source.capture_frame(frame)
            if clip is not None and done is not None:
                now = time.monotonic()
                if index == 0:
                    times["start"] = now
                if index == clip.onset_frame:
                    times["onset"] = now
                    emit("speech_onset", clip=clip.name)
                if index == clip.offset_frame:
                    times["speech_end"] = now
                index += 1
                if index * FRAME >= len(clip.samples):
                    times["end"] = now
                    done.set_result(times)
                    clip = None
            deadline += FRAME / SAMPLE_RATE
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.05:
                deadline = time.monotonic()  # fell behind: resync instead of bursting


class AgentAudio:
    """Spans of the agent's audio above the speech threshold."""

    GAP = 0.1  # quieter stretches shorter than this do not end a span

    def __init__(self) -> None:
        self.spans: list[list[float]] = []

    async def consume(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=AGENT_RATE, num_channels=1, frame_size_ms=10)
        try:
            async for event in stream:
                pcm = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float64)
                if np.sqrt(np.mean((pcm / 32768.0) ** 2)) < AGENT_SPEECH_RMS:
                    continue
                now = time.monotonic()
                if self.spans and now - self.spans[-1][1] <= self.GAP:
                    self.spans[-1][1] = now
                else:
                    self.spans.append([now, now])
        finally:
            await stream.aclose()

    async def first_onset_after(self, t: float, timeout: float) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            starts = [start for start, _ in self.spans if start > t]
            if starts:
                return starts[0]
            await asyncio.sleep(0.01)
        raise TimeoutError("the agent did not start speaking")


def agents(room: rtc.Room) -> list[rtc.RemoteParticipant]:
    return [
        participant
        for participant in room.remote_participants.values()
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    ]


async def until(condition: Any, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        await asyncio.sleep(0.02)


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
    microphone = Microphone()
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
