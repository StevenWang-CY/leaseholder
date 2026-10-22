"""Caller-side audio for voice measurements: a real-time microphone and the agent's speech spans.

Every time is ``time.monotonic()`` in the calling process, so a latency measured from a caller's
microphone to the agent's audio carries no clock offset.
"""

from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from livekit import rtc

SAMPLE_RATE = 16_000
FRAME = SAMPLE_RATE // 100  # 10 ms
AGENT_RATE = 24_000
AGENT_SPEECH_RMS = 0.02
"""Agent audio at or above this RMS (of full scale) counts as speech."""
AGENT_STATE = "lk.agent.state"


@dataclass
class Clip:
    name: str
    samples: np.ndarray
    onset_frame: int
    """First 10 ms frame within 20 dB of the loudest."""
    offset_frame: int
    """First frame after the last such frame: where the speech ends."""

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
    """A continuous 16 kHz track paced in real time: silence, or the clip being played.

    ``on_onset``, when given, is called as each clip's speech onset goes out.
    """

    def __init__(self, on_onset: Callable[[Clip], None] | None = None) -> None:
        # a short queue keeps a frame's capture time within 50 ms of when it is sent
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=50)
        self._on_onset = on_onset
        self.max_lateness_s = 0.0
        """The pacing loop's worst lateness: a measure of whether the caller kept up."""
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
                    if self._on_onset is not None:
                        self._on_onset(clip)
                if index == clip.offset_frame:
                    times["speech_end"] = now
                index += 1
                if index * FRAME >= len(clip.samples):
                    times["end"] = now
                    times.setdefault("speech_end", now)
                    done.set_result(times)
                    clip = None
            deadline += FRAME / SAMPLE_RATE
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                self.max_lateness_s = max(self.max_lateness_s, -delay)
                if delay < -0.05:
                    deadline = time.monotonic()  # fell behind: resync instead of bursting


class AgentAudio:
    """Spans of the agent's audio above the speech threshold."""

    GAP = 0.1
    """Quieter stretches shorter than this do not end a span."""

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
        """When the agent first starts speaking after ``t``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            starts = [start for start, _ in self.spans if start > t]
            if starts:
                return starts[0]
            await asyncio.sleep(0.01)
        raise TimeoutError("the agent did not start speaking")

    def speaking_at(self, t: float) -> bool:
        return any(start <= t <= end + self.GAP for start, end in self.spans)

    def started_between(self, a: float, b: float) -> bool:
        return any(a < start < b for start, _ in self.spans)

    async def end_of_span_at(self, t: float, timeout: float) -> float:
        """When the agent's speech that was under way at ``t`` stopped (at least ``GAP`` of quiet)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            span = next((s for s in self.spans if s[0] <= t <= s[1] + self.GAP), None)
            if span is None:
                return t
            if time.monotonic() - span[1] > self.GAP:
                return span[1]
            await asyncio.sleep(0.01)
        raise TimeoutError("the agent did not stop speaking")

    async def silent_after(self, t: float, quiet: float, timeout: float) -> float:
        """When the agent's audio stops for good after ``t``: the end of the last span that began
        before a ``quiet``-second stretch of silence."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            last_end = max((end for _, end in self.spans if end >= t), default=t)
            if now - last_end >= quiet:
                return last_end
            await asyncio.sleep(0.02)
        raise TimeoutError("the agent did not stop speaking")


def agents(room: rtc.Room) -> list[rtc.RemoteParticipant]:
    return [
        participant
        for participant in room.remote_participants.values()
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    ]


async def until(condition: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        await asyncio.sleep(0.02)
