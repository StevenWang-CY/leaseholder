"""Measurement callers for the voice agent (``lh-voice bots``): latency read from waveforms.

Each caller joins its own room with a token that dispatches the agent, publishes a microphone
track paced in real time and records the agent's audio. A call is a sequence of turns. The caller
plays a line from the corpus and waits for the reply:

- **voice-to-voice** (V2V) runs from the end of the caller's speech to the first agent audio above
  the speech threshold;
- on a share of turns (``--barge-in``) the caller interrupts the reply after a random delay;
  **barge-in to silence** runs from the interruption's speech onset until the agent's audio
  stops, and the reply to the interruption gives a second V2V sample.

The caller then waits until the agent has been quiet for a second, pauses for a think time and
takes the next turn. Every time comes from one clock in one process, so no clock offset enters
the numbers. Each run directory gets ``run.json``, ``turns.jsonl`` and ``summary.json``.

A call may open with warm-up turns (``--warmup-turns``): a line and its reply, without an
interruption, played but not measured. Their lines in turns.jsonl say ``"warmup": true`` and the
summary leaves them out. They draw their lines and pauses from a generator of their own, so the
measured turns after them are those of the same call without them.

A turn fails when the agent does not answer within ``reply_timeout``, or does not stop speaking: its
line in turns.jsonl has an ``error`` and keeps what the turn measured before. In the summary a
sample that the timeout cut short counts as censored at the timeout, never as missing, so that a
failure raises a percentile rather than leaving it out.

With ``--save-audio`` each caller also keeps the frames it already handles, what it sent and the
agent's audio as received, and once every call has ended writes them as ``call-<i>.wav`` with the
times of each turn's events in ``call-<i>.json``, for the demo video (benchmarks/e2/demo.py).
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import random
import statistics
import time
import uuid
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from livekit import api, rtc

from .audio import (
    AGENT_RATE,
    AGENT_STATE,
    SAMPLE_RATE,
    AgentAudio,
    Clip,
    Microphone,
    Recorder,
    agents,
    until,
    upsample_track,
)
from .protocol import AGENT_NAME

QUIET_S = 1.0
"""Agent silence that ends its reply."""


@dataclass(frozen=True)
class BotsConfig:
    url: str
    """LiveKit signalling URL, e.g. ``ws://127.0.0.1:17880``."""
    api_key: str
    api_secret: str
    corpus: Path
    callers: int = 2
    turns: int = 20
    barge_in: float = 0.25
    """Share of turns on which the caller interrupts the reply."""
    barge_in_delay: tuple[float, float] = (0.6, 1.5)
    """Seconds after the reply's first audio at which an interruption starts."""
    think: tuple[float, float] = (1.0, 3.0)
    """Seconds between the end of a reply and the next turn."""
    reply_timeout: float = 30.0
    seed: int = 0
    warmup_turns: int = 0
    """Turns that open each call unmeasured: a line and its reply, no interruption."""
    save_audio: bool = False
    """Write each call's audio and its turns' times after the calls: call-<i>.wav, call-<i>.json."""

    def __post_init__(self) -> None:
        if self.warmup_turns < 0:
            raise ValueError("warmup_turns cannot be negative")


class Corpus:
    def __init__(self, directory: Path) -> None:
        manifest = json.loads((directory / "manifest.json").read_text())
        self.turns = [
            Clip.load(name, str(directory / f"{name}.wav"))
            for name, entry in sorted(manifest.items())
            if entry["kind"] == "turn"
        ]
        self.barge_ins = [
            Clip.load(name, str(directory / f"{name}.wav"))
            for name, entry in sorted(manifest.items())
            if entry["kind"] == "barge_in"
        ]
        if not self.turns or not self.barge_ins:
            raise ValueError(f"{directory}: the corpus needs turn and barge_in clips")


def token(config: BotsConfig, room: str, identity: str) -> str:
    """A join token for ``room`` that also dispatches the agent into it."""
    dispatch = api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
    return (
        api.AccessToken(config.api_key, config.api_secret)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(dispatch)
        .to_jwt()
    )


class Caller:
    def __init__(self, config: BotsConfig, corpus: Corpus, index: int, run_id: str) -> None:
        self.config = config
        self.corpus = corpus
        self.index = index
        self.room_name = f"lh-bot-{run_id}-{index}"
        self.rng = random.Random(config.seed * 1000 + index)
        self.room = rtc.Room()
        self.sent = Recorder(SAMPLE_RATE) if config.save_audio else None
        self.received = Recorder(AGENT_RATE) if config.save_audio else None
        self.agent_audio = AgentAudio(self.received)
        self.microphone = Microphone(recorder=self.sent)
        self.max_lateness_s = 0.0
        self.moments: dict[int, dict[str, float]] = {}
        """When each turn's events happened (its line's start and speech end, the reply, ...), as far
        as the turn got; call-<i>.json gives them."""

    async def turn(self, number: int, warmup: random.Random | None = None) -> dict[str, Any]:
        """One turn of the call; a warm-up turn draws its line from ``warmup`` and is never
        interrupted."""
        config, audio = self.config, self.agent_audio
        clip = (warmup or self.rng).choice(self.corpus.turns)
        at = self.moments[number] = {}
        played = await self.microphone.play(clip)
        at |= {"start": played["start"], "speech_end": played["speech_end"]}
        reply = await audio.first_onset_after(played["speech_end"], config.reply_timeout)
        at["reply"] = reply
        row: dict[str, Any] = {
            "caller": self.index,
            "turn": number,
            "clip": clip.name,
            "v2v_ms": round((reply - played["speech_end"]) * 1000, 1),
            # the agent answered part of the line before the caller finished (a VAD split)
            "spoke_during_turn": audio.started_between(played["onset"], played["speech_end"]),
            "barge_in": False,
        }
        try:
            if warmup is None and self.rng.random() < config.barge_in:
                delay = self.rng.uniform(*config.barge_in_delay)
                await asyncio.sleep(max(0.0, reply + delay - time.monotonic()))
                interruption = self.rng.choice(self.corpus.barge_ins)
                barged = await self.microphone.play(interruption)
                onset = barged["onset"]
                at |= {"barge_in": onset, "barge_in_speech_end": barged["speech_end"]}
                row.update(barge_in=True, barge_in_clip=interruption.name)
                row["interrupted_speech"] = audio.speaking_at(onset)
                if row["interrupted_speech"]:
                    stopped = await audio.end_of_span_at(onset, config.reply_timeout)
                    at["silence"] = stopped
                    row["barge_in_to_silence_ms"] = round(max(0.0, stopped - onset) * 1000, 1)
                second = await audio.first_onset_after(barged["speech_end"], config.reply_timeout)
                at["barge_in_reply"] = second
                row["v2v_after_barge_in_ms"] = round((second - barged["speech_end"]) * 1000, 1)
            at["end"] = await audio.silent_after(
                time.monotonic(), QUIET_S, config.reply_timeout * 2
            )
        except TimeoutError as exc:  # the turn fails, keeping the reply it measured
            row["error"] = str(exc)
        return row

    async def call(self, rows: list[dict[str, Any]], log: TextIO) -> None:
        config = self.config
        tasks: set[asyncio.Task[None]] = set()

        @self.room.on("track_subscribed")
        def _on_track(track: rtc.Track, _: Any, participant: rtc.RemoteParticipant) -> None:
            is_agent = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            if is_agent and track.kind == rtc.TrackKind.KIND_AUDIO:
                tasks.add(asyncio.create_task(self.agent_audio.consume(track)))

        identity = f"caller-{self.index}"
        await self.room.connect(config.url, token(config, self.room_name, identity))
        track = rtc.LocalAudioTrack.create_audio_track("microphone", self.microphone.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)
        tasks.add(asyncio.create_task(self.microphone.run()))
        try:
            await until(
                lambda: any(
                    p.attributes.get(AGENT_STATE) == "listening" for p in agents(self.room)
                ),
                config.reply_timeout,
                "the agent never listened",
            )
            await asyncio.sleep(0.5)  # let the agent's audio input run on silence first
            await self.play(rows, log)
        finally:
            self.max_lateness_s = self.microphone.max_lateness_s
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.microphone.source.aclose()
            await self.room.disconnect()

    async def play(self, rows: list[dict[str, Any]], log: TextIO) -> None:
        """The call's turns, each line to ``rows`` and ``log`` (turns.jsonl) as it ends, numbered in
        the order they are played: first the warm-up turns, marked ``"warmup": true``, whose lines
        and pauses come from a generator of their own, then the measured turns."""
        config = self.config
        warmup = random.Random(f"warmup-{config.seed}-{self.index}")
        for number in range(config.warmup_turns + config.turns):
            warm = number < config.warmup_turns
            try:
                row = await self.turn(number, warmup if warm else None)
            except TimeoutError as exc:
                row = {"caller": self.index, "turn": number, "error": str(exc)}
            if warm:
                row["warmup"] = True
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            await asyncio.sleep((warmup if warm else self.rng).uniform(*config.think))

    def save(self, out_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
        """Write the call as the caller had it (``--save-audio``), from the frames it kept.

        ``call-<i>.wav`` is stereo 16-bit at the agent's rate: on the left what the caller sent,
        upsampled from 16 kHz, on the right the agent's audio as received, both on the caller's
        clock from the first frame of either. ``call-<i>.json`` gives when the WAV starts, on that
        clock and on the wall clock, and each of the caller's turns as turns.jsonl has it, with
        when its events happened in seconds into the WAV (``at``): its line's start and end of
        speech, the reply, an interruption's onset and end of speech, the agent's silence after it
        and its reply to it, and the end of the turn.
        """
        sent, received = self.sent, self.received
        if sent is None or received is None:
            raise ValueError("the caller kept no audio: save_audio is off")
        tracks = [track for track in (sent, received) if track.frames]
        start = min(track.frames[0][0] for track in tracks)
        end = max(track.frames[-1][0] + len(track.frames[-1][1]) / track.rate for track in tracks)
        length = math.ceil((end - start) * AGENT_RATE)
        caller = upsample_track(sent.signal(start, math.ceil((end - start) * SAMPLE_RATE)))
        stereo = np.zeros((length, 2), dtype=np.int16)
        stereo[: len(caller), 0] = caller[:length]
        stereo[:, 1] = received.signal(start, length)
        with wave.open(str(out_dir / f"call-{self.index}.wav"), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(AGENT_RATE)
            wav.writeframes(stereo.tobytes())
        turns = [
            {key: value for key, value in row.items() if key != "caller"}
            | {"at": {event: round(t - start, 4) for event, t in self.moments[row["turn"]].items()}}
            for row in rows
        ]
        call = {
            "caller": self.index,
            "room": self.room_name,
            "rate": AGENT_RATE,
            "channels": ["caller", "agent"],
            "start": {"monotonic": start, "wall": start + time.time() - time.monotonic()},
            "seconds": round(length / AGENT_RATE, 3),
            "turns": turns,
        }
        (out_dir / f"call-{self.index}.json").write_text(json.dumps(call, indent=1) + "\n")


def percentiles(values: Iterable[float], censored: int = 0) -> dict[str, float | int]:
    """Count, p50, p95 and p99 of samples in ms; ``censored`` of them stand at the reply timeout."""
    xs = sorted(values)
    count = {"n": len(xs)} | ({"censored": censored} if censored else {})
    if len(xs) < 2:
        return count | ({"p50": xs[0]} if xs else {})
    q = statistics.quantiles(xs, n=100, method="inclusive")
    return count | {"p50": round(q[49], 1), "p95": round(q[94], 1), "p99": round(q[98], 1)}


def v2v_samples(rows: Iterable[dict[str, Any]], timeout_ms: float) -> tuple[list[float], int]:
    """The callers' voice-to-voice samples and how many of them are censored: each turn's reply and,
    after a barge-in, the reply to the interruption, a reply that never came standing at the
    timeout."""
    samples, censored = [], 0
    for row in rows:
        wanted = ["v2v_ms", *(["v2v_after_barge_in_ms"] if row.get("barge_in") else [])]
        for key in wanted:
            censored += key not in row
            samples.append(row.get(key, timeout_ms))
    return samples, censored


def summarize(
    rows: list[dict[str, Any]], lateness: list[float], timeout_s: float = 30.0
) -> dict[str, Any]:
    """The calls' summary of their measured turns, the warm-up turns left out. A sample a timeout cut
    short counts at the timeout (``timeout_s``): the reply to a turn or to an interruption that never
    came, the agent's audio that never stopped."""
    rows = [row for row in rows if not row.get("warmup")]
    timeout_ms = timeout_s * 1000
    v2v, censored = v2v_samples(rows, timeout_ms)
    silence = [r["barge_in_to_silence_ms"] for r in rows if "barge_in_to_silence_ms" in r]
    cut = sum(1 for r in rows if r.get("interrupted_speech") and "barge_in_to_silence_ms" not in r)
    barge_ins = [r for r in rows if r.get("barge_in")]
    return {
        "turns": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "v2v_ms": percentiles(v2v, censored),
        "barge_in_to_silence_ms": percentiles([*silence, *[timeout_ms] * cut], cut),
        "barge_ins": len(barge_ins),
        "barge_ins_after_the_reply": sum(1 for r in barge_ins if not r["interrupted_speech"]),
        "caller_max_lateness_ms": round(max(lateness, default=0.0) * 1000, 1),
    }


async def run_bots(config: BotsConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(config.corpus)
    run_id = uuid.uuid4().hex[:8]
    callers = [Caller(config, corpus, i, run_id) for i in range(config.callers)]
    rows: list[dict[str, Any]] = []
    started = time.time()
    with (out_dir / "turns.jsonl").open("w") as log:
        await asyncio.gather(*(caller.call(rows, log) for caller in callers))
    summary = summarize(rows, [c.max_lateness_s for c in callers], config.reply_timeout)
    run = {
        "kind": "voice-bots",
        "config": asdict(config) | {"api_secret": "<redacted>", "corpus": str(config.corpus)},
        "host": platform.node(),
        "run_id": run_id,
        "started": started,
        "finished": time.time(),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if config.save_audio:  # once every call has ended, so that writing them delays no measurement
        for caller in callers:
            caller.save(out_dir, [row for row in rows if row["caller"] == caller.index])
    return summary
