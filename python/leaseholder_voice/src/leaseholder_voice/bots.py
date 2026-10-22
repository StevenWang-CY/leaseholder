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
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import statistics
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from livekit import api, rtc

from .audio import AGENT_STATE, AgentAudio, Clip, Microphone, agents, until
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
        self.agent_audio = AgentAudio()
        self.microphone = Microphone()
        self.max_lateness_s = 0.0

    async def turn(self, number: int) -> dict[str, Any]:
        config, audio = self.config, self.agent_audio
        clip = self.rng.choice(self.corpus.turns)
        played = await self.microphone.play(clip)
        reply = await audio.first_onset_after(played["speech_end"], config.reply_timeout)
        row: dict[str, Any] = {
            "caller": self.index,
            "turn": number,
            "clip": clip.name,
            "v2v_ms": round((reply - played["speech_end"]) * 1000, 1),
            # the agent answered part of the line before the caller finished (a VAD split)
            "spoke_during_turn": audio.started_between(played["onset"], played["speech_end"]),
            "barge_in": False,
        }
        if self.rng.random() < config.barge_in:
            delay = self.rng.uniform(*config.barge_in_delay)
            await asyncio.sleep(max(0.0, reply + delay - time.monotonic()))
            interruption = self.rng.choice(self.corpus.barge_ins)
            barged = await self.microphone.play(interruption)
            onset = barged["onset"]
            row.update(barge_in=True, barge_in_clip=interruption.name)
            row["interrupted_speech"] = audio.speaking_at(onset)
            if row["interrupted_speech"]:
                stopped = await audio.end_of_span_at(onset, config.reply_timeout)
                row["barge_in_to_silence_ms"] = round(max(0.0, stopped - onset) * 1000, 1)
            second = await audio.first_onset_after(barged["speech_end"], config.reply_timeout)
            row["v2v_after_barge_in_ms"] = round((second - barged["speech_end"]) * 1000, 1)
        await audio.silent_after(time.monotonic(), QUIET_S, config.reply_timeout * 2)
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
            for number in range(config.turns):
                try:
                    row = await self.turn(number)
                except TimeoutError as exc:
                    row = {"caller": self.index, "turn": number, "error": str(exc)}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                await asyncio.sleep(self.rng.uniform(*config.think))
        finally:
            self.max_lateness_s = self.microphone.max_lateness_s
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.microphone.source.aclose()
            await self.room.disconnect()


def percentiles(values: Iterable[float]) -> dict[str, float | int]:
    xs = sorted(values)
    if len(xs) < 2:
        return {"n": len(xs), **({"p50": xs[0]} if xs else {})}
    q = statistics.quantiles(xs, n=100, method="inclusive")
    return {"n": len(xs), "p50": round(q[49], 1), "p95": round(q[94], 1), "p99": round(q[98], 1)}


def summarize(rows: list[dict[str, Any]], lateness: list[float]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    v2v = [r["v2v_ms"] for r in ok] + [r["v2v_after_barge_in_ms"] for r in ok if r["barge_in"]]
    barge = [r["barge_in_to_silence_ms"] for r in ok if r.get("barge_in_to_silence_ms") is not None]
    return {
        "turns": len(rows),
        "errors": len(rows) - len(ok),
        "v2v_ms": percentiles(v2v),
        "barge_in_to_silence_ms": percentiles(barge),
        "barge_ins": sum(1 for r in ok if r["barge_in"]),
        "barge_ins_after_the_reply": sum(
            1 for r in ok if r["barge_in"] and not r["interrupted_speech"]
        ),
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
    summary = summarize(rows, [c.max_lateness_s for c in callers])
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
    return summary
