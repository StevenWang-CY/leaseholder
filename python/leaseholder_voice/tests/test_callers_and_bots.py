from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
import wave
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from leaseholder_voice import audio, bots
from leaseholder_voice.audio import FRAME, Clip, Recorder, resample, upsample_track
from leaseholder_voice.bots import BotsConfig, Caller, Corpus, summarize
from leaseholder_voice.callers import (
    BARGE_IN_LINES,
    TURN_LINES,
    VOICES,
    render,
)


def tone(hz: float, seconds: float = 1.0, rate: int = 24_000) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * 10_000).astype(np.int16)


def test_resampling_keeps_speech_band_and_rejects_aliases():
    speech = resample(tone(440), 2, 3)  # 24 kHz to the corpus's 16 kHz
    assert len(speech) == 16_000
    assert abs(int(np.abs(speech[2000:14000]).max()) - 10_000) < 100
    alias = resample(tone(11_000), 2, 3)  # above the new 8 kHz Nyquist
    assert int(np.abs(alias[2000:14000]).max()) < 30
    back = resample(speech, 3, 2)  # and up again, as --save-audio records the caller
    assert len(back) == 24_000
    assert abs(int(np.abs(back[3000:21000]).max()) - 10_000) < 150
    assert np.argmax(np.abs(np.fft.rfft(back))) == 440  # 1 Hz bins over the second


def test_a_track_is_upsampled_a_stretch_of_sound_at_a_time_exactly():
    silence = np.zeros(4001, dtype=np.int16)
    line = tone(300, 0.5, rate=16_000)
    track = np.concatenate([silence, line, silence[:100], line[:999], silence])  # odd offsets
    assert np.array_equal(upsample_track(track), resample(track, 3, 2))
    assert np.array_equal(upsample_track(silence), resample(silence, 3, 2))


class FakeTTS(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTTS.requests.append(body)
        pcm = tone(300, 0.5).tobytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)

    def log_message(self, *args: object) -> None:
        pass


def test_render_writes_16k_clips_and_a_manifest(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = render(f"http://127.0.0.1:{server.server_port}/v1", tmp_path)
    finally:
        server.shutdown()
    assert len(manifest) == len(TURN_LINES) + len(BARGE_IN_LINES)
    assert {r["voice"] for r in FakeTTS.requests} == set(VOICES)
    assert all(r["response_format"] == "pcm" and r["model"] == "kokoro" for r in FakeTTS.requests)
    first = manifest["turn-00"]
    assert first["kind"] == "turn" and first["text"] == TURN_LINES[0]
    assert first["duration_s"] == 0.9  # 0.2 s of padding on each side of 0.5 s of speech
    with wave.open(str(tmp_path / "turn-00.wav")) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)
    clip = Clip.load("turn-00", str(tmp_path / "turn-00.wav"))
    assert (clip.onset_frame, clip.offset_frame) == (20, 70)  # speech between the paddings
    corpus = Corpus(tmp_path)
    assert len(corpus.turns) == len(TURN_LINES) and len(corpus.barge_ins) == len(BARGE_IN_LINES)


def test_summary_pools_v2v_samples_and_censors_the_replies_that_never_came():
    rows = [
        {"caller": 0, "turn": 0, "v2v_ms": 800.0, "barge_in": False},
        {"caller": 0, "turn": 1, "v2v_ms": 900.0, "barge_in": True, "interrupted_speech": True,
         "barge_in_to_silence_ms": 620.0, "v2v_after_barge_in_ms": 1000.0},
        {"caller": 1, "turn": 0, "v2v_ms": 1100.0, "barge_in": True, "interrupted_speech": False,
         "v2v_after_barge_in_ms": 950.0},
        {"caller": 1, "turn": 1, "error": "the agent did not start speaking"},
        # The reply came; after the interruption the agent never stopped speaking.
        {"caller": 1, "turn": 2, "v2v_ms": 700.0, "barge_in": True, "interrupted_speech": True,
         "error": "the agent did not stop speaking"},
    ]  # fmt: skip
    summary = summarize(rows, [0.004, 0.012], timeout_s=30.0)
    assert summary["turns"] == 5 and summary["errors"] == 2
    # The failed turn's reply and the unanswered interruption count at the 30 s timeout.
    assert summary["v2v_ms"] == {
        "n": 8, "censored": 2, "p50": 975.0, "p95": 30_000.0, "p99": 30_000.0
    }  # fmt: skip
    silence = summary["barge_in_to_silence_ms"]
    assert (silence["n"], silence["censored"], silence["p50"]) == (2, 1, 15_310.0)
    assert summary["barge_ins"] == 3 and summary["barge_ins_after_the_reply"] == 1
    assert summary["caller_max_lateness_ms"] == 12.0
    # Without a failure the summary is as it always was: no censored count.
    answered = summarize(rows[:3], [0.0])
    assert answered["v2v_ms"] == {"n": 5, "p50": 950.0, "p95": 1080.0, "p99": 1096.0}


def test_the_summary_leaves_the_warm_up_turns_out():
    rows = [
        {"caller": 0, "turn": 0, "v2v_ms": 2500.0, "barge_in": False, "warmup": True},
        {"caller": 0, "turn": 1, "v2v_ms": 800.0, "barge_in": False},
        {"caller": 1, "turn": 0, "error": "the agent did not start speaking", "warmup": True},
        {"caller": 1, "turn": 1, "v2v_ms": 900.0, "barge_in": False},
    ]
    summary = summarize(rows, [0.0])
    assert (summary["turns"], summary["errors"]) == (2, 0)
    assert summary["v2v_ms"] == {"n": 2, "p50": 850.0, "p95": 895.0, "p99": 899.0}
    assert summary == summarize([rows[1], rows[3]], [0.0])
    with pytest.raises(ValueError, match="warmup_turns cannot be negative"):
        BotsConfig("ws://fake", "key", "secret", Path("."), warmup_turns=-1)


def test_a_recorded_track_keeps_its_frames_together_and_its_pauses():
    recorder = Recorder(1000)  # 10-sample frames of 10 ms
    # Frames 0-2 come in one burst at 25 ms, 3 on time, 4 a millisecond late, 5 after a pause.
    for k, at in enumerate((0.025, 0.0251, 0.0252, 0.030, 0.041, 0.080)):
        recorder.add(at, np.full(10, k + 1, dtype=np.int16))
    signal = recorder.signal(0.0, 100)
    assert list(signal[:50]) == [n for n in range(1, 6) for _ in range(10)]  # from 0 ms, no gap
    assert not signal[50:80].any() and list(signal[80:90]) == [6] * 10


REPLY_S, SPOKEN_S = 0.2, 0.4
"""The fake agent answers each line REPLY_S after its end of speech, with SPOKEN_S of tone."""


class Line:
    """The fake room: the caller's microphone reaches a fake agent, which answers each line on the
    track the caller's AgentAudio reads, as LiveKit's AudioSource and AudioStream would carry it."""

    def __init__(self) -> None:
        self.replies: list[float] = []
        self._speaking = False

    def source(self, *_: Any, **__: Any) -> Line:  # rtc.AudioSource(rate, channels, ...)
        return self

    async def capture_frame(self, frame: Any) -> None:
        loud = int(np.abs(np.frombuffer(frame.data, dtype=np.int16)).max()) > 1000
        if self._speaking and not loud:
            self.replies.append(time.monotonic() + REPLY_S)
        self._speaking = loud
        await asyncio.sleep(0)

    async def aclose(self) -> None:
        pass

    def stream(self, *_: Any, **__: Any) -> Line:  # rtc.AudioStream(track, ...)
        return self

    async def __aiter__(self) -> AsyncIterator[Any]:
        """A frame of 10 ms every 10 ms, of the reply's tone or of silence."""
        speech, silence = tone(440, 0.01)[:240], np.zeros(240, dtype=np.int16)
        deadline = time.monotonic()
        while True:
            deadline += 0.01
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            replying = any(start <= now < start + SPOKEN_S for start in self.replies)
            frame = SimpleNamespace(data=memoryview(speech if replying else silence))
            yield SimpleNamespace(frame=frame)


def line_clip(name: str, seconds: float) -> Clip:
    """A 300 Hz line with 0.1 s of silence on each side, as the corpus pads its lines."""
    pad = np.zeros(10 * FRAME, dtype=np.int16)
    samples = np.concatenate([pad, tone(300, seconds, rate=16_000), pad])
    return Clip(name, samples, 10, 10 + round(seconds * 100))


def runs(channel: np.ndarray) -> list[tuple[int, int]]:
    """The first and last loud sample of each stretch of a channel without 10 ms of quiet."""
    loud = np.flatnonzero(np.abs(channel.astype(int)) > 3000)
    breaks = np.flatnonzero(np.diff(loud) > 240)  # longer than a tone's zero crossings
    return list(zip(loud[np.r_[0, breaks + 1]], loud[np.r_[breaks, -1]], strict=True))


def test_a_saved_call_lines_up_what_the_caller_sent_with_what_it_heard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = Line()
    monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
    monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    corpus = SimpleNamespace(turns=[line_clip("turn-00", 0.3)], barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, barge_in=0.0, save_audio=True)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05))

    async def call() -> tuple[Caller, list[dict[str, Any]]]:
        caller = Caller(config, corpus, 0, "test")
        tasks = [
            asyncio.create_task(caller.microphone.run()),
            asyncio.create_task(caller.agent_audio.consume(line)),
        ]
        await asyncio.sleep(0.2)
        rows = [await caller.turn(0)]
        caller.config = dataclasses.replace(config, barge_in=1.0)  # the next one barges in
        rows.append(await caller.turn(1))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return caller, [{"caller": 0} | row for row in rows]

    caller, rows = asyncio.run(call())
    caller.save(tmp_path, rows)
    with wave.open(str(tmp_path / "call-0.wav")) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 2, 24_000)
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).reshape(-1, 2)
    saved = json.loads((tmp_path / "call-0.json").read_text())
    assert (saved["caller"], saved["rate"], saved["channels"]) == (0, 24_000, ["caller", "agent"])
    assert saved["seconds"] == pytest.approx(len(pcm) / 24_000, abs=1e-3)
    wall = time.time() - time.monotonic()  # the wall clock's lead on the monotonic one
    assert saved["start"]["wall"] - saved["start"]["monotonic"] == pytest.approx(wall, abs=0.1)
    first, second = saved["turns"]
    assert {k: v for k, v in first.items() if k != "at"} == {
        k: v for k, v in rows[0].items() if k != "caller"
    }
    assert first["at"].keys() == {"start", "speech_end", "reply", "end"}
    assert second["barge_in"] and second["at"].keys() == {
        "start", "speech_end", "reply", "barge_in", "barge_in_speech_end", "silence",
        "barge_in_reply", "end",
    }  # fmt: skip
    # The caller's lines on the left, the agent's replies on the right, where the times say.
    sent, heard = runs(pcm[:, 0]), runs(pcm[:, 1])
    assert len(sent) == 3 and len(heard) == 3
    turns = [(first, ""), (second, ""), (second, "barge_in_")]
    ends = [turn["at"][f"{event}speech_end"] for turn, event in turns]
    replies = [turn["at"][f"{event}reply"] for turn, event in turns]
    assert [last / 24_000 for _, last in sent] == pytest.approx(ends, abs=0.015)
    assert [onset / 24_000 for onset, _ in heard] == pytest.approx(replies, abs=0.015)
    v2v = [first["v2v_ms"], second["v2v_ms"], second["v2v_after_barge_in_ms"]]
    gaps = [(onset - last) / 24 for (_, last), (onset, _) in zip(sent, heard, strict=True)]
    assert gaps == pytest.approx(v2v, abs=25)  # the waveforms show the V2V the callers measured
    assert all(REPLY_S * 1000 <= ms <= REPLY_S * 1000 + 60 for ms in v2v)


def test_a_call_opens_with_its_warm_up_turns_and_measures_the_same_turns_after_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    lines = [line_clip(f"turn-{n:02}", 0.2) for n in range(8)]
    corpus = SimpleNamespace(turns=lines, barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, turns=2, barge_in=1.0, seed=3)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05), think=(0.0, 0.0))

    def call(warmup_turns: int) -> list[dict[str, Any]]:
        line = Line()
        monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
        monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)

        async def play() -> list[dict[str, Any]]:
            caller = Caller(dataclasses.replace(config, warmup_turns=warmup_turns), corpus, 0, "t")
            tasks = [
                asyncio.create_task(caller.microphone.run()),
                asyncio.create_task(caller.agent_audio.consume(line)),
            ]
            await asyncio.sleep(0.2)
            rows: list[dict[str, Any]] = []
            with (tmp_path / f"turns-{warmup_turns}.jsonl").open("w") as log:
                await caller.play(rows, log)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return rows

        return asyncio.run(play())

    plain, warmed = call(0), call(1)
    # One warm-up turn first, a line and its reply without an interruption, marked as such.
    assert [(row["turn"], row.get("warmup", False), row["barge_in"]) for row in warmed] == [
        (0, True, False),
        (1, False, True),
        (2, False, True),
    ]
    assert REPLY_S * 1000 <= warmed[0]["v2v_ms"] <= REPLY_S * 1000 + 60
    # The measured turns play the lines and interruptions of the same call without it.
    script = [(row["clip"], row["barge_in_clip"]) for row in plain]
    assert [(row["clip"], row["barge_in_clip"]) for row in warmed[1:]] == script
    logged = (tmp_path / "turns-1.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in logged] == warmed


class FirstReplyOnly(Line):
    """The fake agent answers the first line it hears and never the interruption after it."""

    async def capture_frame(self, frame: Any) -> None:
        await super().capture_frame(frame)
        del self.replies[1:]


def test_an_unanswered_interruption_fails_the_turn_and_keeps_its_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = FirstReplyOnly()
    monkeypatch.setattr(audio.rtc, "AudioSource", line.source)
    monkeypatch.setattr(audio.rtc, "AudioStream", line.stream)
    monkeypatch.setattr(bots, "QUIET_S", 0.2)
    corpus = SimpleNamespace(turns=[line_clip("turn-00", 0.3)], barge_ins=[line_clip("barge", 0.2)])
    config = BotsConfig("ws://fake", "key", "secret", tmp_path, barge_in=1.0, reply_timeout=0.5)
    config = dataclasses.replace(config, barge_in_delay=(0.05, 0.05))

    async def call() -> dict[str, Any]:
        caller = Caller(config, corpus, 0, "test")
        tasks = [
            asyncio.create_task(caller.microphone.run()),
            asyncio.create_task(caller.agent_audio.consume(line)),
        ]
        await asyncio.sleep(0.2)
        row = await caller.turn(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return row

    row = asyncio.run(call())
    assert row["error"] == "the agent did not start speaking"
    assert row["barge_in"] and "v2v_after_barge_in_ms" not in row
    assert REPLY_S * 1000 <= row["v2v_ms"] <= REPLY_S * 1000 + 60  # the reply it measured stays
    summary = summarize([row], [0.0], timeout_s=0.5)
    assert summary["errors"] == 1
    assert (summary["v2v_ms"]["n"], summary["v2v_ms"]["censored"]) == (2, 1)
    assert summary["v2v_ms"]["p99"] == pytest.approx(500.0, abs=5)
