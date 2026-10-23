"""The demo video's renderer (benchmarks/e2/demo.py) on a synthetic pair of saved calls: tones for
the caller's lines and the agent's replies, in the files lh-voice bots --save-audio writes.

The excerpt, the timelines, the soundtrack and the frames run here, with this venv's numpy and
matplotlib. Encoding needs the ffmpeg that the script's own dependencies bring: that test runs the
script with uv as the docs do, and is marked integration, so CI leaves it out."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[3] / "benchmarks" / "e2" / "demo.py"
_spec = importlib.util.spec_from_file_location("demo", DEMO)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
sys.modules["demo"] = demo
_spec.loader.exec_module(demo)

RATE = 24_000
WALL = 1_800_000_000.0
LINE_S, REPLY_S, PAUSE_S = 1.6, 2.0, 2.5
"""Each line's speech, each reply, and the quiet then the think time before the next line."""


def tone(hz: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * hz * t) * 12_000).astype(np.int16)


def saved_call(run_dir: Path, v2v_s: list[float], *, line_hz: float, label: str) -> Path:
    """A point's run directory with one saved call: a line, the reply V2V later, a pause; turn 1
    is interrupted and answered again. AIPerf completes 1,000 tokens a second all along."""
    pcm = np.zeros((round(20 * len(v2v_s) * RATE), 2), dtype=np.int16)
    turns, t = [], 1.0
    for number, v2v in enumerate(v2v_s):
        start, speech_end = t, t + 0.2 + LINE_S
        pcm[round((start + 0.2) * RATE) :][: round(LINE_S * RATE), 0] = tone(line_hz, LINE_S)
        reply = speech_end + v2v
        pcm[round(reply * RATE) :][: round(REPLY_S * RATE), 1] = tone(440, REPLY_S)
        row: dict[str, Any] = {"turn": number, "clip": f"turn-{number:02d}", "v2v_ms": v2v * 1000}
        at = {"start": start, "speech_end": speech_end, "reply": reply}
        end = reply + REPLY_S
        if number == 1:
            onset = reply + 0.8
            pcm[round(onset * RATE) :][: round(0.6 * RATE), 0] = tone(line_hz, 0.6)
            pcm[round((onset + 0.3) * RATE) : round(end * RATE), 1] = 0  # the agent stops
            second = onset + 0.6 + v2v
            pcm[round(second * RATE) :][: round(REPLY_S * RATE), 1] = tone(440, REPLY_S)
            row |= {"barge_in": True, "barge_in_clip": "barge_in-40", "interrupted_speech": True}
            row |= {"barge_in_to_silence_ms": 300.0, "v2v_after_barge_in_ms": v2v * 1000}
            at |= {"barge_in": onset, "barge_in_speech_end": onset + 0.6, "silence": onset + 0.3}
            at |= {"barge_in_reply": second}
            end = second + REPLY_S
        row.setdefault("barge_in", False)
        turns.append(row | {"at": at | {"end": end}})
        t = end + PAUSE_S
    pcm = pcm[: round((t + 1) * RATE)]
    (run_dir / "bots").mkdir(parents=True)
    with wave.open(str(run_dir / "bots" / "call-0.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    call = {"caller": 0, "rate": RATE, "channels": ["caller", "agent"], "turns": turns}
    call["start"] = {"monotonic": 5000.0, "wall": WALL}
    (run_dir / "bots" / "call-0.json").write_text(json.dumps(call))
    (run_dir / "run.json").write_text(json.dumps({"arm": {"label": label}}))
    (run_dir / "summary.json").write_text(json.dumps({"background": {"tokens_per_s": 1000.0}}))
    records = [
        {
            "metadata": {"benchmark_phase": "profiling", "request_end_ns": int((WALL + s) * 1e9)},
            "metrics": {
                "usage_prompt_tokens": {"value": 990},
                "usage_completion_tokens": {"value": 10},
            },
        }
        for s in range(-30, 90)
    ]
    (run_dir / "aiperf").mkdir()
    (run_dir / "aiperf" / "profile_export.jsonl").write_text("\n".join(map(json.dumps, records)))
    return run_dir


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    slow = saved_call(tmp_path / "frozen", [3.0, 2.5, 4.0, 3.5], line_hz=300, label="Frozen box")
    fast = saved_call(tmp_path / "a9", [1.0, 0.8, 1.2, 0.9], line_hz=500, label="Leaseholder")
    return slow, fast


def test_both_calls_take_their_turns_together(pair: tuple[Path, Path]) -> None:
    calls = (demo.Call.load(pair[0]), demo.Call.load(pair[1]))
    assert [call.label for call in calls] == ["Frozen box", "Leaseholder"]
    timeline = demo.Timeline.choose(calls, seconds=30)
    # As many turns as fit in 30 s (9.3 and 12.7 s), each as long as the slower call's.
    lengths = [b - a for a, b in timeline.spans[0]]
    assert lengths == pytest.approx([9.3, 12.7]) and timeline.seconds == pytest.approx(22.0)
    assert all(b - a < slow for (a, b), slow in zip(timeline.spans[1], lengths, strict=True))
    # The faster call holds the end of its turn while the slower one finishes.
    held = timeline.spans[1][0][1]
    assert timeline.call_time(1, lengths[0] - 0.01) == pytest.approx(held)
    assert timeline.call_time(1, lengths[0] + 0.5) == pytest.approx(timeline.spans[1][1][0] + 0.5)
    # Each turn's line starts at the same moment on both sides.
    for side in (0, 1):
        speaking = demo.envelope(timeline.track(side)[:, 0], RATE) > 0.1
        starts = np.flatnonzero(np.diff(speaking.astype(int)) == 1)[:2] + 1
        assert starts / 100 == pytest.approx([0.7, lengths[0] + 0.7], abs=0.015)
    assert demo.Timeline.choose(calls, first=1, turns=3).offsets[0] == 0.0
    with pytest.raises(ValueError, match="both calls have 3 turns from turn 1, not 4"):
        demo.Timeline.choose(calls, first=1, turns=4)


def test_the_waits_are_where_the_callers_measured_them(pair: tuple[Path, Path]) -> None:
    timeline = demo.Timeline.choose((demo.Call.load(pair[0]), demo.Call.load(pair[1])), turns=2)
    slow, fast = timeline.waits(0), timeline.waits(1)
    assert [ms for *_, ms in slow] == [3000.0, 2500.0, 2500.0]  # turn 1 is barged in on
    assert [round(until - ended, 3) for ended, until, _ in fast] == [1.0, 0.8, 0.8]
    assert fast[0][0] == pytest.approx(slow[0][0])  # the same line ends together
    # The soundtrack is the left call, then the right one.
    sound, tracks = demo.soundtrack(timeline), [timeline.track(side) for side in (0, 1)]
    mixes = [np.clip(track.astype(float).sum(axis=1) * 0.7, -32768, 32767) for track in tracks]
    half = len(sound) // 2
    assert np.array_equal(sound[: half - 2000], mixes[0][: half - 2000].astype(np.int16))
    assert np.array_equal(sound[half + 2000 :], mixes[1][half + 2000 :].astype(np.int16))


def test_a_frame_shows_the_live_wait_and_each_reply(pair: tuple[Path, Path]) -> None:
    pytest.importorskip("matplotlib")
    timeline = demo.Timeline.choose((demo.Call.load(pair[0]), demo.Call.load(pair[1])), turns=2)
    frames = demo.Frames(timeline)
    ended = timeline.waits(0)[0][0]
    image = frames.draw(ended + 2.0)  # the left still waits; the right replied after 1 s
    assert image.shape == (demo.HEIGHT, demo.WIDTH, 4) and image.dtype == np.uint8
    left, right = frames.panels
    assert (left.wait.get_text(), left.caption.get_text()) == ("2.0 s", "waiting for the agent")
    assert right.wait.get_text() == "1.00 s"
    assert left.bars[0].get_height() == pytest.approx(2.0) and left.bars[0].get_alpha() == 0.45
    assert right.bars[0].get_height() == 1.0 and right.bars[1].get_height() == 0.0
    assert left.tag.get_visible() and not right.tag.get_visible()  # the left one is heard first
    assert left.goodput[1].get_text() == "Background agents: 1,000 tokens/s, last 10 s"
    frames.draw(timeline.seconds - 0.1)
    assert right.tag.get_visible() and not left.tag.get_visible()
    assert left.wait.get_text() == "2.50 s" and len(left.bars) == 3


@pytest.mark.integration
def test_the_script_renders_a_video_and_a_gif(pair: tuple[Path, Path], tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv, which brings the script's dependencies")
    out, gif = tmp_path / "demo.mp4", tmp_path / "demo.gif"
    command = [uv, "run", "-q", str(DEMO), *map(str, pair), "--out", str(out), "--gif", str(gif)]
    done = subprocess.run(
        [*command, "--seconds", "12", "--fps", "10", "--gif-seconds", "3"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    assert f"{out}: " in done.stdout and "turns 0-0" in done.stdout
    assert out.read_bytes()[4:8] == b"ftyp" and gif.read_bytes()[:6] == b"GIF89a"
    assert gif.stat().st_size < 2_000_000
