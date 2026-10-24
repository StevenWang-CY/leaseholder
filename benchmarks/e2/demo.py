# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.5.4",
#     "matplotlib==3.11.2",
#     "imageio-ffmpeg==0.6.0",
# ]
# ///
"""E2's demo video: two calls that ``lh-bench e2 run --save-audio`` saved, side by side.

    uv run benchmarks/e2/demo.py LEFT RIGHT --out demo.mp4 --gif demo.gif

LEFT and RIGHT are the run directories of two points, for example the frozen box and arm 9 in the
discriminating scenario. Each side shows one of its point's calls (``--caller``, by default the
first): its scrolling waveform, the caller above and the agent below; the current turn's wait,
from the end of the caller's speech to the agent's first audio, counting live; each reply's V2V
as the callers measured it, in a bar that grows while the caller waits; and the point's
background goodput, over its calls (summary.json) and over the last 10 s of the wall clock
(AIPerf's profile_export.jsonl).

Both calls show the same turns, from ``--first-turn`` on: ``--turns`` of them, by default as many
as fit in ``--seconds``. Turn k of both starts at the same moment, so the same line is answered
side by side, and the call that is ahead waits in silence before its next line: only a caller's
pause between turns is stretched, never a measured wait. The soundtrack is one call at a time, the
left one for the first half and the right one for the second, and the side playing is
highlighted. ``--gif`` also writes a short, small GIF of the start, for a README.

The video is 1280x720 H.264 with AAC sound, encoded by the ffmpeg that imageio-ffmpeg bundles.
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WIDTH, HEIGHT = 1280, 720
LEAD_S = 0.5
"""How long before its line a turn starts on the video."""
WINDOW_S = 8.0
"""The span of the scrolling waveforms; the newest audio is at their right edge."""
GOODPUT_WINDOW_S = 10.0
"""The trailing window of the background goodput's live rate."""
FADE_S = 0.05
"""The soundtrack's crossfade from the left call to the right one."""

# The docs' chart palette, as e2/figures.py has it: the caller in its first slot, the agent in its
# second, the background agents in its third.
SURFACE, PLANE, INK, SECONDARY, MUTED, GRID = (
    "#fcfcfb", "#f0efec", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)  # fmt: skip
CALLER, AGENT, BACKGROUND = "#2a78d6", "#eb6834", "#1baf7a"


@dataclass
class Call:
    """A call that ``lh-voice bots --save-audio`` saved, and the point it was measured at."""

    label: str
    pcm: np.ndarray
    """(samples, 2) of int16: what the caller sent, and the agent's audio as received."""
    rate: int
    wall: float
    """The WAV's start on the wall clock."""
    turns: list[dict[str, Any]]
    """The caller's turns as turns.jsonl has them, with ``at``: when their events happened, in
    seconds into the WAV."""
    goodput: float | None
    """The point's background tokens/s over its calls."""
    finished: np.ndarray
    """(requests, 2): when each agent request that AIPerf completed ended, on the wall clock, and
    its tokens."""

    @classmethod
    def load(cls, run_dir: Path, caller: int = 0, label: str | None = None) -> Call:
        saved = json.loads((run_dir / "bots" / f"call-{caller}.json").read_text())
        with wave.open(str(run_dir / "bots" / f"call-{caller}.wav")) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).reshape(-1, 2)
        if label is None:
            label = json.loads((run_dir / "run.json").read_text())["arm"]["label"]
        background = json.loads((run_dir / "summary.json").read_text())["background"] or {}
        return cls(
            label,
            pcm,
            saved["rate"],
            saved["start"]["wall"],
            saved["turns"],
            background.get("tokens_per_s"),
            finished(run_dir / "aiperf" / "profile_export.jsonl"),
        )


def finished(records: Path) -> np.ndarray:
    """The agent requests AIPerf completed while profiling, as the point's goodput counts them
    (failed and cancelled ones are lost): when each ended and its prompt and completion tokens, in
    the order they ended; none without the records."""
    rows = []
    if records.exists():
        for line in records.read_text().splitlines():
            record = json.loads(line)
            meta, metrics = record["metadata"], record["metrics"]
            lost = record.get("error") or meta.get("was_cancelled")
            if meta.get("benchmark_phase") != "profiling" or lost:
                continue
            kinds = ("usage_prompt_tokens", "usage_completion_tokens")
            tokens = sum(metrics.get(kind, {}).get("value", 0) for kind in kinds)
            rows.append((meta["request_end_ns"] / 1e9, tokens))
    return np.array(sorted(rows), dtype=float).reshape(-1, 2)


@dataclass
class Timeline:
    """Where the turns both calls show are on the video: turn k of both starts at ``offsets[k]`` and
    lasts as long as the longer of the two; the call that is ahead waits in silence."""

    calls: tuple[Call, Call]
    first: int
    """The first turn shown, by its number in the calls."""
    spans: tuple[list[tuple[float, float]], list[tuple[float, float]]]
    """Each shown turn's stretch of each call, in seconds into its WAV: from LEAD_S before its line
    to LEAD_S before the next one's, or to the end of the call."""
    offsets: list[float]
    """When each shown turn starts on the video, then when the video ends."""

    @classmethod
    def choose(
        cls, calls: tuple[Call, Call], first: int = 0, turns: int | None = None, seconds: float = 90
    ) -> Timeline:
        """``turns`` turns of both calls from turn ``first``, by default as many as fit in
        ``seconds``."""
        if calls[0].rate != calls[1].rate:
            raise ValueError("the two calls are saved at different rates")
        spans = (_spans(calls[0], first), _spans(calls[1], first))
        lengths = [max(b - a for a, b in pair) for pair in zip(*spans, strict=False)]
        offsets = [0.0, *np.cumsum(lengths).tolist()]
        if turns is None:
            turns = max(1, bisect.bisect_right(offsets, seconds) - 1)
        if not 1 <= turns <= len(lengths):
            raise ValueError(f"both calls have {len(lengths)} turns from turn {first}, not {turns}")
        return cls(calls, first, (spans[0][:turns], spans[1][:turns]), offsets[: turns + 1])

    @property
    def seconds(self) -> float:
        return self.offsets[-1]

    def call_time(self, side: int, t: float) -> float:
        """Where in its call a side is at video time t, held at its turn's end while it waits."""
        k = min(max(bisect.bisect_right(self.offsets, t) - 1, 0), len(self.offsets) - 2)
        start, end = self.spans[side][k]
        return min(start + t - self.offsets[k], end)

    def video_time(self, side: int, k: int, t: float) -> float:
        """A moment of a side's shown turn k, given in seconds into its WAV, on the video."""
        return self.offsets[k] + t - self.spans[side][k][0]

    def track(self, side: int) -> np.ndarray:
        """A side's audio on the video, both channels, silent where it waits."""
        call = self.calls[side]
        out = np.zeros((round(self.seconds * call.rate), 2), dtype=np.int16)
        for (start, end), offset in zip(self.spans[side], self.offsets, strict=False):
            a, b, at = (round(x * call.rate) for x in (start, end, offset))
            out[at : at + b - a] = call.pcm[a:b][: len(out) - at]
        return out

    def waits(self, side: int) -> list[tuple[float, float, float | None]]:
        """Each wait of a side's shown turns on the video, in order: when the caller's speech ended,
        when the agent's reply came, and the V2V the callers measured in ms, or for a reply that
        never came the turn's end and None."""
        found = []
        shown = self.calls[side].turns[self.first : self.first + len(self.spans[side])]
        for k, turn in enumerate(shown):
            at = turn["at"]
            pairs = [("speech_end", "reply", "v2v_ms")]
            if turn.get("barge_in"):
                pairs.append(("barge_in_speech_end", "barge_in_reply", "v2v_after_barge_in_ms"))
            for ended, reply, measured in pairs:
                if ended not in at:
                    continue
                until = at.get(reply, self.spans[side][k][1])
                ms = turn.get(measured) if reply in at else None
                found.append(
                    (self.video_time(side, k, at[ended]), self.video_time(side, k, until), ms)
                )
        return found


def _spans(call: Call, first: int) -> list[tuple[float, float]]:
    starts = [max(0.0, turn["at"]["start"] - LEAD_S) for turn in call.turns[first:]]
    return list(zip(starts, [*starts[1:], len(call.pcm) / call.rate], strict=True))


def soundtrack(timeline: Timeline) -> np.ndarray:
    """One call at a time, the caller and the agent mixed: the left call for the first half of the
    video, the right one for the second."""
    mixes = [timeline.track(side).astype(np.float64).sum(axis=1) * 0.7 for side in (0, 1)]
    half, ramp = len(mixes[0]) / 2, FADE_S * timeline.calls[0].rate
    fade = np.clip((np.arange(len(mixes[0])) - half) / ramp + 0.5, 0.0, 1.0)
    return np.clip(mixes[0] * (1 - fade) + mixes[1] * fade, -32768, 32767).astype(np.int16)


def envelope(signal: np.ndarray, rate: int) -> np.ndarray:
    """A channel's peak in each 10 ms, as a share of full scale."""
    bins = rate // 100
    peaks = np.abs(signal[: len(signal) // bins * bins].astype(np.int32)).reshape(-1, bins)
    return peaks.max(axis=1) / 32768


def goodput(timeline: Timeline, side: int, times: np.ndarray) -> np.ndarray:
    """A side's background goodput at each video time: the tokens of the agent requests that
    completed in the GOODPUT_WINDOW_S before that moment of its call, per second."""
    call = timeline.calls[side]
    walls = call.wall + np.array([timeline.call_time(side, t) for t in times])
    ends, tokens = call.finished[:, 0], np.concatenate([[0.0], np.cumsum(call.finished[:, 1])])
    done = tokens[np.searchsorted(ends, walls, side="right")]
    before = tokens[np.searchsorted(ends, walls - GOODPUT_WINDOW_S, side="right")]
    return (done - before) / GOODPUT_WINDOW_S


@dataclass
class Panel:
    """One side's artists that change from frame to frame."""

    box: Any
    tag: Any
    wait: Any
    caption: Any
    bars: Any
    waveform: Any
    goodput: Any | None
    """The goodput's axes and its label; None for a point without AIPerf's records."""
    rates: np.ndarray


class Frames:
    """The video's frames: both calls' panels at a moment of the video, as RGBA pixels."""

    STEP_S = 0.1
    """The time step of the goodput's live rate."""

    def __init__(self, timeline: Timeline) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.timeline = timeline
        self.figure = Figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor=SURFACE)
        self.canvas = FigureCanvasAgg(self.figure)
        self.waits = [timeline.waits(side) for side in (0, 1)]
        self.times = np.arange(0.0, timeline.seconds, self.STEP_S)
        rates = [goodput(timeline, side, self.times) for side in (0, 1)]
        waited = [until - ended for side in self.waits for ended, until, _ in side]
        scale = (max([1.0, *waited]) * 1.15, max(float(r.max(initial=0.0)) for r in rates) * 1.15)
        self.panels = [
            self._panel(side, x, rates[side], scale) for side, x in ((0, 0.015), (1, 0.51))
        ]
        self.figure.text(
            0.5,
            0.04,
            "V2V: from the end of the caller's speech to the agent's first audio, as the callers "
            "measured it.\nEach pair of turns starts together, and the call that is ahead waits in "
            "silence. Sound: the left call, then the right one.",
            color=MUTED,
            fontsize=9,
            ha="center",
            va="center",
            linespacing=1.6,
        )

    def _panel(self, side: int, x: float, rates: np.ndarray, scale: tuple[float, float]) -> Panel:
        from matplotlib.patches import FancyBboxPatch

        figure, call = self.figure, self.timeline.calls[side]
        box = FancyBboxPatch(
            (x, 0.09), 0.475, 0.885, boxstyle="round,pad=0,rounding_size=0.01",
            transform=figure.transFigure, facecolor=SURFACE, edgecolor=GRID, zorder=-1,
        )  # fmt: skip
        figure.patches.append(box)
        figure.text(x + 0.012, 0.935, call.label, color=INK, fontsize=14, va="center")
        tag = figure.text(
            x + 0.463, 0.935, "SOUND", color=SURFACE, fontsize=9, ha="right", va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": INK, "edgecolor": INK},
        )  # fmt: skip
        wait = figure.text(x + 0.012, 0.815, "", color=INK, fontsize=36, va="center")
        caption = figure.text(x + 0.014, 0.74, "", color=SECONDARY, fontsize=10)
        return Panel(
            box,
            tag,
            wait,
            caption,
            self._bars(x, side, scale[0]),
            self._waveform(x, side),
            self._goodput(x, call, rates, scale[1]),
            rates,
        )

    def _bars(self, x: float, side: int, top: float) -> Any:
        ax = self.figure.add_axes((x + 0.25, 0.715, 0.21, 0.185), facecolor="none")
        count = len(self.waits[side])
        bars = ax.bar(range(count), [0.0] * count, width=0.7, color=AGENT)
        ax.set_xlim(-0.6, max(count, 8) - 0.4)
        ax.set_ylim(0, top)
        ax.set_xticks([])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.yaxis.grid(color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_title("V2V of each reply (s)", color=SECONDARY, fontsize=9, loc="left")
        return bars

    def _waveform(self, x: float, side: int) -> Any:
        ax = self.figure.add_axes((x + 0.06, 0.37, 0.405, 0.32), facecolor="none")
        track, rate = self.timeline.track(side), self.timeline.calls[side].rate
        times = np.arange(len(track) // (rate // 100)) / 100
        ax.fill_between(times, 0, envelope(track[:, 0], rate), color=CALLER, linewidth=0)
        ax.fill_between(times, 0, -envelope(track[:, 1], rate), color=AGENT, linewidth=0)
        for ended, until, _ in self.waits[side]:
            ax.axvspan(ended, until, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax.axhline(0, color=GRID, linewidth=0.75)
        ax.set_ylim(-1, 1)
        ax.set_yticks([0.5, -0.5], ["caller", "agent"])
        ax.tick_params(axis="y", colors=SECONDARY, labelsize=9, length=0)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def _goodput(self, x: float, call: Call, rates: np.ndarray, top: float) -> Any | None:
        from matplotlib.ticker import MaxNLocator

        figure = self.figure
        if call.goodput is not None:
            average = f"{call.goodput:,.0f} tokens/s over the calls"
            figure.text(x + 0.463, 0.3, average, color=MUTED, fontsize=9, ha="right")
        if not len(call.finished):
            alone = "Background agents" if call.goodput else "No background agents"
            figure.text(x + 0.014, 0.3, alone, color=SECONDARY, fontsize=10)
            return None
        ax = figure.add_axes((x + 0.06, 0.125, 0.405, 0.155), facecolor="none")
        ax.fill_between(self.times, 0, rates, color=BACKGROUND, alpha=0.1, linewidth=0)
        ax.plot(self.times, rates, color=BACKGROUND, linewidth=2)
        ax.set_ylim(0, max(top, 1.0))
        ax.set_xticks([])
        ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
        ax.yaxis.grid(color=GRID, linewidth=0.75)
        ax.yaxis.set_major_formatter("{x:,.0f}")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax, figure.text(x + 0.014, 0.3, "", color=SECONDARY, fontsize=10)

    def draw(self, t: float) -> np.ndarray:
        """The frame at video time t."""
        playing = 0 if t < self.timeline.seconds / 2 else 1
        for side, panel in enumerate(self.panels):
            panel.box.set_facecolor(PLANE if side == playing else SURFACE)
            panel.box.set_edgecolor(SECONDARY if side == playing else GRID)
            panel.tag.set_visible(side == playing)
            panel.waveform.set_xlim(t - WINDOW_S, t)
            wait, caption = "—", ""
            for bar, (ended, until, ms) in zip(panel.bars, self.waits[side], strict=True):
                if ended <= t < until:
                    wait, caption = f"{t - ended:.1f} s", "waiting for the agent"
                    bar.set_height(t - ended)
                    bar.set_alpha(0.45)
                elif t >= until:
                    wait = "no reply" if ms is None else f"{ms / 1000:.2f} s"
                    caption = "V2V, as the caller measured it"
                    bar.set_height(until - ended if ms is None else ms / 1000)
                    bar.set_alpha(1.0)
                else:
                    bar.set_height(0.0)
            panel.wait.set_text(wait)
            panel.caption.set_text(caption)
            if panel.goodput is not None:
                ax, label = panel.goodput
                ax.set_xlim(t - WINDOW_S, t)
                rate = panel.rates[min(int(t / self.STEP_S), len(panel.rates) - 1)]
                label.set_text(f"Background agents: {rate:,.0f} tokens/s, last 10 s")
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())


def render(
    timeline: Timeline, out: Path, *, fps: float = 30, gif: Path | None = None, gif_s: float = 10
) -> int:
    """Write the video, and the GIF of its first ``gif_s`` seconds; returns the video's frames."""
    import imageio_ffmpeg

    frames = Frames(timeline)
    count = round(timeline.seconds * fps)
    with tempfile.TemporaryDirectory() as scratch:
        sound = Path(scratch) / "soundtrack.wav"
        with wave.open(str(sound), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(timeline.calls[0].rate)
            wav.writeframes(soundtrack(timeline).tobytes())
        writer = imageio_ffmpeg.write_frames(
            str(out),
            (WIDTH, HEIGHT),
            pix_fmt_in="rgba",
            fps=fps,
            quality=7,
            ffmpeg_log_level="error",
            audio_path=str(sound),
            audio_codec="aac",
        )
        writer.send(None)
        for index in range(count):
            writer.send(frames.draw(index / fps))
        writer.close()
    if gif is not None:
        palette = (
            "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:"
            "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        )
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-t", f"{gif_s:g}"]
        subprocess.run(
            [*command, "-i", str(out), "-vf", palette, "-loop", "0", str(gif)], check=True
        )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("left", type=Path, help="run directory of the left point")
    parser.add_argument("right", type=Path, help="run directory of the right point")
    parser.add_argument("--out", type=Path, required=True, help="the MP4 to write")
    parser.add_argument("--gif", type=Path, help="also write a GIF of the video's start")
    parser.add_argument("--gif-seconds", type=float, default=10.0, help="its length (default 10)")
    parser.add_argument("--caller", type=int, default=0, help="each point's call (default 0)")
    parser.add_argument("--labels", nargs=2, metavar=("LEFT", "RIGHT"), help="instead of the arms'")
    parser.add_argument("--first-turn", type=int, default=0, help="the first turn shown")
    parser.add_argument("--turns", type=int, help="the turns shown (default: as many as fit)")
    parser.add_argument("--seconds", type=float, default=90.0, help="the video's most (default 90)")
    parser.add_argument("--fps", type=float, default=30.0, help="frames a second (default 30)")
    args = parser.parse_args(argv)
    labels = args.labels or (None, None)
    try:
        left, right = (
            Call.load(run_dir, args.caller, label)
            for run_dir, label in zip((args.left, args.right), labels, strict=True)
        )
        timeline = Timeline.choose((left, right), args.first_turn, args.turns, args.seconds)
    except (OSError, KeyError, ValueError) as exc:
        print(f"demo.py: {exc}", file=sys.stderr)
        return 1
    count = render(timeline, args.out, fps=args.fps, gif=args.gif, gif_s=args.gif_seconds)
    last = args.first_turn + len(timeline.spans[0]) - 1
    print(f"{args.out}: {count} frames, {timeline.seconds:.1f} s, turns {args.first_turn}-{last}")
    if args.gif is not None:
        print(f"{args.gif}: its first {min(args.gif_seconds, timeline.seconds):g} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
