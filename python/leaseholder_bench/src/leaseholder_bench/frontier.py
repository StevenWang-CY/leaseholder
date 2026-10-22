"""E2's results: a point's summary over its calls, and the frontier across points (``lh-bench e2
report``).

:func:`summarize_point` reads one run directory of ``lh-bench e2 run`` (:mod:`.e2`): the callers'
summary, the per-request records of the AIPerf replay and the ``/metrics`` snapshots taken when the
calls started and ended. Every metric covers that window. A metric that the gateway or the engine
does not export is null, never zero, and a counter that went down (its process restarted during the
calls) is an error.

:func:`write_report` places the points on E2's frontier: background goodput on x, voice-to-voice
p95 on y, one frontier per arm. A point is on its arm's frontier when no other point of the arm
harvests at least as much background goodput at a V2V p95 at least as low, and one of the two
strictly better.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson

from leaseholder_bench.kv_metrics import Histogram, Labels, metric_samples, samples
from leaseholder_bench.load import summarize

FIRST_BYTE = "lh_gateway_first_byte_seconds"
OUTCOMES = "lh_gateway_request_outcomes_total"
"""By class and outcome; ``shed`` is a request the admission point gave up for a stricter class."""
REFUSED = "lh_gateway_admission_refused_total"
RESUBMITS = "lh_gateway_admission_resubmits_total"
DISCARDED = "lh_gateway_admission_discarded_tokens_total"
"""The prompt tokens of the attempts preempt-resubmit dropped: at most the prefill they discarded."""
USAGE = "lh_gateway_usage_tokens_total"
"""Tokens from response usage, by class and kind, counted as each response ends."""


def summarize_point(run_dir: Path, start: float, end: float, *, agents: bool) -> dict[str, Any]:
    """The E2 metrics of the window ``start``..``end`` (epoch seconds) from a run directory. Raises
    ``ValueError`` when the point measured nothing to place on the frontier."""
    bots = orjson.loads((run_dir / "bots" / "summary.json").read_bytes())
    if not bots["v2v_ms"].get("n"):
        raise ValueError(f"the calls measured no voice-to-voice latency ({bots['errors']} failed)")
    metrics = run_dir / "metrics"
    gateway = [(metrics / f"gateway-{when}.prom").read_text() for when in ("start", "end")]
    engine = [(metrics / f"engine-{when}.prom").read_text() for when in ("start", "end")]
    records = run_dir / "aiperf" / "profile_export.jsonl"
    voice = ("turns", "errors", "v2v_ms", "barge_in_to_silence_ms", "barge_ins")
    return {
        "window_s": round(end - start, 1),
        "voice": {key: bots[key] for key in voice},
        "background": background(records, start, end) if agents else None,
        "gateway": gateway_counts(*gateway, end - start),
        "engine": engine_counts(*engine),
    }


def background(records: Path, start: float, end: float) -> dict[str, Any]:
    """The agent requests AIPerf completed during the window: the background goodput, which counts
    failed and cancelled requests as lost, and their time to first token."""
    if not records.exists():
        raise ValueError(f"AIPerf wrote no per-request records ({records})")
    ended = []
    for line in records.read_bytes().splitlines():
        record = orjson.loads(line)
        meta = record["metadata"]
        if meta.get("benchmark_phase") != "profiling":
            continue
        if start <= meta["request_end_ns"] / 1e9 <= end:
            ended.append(record)
    ok = [r for r in ended if not r.get("error") and not r["metadata"].get("was_cancelled")]
    if not ok:
        raise ValueError(f"AIPerf completed no request during the calls ({len(ended)} failed)")
    usage = [r["metrics"] for r in ok if "usage_prompt_tokens" in r["metrics"]]
    prompt = sum(m["usage_prompt_tokens"]["value"] for m in usage)
    completion = sum(m.get("usage_completion_tokens", {}).get("value", 0) for m in usage)
    seconds = end - start
    ttft = [
        r["metrics"]["time_to_first_token"]["value"]
        for r in ok
        if "time_to_first_token" in r["metrics"]
    ]
    return {
        "requests": len(ok),
        "errors": len(ended) - len(ok),
        "without_usage": len(ok) - len(usage),
        "requests_per_s": round(len(ok) / seconds, 3),
        "prompt_tokens_per_s": round(prompt / seconds, 1),
        "completion_tokens_per_s": round(completion / seconds, 1),
        "tokens_per_s": round((prompt + completion) / seconds, 1),
        "ttft_ms": summarize(ttft),
    }


def gateway_counts(before: str, after: str, seconds: float) -> dict[str, Any]:
    """The gateway's realtime first byte, what its admission point shed, refused and resubmitted, and
    its own count of the background tokens served over the window of ``seconds``; every counter that
    moved is listed under ``counters``. The admission counts are null for a gateway without an
    admission point, the background rate for one without usage counters."""
    names = {name for name, _, _ in metric_samples(after) if name.startswith("lh_gateway_")}
    counters = {
        name: deltas(before, after, name, "gateway") or {}
        for name in sorted(names)
        if name.endswith("_total")
    }

    def total(name: str, match: Mapping[str, str]) -> int | None:
        if name not in counters:
            return None
        return round(sum(v for k, v in counters[name].items() if dict(k).items() >= match.items()))

    admission = REFUSED in counters  # its series exist from the gateway's start
    prompt = total(USAGE, {"class": "background", "kind": "prompt"})
    completion = total(USAGE, {"class": "background", "kind": "completion"})
    return {
        "realtime_first_byte_ms": histogram_ms(before, after, FIRST_BYTE, {"class": "realtime"}),
        "shed": total(OUTCOMES, {"outcome": "shed"}) if admission else None,
        "refused": total(REFUSED, {}),
        "resubmitted": total(RESUBMITS, {}),
        "discarded_tokens": total(DISCARDED, {}),
        "background_tokens_per_s": (
            None
            if prompt is None or completion is None
            else round((prompt + completion) / seconds, 1)
        ),
        "counters": {
            _series(name, labels): round(value)
            for name, series in counters.items()
            for labels, value in series.items()
            if value
        },
    }


def engine_counts(before: str, after: str) -> dict[str, Any]:
    """vLLM's preemptions, prompt tokens by source and refused requests over the window."""

    def by(label: str, name: str) -> dict[str, int] | None:
        series = deltas(before, after, name, "engine")
        if series is None:
            return None
        totals: dict[str, int] = {}
        for labels, value in series.items():
            key = dict(labels).get(label, "")
            totals[key] = totals.get(key, 0) + round(value)
        return totals

    preemptions = deltas(before, after, "vllm:num_preemptions_total", "engine")
    return {
        "preemptions": None if preemptions is None else round(sum(preemptions.values())),
        "prompt_tokens_by_source": by("source", "vllm:prompt_tokens_by_source_total"),
        "refused": by("reason", "vllm:admission_rejections_total"),
    }


def deltas(before: str, after: str, name: str, source: str) -> dict[Labels, float] | None:
    """How much each series of counter ``name`` grew; None when the exposition has no such counter.
    A counter that went down means its process restarted during the window, which voids it."""
    now = samples(after, name)
    if not now:
        return None
    then = samples(before, name)
    grown = {}
    for labels, value in now.items():
        grown[labels] = value - then.get(labels, 0.0)
        if grown[labels] < 0:
            raise ValueError(f"{name} went down during the calls: the {source} restarted")
    return grown


def histogram_ms(
    before: str, after: str, name: str, match: Mapping[str, str]
) -> dict[str, float | int]:
    """Count, mean and p50/p95/p99 in ms over the window; quantiles interpolate inside buckets."""
    hist = Histogram.parse(after, name, match) - Histogram.parse(before, name, match)
    if hist.total <= 0:
        return {"n": 0}
    pick = tuple(sorted(match.items()))
    seconds = samples(after, f"{name}_sum").get(pick, 0.0)
    seconds -= samples(before, f"{name}_sum").get(pick, 0.0)
    cell: dict[str, float | int] = {"n": round(hist.total)}
    cell["mean"] = round(seconds / hist.total * 1e3, 1)
    for q in (0.5, 0.95, 0.99):
        cell[f"p{round(q * 100)}"] = round((hist.quantile(q) or 0.0) * 1e3, 1)
    return cell


def _series(name: str, labels: Labels) -> str:
    return name + ("{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else "")


def render(summary: Mapping[str, Any]) -> str:
    """One line of a point's headline numbers."""
    voice = summary["voice"]
    v2v = "/".join(str(voice["v2v_ms"].get(p)) for p in ("p50", "p95", "p99"))
    line = f"{summary['name']}: V2V p50/p95/p99 {v2v} ms"
    line += f", {voice['errors']} of {voice['turns']} turns failed"
    if background := summary["background"]:
        ttft = "/".join(str(background["ttft_ms"].get(p)) for p in ("p50", "p99"))
        line += f"; background {background['requests_per_s']} requests/s"
        line += f", {background['tokens_per_s']:,.0f} tokens/s, agent TTFT p50/p99 {ttft} ms"
    return line


# The frontier

COLUMNS = (
    "arm",
    "point",
    "label",
    "frontier",
    "callers",
    "sessions",
    "preemptive_generation",
    "engine",
    "router",
    "background_tokens_per_s",
    "background_requests_per_s",
    "v2v_p50_ms",
    "v2v_p95_ms",
    "v2v_p99_ms",
    "voice_turns",
    "voice_errors",
    "barge_in_to_silence_p95_ms",
    "agent_ttft_p50_ms",
    "agent_ttft_p99_ms",
    "realtime_first_byte_p95_ms",
    "engine_preemptions",
    "shed",
    "refused",
    "resubmitted",
    "gateway_background_tokens_per_s",
)
"""frontier.csv's columns; goodput is in tokens and requests per second, latencies in ms. Refused
adds up the gateway's admission point and the engine's queue limits; the gateway's own count of the
background tokens it served checks AIPerf's goodput."""


def load_points(run_dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    """One row per complete point, with its arm's frontier marked, and the run directories whose
    point did not complete."""
    rows, incomplete = [], []
    for run_dir in run_dirs:
        run_file = run_dir / "run.json"
        run = orjson.loads(run_file.read_bytes()) if run_file.exists() else {}
        if run.get("status") != "complete":
            incomplete.append(run_dir)
            continue
        rows.append(_row(run, orjson.loads((run_dir / "summary.json").read_bytes())))
    mark_frontier(rows)
    rows.sort(key=lambda r: (_arm_order(r["arm"]), r["background_tokens_per_s"], r["point"]))
    return rows, incomplete


def _row(run: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    arm, voice = run["arm"], summary["voice"]
    background = summary["background"] or {}
    gateway, engine = summary["gateway"], summary["engine"]
    return {
        "arm": str(arm["arm"]),
        "point": run["name"],
        "label": arm["label"],
        "frontier": False,
        "callers": arm["voice"]["callers"],
        "sessions": arm["agents"]["sessions"],
        "preemptive_generation": arm["voice"]["preemptive_generation"],
        "engine": " ".join([arm["engine"]["variant"], *arm["engine"]["flags"]]),
        "router": arm["router"] or "",
        "background_tokens_per_s": background.get("tokens_per_s", 0.0),
        "background_requests_per_s": background.get("requests_per_s", 0.0),
        "v2v_p50_ms": voice["v2v_ms"].get("p50"),
        "v2v_p95_ms": voice["v2v_ms"].get("p95"),
        "v2v_p99_ms": voice["v2v_ms"].get("p99"),
        "voice_turns": voice["turns"],
        "voice_errors": voice["errors"],
        "barge_in_to_silence_p95_ms": voice["barge_in_to_silence_ms"].get("p95"),
        "agent_ttft_p50_ms": background.get("ttft_ms", {}).get("p50"),
        "agent_ttft_p99_ms": background.get("ttft_ms", {}).get("p99"),
        "realtime_first_byte_p95_ms": gateway["realtime_first_byte_ms"].get("p95"),
        "engine_preemptions": engine["preemptions"],
        "shed": gateway["shed"],
        "refused": _added(gateway["refused"], *(engine["refused"] or {}).values()),
        "resubmitted": gateway["resubmitted"],
        "gateway_background_tokens_per_s": gateway["background_tokens_per_s"],
    }


def _added(*counts: int | None) -> int | None:
    """The sum of the counts that were measured; None when none was."""
    measured = [c for c in counts if c is not None]
    return sum(measured) if measured else None


def mark_frontier(rows: Sequence[dict[str, Any]]) -> None:
    """Mark the points that no other point of their arm dominates."""

    def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        ax, ay = a["background_tokens_per_s"], a["v2v_p95_ms"]
        bx, by = b["background_tokens_per_s"], b["v2v_p95_ms"]
        return ax >= bx and ay <= by and (ax > bx or ay < by)

    placed = [r for r in rows if r["v2v_p95_ms"] is not None]
    for row in placed:
        rivals = (r for r in placed if r is not row and r["arm"] == row["arm"])
        row["frontier"] = not any(dominates(rival, row) for rival in rivals)


def _arm_order(arm: str) -> tuple[int, str]:
    """Arms in E2's order: 2 before 10, 7a before 7b."""
    number = re.match(r"\d+", arm)
    return (int(number[0]) if number else 1000, arm)


def write_report(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """frontier.csv, frontier.md and, when matplotlib is installed, frontier.svg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "frontier.csv", out_dir / "frontier.md"]
    with written[0].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    written[1].write_text(markdown(rows))
    if plot(rows, out_dir / "frontier.svg"):
        written.append(out_dir / "frontier.svg")
    return written


def markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """The frontier as a table per arm, ordered by background goodput."""
    lines = [
        "# E2 frontier",
        "",
        "Background goodput (prompt and completion tokens per second of the agent requests that "
        "completed during the calls) against voice-to-voice latency in ms. • marks the points on "
        "their arm's frontier. Refused counts the gateway's admission point and the engine's queue "
        "limits; — is a count the point's gateway or engine does not keep.",
    ]
    header = (
        "| Point | | Sessions | Background tokens/s | Requests/s | V2V p50 | V2V p95 | V2V p99 "
        "| Failed turns | Agent TTFT p50 / p99 | Preemptions | Shed / refused / resubmitted |"
    )
    arms = sorted({r["arm"] for r in rows}, key=_arm_order)
    for arm in arms:
        lines += [
            "",
            f"## Arm {arm}",
            "",
            header,
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in (r for r in rows if r["arm"] == arm):
            counts = " / ".join(_cell(r[k]) for k in ("shed", "refused", "resubmitted"))
            ttft = f"{_cell(r['agent_ttft_p50_ms'])} / {_cell(r['agent_ttft_p99_ms'])}"
            lines.append(
                f"| {r['point']}: {r['label']} | {'•' if r['frontier'] else ''} | {r['sessions']} "
                f"| {_cell(r['background_tokens_per_s'])} | {_cell(r['background_requests_per_s'], 2)} "
                f"| {_cell(r['v2v_p50_ms'])} | {_cell(r['v2v_p95_ms'])} | {_cell(r['v2v_p99_ms'])} "
                f"| {r['voice_errors']} of {r['voice_turns']} | {ttft} "
                f"| {_cell(r['engine_preemptions'])} | {counts} |"
            )
    return "\n".join(lines) + "\n"


def _cell(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


# Light chart surface, ink and the categorical order of the docs' figures.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_MARKERS = ("o", "s", "^", "D")
_SURFACE, _INK, _MUTED, _GRID, _AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def plot(rows: Sequence[Mapping[str, Any]], path: Path) -> bool:
    """frontier.svg: each arm's frontier as a line, its dominated points as open markers. Arms
    past the eighth take the colors again with another marker. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    arms = sorted({r["arm"] for r in rows if r["v2v_p95_ms"] is not None}, key=_arm_order)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    for i, arm in enumerate(arms):
        color, marker = _SERIES[i % len(_SERIES)], _MARKERS[i // len(_SERIES) % len(_MARKERS)]
        points = [r for r in rows if r["arm"] == arm and r["v2v_p95_ms"] is not None]
        front = sorted(
            (r for r in points if r["frontier"]), key=lambda r: r["background_tokens_per_s"]
        )
        others = [r for r in points if not r["frontier"]]
        ax.plot(
            [r["background_tokens_per_s"] for r in front],
            [r["v2v_p95_ms"] for r in front],
            color=color,
            linewidth=1.5,
            marker=marker,
            markersize=6,
            markeredgecolor=_SURFACE,
            label=f"arm {arm}",
        )
        ax.scatter(
            [r["background_tokens_per_s"] for r in others],
            [r["v2v_p95_ms"] for r in others],
            s=36,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
        )
    ax.set_xlabel("Background goodput (tokens/s)", color=_INK)
    ax.set_ylabel("Voice V2V p95 (ms)", color=_INK)
    ax.set_title("E2 frontier: open markers are dominated points", color=_INK, loc="left")
    ax.grid(color=_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(_AXIS)
    if len(arms) > 1:
        legend = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        for text in legend.get_texts():
            text.set_color(_INK)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=_SURFACE)
    plt.close(fig)
    return True
