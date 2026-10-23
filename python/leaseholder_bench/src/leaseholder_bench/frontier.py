"""E2's results: a point's summary over its calls, and the frontier across points (``lh-bench e2
report``).

:func:`summarize_point` reads one run directory of ``lh-bench e2 run`` (:mod:`.e2`): the callers'
summary, the per-request records of the AIPerf replay, the ``/metrics`` snapshots taken when the
calls started and ended, and the voice worker's journal of the calls. Every metric covers that
window. A metric that the gateway or the engine does not export is null, never zero, and a counter
that went down (its process restarted during the calls) is an error.

:func:`write_report` places the points on E2's frontier: background goodput on x, voice-to-voice
p95 on y, one frontier per arm. A point is on its arm's frontier when no other point of the arm
harvests at least as much background goodput at a V2V p95 at least as low, and one of the two
strictly better.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
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

TURNS_LOGGER = "leaseholder_voice.turns"
SESSIONS_LOGGER = "leaseholder_voice.events"
"""The voice worker's loggers (leaseholder_voice.events): a line per turn, and one per session as it
ends ("session ended")."""
WATERFALL = (
    "transcript_ms",
    "end_of_turn_ms",
    "llm_ttft_ms",
    "llm_first_sentence_ms",
    "turn_to_first_sentence_ms",
    "tts_first_audio_ms",
    "e2e_ms",
)
"""The stage times of the worker's turn lines, in the order of a turn (its ``STAGES``)."""
GPU_TIME = "%Y/%m/%d %H:%M:%S.%f"
"""nvidia-smi's timestamp, which campaign.sh has it write in UTC."""


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
    requests = gateway_requests(run_dir, start, end)
    voice = ("turns", "errors", "v2v_ms", "barge_in_to_silence_ms", "barge_ins")
    return {
        "window_s": round(end - start, 1),
        "voice": {key: bots[key] for key in voice},
        "worker": worker(run_dir / "voice-worker.log"),
        "background": background(records, start, end) if agents else None,
        "gateway": gateway_counts(*gateway, end - start, requests),
        "engine": engine_counts(*engine),
    }


def worker(journal: Path) -> dict[str, Any] | None:
    """The voice worker's own account of the calls, from its journal (voice-worker.log): each stage
    of the waterfall over the turns, and the lifecycle events its sessions sent, dropped and failed
    to deliver. None without a turn or session line in it; the events are None when no session sent
    any (``LH_EVENTS=off``, or an LLM that bypasses the gateway)."""
    if not journal.exists():
        return None
    turns, sessions = worker_lines(journal.read_text(errors="replace"))
    if not turns and not sessions:
        return None
    counted = [s for s in sessions if "events_sent" in s]
    events = {
        kind: sum(s[f"events_{kind}"] for s in counted) for kind in ("sent", "dropped", "failed")
    }
    return {
        "sessions": len(sessions),
        "turns": len(turns),
        "waterfall": {
            stage: summarize(t[stage] for t in turns if stage in t) for stage in WATERFALL
        },
        "events": events if counted else None,
    }


def worker_lines(journal: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The voice worker's turn lines and session-end lines among its journal's JSON lines."""
    turns, sessions = [], []
    for line in journal.splitlines():
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue  # not one of the worker's JSON log lines
        if not isinstance(record, dict):
            continue
        if record.get("name") == TURNS_LOGGER:
            turns.append(record)
        elif record.get("name") == SESSIONS_LOGGER and record.get("message") == "session ended":
            sessions.append(record)
    return turns, sessions


def gateway_request_lines(journal: str) -> list[str]:
    """The gateway's "request" lines among its journal's lines: one JSON object per request that
    ended, with its class, status, outcome and times (``log_format = "json"``, as on cyan)."""
    return [line for line in journal.splitlines() if _request(line) is not None]


def gateway_requests(run_dir: Path, start: float, end: float) -> list[dict[str, Any]] | None:
    """The requests the gateway logged as ending within ``start``..``end``, from the run's
    gateway.jsonl; None without one, or with one that holds no request (the calls make some)."""
    path = run_dir / "gateway.jsonl"
    if not path.exists():
        return None
    logged = [r for r in map(_request, path.read_text(errors="replace").splitlines()) if r]
    if not logged:
        return None
    return [record for ended, record in logged if start <= ended <= end]


def _request(line: str) -> tuple[float, dict[str, Any]] | None:
    """A request line of the gateway's, with when the request ended; None for any other line."""
    try:
        record = orjson.loads(line)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("message") != "request":
        return None
    try:
        return datetime.fromisoformat(record["timestamp"]).timestamp(), record
    except (KeyError, TypeError, ValueError):
        return None


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


def gateway_counts(
    before: str, after: str, seconds: float, requests: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The gateway's first byte by class, what its admission point shed, refused and resubmitted, and
    its own count of the background tokens served over the window of ``seconds``; every counter that
    moved is listed under ``counters``. The admission counts are null for a gateway without an
    admission point, the background rate for one without usage counters. ``requests`` are the
    window's lines of the gateway's request log, when the run kept it."""
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
        "realtime_first_byte_ms": first_byte_ms(before, after, "realtime", requests),
        "background_first_byte_ms": first_byte_ms(before, after, "background", requests),
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


def first_byte_ms(
    before: str, after: str, slo_class: str, requests: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """A class's time to first byte at the gateway over the window. From its request log when the run
    has one (``source`` "log"), which is exact; otherwise interpolated inside its histogram's buckets
    ("histogram"), whose bounds are 0.1, 0.25, 0.5 and 1 s in the range that matters for voice."""
    if requests is None:
        cell = histogram_ms(before, after, FIRST_BYTE, {"class": slo_class})
        return cell | {"source": "histogram"}
    return logged_first_byte(requests, slo_class)


def logged_first_byte(requests: Iterable[Mapping[str, Any]], slo_class: str) -> dict[str, Any]:
    """Count, mean and p50/p95/p99 of a class's first byte in ms, from the gateway's request lines:
    the requests it sent upstream (with an outcome) that had a first byte, as its histogram counts."""
    first_bytes = [
        r["first_byte_ms"]
        for r in requests
        if r.get("class") == slo_class and "outcome" in r and "first_byte_ms" in r
    ]
    cell = summarize(first_bytes)
    return {key: cell[key] for key in ("n", "mean", "p50", "p95", "p99") if key in cell} | {
        "source": "log"
    }


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
    "realtime_first_byte_source",
    "engine_preemptions",
    "shed",
    "refused",
    "resubmitted",
    "gateway_background_tokens_per_s",
)
"""frontier.csv's columns; goodput is in tokens and requests per second, latencies in ms. The
realtime first byte is the gateway's, from its request log ("log", exact) or its histogram
("histogram", interpolated). Refused adds up the gateway's admission point and the engine's queue
limits; the gateway's own count of the background tokens it served checks AIPerf's goodput."""
WORKER_COLUMNS = (
    *(f"{stage.removesuffix('_ms')}_{q}_ms" for stage in WATERFALL for q in ("p50", "p95")),
    "events_sent",
    "events_dropped",
    "events_failed",
)
GPU_COLUMNS = ("gpu_sm_clock_p50_mhz", "gpu_temperature_max_c")
"""Columns that only some points have, each group shown when one of them has it: the voice worker's
stage times and lifecycle events, and the GPU's median SM clock and highest temperature during the
calls."""


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
        rows.append(_row(run_dir, run, orjson.loads((run_dir / "summary.json").read_bytes())))
    mark_frontier(rows)
    rows.sort(key=lambda r: (_arm_order(r["arm"]), r["background_tokens_per_s"], r["point"]))
    return rows, incomplete


def _row(run_dir: Path, run: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    arm, voice = run["arm"], summary["voice"]
    background = summary["background"] or {}
    gateway, engine = summary["gateway"], summary["engine"]
    window = run.get("window")
    first_byte = gateway["realtime_first_byte_ms"]
    # A request log fetched into the run directory after its summary was written counts as well.
    if (
        window
        and (requests := gateway_requests(run_dir, window["start"], window["end"])) is not None
    ):
        first_byte = logged_first_byte(requests, "realtime")
    worker = summary.get("worker") or {}
    waterfall, events = worker.get("waterfall") or {}, worker.get("events") or {}
    row = {
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
        "realtime_first_byte_p95_ms": first_byte.get("p95"),
        # summaries from before the request log was kept took it from the histogram
        "realtime_first_byte_source": first_byte.get("source", "histogram"),
        "engine_preemptions": engine["preemptions"],
        "shed": gateway["shed"],
        "refused": _added(gateway["refused"], *(engine["refused"] or {}).values()),
        "resubmitted": gateway["resubmitted"],
        "gateway_background_tokens_per_s": gateway["background_tokens_per_s"],
    }
    row |= {
        f"{stage.removesuffix('_ms')}_{q}_ms": waterfall.get(stage, {}).get(q)
        for stage in WATERFALL
        for q in ("p50", "p95")
    }
    row |= {f"events_{kind}": events.get(kind) for kind in ("sent", "dropped", "failed")}
    return row | gpu(run_dir / "gpu.csv", window)


def gpu(samples: Path, window: Mapping[str, float] | None) -> dict[str, int | None]:
    """The GPU's median SM clock (MHz) and highest temperature (°C) during the calls, from the samples
    campaign.sh takes once a second on the GPU node (gpu.csv); None without them."""
    clocks: list[float] = []
    temperatures: list[float] = []
    if samples.exists() and window:
        for line in samples.read_text(errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            try:
                at = datetime.strptime(fields[0], GPU_TIME).replace(tzinfo=UTC).timestamp()
            except ValueError:
                continue  # a line cut short when the sampling stopped
            if not window["start"] <= at <= window["end"]:
                continue
            if (clock := _number(fields, 1)) is not None:
                clocks.append(clock)
            if (temperature := _number(fields, 3)) is not None:
                temperatures.append(temperature)
    return {
        "gpu_sm_clock_p50_mhz": round(statistics.median(clocks)) if clocks else None,
        "gpu_temperature_max_c": round(max(temperatures)) if temperatures else None,
    }


def _number(fields: Sequence[str], index: int) -> float | None:
    """A field of an nvidia-smi line without its unit ("2805 MHz"); None for "[N/A]" and the like."""
    try:
        return float(fields[index].split()[0])
    except (IndexError, ValueError):
        return None


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
        columns = [*COLUMNS, *(c for group in _measured(rows) for c in group)]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    written[1].write_text(markdown(rows))
    if plot(rows, out_dir / "frontier.svg"):
        written.append(out_dir / "frontier.svg")
    return written


def _measured(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    """The optional column groups that some point has."""
    groups = (WORKER_COLUMNS, GPU_COLUMNS)
    return [group for group in groups if any(r[c] is not None for r in rows for c in group)]


def markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """The frontier as a table per arm, ordered by background goodput, then the voice worker's stage
    times when the points have them."""
    measured = _measured(rows)
    gpu_columns = GPU_COLUMNS in measured
    intro = (
        "Background goodput (prompt and completion tokens per second of the agent requests that "
        "completed during the calls) against voice-to-voice latency in ms. • marks the points on "
        "their arm's frontier. The realtime first byte is the gateway's, from its request log; ~ "
        "marks one interpolated inside its histogram's buckets, for a point without the log. Refused "
        "counts the gateway's admission point and the engine's queue limits; — is a count the "
        "point's gateway or engine does not keep."
    )
    if gpu_columns:
        intro += " SM clock (MHz) and temperature (°C) are the GPU's during the calls."
    lines = ["# E2 frontier", "", intro]
    header = (
        "| Point | | Sessions | Background tokens/s | Requests/s | V2V p50 | V2V p95 | V2V p99 "
        "| Failed turns | Agent TTFT p50 / p99 | Realtime first byte p95 | Preemptions "
        "| Shed / refused / resubmitted |"
    )
    align = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    if gpu_columns:
        header += " SM clock p50 | Max temp |"
        align += "---:|---:|"
    arms = sorted({r["arm"] for r in rows}, key=_arm_order)
    for arm in arms:
        lines += ["", f"## Arm {arm}", "", header, align]
        for r in (r for r in rows if r["arm"] == arm):
            counts = " / ".join(_cell(r[k]) for k in ("shed", "refused", "resubmitted"))
            ttft = f"{_cell(r['agent_ttft_p50_ms'])} / {_cell(r['agent_ttft_p99_ms'])}"
            first_byte = _cell(r["realtime_first_byte_p95_ms"])
            if r["realtime_first_byte_source"] == "histogram" and first_byte != "—":
                first_byte = f"~{first_byte}"
            line = (
                f"| {r['point']}: {r['label']} | {'•' if r['frontier'] else ''} | {r['sessions']} "
                f"| {_cell(r['background_tokens_per_s'])} | {_cell(r['background_requests_per_s'], 2)} "
                f"| {_cell(r['v2v_p50_ms'])} | {_cell(r['v2v_p95_ms'])} | {_cell(r['v2v_p99_ms'])} "
                f"| {r['voice_errors']} of {r['voice_turns']} | {ttft} | {first_byte} "
                f"| {_cell(r['engine_preemptions'])} | {counts} |"
            )
            if gpu_columns:
                line += (
                    f" {_cell(r['gpu_sm_clock_p50_mhz'])} | {_cell(r['gpu_temperature_max_c'])} |"
                )
            lines.append(line)
    if WORKER_COLUMNS in measured:
        lines += _stages(rows)
    return "\n".join(lines) + "\n"


def _stages(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The voice worker's stage times of each point, p50 / p95."""
    lines = [
        "",
        "## Voice turn stages",
        "",
        "The voice worker's own timing of each turn, p50 / p95 in ms: from the end of the caller's "
        "speech (VAD) to the final transcript and to the end-of-turn decision; from the LLM request "
        "to its first token and to the first sentence handed to TTS; from the end-of-turn decision "
        "to that sentence; from that sentence to its first audio; and from the end of speech to the "
        "reply's first audio frame on the track. Events are those the worker sent the gateway, "
        "dropped from its queue and failed to deliver.",
        "",
        "| Point | Transcript | End of turn | LLM first token | LLM first sentence "
        "| Turn to first sentence | TTS first audio | End to end | Events sent / dropped / failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        stages = [
            " / ".join(_cell(r[f"{stage.removesuffix('_ms')}_{q}_ms"]) for q in ("p50", "p95"))
            for stage in WATERFALL
        ]
        events = " / ".join(_cell(r[f"events_{kind}"]) for kind in ("sent", "dropped", "failed"))
        lines.append(f"| {r['point']} | {' | '.join(stages)} | {events} |")
    return lines


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
