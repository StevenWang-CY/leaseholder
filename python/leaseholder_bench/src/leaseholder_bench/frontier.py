"""E2's results: a point's summary over its calls, and the frontier across points (``lh-bench e2
report``).

:func:`summarize_point` reads one run directory of ``lh-bench e2 run`` (:mod:`.e2`): the callers'
summary, the per-request records of the AIPerf replay, the ``/metrics`` snapshots taken when the
calls started and ended, and the voice worker's journal of the calls. Every metric covers that
window. A metric that the gateway or the engine does not export is null, never zero, and a counter
that went down (its process restarted during the calls) is an error. A point on several GPU nodes
reports its totals as a point on one does, its engines' counters added up, and each node's share.

:func:`write_report` places the points on E2's frontier: background goodput on x, voice-to-voice
p95 on y, one frontier per arm and workload (the agents' corpus and the number of callers, since
points of different workloads are never compared). A point is on its arm's frontier when no other
point of the arm and workload harvests at least as much background goodput at a V2V p95 at least as
low, and one of the two strictly better. A point on several GPU nodes is placed by its totals.
Points measured from the same arm file in several campaigns are repeats of one configuration, which
the frontier places at their medians; the tables keep every run. The stage waterfall splits each
configuration's voice turn into the stages that add up to its mean V2V.

:func:`export_point` writes a point's compact record (``lh-bench e2 export``), the few small files
from which the report gives the same tables as from the whole run directory.
"""

from __future__ import annotations

import csv
import re
import shutil
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
PLACEMENT_ROUTE = "/v1/sessions"
"""The gateway's placement route, whose request lines name a class but no inference request."""

TURNS_LOGGER = "leaseholder_voice.turns"
SESSIONS_LOGGER = "leaseholder_voice.events"
"""The voice worker's loggers (leaseholder_voice.events): a line per turn, and one per session as it
ends ("session ended")."""
PLACEMENT_LOGGER = "leaseholder_voice.placement"
"""The voice worker's logger of each call's GPU node on several ("session placed", with its room)."""
CALLER_ROOM = re.compile(r"lh-bot-(?P<run_id>[0-9a-f]+)-(?P<caller>\d+)")
"""How ``lh-voice bots`` names each caller's room: after its run (bots/run.json) and the caller."""
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


def summarize_point(
    run_dir: Path, start: float, end: float, *, agents: bool, nodes: Sequence[str] = ()
) -> dict[str, Any]:
    """The E2 metrics of the window ``start``..``end`` (epoch seconds) from a run directory. Raises
    ``ValueError`` when the point measured nothing to place on the frontier. A point on several GPU
    ``nodes`` adds up their engines' counters and reports each node's share under ``nodes``."""
    bots = orjson.loads((run_dir / "bots" / "summary.json").read_bytes())
    if not bots["v2v_ms"].get("n"):
        raise ValueError(f"the calls measured no voice-to-voice latency ({bots['errors']} failed)")
    metrics = run_dir / "metrics"
    records = run_dir / "aiperf" / "profile_export.jsonl"
    requests = gateway_requests(run_dir, start, end)
    engines = {node: engine_counts(*_snapshots(metrics, f"engine-{node}")) for node in nodes}
    voice = ("turns", "errors", "v2v_ms", "barge_in_to_silence_ms", "barge_ins")
    summary = {
        "window_s": round(end - start, 1),
        "voice": {key: bots[key] for key in voice},
        "worker": worker(run_dir / "voice-worker.log"),
        "background": background(records, start, end) if agents else None,
        "gateway": gateway_counts(*_snapshots(metrics, "gateway"), end - start, requests),
        "engine": (
            added_engine_counts(engines.values())
            if nodes
            else engine_counts(*_snapshots(metrics, "engine"))
        ),
    }
    if nodes:
        summary["nodes"] = node_shares(run_dir, end - start, requests, engines)
    return summary


def _snapshots(metrics: Path, name: str) -> tuple[str, str]:
    """The ``/metrics`` of ``name`` (``gateway``, ``engine``, ``engine-NODE``) as the calls started and
    ended."""
    start, end = ((metrics / f"{name}-{when}.prom").read_text() for when in ("start", "end"))
    return start, end


def node_shares(
    run_dir: Path,
    seconds: float,
    requests: Sequence[Mapping[str, Any]] | None,
    engines: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each GPU node's share of a point on several: the calls the voice worker placed on it (None when
    its journal places none), the agent requests and the first bytes of the realtime requests the
    gateway sent it (None without its request log), and its engine's counters."""
    placed = call_nodes(run_dir)
    turns = list(_json_lines((run_dir / "bots" / "turns.jsonl").read_text())) if placed else []
    shares = {}
    for node, engine in engines.items():
        sent = None if requests is None else [r for r in requests if r.get("node") == node]
        callers = {caller for caller, on in placed.items() if on == node}
        shares[node] = {
            "voice": (
                call_turns([t for t in turns if t.get("caller") in callers], len(callers))
                if placed
                else None
            ),
            "background": None if sent is None else logged_background(sent, seconds),
            "realtime_first_byte_ms": None if sent is None else logged_first_byte(sent, "realtime"),
            "engine": engine,
        }
    return shares


def call_nodes(run_dir: Path) -> dict[int, str]:
    """Each call's GPU node, by caller: the voice worker's journal names the room of each call it placed,
    and the callers named their rooms after their run."""
    journal, bots = run_dir / "voice-worker.log", run_dir / "bots" / "run.json"
    if not journal.exists() or not bots.exists():
        return {}
    run_id = orjson.loads(bots.read_bytes())["run_id"]
    nodes = {}
    for record in _json_lines(journal.read_text(errors="replace")):
        if record.get("name") != PLACEMENT_LOGGER or record.get("message") != "session placed":
            continue
        room = CALLER_ROOM.fullmatch(str(record.get("room")))
        if room and room["run_id"] == run_id:
            nodes[int(room["caller"])] = record["node"]
    return nodes


def call_turns(turns: Sequence[Mapping[str, Any]], calls: int) -> dict[str, Any]:
    """Some of the calls' numbers from their callers' turns (bots/turns.jsonl), counted as ``lh-voice
    bots`` sums up all of them: voice-to-voice over every reply, the reply to an interruption too."""
    ok = [t for t in turns if "error" not in t]
    v2v = [t["v2v_ms"] for t in ok] + [t["v2v_after_barge_in_ms"] for t in ok if t["barge_in"]]
    silence = [
        t["barge_in_to_silence_ms"] for t in ok if t.get("barge_in_to_silence_ms") is not None
    ]
    return {
        "calls": calls,
        "turns": len(turns),
        "errors": len(turns) - len(ok),
        "v2v_ms": summarize(v2v),
        "barge_in_to_silence_ms": summarize(silence),
        "barge_ins": sum(1 for t in ok if t["barge_in"]),
    }


def logged_background(requests: Iterable[Mapping[str, Any]], seconds: float) -> dict[str, Any]:
    """The agent requests among the gateway's request lines, counted as :func:`background` counts
    AIPerf's records: the goodput of those served (completed with status 200: the gateway completes a
    response it relays in full whatever its status), the others as lost, and their first byte."""
    ended = [
        r for r in requests if r.get("class") == "background" and r.get("route") != PLACEMENT_ROUTE
    ]
    ok = [r for r in ended if r.get("outcome") == "completed" and r.get("status") == 200]
    tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in ok)
    return {
        "requests": len(ok),
        "errors": len(ended) - len(ok),
        "requests_per_s": round(len(ok) / seconds, 3),
        "tokens_per_s": round(tokens / seconds, 1),
        "first_byte_ms": logged_first_byte(ok, "background"),
    }


def _json_lines(text: str) -> Iterator[dict[str, Any]]:
    """The JSON objects among the lines of a log."""
    for line in text.splitlines():
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


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
    for record in _json_lines(journal):
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


def added_engine_counts(counts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Several engines' :func:`engine_counts` added up; a count that no engine keeps stays None."""
    counts = list(counts)

    def added(key: str) -> dict[str, int] | None:
        kept = [c[key] for c in counts if c[key] is not None]
        if not kept:
            return None
        totals: dict[str, int] = {}
        for by in kept:
            for name, value in by.items():
                totals[name] = totals.get(name, 0) + value
        return totals

    return {
        "preemptions": _added(*(c["preemptions"] for c in counts)),
        "prompt_tokens_by_source": added("prompt_tokens_by_source"),
        "refused": added("refused"),
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
    for node, share in (summary.get("nodes") or {}).items():
        parts = []
        if calls := share["voice"]:
            parts.append(f"{calls['calls']} calls, V2V p95 {calls['v2v_ms'].get('p95')} ms")
        if served := share["background"]:
            parts.append(f"{served['tokens_per_s']:,.0f} background tokens/s")
        if parts:
            line += f"; {node}: {', '.join(parts)}"
    return line


# The frontier

COLUMNS = (
    "arm",
    "workload",
    "point",
    "campaign",
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
"""frontier.csv's columns; goodput is in tokens and requests per second, latencies in ms. A point's
workload is the agents' corpus and the number of callers ("weka-x32, 2 callers"). A run's campaign
is the directory that holds its run directory (campaign.sh measures a campaign's points into
CAMPAIGN/POINT). The realtime first byte is the gateway's, from its request log ("log", exact) or
its histogram ("histogram", interpolated). Refused adds up the gateway's admission point and the
engine's queue limits; the gateway's own count of the background tokens it served checks AIPerf's
goodput."""
STANDARD_WORKLOAD = ("weka-x32", 2)
"""The corpus and callers of an arm file that names neither (e2's Agents.data and Voice.callers):
the workload drawn in frontier.svg."""
NODE_COLUMNS = ("nodes",)
WORKER_COLUMNS = (
    *(f"{stage.removesuffix('_ms')}_{q}_ms" for stage in WATERFALL for q in ("p50", "p95")),
    "events_sent",
    "events_dropped",
    "events_failed",
)
GPU_COLUMNS = ("gpu_sm_clock_p50_mhz", "gpu_temperature_max_c")
"""Columns that only some points have, each group shown when one of them has it: the GPU nodes of a
point on several, which every other column then totals; the voice worker's stage times and lifecycle
events; and the GPU's median SM clock and highest temperature during the calls (over every node's
GPU on several)."""


def load_points(run_dirs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    """One row per complete point, with its arm's frontier in its workload marked, and the run
    directories whose point did not complete. The rows go by workload, the arm file's own first,
    then by arm. The runs of a configuration measured in several campaigns go together, and the
    frontier places the configuration at their medians."""
    rows, incomplete = [], []
    for run_dir in run_dirs:
        run_file = run_dir / "run.json"
        run = orjson.loads(run_file.read_bytes()) if run_file.exists() else {}
        if run.get("status") != "complete":
            incomplete.append(run_dir)
            continue
        rows.append(_row(run_dir, run, orjson.loads((run_dir / "summary.json").read_bytes())))
    groups = configurations(rows)
    mark_frontier(groups)
    placed = {_configuration(g): g for g in groups}
    for row in rows:
        row["frontier"] = placed[_configuration(row)]["frontier"]

    def order(r: Mapping[str, Any]) -> tuple[Any, ...]:
        goodput = placed[_configuration(r)]["background_tokens_per_s"]
        # The arm file's own workload first, then the rest of its corpus, then the other corpora.
        data, callers = r["data"], r["callers"]
        first = ((data, callers) != STANDARD_WORKLOAD, data != STANDARD_WORKLOAD[0], data, callers)
        return (*first, _arm_order(r["arm"]), goodput, r["point"], r["campaign"])

    rows.sort(key=order)
    return rows, incomplete


def _configuration(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """What makes the runs of one configuration: its workload, its arm and its arm file's name."""
    return row["workload"], row["arm"], row["point"]


def _workload(data: str, callers: int) -> str:
    """A workload's name, as frontier.csv and frontier.md give it."""
    return f"{data}, {callers} caller{'' if callers == 1 else 's'}"


def _row(run_dir: Path, run: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """A point's row of frontier.csv; a point on several GPU nodes also carries each node's numbers
    (``by_node``) for frontier.md."""
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
        "workload": _workload(arm["agents"]["data"], arm["voice"]["callers"]),
        "point": run["name"],
        "campaign": run_dir.absolute().parent.name,
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
    # What frontier.csv leaves out: the corpus, which the workload names, and the stage waterfall's
    # inputs, the count and mean of each of the worker's stages and the callers' V2V samples.
    row["data"] = arm["agents"]["data"]
    row["stage_means"] = {s: (c["n"], c["mean"]) for s, c in waterfall.items() if "mean" in c}
    row["v2v_samples"] = v2v_samples(run_dir / "bots" / "turns.jsonl")
    nodes = summary.get("nodes") or {}
    row["nodes"] = " ".join(nodes) or None
    row["by_node"] = [_node_row(run_dir, node, share, window) for node, share in nodes.items()]
    return row | _gpu(run_dir, _gpu_samples(nodes), summary, window)


def v2v_samples(turns: Path) -> list[float]:
    """The callers' voice-to-voice samples in their turns.jsonl, pooled as their summary pools them:
    each answered turn's, and the reply's after each barge-in. None without the file."""
    if not turns.exists():
        return []
    samples = []
    for row in map(orjson.loads, filter(str.strip, turns.read_text().splitlines())):
        if "error" not in row:
            samples.append(row["v2v_ms"])
            if row["barge_in"]:
                samples.append(row["v2v_after_barge_in_ms"])
    return samples


def _node_row(
    run_dir: Path, node: str, share: Mapping[str, Any], window: Mapping[str, float] | None
) -> dict[str, Any]:
    """One node's share of a point, for frontier.md's table of GPU nodes."""
    voice, background = share["voice"] or {}, share["background"] or {}
    v2v, first_byte = voice.get("v2v_ms", {}), background.get("first_byte_ms", {})
    return {
        "node": node,
        "calls": voice.get("calls"),
        "v2v_p50_ms": v2v.get("p50"),
        "v2v_p95_ms": v2v.get("p95"),
        "v2v_p99_ms": v2v.get("p99"),
        "voice_turns": voice.get("turns"),
        "voice_errors": voice.get("errors"),
        "background_tokens_per_s": background.get("tokens_per_s"),
        "background_requests_per_s": background.get("requests_per_s"),
        "agent_first_byte_p50_ms": first_byte.get("p50"),
        "agent_first_byte_p99_ms": first_byte.get("p99"),
        "realtime_first_byte_p95_ms": (share["realtime_first_byte_ms"] or {}).get("p95"),
        "engine_preemptions": share["engine"]["preemptions"],
    } | _gpu(run_dir, _gpu_samples([node]), share, window)


def _gpu_samples(nodes: Iterable[str]) -> list[str]:
    """The files of GPU samples campaign.sh leaves in a run directory: gpu.csv, or gpu-NODE.csv for
    each of several GPU nodes."""
    return [f"gpu-{node}.csv" for node in nodes] or ["gpu.csv"]


def _gpu(
    run_dir: Path, names: Sequence[str], kept: Mapping[str, Any], window: Mapping[str, float] | None
) -> dict[str, int | None]:
    """The GPU columns from the samples ``names`` in the run directory; in a compact record, which
    leaves them out, from what its summary (or a node's share in it) kept of them, ``gpu``."""
    files = [run_dir / name for name in names]
    if not any(path.exists() for path in files) and "gpu" in kept:
        return {f"gpu_{key}": value for key, value in kept["gpu"].items()}
    return gpu(files, window)


def gpu(files: Sequence[Path], window: Mapping[str, float] | None) -> dict[str, int | None]:
    """The GPUs' median SM clock (MHz) and highest temperature (°C) during the calls, from the samples
    campaign.sh takes once a second on each GPU node (gpu.csv; gpu-NODE.csv for a point on several);
    None without them."""
    clocks: list[float] = []
    temperatures: list[float] = []
    for path in files:
        if not path.exists() or not window:
            continue
        for line in path.read_text(errors="replace").splitlines():
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


REPEATED = (
    "background_tokens_per_s",
    "v2v_p50_ms",
    "v2v_p95_ms",
    "v2v_p99_ms",
    "realtime_first_byte_p95_ms",
    "voice_errors",
)
"""The columns a configuration measured in several campaigns takes the median of over its runs, and
frontier.md gives with their range."""


def configurations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The configurations the rows measured, in the order of their first runs: each arm file's runs,
    one per campaign that measured it, with the median of each REPEATED column over the runs that
    measured it (the run's own value for a configuration measured once)."""
    runs: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        runs.setdefault(_configuration(row), []).append(row)
    return [
        {"workload": workload, "arm": arm, "point": point, "runs": measured}
        | {"frontier": all(r["frontier"] for r in measured)}
        | {column: _median(r[column] for r in measured) for column in REPEATED}
        for (workload, arm, point), measured in runs.items()
    ]


def _median(values: Iterable[float | None]) -> float | None:
    measured = [v for v in values if v is not None]
    return statistics.median(measured) if measured else None


def mark_frontier(rows: Sequence[dict[str, Any]]) -> None:
    """Mark the points that no other point of their arm and workload dominates."""

    def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        ax, ay = a["background_tokens_per_s"], a["v2v_p95_ms"]
        bx, by = b["background_tokens_per_s"], b["v2v_p95_ms"]
        return ax >= bx and ay <= by and (ax > bx or ay < by)

    placed = [r for r in rows if r["v2v_p95_ms"] is not None]
    for row in placed:
        rivals = (
            r
            for r in placed
            if r is not row and (r["arm"], r["workload"]) == (row["arm"], row["workload"])
        )
        row["frontier"] = not any(dominates(rival, row) for rival in rivals)


SLO_MARGIN_MS = 250.0
"""The voice V2V target (docs/SLO.md): p95 at most 250 ms above voice alone."""


def voice_slo(groups: Sequence[Mapping[str, Any]], callers: int) -> tuple[float, list[str]] | None:
    """The V2V p95 that the voice SLO allows with ``callers`` callers, and the configurations of
    voice alone it is read against: arm 1's with as many callers, from whichever corpus, since
    voice alone replays none. Their median V2V p95 plus SLO_MARGIN_MS; None without one."""
    alone = [
        g
        for g in groups
        if _arm_order(g["arm"])[0] == 1
        and g["runs"][0]["callers"] == callers
        and g["v2v_p95_ms"] is not None
    ]
    if not alone:
        return None
    p95 = statistics.median(g["v2v_p95_ms"] for g in alone)
    return p95 + SLO_MARGIN_MS, [g["point"] for g in alone]


def _arm_order(arm: str) -> tuple[int, str]:
    """Arms in E2's order: 2 before 10, 7a before 7b."""
    number = re.match(r"\d+", arm)
    return (int(number[0]) if number else 1000, arm)


STACKED = (
    ("end_of_turn_ms", "Endpointing"),
    ("turn_to_first_sentence_ms", "Reply to first sentence"),
    ("tts_first_audio_ms", "TTS first audio"),
)
"""The worker's stages that follow one another from the end of the caller's speech, as the stage
waterfall stacks them; the rest of the callers' V2V is transport and playout."""


def stage_waterfall(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each configuration's voice turn, over the runs that have turn lines: the mean of each STACKED
    stage over their turns, the rest of the mean V2V (``transport_ms``: the callers' V2V less the
    worker's stages), the LLM's mean time to first token and when its request started (at the final
    transcript under preemptive generation, otherwise at the end of the turn), and the mean, median
    and p95 of the callers' V2V samples. Configurations without turn lines are left out."""
    cells = []
    for group in configurations(rows):
        runs = [r for r in group["runs"] if r["stage_means"] and r["v2v_samples"]]
        stages = {stage: _pooled(runs, stage) for stage, _ in STACKED}
        if not runs or None in stages.values():
            continue
        v2v = summarize(sample for r in runs for sample in r["v2v_samples"])
        request = _pooled(runs, "transcript_ms") if runs[0]["preemptive_generation"] else None
        cells.append(
            {
                "point": group["point"],
                "label": runs[0]["label"],
                "runs": len(runs),
                "replies": v2v["n"],
                **stages,
                "transport_ms": v2v["mean"] - sum(stages.values()),
                "llm_request_ms": stages["end_of_turn_ms"] if request is None else request,
                "llm_ttft_ms": _pooled(runs, "llm_ttft_ms"),
                "v2v_mean_ms": v2v["mean"],
                "v2v_p50_ms": v2v["p50"],
                "v2v_p95_ms": v2v["p95"],
            }
        )
    return cells


def _pooled(runs: Iterable[Mapping[str, Any]], stage: str) -> float | None:
    """A stage's mean over the turns of several runs: each run's mean weighted by its count."""
    counted = [r["stage_means"][stage] for r in runs if stage in r["stage_means"]]
    turns = sum(n for n, _ in counted)
    return sum(n * mean for n, mean in counted) / turns if turns else None


def write_report(
    rows: Sequence[Mapping[str, Any]], out_dir: Path, waterfall: Sequence[str] = ()
) -> list[Path]:
    """frontier.csv, frontier.md and, when matplotlib is installed, a frontier figure per workload
    (:func:`figure_name`) and waterfall.svg: the stage waterfall of the points ``waterfall`` names,
    in its order, or of every configuration with turn lines. Naming a point without turn lines in
    the report is a ValueError."""
    cells = stage_waterfall(rows)
    if waterfall:
        missing = [name for name in waterfall if name not in {c["point"] for c in cells}]
        if missing:
            raise ValueError(f"the report has no turn lines of {', '.join(missing)}")
        cells = [c for name in waterfall for c in cells if c["point"] == name]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "frontier.csv", out_dir / "frontier.md"]
    with written[0].open("w", newline="") as f:
        columns = [*COLUMNS, *(c for group in _measured(rows) for c in group)]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    written[1].write_text(markdown(rows))
    groups = configurations(rows)
    for members in _by_workload(rows):
        slo = voice_slo(groups, members[0]["callers"])
        figure = out_dir / figure_name(members[0]["data"], members[0]["callers"])
        if plot(members, figure, None if slo is None else slo[0]):
            written.append(figure)
    if cells and plot_waterfall(cells, out_dir / "waterfall.svg"):
        written.append(out_dir / "waterfall.svg")
    return written


def _by_workload(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """The rows of each workload, in the rows' order."""
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        members.setdefault(row["workload"], []).append(row)
    return list(members.values())


def figure_name(data: str, callers: int) -> str:
    """A workload's frontier figure: frontier.svg for STANDARD_WORKLOAD, else named after the
    corpus when it is another (frontier-weka-x32-long.svg) and the callers when they are."""
    name = "frontier"
    if data != STANDARD_WORKLOAD[0]:
        name += "-" + data.replace("/", "-")
    if callers != STANDARD_WORKLOAD[1]:
        name += f"-{callers}-callers"
    return name + ".svg"


def _measured(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    """The optional column groups that some point has."""
    groups = (NODE_COLUMNS, WORKER_COLUMNS, GPU_COLUMNS)
    return [group for group in groups if any(r[c] is not None for r in rows for c in group)]


def markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """The frontier as a section per workload, with its voice SLO, and in it a table per arm,
    ordered by background goodput; then the voice worker's stage times when the points have them,
    and the stage waterfall of the configurations with turn lines."""
    measured = _measured(rows)
    gpu_columns = GPU_COLUMNS in measured
    groups = {_configuration(g): g for g in configurations(rows)}
    intro = (
        "Background goodput (prompt and completion tokens per second of the agent requests that "
        "completed during the calls) against voice-to-voice latency in ms, by workload: the agents' "
        "corpus (`[agents] data`) and the number of callers, since points of different workloads "
        "are not compared. • marks the points on their arm's frontier in their workload. The "
        "realtime first byte is the gateway's, from its request log; ~ marks one interpolated "
        "inside its histogram's buckets, for a point without the log. Refused counts the gateway's "
        "admission point and the engine's queue limits; — is a count the point's gateway or engine "
        "does not keep."
    )
    if any(len(g["runs"]) > 1 for g in groups.values()):
        intro += (
            " A point measured in several campaigns lists each run by its campaign, then the median "
            "and range (min\N{EN DASH}max) over its runs, where the frontier places it."
        )
    if gpu_columns:
        intro += " SM clock (MHz) and temperature (°C) are the GPU's during the calls."
    if NODE_COLUMNS in measured:
        intro += (
            " A point on several GPU nodes (its label says how many) is placed by its totals: all its "
            "calls, all its agent requests and its engines' counters added up; GPU nodes below gives "
            "each node's share."
        )
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
    for members in _by_workload(rows):
        lines += ["", f"## Workload: {members[0]['workload']}", "", _slo_line(members[0], groups)]
        for arm in sorted({r["arm"] for r in members}, key=_arm_order):
            lines += ["", f"### Arm {arm}", "", header, align]
            lines += _arm_rows([r for r in members if r["arm"] == arm], groups, gpu_columns)
    if NODE_COLUMNS in measured:
        lines += _nodes(rows, gpu_columns)
    if WORKER_COLUMNS in measured:
        lines += _stages(rows)
    if cells := stage_waterfall(rows):
        lines += _waterfall_table(cells)
    return "\n".join(lines) + "\n"


def _slo_line(
    row: Mapping[str, Any], groups: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> str:
    """The sentence that opens the section of ``row``'s workload: its voice SLO, and its figure."""
    callers, figure = row["callers"], figure_name(row["data"], row["callers"])
    slo = voice_slo(list(groups.values()), callers)
    if slo is None:
        return (
            f"No point of voice alone with {callers} callers sets a voice SLO. Figure: `{figure}`."
        )
    allowed, alone = slo
    return (
        f"The voice SLO allows a V2V p95 of {allowed:,.0f} ms, {SLO_MARGIN_MS:.0f} ms above voice "
        f"alone with as many callers ({', '.join(alone)}: {allowed - SLO_MARGIN_MS:,.0f} ms). "
        f"Figure: `{figure}`."
    )


def _arm_rows(
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[tuple[str, str, str], Mapping[str, Any]],
    gpu_columns: bool,
) -> list[str]:
    """An arm's table rows in a workload: a line per run, and a repeated configuration's medians."""
    lines = []
    for r in rows:
        group = groups[_configuration(r)]
        counts = " / ".join(_cell(r[k]) for k in ("shed", "refused", "resubmitted"))
        ttft = f"{_cell(r['agent_ttft_p50_ms'])} / {_cell(r['agent_ttft_p99_ms'])}"
        first_byte = _cell(r["realtime_first_byte_p95_ms"])
        if r["realtime_first_byte_source"] == "histogram" and first_byte != "—":
            first_byte = f"~{first_byte}"
        # A repeated configuration's frontier mark goes on its medians' line.
        name, mark = f"{r['point']}: {_label(r)}", "•" if r["frontier"] else ""
        if len(group["runs"]) > 1:
            name, mark = f"{r['point']} ({r['campaign']})", ""
        line = (
            f"| {name} | {mark} | {r['sessions']} "
            f"| {_cell(r['background_tokens_per_s'])} | {_cell(r['background_requests_per_s'], 2)} "
            f"| {_cell(r['v2v_p50_ms'])} | {_cell(r['v2v_p95_ms'])} | {_cell(r['v2v_p99_ms'])} "
            f"| {r['voice_errors']} of {r['voice_turns']} | {ttft} | {first_byte} "
            f"| {_cell(r['engine_preemptions'])} | {counts} |"
        )
        if gpu_columns:
            line += f" {_cell(r['gpu_sm_clock_p50_mhz'])} | {_cell(r['gpu_temperature_max_c'])} |"
        lines.append(line)
        if len(group["runs"]) > 1 and r is group["runs"][-1]:
            lines.append(_spread(group, gpu_columns))
    return lines


def _spread(group: Mapping[str, Any], gpu_columns: bool) -> str:
    """A repeated configuration's line in its arm's table: the median and range of each REPEATED
    column over its runs, and its frontier mark."""
    runs = group["runs"]

    def spread(column: str) -> str:
        values = [r[column] for r in runs if r[column] is not None]
        if not values:
            return "—"
        median = group[column]
        # A median between two counts of failed turns keeps its half.
        digits = 1 if column == "voice_errors" and not float(median).is_integer() else 0
        return f"{_cell(median, digits)} ({_cell(min(values))}\N{EN DASH}{_cell(max(values))})"

    first_byte = spread("realtime_first_byte_p95_ms")
    if any(r["realtime_first_byte_source"] == "histogram" for r in runs) and first_byte != "—":
        first_byte = f"~{first_byte}"
    first = runs[0]
    line = (
        f"| {first['point']}: {_label(first)}; median (range) of {len(runs)} runs "
        f"| {'•' if group['frontier'] else ''} | {first['sessions']} "
        f"| {spread('background_tokens_per_s')} | | {spread('v2v_p50_ms')} "
        f"| {spread('v2v_p95_ms')} | {spread('v2v_p99_ms')} | {spread('voice_errors')} | "
        f"| {first_byte} | | |"
    )
    return line + (" | |" if gpu_columns else "")


def _label(row: Mapping[str, Any]) -> str:
    """A point's label in frontier.md, which gives the number of GPU nodes of a point on several."""
    nodes = len(row["by_node"])
    return row["label"] + (f" ({nodes} GPU nodes)" if nodes else "")


def _nodes(rows: Sequence[Mapping[str, Any]], gpu_columns: bool) -> list[str]:
    """Each GPU node's share of the points on several, a line per run and node."""
    lines = [
        "",
        "## GPU nodes",
        "",
        "Each node's share of the points on several GPU nodes: the calls the voice worker placed on it, "
        "with their voice-to-voice latency in ms; the agent requests the gateway sent it, with their "
        "goodput by the gateway's request log and their first byte at the gateway; the first byte of "
        "its realtime requests; its engine's preemptions.",
        "",
    ]
    header = (
        "| Point | Node | Calls | V2V p50 | V2V p95 | V2V p99 | Failed turns | Background tokens/s "
        "| Requests/s | Agent first byte p50 / p99 | Realtime first byte p95 | Preemptions |"
    )
    align = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    if gpu_columns:
        header += " SM clock p50 | Max temp |"
        align += "---:|---:|"
    lines += [header, align]
    runs = Counter(map(_configuration, rows))
    for r in rows:
        name = f"{r['point']} ({r['campaign']})" if runs[_configuration(r)] > 1 else r["point"]
        for n in r["by_node"]:
            failed = "—"
            if n["voice_turns"] is not None:
                failed = f"{n['voice_errors']} of {n['voice_turns']}"
            first_byte = " / ".join(_cell(n[f"agent_first_byte_{q}_ms"]) for q in ("p50", "p99"))
            line = (
                f"| {name} | {n['node']} | {_cell(n['calls'])} | {_cell(n['v2v_p50_ms'])} "
                f"| {_cell(n['v2v_p95_ms'])} | {_cell(n['v2v_p99_ms'])} | {failed} "
                f"| {_cell(n['background_tokens_per_s'])} | {_cell(n['background_requests_per_s'], 2)} "
                f"| {first_byte} | {_cell(n['realtime_first_byte_p95_ms'])} "
                f"| {_cell(n['engine_preemptions'])} |"
            )
            if gpu_columns:
                line += (
                    f" {_cell(n['gpu_sm_clock_p50_mhz'])} | {_cell(n['gpu_temperature_max_c'])} |"
                )
            lines.append(line)
    return lines


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
    runs = Counter(map(_configuration, rows))
    for r in rows:
        stages = [
            " / ".join(_cell(r[f"{stage.removesuffix('_ms')}_{q}_ms"]) for q in ("p50", "p95"))
            for stage in WATERFALL
        ]
        events = " / ".join(_cell(r[f"events_{kind}"]) for kind in ("sent", "dropped", "failed"))
        name = f"{r['point']} ({r['campaign']})" if runs[_configuration(r)] > 1 else r["point"]
        lines.append(f"| {name} | {' | '.join(stages)} | {events} |")
    return lines


def _waterfall_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """The stage waterfall of each configuration with turn lines, as waterfall.svg draws it."""
    lines = [
        "",
        "## Stage waterfall",
        "",
        "Each configuration's voice turn as stages that follow one another, in ms, the mean over the "
        "replies of its runs: the end-of-turn decision after the end of speech (endpointing), the "
        "reply's first sentence after that decision and its first audio from TTS, as the voice "
        "worker times them, then the rest of the callers' V2V, transport and playout. The four add "
        "up to the mean V2V. The LLM's own first token is timed from its request, which under "
        "preemptive generation starts at the final transcript, before the end of the turn, so it is "
        "not a stage of its own. waterfall.svg stacks the four and marks the median and p95 V2V.",
        "",
        "| Configuration | Runs | Replies | Endpointing | Reply to first sentence "
        "| TTS first audio | Transport and playout | Mean V2V | V2V p50 | V2V p95 "
        "| LLM first token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stages = (*(key for key, _ in STACKED), "transport_ms")
    columns = (*stages, "v2v_mean_ms", "v2v_p50_ms", "v2v_p95_ms")
    for c in cells:
        times = " | ".join(_cell(c[key]) for key in columns)
        lines.append(
            f"| {c['point']}: {c['label']} | {c['runs']} | {c['replies']} | {times} "
            f"| {_cell(c['llm_ttft_ms'])} |"
        )
    return lines


def _cell(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


# Light chart surface, ink and the categorical palette, in its order.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_SURFACE, _INK, _MUTED, _GRID, _AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
_SECONDARY = "#52514e"

FAMILIES = (
    ("stock vLLM", 1, 5, _SECONDARY),
    ("operator answers", 6, 6, "#1baf7a"),
    ("engines", 7, 7, "#eda100"),
    ("routers", 8, 8, "#e87ba4"),
    ("Leaseholder", 9, 9, "#2a78d6"),
    ("MPS", 10, 10, "#008300"),
)
"""E2's arms by family (docs/EXPERIMENTS.md): its name, its first and last arm, and the color the
frontier figures draw it in. Any two families can meet on a scatter, and no six of the palette's
hues stay apart pair by pair, so stock vLLM, the baseline the others are read against, takes the
secondary ink, and five hues that do (with the markers and the arms' names beside them) take the
rest: blue, the first, for Leaseholder, then the others in E2's order."""
_MARKERS = (("o", 8.0), ("s", 7.1), ("^", 10.0), ("D", 7.1), ("v", 10.0), ("P", 9.5))
"""The markers of the arms of a family, counted by number (stock vLLM's 1 to 5), then by letter,
each in the size (points) that gives it the circle's area: 8 px inside its 2 px ring."""


def _save_svg(fig: Any, path: Path, **options: Any) -> None:
    """Write a figure as SVG, the same bytes for the same figure: no date, and the ids salted by
    the file's name rather than at random. ``options`` go to savefig."""
    import matplotlib

    with matplotlib.rc_context({"svg.hashsalt": path.name}):
        fig.savefig(path, format="svg", facecolor=_SURFACE, metadata={"Date": None}, **options)


def _style(arm: str) -> tuple[str, str, tuple[str, float]]:
    """An arm's family as the legend names it, its color and the arm's marker (FAMILIES, _MARKERS).
    An arm with a factor (4-no-preemptive) takes its arm's marker, and its name tells them apart."""
    found = re.match(r"(\d+)([a-z]?)", arm)
    number, letter = (int(found[1]), found[2]) if found else (0, "")
    for name, first, last, color in FAMILIES:
        if first <= number <= last:
            arms = f"arm {first}" if first == last else f"arms {first}\N{EN DASH}{last}"
            index = number - first + (ord(letter) - ord("a") if letter else 0)
            return f"{name} ({arms})", color, _MARKERS[index % len(_MARKERS)]
    return "other arms", _MUTED, _MARKERS[0]


def plot(rows: Sequence[Mapping[str, Any]], path: Path, slo: float | None = None) -> bool:
    """A workload's frontier figure: each arm's frontier as a line in its family's color and with
    the arm's marker, its dominated configurations as open markers, and the arm's name by the last
    point of its frontier. A configuration measured in several campaigns sits at its runs' medians,
    with whiskers over their range on both axes: one marker per configuration keeps each frontier a
    line through what frontier.md marks, and the whiskers still show a run that collapsed. A dashed
    line marks the voice SLO, ``slo`` ms. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter

    groups = [g for g in configurations(rows) if g["v2v_p95_ms"] is not None]
    arms = sorted({g["arm"] for g in groups}, key=_arm_order)
    fence = _fence([g["v2v_p95_ms"] for g in groups])
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=_SURFACE, layout="constrained")
    ax.set_facecolor(_SURFACE)
    ends = []
    for arm in arms:
        _, color, (marker, size) = _style(arm)
        points = [g for g in groups if g["arm"] == arm]
        front = sorted(
            (g for g in points if g["frontier"]), key=lambda g: g["background_tokens_per_s"]
        )
        others = [g for g in points if not g["frontier"]]
        ax.plot(
            [g["background_tokens_per_s"] for g in front],
            [g["v2v_p95_ms"] for g in front],
            color=color,
            linewidth=1.5,
            marker=marker,
            markersize=size,
            markeredgecolor=_SURFACE,
            markeredgewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            [g["background_tokens_per_s"] for g in others],
            [g["v2v_p95_ms"] for g in others],
            s=(size - 2.75) ** 2,  # with its edge, as large as a frontier marker inside its ring
            marker=marker,
            facecolors=_SURFACE,
            edgecolors=color,
            linewidths=1.25,
            zorder=2,
        )
        for g in (g for g in points if len(g["runs"]) > 1):
            x, y = g["background_tokens_per_s"], g["v2v_p95_ms"]
            xs = [r["background_tokens_per_s"] for r in g["runs"]]
            ys = [r["v2v_p95_ms"] for r in g["runs"] if r["v2v_p95_ms"] is not None]
            ax.errorbar(
                x,
                y,
                xerr=[[x - min(xs)], [max(xs) - x]],
                yerr=[[y - min(ys)], [max(ys) - y]],
                fmt="none",
                ecolor=color,
                elinewidth=1,
                alpha=0.6,
                zorder=1,
            )
        if drawn := [g for g in front if fence is None or g["v2v_p95_ms"] <= fence]:
            ends.append((drawn[-1]["background_tokens_per_s"], drawn[-1]["v2v_p95_ms"], arm))
    notes = ["A line joins each arm's frontier; open markers are dominated configurations"]
    if any(len(g["runs"]) > 1 for g in groups):
        notes[0] += ", whiskers span repeated runs"
    if fence is not None:  # the scale of the rest and of the SLO, with matplotlib's margins
        shown = [g["v2v_p95_ms"] for g in groups if g["v2v_p95_ms"] <= fence]
        shown += [] if slo is None else [slo]
        low, high = min(shown), max(shown)
        ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        notes += [
            f"Off the scale: {g['point']}, {g['v2v_p95_ms']:,.0f} ms at "
            f"{g['background_tokens_per_s']:,.0f} tokens/s"
            for g in groups
            if g["v2v_p95_ms"] > fence
        ]
    taken = []  # what the arms' names keep clear of
    if slo is None:
        notes.append(f"No point of voice alone with {rows[0]['callers']} callers sets a voice SLO")
    else:  # said above the plot: on the line, it would cover the points that reach it
        notes.append(
            f"The dashed line is the voice SLO, {slo:,.0f} ms: voice alone + {SLO_MARGIN_MS:.0f} ms"
        )
        taken.append(ax.axhline(slo, color=_INK, linewidth=0.75, linestyle=(0, (4, 3)), zorder=1.5))
    title = f"E2 frontier: {rows[0]['workload']}"
    ax.set_title(title, color=_INK, loc="left", pad=12 + 11 * len(notes))
    ax.annotate(
        ".\n".join(notes) + ".",
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=_SECONDARY,
    )
    ax.set_xlabel("Background goodput (tokens/s)", color=_INK)
    ax.set_ylabel("Voice V2V p95 (ms)", color=_INK)
    ax.grid(color=_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(_AXIS)
    families = list(dict.fromkeys(_style(arm)[:2] for arm in arms))
    if len(families) > 1:  # one family needs no legend: the arms' names say whose they are
        handles = [Line2D([], [], color=color, linewidth=2.5) for _, color in families]
        labels = [family for family, _ in families]
        legend = fig.legend(handles, labels, loc="outside lower left", ncols=3, frameon=False)
        for text in legend.get_texts():
            text.set_color(_INK)
    fig.draw_without_rendering()  # lays the axes out, so that the names can be placed in them
    marks = [(g["background_tokens_per_s"], g["v2v_p95_ms"]) for g in groups]
    _name_ends(ax, ends, marks, taken)
    _save_svg(fig, path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def _fence(values: Sequence[float]) -> float | None:
    """The V2V p95 past which a frontier figure leaves a configuration off its scale, so that one
    collapse does not flatten the rest: Tukey's fence for far outliers, three interquartile ranges
    above the upper quartile, of four or more configurations. None when none is past it."""
    if len(values) < 4:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    fence = q3 + 3 * (q3 - q1)
    return fence if max(values) > fence else None


_PLACES = (
    *((7.5, angle) for angle in (0, 35, -35, 90, -90, 145, -145, 180)),
    *((radius, angle) for radius in (20.0, 34.0) for angle in range(0, 360, 30)),
)
"""Where an arm's name may go around the last point of its frontier, in the order tried: beside it,
then farther out on a leader line. Each is a distance in points and an angle in degrees."""


def _name_ends(
    ax: Any,
    ends: Sequence[tuple[float, float, str]],
    marks: Sequence[tuple[float, float]],
    taken: Sequence[Any],
) -> None:
    """Write each arm's name by the last point of its frontier (``ends``: x, y and the name), at
    the first of _PLACES where it stays in the axes, clear of every marker (``marks``), of the texts
    ``taken`` and of the names placed before it; where none is, beside the point anyway. Arms whose
    ends share a marker's place share a name ("6, 8b"), and the most crowded go first."""
    import math

    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    half = 6.0 * ax.figure.dpi / 72  # the largest marker with its ring, in pixels
    boxes = [t.get_window_extent(renderer).padded(2) for t in taken]
    centers = [ax.transData.transform(mark) for mark in marks]
    for cx, cy in centers:
        boxes.append(Bbox.from_extents(cx - half, cy - half, cx + half, cy + half))
    shared: list[tuple[float, float, list[str], Any]] = []
    for x, y, name in ends:
        at = ax.transData.transform((x, y))
        same = next((s for s in shared if math.dist(s[3], at) <= half), None)
        if same is None:
            shared.append((x, y, [name], at))
        else:
            same[2].append(name)

    def crowd(end: tuple[float, float, list[str], Any]) -> int:
        """Less for an end with more markers around it."""
        return -sum(math.dist(center, end[3]) <= 6 * half for center in centers)

    def name_at(x: float, y: float, name: str, distance: float, angle: float) -> tuple[Any, Any]:
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        leader = {"arrowstyle": "-", "color": _MUTED, "linewidth": 0.6, "shrinkB": 5}
        label = ax.annotate(
            name,
            (x, y),
            xytext=(distance * dx, distance * dy),
            textcoords="offset points",
            ha="left" if dx > 0.3 else "right" if dx < -0.3 else "center",
            va="bottom" if dy > 0.3 else "top" if dy < -0.3 else "center",
            fontsize=8,
            color=_INK,
            arrowprops=None if distance == _PLACES[0][0] else leader,
        )
        label.set_in_layout(False)
        label.update_positions(renderer)
        return label, Text.get_window_extent(label, renderer).padded(1.5)

    for x, y, names, _ in sorted(shared, key=crowd):
        name = ", ".join(names)
        for place in _PLACES:
            label, box = name_at(x, y, name, *place)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0
            if inside and box.y1 <= frame.y1 and not any(box.overlaps(b) for b in boxes):
                break
            label.remove()
        else:  # nowhere clear: beside the point anyway
            label, box = name_at(x, y, name, *_PLACES[0])
        boxes.append(box)


def plot_waterfall(cells: Sequence[Mapping[str, Any]], path: Path) -> bool:
    """waterfall.svg: a bar per configuration that stacks the means of its stages from the end of the
    caller's speech, so that its length is the mean V2V, with a line at the median V2V and a
    triangle at the p95, and under it a thin bar for the LLM's time to first token from its
    request. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    ys = list(range(len(cells)))
    fig, ax = plt.subplots(
        figsize=(7.5, 1.9 + 0.5 * len(cells)), facecolor=_SURFACE, layout="constrained"
    )
    ax.set_facecolor(_SURFACE)
    handles, left = [], [0.0] * len(cells)
    segments = [*STACKED, ("transport_ms", "Transport and playout")]
    for color, (key, label) in zip(_SERIES, segments, strict=False):
        widths = [max(0.0, c[key]) for c in cells]
        bars = ax.barh(ys, widths, left=left, height=0.42, color=color, edgecolor=_SURFACE)
        handles.append((bars, label))
        left = [x + width for x, width in zip(left, widths, strict=True)]
    timed = [(y, c) for y, c in zip(ys, cells, strict=True) if c["llm_ttft_ms"] is not None]
    llm = ax.barh(
        [y + 0.32 for y, _ in timed],
        [c["llm_ttft_ms"] for _, c in timed],
        left=[c["llm_request_ms"] for _, c in timed],
        height=0.08,
        color=_SECONDARY,
    )
    median = ax.vlines(
        [c["v2v_p50_ms"] for c in cells],
        [y - 0.27 for y in ys],
        [y + 0.27 for y in ys],
        colors=_INK,
        linewidth=2,
    )
    p95 = ax.scatter(
        [c["v2v_p95_ms"] for c in cells], [y - 0.34 for y in ys], s=36, marker="v", color=_INK
    )
    handles += [
        (llm, "LLM first token, from its request"),
        (median, "V2V median"),
        (p95, "V2V p95"),
    ]
    ax.set_yticks(
        ys, [c["point"] + (f" ({c['runs']} runs)" if c["runs"] > 1 else "") for c in cells]
    )
    ax.set_ylim(len(cells) - 0.5, -0.55)  # the report's order, top to bottom
    ax.set_xlabel("ms from the end of the caller's speech", color=_INK)
    ax.set_title(
        "Voice turn stages, stacked: a bar's length is the mean V2V", color=_INK, loc="left"
    )
    ax.grid(axis="x", color=_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    ax.tick_params(axis="y", length=0, labelcolor=_INK)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    artists, labels = zip(*handles, strict=True)
    legend = fig.legend(artists, labels, loc="outside lower left", ncols=3, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK)
    _save_svg(fig, path, bbox_inches="tight", pad_inches=0.1)  # the rows and the legend, no more
    plt.close(fig)
    return True


# The compact record

RECORD = ("run.json", "arm.toml", "bots/summary.json", "bots/turns.jsonl")
"""What a point's compact record (``lh-bench e2 export``) copies from its run directory, beside its
summary.json and a reduced engine.json (engine-NODE.json for each of several GPU nodes). Left out:
the gateway's request log and the GPU's samples, whose numbers the record's summary keeps; AIPerf's
exports, the metric snapshots, the worker's journal and the tools' logs; and the callers' run.json,
which names LiveKit's API key."""

VLLM_ENGINE = {
    "model_config": ("model", "revision", "max_model_len", "dtype", "quantization"),
    "cache_config": (
        "block_size",
        "enable_prefix_caching",
        "cache_dtype",
        "kv_cache_memory_bytes",
        "num_gpu_blocks",
    ),
    "scheduler_config": (
        "policy",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
        "max_num_seqs",
        "async_scheduling",
        "watermark",
        "max_num_queued_tokens",
    ),
    "kv_transfer_config": ("kv_connector", "kv_connector_extra_config"),
}
"""What a record keeps of a vLLM engine's configuration, beside all of its additional_config: the
settings lh-bench e2 run checks the arms' flags against, and what it takes to serve the same engine
again (the model and its revision, the context, dtype and quantization, the KV pool, the CPU tier)."""
SGLANG_ENGINE = (
    "model_path",
    "revision",
    "context_length",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "page_size",
    "disable_radix_cache",
    "mem_fraction_static",
    "max_total_tokens",
    "max_total_num_tokens",
    "schedule_policy",
    "chunked_prefill_size",
    "max_running_requests",
    "enable_priority_scheduling",
    "schedule_low_priority_values_first",
    "priority_scheduling_preemption_threshold",
    "enable_hierarchical_cache",
    "hicache_ratio",
    "hicache_write_policy",
)
"""The same of an SGLang engine's server arguments."""
_SNAPSHOT = re.compile(r"models--(?P<name>[^/]+)/snapshots/(?P<revision>[0-9a-f]+)")
"""A model in the Hugging Face cache: models--ORG--NAME/snapshots/REVISION."""
_HOME = re.compile(r"/(?:home|Users)/[^/\s]+|/root(?=/)")


def export_point(run_dir: Path, out: Path) -> Path | None:
    """Write the compact record of the point in ``run_dir`` into ``out``, at CAMPAIGN/POINT as the
    run directory sits in its campaign, and return its directory; None for a point that did not
    complete. Its summary.json takes the first bytes exactly from the request log, as the report
    does, and the GPU's median SM clock and highest temperature from gpu.csv (``gpu``; on several
    GPU nodes over all of their gpu-NODE.csv, and each node's in its share), so the report gives the
    same tables from the record as from the run directory. Its engine.json (engine-NODE.json for
    each of several nodes) keeps only what says which engine the point ran on
    (:func:`engine_settings`)."""
    run_file = run_dir / "run.json"
    run = orjson.loads(run_file.read_bytes()) if run_file.exists() else {}
    if run.get("status") != "complete":
        return None
    summary = orjson.loads((run_dir / "summary.json").read_bytes())
    window = run.get("window")
    requests = gateway_requests(run_dir, window["start"], window["end"]) if window else None
    if requests is not None:
        for slo_class in ("realtime", "background"):
            summary["gateway"][f"{slo_class}_first_byte_ms"] = logged_first_byte(
                requests, slo_class
            )

    def clocks(names: Sequence[str]) -> dict[str, int | None]:
        columns = gpu([run_dir / name for name in names], window)
        return {column.removeprefix("gpu_"): value for column, value in columns.items()}

    nodes = summary.get("nodes") or {}
    if any((run_dir / name).exists() for name in _gpu_samples(nodes)):
        summary["gpu"] = clocks(_gpu_samples(nodes))
    for node, share in nodes.items():
        if (run_dir / f"gpu-{node}.csv").exists():
            share["gpu"] = clocks([f"gpu-{node}.csv"])
    record = out / run_dir.absolute().parent.name / run_dir.absolute().name
    (record / "bots").mkdir(parents=True, exist_ok=True)
    for name in RECORD:
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, record / name)
    (record / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2) + b"\n")
    for name in [f"engine-{node}.json" for node in nodes] or ["engine.json"]:
        if (engine := run_dir / name).exists():
            settings = engine_settings(orjson.loads(engine.read_bytes()))
            (record / name).write_bytes(orjson.dumps(settings, option=orjson.OPT_INDENT_2) + b"\n")
    return record


def engine_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """An engine's configuration (engine.json, from its /server_info) reduced to VLLM_ENGINE or
    SGLANG_ENGINE, with the model named by its Hugging Face id and revision rather than its path in
    the cache, and every path under a home directory from ``~``."""
    if "model_config" not in config:  # SGLang's server arguments
        return _home({key: config.get(key) for key in SGLANG_ENGINE})
    settings: dict[str, Any] = {
        section: None if config.get(section) is None else {k: config[section].get(k) for k in keys}
        for section, keys in VLLM_ENGINE.items()
    }
    settings["additional_config"] = config.get("additional_config")
    model = settings["model_config"]
    if model and (cached := _SNAPSHOT.search(str(model["model"]))):
        model |= {"model": cached["name"].replace("--", "/", 1), "revision": cached["revision"]}
    return _home(settings)


def _home(value: Any) -> Any:
    """``value`` with every path under a home directory written from ``~``."""
    if isinstance(value, str):
        return _HOME.sub("~", value)
    if isinstance(value, Mapping):
        return {key: _home(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_home(item) for item in value]
    return value
