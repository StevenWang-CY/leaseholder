"""A point's results from its run directory (:mod:`.run`): its summary and its compact record.

:func:`summarize_point` reads one run directory of ``lh-bench e2 run``: the callers' summary, the
per-request records of the AIPerf replay, the ``/metrics`` snapshots taken when the calls started
and ended, and the voice worker's journal of the calls. Every metric covers that window. A metric
that the gateway or the engine does not export is null, never zero, and a counter that went down
(its process restarted during the calls) is an error. A point on several GPU nodes reports its
totals as a point on one does, its engines' counters added up, and each node's share.

:func:`export_point` writes a point's compact record (``lh-bench e2 export``): the few small files
the report reads. :func:`compact` gives the record's run.json and summary.json from a run directory
or from a record alike, so the report gives the same tables from either.
"""

from __future__ import annotations

import re
import shutil
import statistics
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
REPLY_TIMEOUT_MS = 30_000.0
"""How long ``lh-voice bots`` waits for a reply (its ``reply_timeout``), after which the turn fails: a
reply that never came counts as a sample censored at this value, in every V2V percentile."""


def summarize_point(
    run_dir: Path,
    start: float,
    end: float,
    *,
    agents: bool,
    nodes: Sequence[str] = (),
    warmup_turns: int = 0,
) -> dict[str, Any]:
    """The E2 metrics of the window ``start``..``end`` (epoch seconds) from a run directory. Raises
    ``ValueError`` when the point measured nothing to place on the frontier. A point on several GPU
    ``nodes`` adds up their engines' counters and reports each node's share under ``nodes``. The
    ``warmup_turns`` that open each call are measured by nothing: the callers leave them out, and so
    does the worker's account of the turns."""
    bots = orjson.loads((run_dir / "bots" / "summary.json").read_bytes())
    if bots["v2v_ms"].get("n", 0) <= bots["v2v_ms"].get("censored", 0):
        raise ValueError(f"the calls measured no voice-to-voice latency ({bots['errors']} failed)")
    metrics = run_dir / "metrics"
    records = run_dir / "aiperf" / "profile_export.jsonl"
    requests = gateway_requests(run_dir, start, end)
    engines = {node: engine_counts(*_snapshots(metrics, f"engine-{node}")) for node in nodes}
    voice = ("turns", "errors", "v2v_ms", "barge_in_to_silence_ms", "barge_ins")
    summary = {
        "window_s": round(end - start, 1),
        "voice": {key: bots[key] for key in voice},
        "worker": worker(run_dir / "voice-worker.log", warmup_turns),
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
    turns = _json_lines((run_dir / "bots" / "turns.jsonl").read_text()) if placed else ()
    turns = measured_turns(turns)
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


def measured_turns(turns: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The callers' measured turns (bots/turns.jsonl): all but the warm-up turns that open each call,
    whose lines say ``"warmup": true`` (``[voice] warmup_turns``)."""
    return [turn for turn in turns if not turn.get("warmup")]


def call_turns(turns: Sequence[Mapping[str, Any]], calls: int) -> dict[str, Any]:
    """Some of the calls' numbers from their callers' measured turns (bots/turns.jsonl), counted as
    ``lh-voice bots`` sums up all of them: voice-to-voice over every reply, the reply to an
    interruption too, and a reply or a silence a timeout cut short censored at REPLY_TIMEOUT_MS."""
    turns = measured_turns(turns)
    silence = [t["barge_in_to_silence_ms"] for t in turns if "barge_in_to_silence_ms" in t]
    silence += [
        REPLY_TIMEOUT_MS
        for t in turns
        if t.get("interrupted_speech") and "barge_in_to_silence_ms" not in t
    ]
    return {
        "calls": calls,
        "turns": len(turns),
        "errors": sum(1 for t in turns if "error" in t),
        "v2v_ms": summarize(turn_samples(turns)[0]),
        "barge_in_to_silence_ms": summarize(silence),
        "barge_ins": sum(1 for t in turns if t.get("barge_in")),
    }


def turn_samples(turns: Iterable[Mapping[str, Any]]) -> tuple[list[float], int]:
    """The voice-to-voice samples of the callers' measured turns (bots/turns.jsonl) and how many are
    censored: each turn's reply and, after a barge-in, the reply to the interruption, as ``lh-voice
    bots`` pools them; a reply that never came (a failed turn) stands at REPLY_TIMEOUT_MS."""
    samples, censored = [], 0
    for turn in measured_turns(turns):
        wanted = ["v2v_ms", *(["v2v_after_barge_in_ms"] if turn.get("barge_in") else [])]
        for key in wanted:
            censored += key not in turn
            samples.append(turn.get(key, REPLY_TIMEOUT_MS))
    return samples, censored


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


def worker(journal: Path, warmup_turns: int = 0) -> dict[str, Any] | None:
    """The voice worker's own account of the calls, from its journal (voice-worker.log): each stage
    of the waterfall over the turns, and the lifecycle events its sessions sent, dropped and failed
    to deliver. None without a turn or session line in it; the events are None when no session sent
    any (``LH_EVENTS=off``, or an LLM that bypasses the gateway). The first ``warmup_turns`` turn
    lines of each session are the replies to the call's warm-up turns, which have no interruption to
    add a reply, and are left out."""
    if not journal.exists():
        return None
    turns, sessions = worker_lines(journal.read_text(errors="replace"))
    if not turns and not sessions:
        return None
    turns = [t for t in turns if t.get("turn", warmup_turns) >= warmup_turns]
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


CPU_TIER = {
    "kv_offload_load_bytes": "vllm:kv_offload_load_bytes_total",
    "kv_offload_store_bytes": "vllm:kv_offload_store_bytes_total",
}
"""The CPU tier's own counts (vLLM's OffloadingConnector): the bytes of every load into the GPU pool and
of every store from it, a preempted request's resume included, which the prompt-token sources leave
out."""


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
        "preemptions": measured_sum(*(c["preemptions"] for c in counts)),
        "prompt_tokens_by_source": added("prompt_tokens_by_source"),
        "refused": added("refused"),
        **{key: measured_sum(*(c.get(key) for c in counts)) for key in CPU_TIER},
    }


def engine_counts(before: str, after: str) -> dict[str, Any]:
    """vLLM's preemptions, prompt tokens by source, refused requests and the CPU tier's loads and
    stores in bytes over the window. vLLM 0.32 counts a request's prompt tokens by source only at its
    first scheduling, so a preempted request's resume is in none of them; the CPU tier's counts have
    every load and store."""

    def by(label: str, name: str) -> dict[str, int] | None:
        series = deltas(before, after, name, "engine")
        if series is None:
            return None
        totals: dict[str, int] = {}
        for labels, value in series.items():
            key = dict(labels).get(label, "")
            totals[key] = totals.get(key, 0) + round(value)
        return totals

    def total(name: str) -> int | None:
        series = deltas(before, after, name, "engine")
        return None if series is None else round(sum(series.values()))

    return {
        "preemptions": total("vllm:num_preemptions_total"),
        "prompt_tokens_by_source": by("source", "vllm:prompt_tokens_by_source_total"),
        "refused": by("reason", "vllm:admission_rejections_total"),
        **{key: total(name) for key, name in CPU_TIER.items()},
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


def _gpu_samples(nodes: Iterable[str]) -> list[str]:
    """The files of GPU samples campaign.sh leaves in a run directory: gpu.csv, or gpu-NODE.csv for
    each of several GPU nodes."""
    return [f"gpu-{node}.csv" for node in nodes] or ["gpu.csv"]


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


def measured_sum(*counts: int | None) -> int | None:
    """The sum of the counts that were measured; None when none was."""
    measured = [c for c in counts if c is not None]
    return sum(measured) if measured else None


# The compact record

RECORD = ("run.json", "arm.toml", "bots/summary.json", "bots/turns.jsonl")
"""What a point's compact record (``lh-bench e2 export``) keeps of its run directory's files, its
run.json as :func:`compact` gives it, beside its summary.json and a reduced engine.json
(engine-NODE.json for each of several GPU nodes). Left out:
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
        "scheduler_cls",
    ),
    "kv_transfer_config": ("kv_connector", "kv_connector_extra_config"),
}
"""What a record keeps of a vLLM engine's configuration, beside all of its additional_config: the
settings lh-bench e2 run checks the arms' flags against, the scheduler class among them (a variant's
module, or None for vLLM's own), and what it takes to serve the same engine again (the model and its
revision, the context, dtype and quantization, the KV pool, the CPU tier)."""
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


def compact(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The run.json and summary.json of the compact record of the point in ``run_dir``, a run
    directory or a record (whose own it returns); None for a point that did not complete. The
    run.json is the run's, its ``versions`` naming the GPU nodes' deployed revisions as null where
    the run did not record them (``gpu_source``). The summary takes the first bytes of both classes
    exactly from the request log, and the GPU's median SM clock and highest temperature from gpu.csv
    (``gpu``; on several GPU nodes over all of their gpu-NODE.csv, and each node's in its share): what
    a record keeps of the files it leaves out. The report reads every point through it, so a run
    directory and its record give the same tables."""
    run_file = run_dir / "run.json"
    run = orjson.loads(run_file.read_bytes()) if run_file.exists() else {}
    if run.get("status") != "complete":
        return None
    if "versions" in run and "gpu_source" not in run["versions"]:
        # Points measured before e2 run recorded it: beside cyan's revision, as e2 run writes it.
        versions = list(run["versions"].items())
        at = 1 + next((i for i, (key, _) in enumerate(versions) if key == "source"), -1)
        run["versions"] = dict([*versions[:at], ("gpu_source", None), *versions[at:]])
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
    # The CPU tier's loads and stores, from the engines' /metrics snapshots: points measured before
    # the summary kept them gain them, and the summary's other counts stay as they are.
    snapshots = {node: f"engine-{node}" for node in nodes} or {None: "engine"}
    metrics = run_dir / "metrics"
    if all(
        (metrics / f"{name}-{when}.prom").exists()
        for name in snapshots.values()
        for when in ("start", "end")
    ):
        counts = {
            node: engine_counts(*_snapshots(metrics, name)) for node, name in snapshots.items()
        }
        for node, share in nodes.items():
            share["engine"] |= {key: counts[node][key] for key in CPU_TIER}
        totals = added_engine_counts(counts.values()) if nodes else counts[None]
        summary["engine"] |= {key: totals[key] for key in CPU_TIER}
    return run, summary


def point_engines(run_dir: Path, nodes: Sequence[str]) -> dict[str | None, dict[str, Any] | None]:
    """The engine of a point (by node on several GPU nodes, else under None) as its record keeps it
    (:func:`engine_settings`), from a run directory's engine.json or a record's; None without one."""
    engines: dict[str | None, dict[str, Any] | None] = {}
    for node in [*nodes] or [None]:
        path = run_dir / (f"engine-{node}.json" if node else "engine.json")
        engines[node] = engine_settings(orjson.loads(path.read_bytes())) if path.exists() else None
    return engines


def kv_token_bytes(engine: Mapping[str, Any] | None) -> float | None:
    """The bytes of KV a token takes in an engine's pool, which the CPU tier moves as they are: its KV
    cache's bytes over its tokens, from vLLM's cache_config (``kv_cache_memory_bytes`` over
    ``num_gpu_blocks`` times ``block_size``). The bytes left over by the whole blocks that fit, less
    than one block's, make it high by under one part in ``num_gpu_blocks`` (0.02% on the frozen box:
    147,492 bytes against Qwen3-4B's 147,456). None without them (SGLang)."""
    cache = (engine or {}).get("cache_config") or {}
    try:
        return cache["kv_cache_memory_bytes"] / (cache["num_gpu_blocks"] * cache["block_size"])
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def export_point(run_dir: Path, out: Path) -> Path | None:
    """Write the compact record of the point in ``run_dir`` into ``out``, at CAMPAIGN/POINT as the
    run directory sits in its campaign, and return its directory; None for a point that did not
    complete. Its summary.json is :func:`compact`'s. Its engine.json (engine-NODE.json for each of
    several nodes) keeps only what says which engine the point ran on (:func:`engine_settings`)."""
    if (point := compact(run_dir)) is None:
        return None
    run, summary = point
    nodes = summary.get("nodes") or {}
    record = out / run_dir.absolute().parent.name / run_dir.absolute().name
    (record / "bots").mkdir(parents=True, exist_ok=True)
    for name in RECORD:
        if name != "run.json" and (run_dir / name).exists():
            shutil.copyfile(run_dir / name, record / name)
    # As e2 run writes them, so that a run.json the record keeps as it was stays the same bytes.
    (record / "run.json").write_bytes(orjson.dumps(run, option=orjson.OPT_INDENT_2) + b"\n")
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
