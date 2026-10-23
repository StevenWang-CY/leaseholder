"""E2's results: a point's summary from its run directory, and the frontier across points."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from leaseholder_bench.cli import main
from leaseholder_bench.e2 import SECRETS
from leaseholder_bench.frontier import (
    COLUMNS,
    GPU_COLUMNS,
    RECORD,
    SESSIONS_LOGGER,
    STACKED,
    TURNS_LOGGER,
    WATERFALL,
    WORKER_COLUMNS,
    background,
    configurations,
    engine_counts,
    first_byte_ms,
    gateway_counts,
    gateway_requests,
    load_points,
    mark_frontier,
    stage_waterfall,
    summarize_point,
    write_report,
)

T0 = 1_800_000_000.0  # the calls run from T0 to T0 + 100 s
REPO = Path(__file__).resolve().parents[3]
VOICE = REPO / "python" / "leaseholder_voice"
VOICE_EVENTS = VOICE / "src" / "leaseholder_voice" / "events.py"
RESULTS = REPO / "benchmarks" / "results" / "2026-10-23-e2"


def record(end: float, *, phase: str = "profiling", ttft: float = 100.0, **extra: Any) -> str:
    metrics = {
        "time_to_first_token": {"value": ttft},
        "usage_prompt_tokens": {"value": 1000},
        "usage_completion_tokens": {"value": 20},
    }
    meta = {"benchmark_phase": phase, "request_end_ns": int(end * 1e9)}
    return json.dumps({"metadata": meta | extra.pop("meta", {}), "metrics": metrics} | extra)


def test_background_goodput_counts_the_requests_completed_during_the_calls(tmp_path: Path) -> None:
    lines = [
        record(T0 - 1),  # before the calls
        record(T0 + 1, phase="warmup"),
        *(record(T0 + i, ttft=100.0 + i) for i in range(1, 101)),
        record(T0 + 50, error={"code": 503, "message": "queue full"}),
        record(T0 + 60, meta={"was_cancelled": True}),
        record(T0 + 101),  # after
    ]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("\n".join(lines) + "\n")
    cell = background(path, T0, T0 + 100)
    assert (cell["requests"], cell["errors"], cell["without_usage"]) == (100, 2, 0)
    assert cell["requests_per_s"] == 1.0
    assert (cell["prompt_tokens_per_s"], cell["completion_tokens_per_s"]) == (1000.0, 20.0)
    assert cell["tokens_per_s"] == 1020.0
    assert cell["ttft_ms"]["n"] == 100 and cell["ttft_ms"]["p50"] == 150.5
    with pytest.raises(ValueError, match="completed no request"):
        background(path, T0 + 200, T0 + 300)
    with pytest.raises(ValueError, match="no per-request records"):
        background(tmp_path / "missing.jsonl", T0, T0 + 100)


GATEWAY_BEFORE = """\
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.1"} 10
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.5"} 10
lh_gateway_first_byte_seconds_bucket{class="realtime",le="+Inf"} 10
lh_gateway_first_byte_seconds_sum{class="realtime"} 0.5
lh_gateway_first_byte_seconds_bucket{class="background",le="0.1"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="0.5"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="+Inf"} 50
lh_gateway_request_outcomes_total{class="background",outcome="completed"} 40
lh_gateway_request_outcomes_total{class="background",outcome="shed"} 0
lh_gateway_usage_tokens_total{class="background",kind="prompt"} 1000
lh_gateway_usage_tokens_total{class="background",kind="completion"} 10
lh_gateway_usage_tokens_total{class="background",kind="cached"} 0
lh_gateway_usage_tokens_total{class="realtime",kind="prompt"} 5
lh_gateway_admission_refused_total{class="background",reason="queue_full"} 1
lh_gateway_admission_refused_total{class="background",reason="timeout"} 0
lh_gateway_admission_resubmits_total{class="background"} 0
lh_gateway_admission_discarded_tokens_total{class="background"} 0
"""
GATEWAY_AFTER = """\
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.1"} 18
lh_gateway_first_byte_seconds_bucket{class="realtime",le="0.5"} 20
lh_gateway_first_byte_seconds_bucket{class="realtime",le="+Inf"} 20
lh_gateway_first_byte_seconds_sum{class="realtime"} 1.4
lh_gateway_first_byte_seconds_bucket{class="background",le="0.1"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="0.5"} 0
lh_gateway_first_byte_seconds_bucket{class="background",le="+Inf"} 90
lh_gateway_request_outcomes_total{class="background",outcome="completed"} 140
lh_gateway_request_outcomes_total{class="background",outcome="shed"} 3
lh_gateway_usage_tokens_total{class="background",kind="prompt"} 101000
lh_gateway_usage_tokens_total{class="background",kind="completion"} 2010
lh_gateway_usage_tokens_total{class="background",kind="cached"} 50000
lh_gateway_usage_tokens_total{class="realtime",kind="prompt"} 505
lh_gateway_admission_refused_total{class="background",reason="queue_full"} 3
lh_gateway_admission_refused_total{class="background",reason="timeout"} 4
lh_gateway_admission_resubmits_total{class="background"} 5
lh_gateway_admission_discarded_tokens_total{class="background"} 2500
"""


def test_the_gateway_counts_cover_the_window() -> None:
    cell = gateway_counts(GATEWAY_BEFORE, GATEWAY_AFTER, 100.0)
    # Realtime only: 8 of 10 within 100 ms, 2 more within 500 ms, interpolated inside the buckets.
    assert cell["realtime_first_byte_ms"] == {
        "n": 10,
        "mean": 90.0,
        "p50": 62.5,
        "p95": 400.0,
        "p99": 480.0,
        "source": "histogram",
    }
    assert cell["background_first_byte_ms"]["n"] == 40
    assert (cell["shed"], cell["refused"], cell["resubmitted"]) == (3, 6, 5)
    assert cell["discarded_tokens"] == 2500
    assert cell["background_tokens_per_s"] == 1020.0  # prompt and completion, background only
    assert (
        cell["counters"]['lh_gateway_request_outcomes_total{class="background",outcome="shed"}']
        == 3
    )
    assert (
        'lh_gateway_admission_refused_total{class="background",reason="timeout"}'
        in cell["counters"]
    )
    assert len(cell["counters"]) == 10  # every counter series that moved
    restarted = GATEWAY_AFTER.replace('completed"} 140', 'completed"} 2')
    with pytest.raises(ValueError, match="went down during the calls: the gateway restarted"):
        gateway_counts(GATEWAY_BEFORE, restarted, 100.0)


def test_a_gateway_without_an_admission_point_counts_none_of_its_outcomes() -> None:
    before, after = (
        "\n".join(
            line for line in text.splitlines() if "admission" not in line and "shed" not in line
        )
        for text in (GATEWAY_BEFORE, GATEWAY_AFTER)
    )
    cell = gateway_counts(before, after, 100.0)
    assert (cell["shed"], cell["refused"], cell["resubmitted"], cell["discarded_tokens"]) == (
        None,
        None,
        None,
        None,
    )
    assert cell["background_tokens_per_s"] == 1020.0
    empty = gateway_counts("", "", 100.0)
    assert empty["realtime_first_byte_ms"] == {"n": 0, "source": "histogram"}
    assert empty["background_tokens_per_s"] is None


def logged(at: float, slo_class: str | None, first_byte_ms: float | None, **fields: Any) -> str:
    """One of the gateway's JSON request lines, for a request that ended at ``at``."""
    stamp = datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds")
    line = {
        "timestamp": stamp.replace("+00:00", "Z"),
        "level": "INFO",
        "message": "request",
        "class": slo_class,
        "status": 200,
        "outcome": "completed",
        "first_byte_ms": first_byte_ms,
    }
    return json.dumps({k: v for k, v in (line | fields).items() if v is not None})


# The calls' realtime requests had their first byte at 100, 200, ... 1000 ms.
REQUESTS = "\n".join(
    [
        logged(T0 - 1, "realtime", 9999.0),  # before the calls
        *(logged(T0 + n, "realtime", 100.0 * n) for n in range(1, 11)),
        logged(T0 + 50, "realtime", None, outcome="cancelled"),  # left before its first byte
        logged(T0 + 60, "realtime", 5.0, outcome=None),  # never went upstream
        logged(T0 + 70, "background", 2500.0),
        '{"timestamp": "2027-01-15T08:00:00.000000Z", "message": "lifecycle event"}',
        "a line that is not JSON",
    ]
)


def test_the_first_bytes_come_exactly_from_the_request_log(tmp_path: Path) -> None:
    (tmp_path / "gateway.jsonl").write_text(REQUESTS + "\n")
    requests = gateway_requests(tmp_path, T0, T0 + 100)
    assert requests is not None and len(requests) == 13
    # The histogram of the same window says p95 400 ms, interpolated inside its 0.1-0.5 s bucket.
    assert first_byte_ms(GATEWAY_BEFORE, GATEWAY_AFTER, "realtime", requests) == {
        "n": 10,
        "mean": 550.0,
        "p50": 550.0,
        "p95": 955.0,
        "p99": 991.0,
        "source": "log",
    }
    assert first_byte_ms("", "", "background", requests)["p50"] == 2500.0
    assert gateway_requests(tmp_path / "missing", T0, T0 + 100) is None
    (tmp_path / "gateway.jsonl").write_text("lh-gateway serving\n")  # a gateway that logs text
    assert gateway_requests(tmp_path, T0, T0 + 100) is None


ENGINE = """\
vllm:num_preemptions_total{{engine="0",model_name="m"}} {preemptions}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_compute"}} {computed}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_cache_hit"}} {gpu}
vllm:admission_rejections_total{{model_name="m",reason="max_num_queued_tokens"}} {refused}
"""


def test_the_engine_counts_cover_the_window_and_null_what_it_does_not_export() -> None:
    before = ENGINE.format(preemptions=2, computed=100, gpu=1000, refused=0)
    after = ENGINE.format(preemptions=5, computed=400, gpu=9000, refused=7)
    assert engine_counts(before, after) == {
        "preemptions": 3,
        "prompt_tokens_by_source": {"local_compute": 300, "local_cache_hit": 8000},
        "refused": {"max_num_queued_tokens": 7},
    }
    sglang = "sglang:num_running_reqs 3\n"  # SGLang exports none of vLLM's counters
    assert engine_counts(sglang, sglang) == {
        "preemptions": None,
        "prompt_tokens_by_source": None,
        "refused": None,
    }


def point(
    root: Path,
    name: str,
    arm: str,
    tokens_per_s: float | None,
    p95: float,
    status: str = "complete",
) -> Path:
    """A run directory as lh-bench e2 run leaves it."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    settings = {
        "arm": arm,
        "label": f"{name} label",
        "router": None,
        "voice": {"callers": 2, "preemptive_generation": True},
        "agents": {"sessions": 0 if tokens_per_s is None else 16},
        "engine": {"variant": "frozen", "flags": ["--max-num-seqs", "12"]},
    }
    run = {"kind": "e2-point", "name": name, "status": status, "arm": settings}
    run["window"] = {"start": T0, "end": T0 + 100}
    (run_dir / "run.json").write_text(json.dumps(run))
    bg = None
    if tokens_per_s is not None:
        bg = {
            "tokens_per_s": tokens_per_s,
            "requests_per_s": tokens_per_s / 1000,
            "ttft_ms": {"p50": 140.0, "p99": 8000.0},
        }
    summary = {
        "name": name,
        "arm": arm,
        "label": f"{name} label",
        "voice": {
            "turns": 50,
            "errors": 0,
            "v2v_ms": {"n": 60, "p50": 1600.0, "p95": p95, "p99": p95 + 500},
            "barge_in_to_silence_ms": {"n": 9, "p95": 650.0},
        },
        "background": bg,
        "gateway": {
            "realtime_first_byte_ms": {"n": 60, "p95": 120.0},
            "shed": None,
            "refused": 2,
            "resubmitted": None,
            "background_tokens_per_s": tokens_per_s,
        },
        "engine": {"preemptions": 4, "refused": {"max_num_queued_tokens": 1}},
    }
    if status == "complete":
        (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def test_each_arm_has_its_own_frontier(tmp_path: Path) -> None:
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s8", "2", 3000, 2000),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
        point(tmp_path, "a2-s24", "2", 4500, 3000),  # less goodput at a worse p95
        point(tmp_path, "a7a-seqs12", "7a", 4000, 3100),  # beaten by arm 2, but on its own frontier
        point(tmp_path, "a2-failed", "2", None, 0, status="failed"),
    ]
    rows, incomplete = load_points(dirs)
    assert incomplete == [tmp_path / "a2-failed"]
    assert [(r["point"], r["frontier"]) for r in rows] == [
        ("a1-voice", True),
        ("a2-s8", True),
        ("a2-s24", False),
        ("a2-s16", True),
        ("a7a-seqs12", True),
    ]
    assert rows[0]["background_tokens_per_s"] == 0.0 and rows[0]["agent_ttft_p50_ms"] is None
    assert rows[1]["engine"] == "frozen --max-num-seqs 12"


def test_ties_are_not_dominated() -> None:
    rows = [
        {"arm": "4", "background_tokens_per_s": 100.0, "v2v_p95_ms": 2000.0},
        {"arm": "4", "background_tokens_per_s": 100.0, "v2v_p95_ms": 2000.0},
        {"arm": "4", "background_tokens_per_s": 50.0, "v2v_p95_ms": None},
    ]
    mark_frontier(rows)
    assert [r.get("frontier") for r in rows] == [True, True, None]


def test_the_report_writes_the_table_and_the_csv(tmp_path: Path, capsys) -> None:
    dirs = [
        point(tmp_path / "runs", "a2-s8", "2", 3000, 2000),
        point(tmp_path / "runs", "a10-mps", "10", 2000, 2100),
        point(tmp_path / "runs", "a2-failed", "2", None, 0, status="failed"),
    ]
    out = tmp_path / "report"
    assert main(["e2", "report", *map(str, dirs), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "a2-failed: the point did not complete" in captured.err
    assert str(out / "frontier.md") in captured.out
    with (out / "frontier.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(COLUMNS)  # no point has the optional columns
    assert [r["point"] for r in rows] == ["a2-s8", "a10-mps"]  # arm 2 before arm 10
    assert rows[0]["v2v_p95_ms"] == "2000" and rows[0]["frontier"] == "True"
    assert rows[0]["refused"] == "3"  # 2 by the gateway's admission point, 1 by the engine
    assert rows[0]["gateway_background_tokens_per_s"] == "3000"
    # A summary from before the request log took the first byte from the histogram.
    assert (rows[0]["realtime_first_byte_p95_ms"], rows[0]["realtime_first_byte_source"]) == (
        "120.0",
        "histogram",
    )
    table = (out / "frontier.md").read_text()
    assert "## Arm 2" in table and "## Arm 10" in table
    assert "| a2-s8: a2-s8 label | • | 16 | 3,000 | 3.00 | 1,600 | 2,000 | 2,500 |" in table
    assert "| 0 of 50 | 140 / 8,000 | ~120 | 4 | — / 3 / — |" in table
    assert "Voice turn stages" not in table and "SM clock" not in table
    assert main(["e2", "report", str(dirs[2]), "--out", str(out)]) == 1


def test_the_report_takes_a_request_log_fetched_after_the_run(tmp_path: Path) -> None:
    run_dir = point(tmp_path, "a2-s8", "2", 3000, 2000)
    (run_dir / "gateway.jsonl").write_text(REQUESTS + "\n")  # backfilled from the journal
    [row], _ = load_points([run_dir])
    assert (row["realtime_first_byte_p95_ms"], row["realtime_first_byte_source"]) == (955.0, "log")
    assert "| 140 / 8,000 | 955 | 4 |" in write_report([row], tmp_path / "report")[1].read_text()


def smi(at: float, clock: str, temperature: str) -> str:
    """One sample of campaign.sh's nvidia-smi query, timestamped in UTC."""
    stamp = datetime.fromtimestamp(at, UTC).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    return f"{stamp}, {clock}, 13801 MHz, {temperature}, 150.21 W, 0x0000000000000000"


def test_the_report_shows_stages_and_clocks_only_for_points_that_have_them(tmp_path: Path) -> None:
    dirs = [point(tmp_path, "a2-s8", "2", 3000, 2000), point(tmp_path, "a2-s16", "2", 5000, 2600)]
    summary = json.loads((dirs[1] / "summary.json").read_text())
    stages = {s: {"n": 50, "p50": 100.0 * n, "p95": 150.0 * n} for n, s in enumerate(WATERFALL, 1)}
    events = {"sent": 300, "dropped": 0, "failed": 1}
    summary["worker"] = {"sessions": 2, "turns": 50, "waterfall": stages, "events": events}
    (dirs[1] / "summary.json").write_text(json.dumps(summary))
    samples = [
        smi(T0 - 10, "1530 MHz", "40"),  # before the calls: not counted
        smi(T0 + 10, "2805 MHz", "61"),
        smi(T0 + 20, "2745 MHz", "70"),
        smi(T0 + 30, "2685 MHz", "71"),
        smi(T0 + 40, "[N/A]", "[N/A]"),
        "2027/01/15 08:0",  # cut short when the sampling stopped
    ]
    (dirs[1] / "gpu.csv").write_text("\n".join(samples) + "\n")
    rows, _ = load_points(dirs)
    out = tmp_path / "report"
    write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        written = {r["point"]: r for r in csv.DictReader(f)}
    assert list(written["a2-s8"]) == [*COLUMNS, *WORKER_COLUMNS, *GPU_COLUMNS]
    assert {c: written["a2-s8"][c] for c in GPU_COLUMNS} == dict.fromkeys(GPU_COLUMNS, "")
    sixteen = written["a2-s16"]
    assert (sixteen["transcript_p50_ms"], sixteen["e2e_p95_ms"], sixteen["events_failed"]) == (
        "100.0",
        "1050.0",
        "1",
    )
    assert (sixteen["gpu_sm_clock_p50_mhz"], sixteen["gpu_temperature_max_c"]) == ("2745", "71")
    table = (out / "frontier.md").read_text()
    assert "| SM clock p50 | Max temp |" in table and "| — / 3 / — | 2,745 | 71 |" in table
    assert "| — / 3 / — | — | — |" in table  # a2-s8, sampled by no campaign
    assert "| a2-s16 | 100 / 150 | 200 / 300 | 300 / 450 |" in table
    assert "| 700 / 1,050 | 300 / 0 / 1 |" in table
    assert "| a2-s8 | — / — |" in table


def rewrite(run_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    change(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary))


def spread(median: str, low: str, high: str) -> str:
    """A repeated configuration's cell in frontier.md: its median, then its runs' range."""
    return f"{median} ({low}\N{EN DASH}{high})"


def test_repeats_of_a_point_are_one_configuration_placed_at_their_medians(tmp_path: Path) -> None:
    # Three runs of disc-frozen; one of them alone would beat disc-wide, their medians do not.
    dirs = [
        point(tmp_path / "w2c", "disc-frozen", "4", 5000, 2000),
        point(tmp_path / "r1", "disc-frozen", "4", 3000, 4000),
        point(tmp_path / "r2", "disc-frozen", "4", 4000, 3000),
        point(tmp_path / "w2c", "disc-wide", "4", 4500, 2500),
        point(tmp_path / "w2c", "disc-cap2", "6", 2000, 2200),
    ]
    rewrite(dirs[1], lambda s: s["voice"].update(errors=3))
    rewrite(dirs[2], lambda s: s["voice"].update(errors=2))
    stages = {s: {"n": 50, "mean": 100.0, "p50": 100.0, "p95": 150.0} for s in WATERFALL}
    rewrite(dirs[0], lambda s: s.update(worker={"turns": 50, "waterfall": stages}))
    rows, _ = load_points(dirs)
    assert [(r["point"], r["campaign"], r["frontier"]) for r in rows] == [
        ("disc-frozen", "r1", False),  # the configuration's runs together, at its median goodput
        ("disc-frozen", "r2", False),
        ("disc-frozen", "w2c", False),
        ("disc-wide", "w2c", True),
        ("disc-cap2", "w2c", True),
    ]
    [frozen, wide, cap2] = configurations(rows)
    assert (frozen["background_tokens_per_s"], frozen["v2v_p95_ms"], frozen["voice_errors"]) == (
        4000,
        3000,
        2,
    )
    assert (len(wide["runs"]), wide["v2v_p95_ms"], cap2["frontier"]) == (1, 2500, True)
    out = tmp_path / "report"
    write_report(rows, out)
    with (out / "frontier.csv").open() as f:
        assert [r["campaign"] for r in csv.DictReader(f)] == ["r1", "r2", "w2c", "w2c", "w2c"]
    table = (out / "frontier.md").read_text()
    assert "lists each run by its campaign, then the median and range" in table
    # Each run by its campaign, every one of them kept, then the configuration's medians and range.
    assert "| disc-frozen (r1) |  | 16 | 3,000 | 3.00 | 1,600 | 4,000 | 4,500 | 3 of 50 |" in table
    assert "| disc-frozen (w2c) |  | 16 | 5,000 | 5.00 | 1,600 | 2,000 | 2,500 | 0 of 50 |" in table
    medians = (
        f"| disc-frozen: disc-frozen label; median (range) of 3 runs |  | 16 "
        f"| {spread('4,000', '3,000', '5,000')} | | {spread('1,600', '1,600', '1,600')} "
        f"| {spread('3,000', '2,000', '4,000')} | {spread('3,500', '2,500', '4,500')} "
        f"| {spread('2', '0', '3')} | | ~{spread('120', '120', '120')} | | |"
    )
    lines = table.splitlines()
    assert lines[lines.index(medians) - 1].startswith("| disc-frozen (w2c) |")  # after its runs
    assert "| disc-wide: disc-wide label | • | 16 | 4,500 |" in table  # measured once: as before
    assert "| disc-frozen (w2c) | 100 / 150 |" in table and "| disc-wide | — / — |" in table


def test_a_median_between_two_runs_keeps_its_half(tmp_path: Path) -> None:
    dirs = [point(tmp_path / c, "disc-frozen", "4", 4000, 3000) for c in ("w2c", "r1")]
    rewrite(dirs[1], lambda s: s["voice"].update(errors=1))
    rows, _ = load_points(dirs)
    table = write_report(rows, tmp_path / "report")[1].read_text()
    assert f"| {spread('0.5', '0', '1')} |" in table
    assert f"| • | 16 | {spread('4,000', '4,000', '4,000')} |" in table


def turn_lines(run_dir: Path, stages: dict[str, float], turns: int, v2v: list[dict]) -> None:
    """A run directory's turn lines: the worker's stage means over ``turns`` turns, and the callers'
    turns.jsonl."""
    waterfall = {stage: {"n": turns, "mean": mean} for stage, mean in stages.items()}
    rewrite(run_dir, lambda s: s.update(worker={"turns": turns, "waterfall": waterfall}))
    (run_dir / "bots").mkdir()
    (run_dir / "bots" / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in v2v))


def answered(*v2v_ms: float) -> list[dict[str, Any]]:
    return [
        {"caller": 0, "turn": n, "v2v_ms": ms, "barge_in": False} for n, ms in enumerate(v2v_ms)
    ]


def test_the_stage_waterfall_stacks_each_configurations_stages_to_its_mean_v2v(
    tmp_path: Path, capsys
) -> None:
    frozen = [point(tmp_path / c, "disc-frozen", "4", 3700, 10800) for c in ("w2c", "r1")]
    barged = {"caller": 1, "turn": 0, "v2v_ms": 5000.0, "barge_in": True}
    turn_lines(
        frozen[0],
        {"transcript_ms": 790, "end_of_turn_ms": 800, "llm_ttft_ms": 2500}
        | {"turn_to_first_sentence_ms": 3000, "tts_first_audio_ms": 300},
        4,
        [*answered(4000, 4400), barged | {"v2v_after_barge_in_ms": 4600}, {"error": "timeout"}],
    )
    turn_lines(
        frozen[1],
        {"transcript_ms": 810, "end_of_turn_ms": 820, "llm_ttft_ms": 2100}
        | {"turn_to_first_sentence_ms": 2600, "tts_first_audio_ms": 280},
        4,
        answered(3800, 4200, 4100, 4300),
    )
    cap2 = point(tmp_path / "w2c", "disc-cap2", "6", 3500, 8500)
    turn_lines(
        cap2,
        {"transcript_ms": 820, "end_of_turn_ms": 830, "llm_ttft_ms": 1500}
        | {"turn_to_first_sentence_ms": 2000, "tts_first_audio_ms": 250},
        2,
        answered(3300, 3500),
    )
    run = json.loads((cap2 / "run.json").read_text())
    run["arm"]["voice"]["preemptive_generation"] = False
    (cap2 / "run.json").write_text(json.dumps(run))
    voice = point(tmp_path / "w2", "a1-voice", "1", None, 1760)  # no turn lines
    rows, _ = load_points([*frozen, cap2, voice])
    [disc, capped] = stage_waterfall(rows)
    # Each stage's mean over both runs' turns; the V2V samples pooled as the callers pool them.
    assert {k: disc[k] for k in ("runs", "replies", *(s for s, _ in STACKED))} == {
        "runs": 2,
        "replies": 8,
        "end_of_turn_ms": 810,
        "turn_to_first_sentence_ms": 2800,
        "tts_first_audio_ms": 290,
    }
    assert (disc["v2v_mean_ms"], disc["v2v_p50_ms"], disc["v2v_p95_ms"]) == (4300, 4250, 4860)
    assert disc["transport_ms"] == 400  # the rest of the mean V2V
    # Under preemptive generation the LLM request starts at the final transcript.
    assert (disc["llm_request_ms"], disc["llm_ttft_ms"]) == (800, 2300)
    assert (capped["llm_request_ms"], capped["transport_ms"]) == (830, 320)
    out = tmp_path / "report"
    write_report(rows, out)
    table = (out / "frontier.md").read_text()
    assert (
        "| disc-frozen: disc-frozen label | 2 | 8 | 810 | 2,800 | 290 | 400 | 4,300 | 4,250 | 4,860 "
        "| 2,300 |"
    ) in table
    assert "| disc-cap2: disc-cap2 label | 1 | 2 | 830 | 2,000 | 250 | 320 | 3,400 |" in table
    assert "a1-voice: a1-voice label | 1 |" not in table
    with pytest.raises(ValueError, match="no turn lines of a1-voice"):
        write_report(rows, out, ["disc-cap2", "a1-voice"])
    args = ["e2", "report", *map(str, [*frozen, cap2, voice]), "--out", str(out)]
    assert main([*args, "--waterfall", "a1-voice"]) == 1
    assert "lh-bench e2: the report has no turn lines of a1-voice" in capsys.readouterr().err
    if importlib.util.find_spec("matplotlib"):
        assert main([*args, "--waterfall", "disc-cap2"]) == 0
        figure = (out / "waterfall.svg").read_text()
        assert "disc-cap2" in figure and "disc-frozen" not in figure
        write_report(rows, out)
        figure = (out / "waterfall.svg").read_text()
        assert "disc-frozen (2 runs)" in figure and "a1-voice" not in figure


def whole(run_dir: Path) -> Path:
    """A run directory with everything lh-bench e2 run and campaign.sh leave in one."""
    turn_lines(
        run_dir,
        {"transcript_ms": 790, "end_of_turn_ms": 800, "llm_ttft_ms": 2500}
        | {"turn_to_first_sentence_ms": 3000, "tts_first_audio_ms": 300, "e2e_ms": 4110},
        2,
        answered(4400, 4500),
    )
    (run_dir / "gateway.jsonl").write_text(REQUESTS + "\n")  # exact first bytes; p95 955 ms
    samples = [smi(T0 + 10, "2805 MHz", "61"), smi(T0 + 20, "2745 MHz", "71")]
    (run_dir / "gpu.csv").write_text("\n".join(samples) + "\n")
    (run_dir / "voice-worker.log").write_text("registered worker\n")
    (run_dir / "arm.toml").write_text('arm = 4\nlabel = "x"\n')
    (run_dir / "engine.json").write_text(json.dumps({"model_config": {"hf_token": None}}))
    (run_dir / "bots" / "summary.json").write_text(json.dumps({"turns": 2, "errors": 0}))
    callers = {"config": {"api_key": "lh2a17802e4561", "api_secret": "<redacted>"}}
    (run_dir / "bots" / "run.json").write_text(json.dumps(callers))
    for name in ("aiperf/profile_export.jsonl", "metrics/gateway-end.prom", "bots.log"):
        (run_dir / name).parent.mkdir(exist_ok=True)
        (run_dir / name).write_text("raw\n")
    return run_dir


def test_a_compact_record_gives_the_report_of_its_run_directory(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    dirs = [
        whole(point(runs / "w2c", "disc-frozen", "4", 3700, 10800)),
        whole(point(runs / "r1", "disc-frozen", "4", 3500, 9800)),
        point(runs / "w2c", "disc-cap2", "6", 3400, 8500),  # a summary from before the logs
        point(runs / "w2c", "disc-cap8", "6", None, 0, status="failed"),
    ]
    out = tmp_path / "records"
    assert main(["e2", "export", *map(str, dirs), "--out", str(out)]) == 0
    records = [out / "w2c" / "disc-frozen", out / "r1" / "disc-frozen", out / "w2c" / "disc-cap2"]
    printed = capsys.readouterr()
    assert printed.out.split() == [str(record) for record in records]  # CAMPAIGN/POINT
    assert "disc-cap8: the point did not complete" in printed.err
    record = records[0]
    kept = sorted(str(p.relative_to(record)) for p in record.rglob("*") if p.is_file())
    assert kept == sorted([*RECORD, "summary.json"])
    assert not any("lh2a17802e4561" in p.read_text() for p in out.rglob("*") if p.is_file())
    summary = json.loads((record / "summary.json").read_text())
    assert summary["gateway"]["realtime_first_byte_ms"]["p95"] == 955.0  # exact, from the log
    assert summary["gateway"]["background_first_byte_ms"]["source"] == "log"
    assert summary["gpu"] == {"sm_clock_p50_mhz": 2775, "temperature_max_c": 71}
    full, _ = load_points(dirs[:3])
    compact, _ = load_points(records)
    assert compact == full
    assert full[0]["campaign"] == "r1" and full[0]["gpu_sm_clock_p50_mhz"] == 2775
    reports = [write_report(rows, tmp_path / name) for name, rows in (("a", full), ("b", compact))]
    for a, b in zip(*reports, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name  # the figures too: same bytes every time
    assert main(["e2", "export", str(dirs[3]), "--out", str(out)]) == 1
    assert "no complete point to export" in capsys.readouterr().err


def unredacted(data: Any) -> list[str]:
    """The credential settings (e2.SECRETS) in a JSON document that hold anything but a redaction."""
    if isinstance(data, list):
        return [key for item in data for key in unredacted(item)]
    if not isinstance(data, dict):
        return []
    found = [
        key
        for key, value in data.items()
        if key in SECRETS and value and value not in ("<redacted>", ["<redacted>"] * len(value))
    ]
    return found + [key for value in data.values() for key in unredacted(value)]


def test_the_committed_records_give_the_report_and_hold_no_credential(tmp_path: Path) -> None:
    assert unredacted({"tenants": {"a": {"keys_sha256": ["<sha256>"]}}}) == ["keys_sha256"]
    dirs = sorted(run.parent for run in RESULTS.glob("*/*/run.json"))  # as make reproduce-e2 does
    assert dirs
    rows, incomplete = load_points(dirs)
    assert len(rows) == len(dirs) and not incomplete
    for run_dir in dirs:
        kept = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
        assert kept == sorted([*RECORD, "summary.json"]), run_dir
        for path in run_dir.rglob("*.json"):
            assert not unredacted(json.loads(path.read_text())), path
    write_report(rows, tmp_path)
    assert (tmp_path / "frontier.md").read_text().startswith("# E2 frontier")


def test_the_report_reads_the_lines_the_voice_worker_writes() -> None:
    source = VOICE_EVENTS.read_text()
    [stages] = [
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets)
    ]
    assert stages == WATERFALL
    assert f'logging.getLogger("{TURNS_LOGGER}")' in source
    # The sessions' end lines come from the module's own logger.
    assert SESSIONS_LOGGER == "leaseholder_voice.events" and "logging.getLogger(__name__)" in source
    assert '"session ended"' in source


def test_the_figure_needs_matplotlib(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    dirs = [
        point(tmp_path, "a1-voice", "1", None, 1760),
        point(tmp_path, "a2-s8", "2", 3000, 2000),
        point(tmp_path, "a2-s16", "2", 5000, 2600),
    ]
    rows, _ = load_points(dirs)
    written = write_report(rows, tmp_path / "report")
    assert (tmp_path / "report" / "frontier.svg") in written
    figure = (tmp_path / "report" / "frontier.svg").read_text()
    assert "<svg" in figure and "whiskers" not in figure
    repeated = point(tmp_path / "r1", "a2-s16", "2", 4000, 3000)
    rows, _ = load_points([*dirs, repeated])
    write_report(rows, tmp_path / "report")
    assert "whiskers span repeated runs" in (tmp_path / "report" / "frontier.svg").read_text()


def test_a_point_summary_reads_its_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "a1"
    (run_dir / "bots").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    bots = {
        "turns": 50,
        "errors": 1,
        "v2v_ms": {"n": 59, "p50": 1600.0, "p95": 1760.0, "p99": 1890.0},
        "barge_in_to_silence_ms": {"n": 9, "p50": 630.0, "p95": 690.0, "p99": 700.0},
        "barge_ins": 9,
        "caller_max_lateness_ms": 0.0,
    }
    (run_dir / "bots" / "summary.json").write_text(json.dumps(bots))
    for when, text in (("start", GATEWAY_BEFORE), ("end", GATEWAY_AFTER)):
        (run_dir / "metrics" / f"gateway-{when}.prom").write_text(text)
        (run_dir / "metrics" / f"engine-{when}.prom").write_text("")
    summary = summarize_point(run_dir, T0, T0 + 321.04, agents=False)
    assert summary["window_s"] == 321.0 and summary["background"] is None
    assert summary["gateway"]["background_tokens_per_s"] == 317.7  # 102,000 tokens over the window
    assert summary["voice"] == {k: bots[k] for k in summary["voice"]}
    assert summary["engine"]["preemptions"] is None
    bots["v2v_ms"] = {"n": 0}
    (run_dir / "bots" / "summary.json").write_text(json.dumps(bots))
    with pytest.raises(ValueError, match="measured no voice-to-voice latency"):
        summarize_point(run_dir, T0, T0 + 321, agents=False)
