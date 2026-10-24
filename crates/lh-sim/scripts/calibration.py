"""The simulator against the measured runs: every number of the lh-sim README's sections "Calibration against the GPU
operating points" and "Validation against W0", and of docs/BENCHMARKS.md's table of the W2 operating points.

    uv run --no-project python crates/lh-sim/scripts/calibration.py report EXPORTS CORPUS LH_SIM OUT

EXPORTS holds one directory per measured run (`RUNS`), each with AIPerf's `aiperf/profile_export.jsonl` and the
engine's /metrics before and after the run, `before.prom` and `after.prom`. CORPUS is the 32x corpus and LH_SIM the
release binary. The simulator's outputs go to OUT, and `OUT/calibration.md` holds the tables and the figures the
sections quote. `make reproduce-calibration` runs it.

    uv run --no-project python crates/lh-sim/scripts/calibration.py export RUNS_DIR EXPORTS

copies those files of each run from the harness's run directories under RUNS_DIR to EXPORTS, with the engine's local
path to the model in the snapshots replaced by the model's name, for publishing.

Percentiles interpolate between ranks, as numpy's and lh-sim's do. Shares of prompt tokens over a run's profiling
phase are the engine's counters less what its warm-up requests counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

# W2's operating points, each on a freshly started engine: the calibration's data. W0, the validation's. The repeats
# on a warm engine: 16 sessions after the 8-session point, 24 after the 16-session one.
POINTS = (
    ("8 sessions", 8, "w2-point-s8"),
    ("16 sessions", 16, "w2-point-s16-fresh"),
    ("24 sessions", 24, "w2-point-s24-fresh"),
)
W0 = "w0-s16"
WARM = (("16 sessions", 16, "w2-point-s16"), ("24 sessions", 24, "w2-point-s24"))
RUNS = (*(run for _, _, run in POINTS), W0, *(run for _, _, run in WARM))
FILES = ("aiperf/profile_export.jsonl", "before.prom", "after.prom")
# The seeds whose spread the validation states.
SEEDS = range(11, 16)
FIT = Path(__file__).with_name("fit_step_cost.py")
SUBSET = Path(__file__).resolve().parents[1] / "testdata" / "weka-x32-sub16"
# A model path in the engine's labels, as vLLM serves a Hugging Face snapshot.
SNAPSHOT = re.compile(r'"[^"]*/models--(?P<org>[^/"]+?)--(?P<name>[^/"]+)/snapshots/[0-9a-f]+/?"')
# The README's minus sign and range dash.
MINUS, DASH = "\u2212", "\u2013"


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    rank = p / 100 * (len(v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (rank - lo) * (v[hi] - v[lo])


def p3(values: list[float]) -> list[float]:
    return [pct(values, p) for p in (50, 90, 99)]


def counters(path: Path) -> dict[str, float]:
    """vLLM's counters in a /metrics snapshot, summed over labels but the prompt tokens' source."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+(\S+)$", line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        source = re.search(r'source="([^"]+)"', labels)
        key = name + (f"[{source.group(1)}]" if source else "")
        out[key] = out.get(key, 0.0) + value
    return out


def records(path: Path) -> list[dict[str, Any]]:
    """AIPerf's records, one row each."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta, metrics = r["metadata"], r["metrics"]

        def value(name: str, default: Any = 0.0, metrics: dict[str, Any] = metrics) -> Any:
            return metrics[name]["value"] if name in metrics else default

        rows.append(
            {
                "warmup": meta["benchmark_phase"] == "warmup",
                "play": meta["root_correlation_id"],
                "turn": meta["turn_index"],
                "start_ms": meta["request_start_ns"] / 1e6,
                "ttft": value("time_to_first_token", None),
                "latency": value("request_latency", None),
                "itl": value("inter_token_latency", None),
                "prompt": value("usage_prompt_tokens"),
                "cached": value("usage_prompt_cache_read_tokens"),
            }
        )
    return rows


def measured(run: Path) -> dict[str, Any]:
    """A measured run: its profiling requests' latencies and the engine's prompt-token split."""
    rows = records(run / FILES[0])
    prof = [r for r in rows if not r["warmup"]]
    warm = [r for r in rows if r["warmup"]]
    before, after = counters(run / FILES[1]), counters(run / FILES[2])

    def delta(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    total, gpu = (
        delta("vllm:prompt_tokens_total"),
        delta("vllm:prompt_tokens_cached_by_source_total[device]"),
    )
    cpu = delta("vllm:prompt_tokens_cached_by_source_total[host]")
    # A fresh engine computes the warm-up's prompts, but for the few tokens one warm-up request hits from another.
    prompt = total - sum(r["prompt"] for r in warm)
    gpu_prof = gpu - sum(r["cached"] for r in warm)
    return {
        "requests": len(prof),
        "plays": len({r["play"] for r in rows}),
        "warmup": len(warm),
        "ttft": p3([r["ttft"] for r in prof if r["ttft"] is not None]),
        "resume": p3([r["ttft"] for r in prof if r["ttft"] is not None and r["turn"] > 0]),
        "itl50": pct([r["itl"] for r in prof if r["itl"] is not None], 50),
        "run_shares": [100 * gpu / total, 100 * cpu / total, 100 * (total - gpu - cpu) / total],
        "prompt": prompt,
        "shares": [
            100 * gpu_prof / prompt,
            100 * cpu / prompt,
            100 * (prompt - gpu_prof - cpu) / prompt,
        ],
        "cached_share": 100 * sum(r["cached"] for r in prof) / sum(r["prompt"] for r in prof),
        "mean_ttft": statistics.mean(r["ttft"] for r in prof if r["ttft"] is not None),
        "mean_latency": statistics.mean(r["latency"] for r in prof if r["latency"] is not None),
        "preemptions": round(delta("vllm:num_preemptions_total")),
    }


class Sim:
    """The release binary on the corpus, its outputs in `out`."""

    def __init__(self, binary: Path, corpus: Path, out: Path) -> None:
        self.binary, self.corpus, self.out = binary, corpus, out

    def run(self, name: str, *options: str, corpus: Path | None = None) -> dict[str, Any]:
        """`lh-sim run` of `lru-cpu`: its summary, and its per-request rows."""
        summary, requests = self.out / f"{name}.json", self.out / f"{name}-requests.csv"
        cmd = [self.binary, "run", corpus or self.corpus, "--arms", "lru-cpu", *options]
        subprocess.run(
            [*map(str, cmd), "--json", str(summary), "--requests", str(requests)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(summary.read_text())["results"][0]
        result["rows"] = list(csv.DictReader(requests.open()))
        return result

    def replay_log(self, name: str, export: Path, *options: str) -> list[dict[str, str]]:
        """`lh-sim replay-log` of a measured run: one row per record."""
        path = self.out / f"{name}.csv"
        cmd = [self.binary, "replay-log", self.corpus, export / FILES[0], *options, "--csv", path]
        subprocess.run(list(map(str, cmd)), check=True, stdout=subprocess.DEVNULL)
        return list(csv.DictReader(path.open()))


def fit(runs: list[tuple[Path, Path]], out: Path) -> tuple[Path, str]:
    """fit_step_cost.py on `runs` (each an export and its replay-log CSV): the fit's path and its report."""
    args = [str(p) for run in runs for p in run]
    done = subprocess.run(
        [sys.executable, str(FIT), *args], check=True, capture_output=True, text=True
    )
    out.write_text(done.stdout)
    return out, done.stderr


def sim_row(s: dict[str, Any]) -> tuple[int, list[float], list[float]]:
    sh = s["shares"]
    return (
        s["requests"],
        [s["resume_ttft_ms"][k] for k in ("p50", "p90", "p99")],
        [100 * sh[k] for k in ("gpu", "cpu", "computed")],
    )


def n(x: float) -> str:
    """A whole number with thousands separators."""
    return f"{x:,.0f}"


def signed(x: float, digits: int = 0) -> str:
    return f"{x:+.{digits}f}".replace("-", MINUS)


def errors(sim: list[float], meas: list[float]) -> str:
    return " / ".join(signed(100 * (s / m - 1)) for s, m in zip(sim, meas, strict=True))


def shares(values: list[float], digits: int = 1) -> str:
    return " / ".join(f"{v:.{digits}f}" for v in values)


def resume_kind(row: dict[str, str]) -> str:
    """How the engine model found a request's prompt: restored, a GPU hit, or computed (more than a step's 512)."""
    if int(row["cpu_tokens"]):
        return "restore"
    return "hit" if int(row["computed_tokens"]) <= 512 else "computed"


def profiling(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["warmup"] == "false"]


def cache_model(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    """The engine model on a run's own timeline: its GPU and CPU shares of the profiling requests' prompts, and the
    mean absolute difference between its cached share and the engine's, per request, in points."""
    prof = profiling(rows)
    prompt = sum(int(r["prompt_tokens"]) for r in prof)
    gpu, cpu = (100 * sum(int(r[k]) for r in prof) / prompt for k in ("gpu_tokens", "cpu_tokens"))
    error = statistics.mean(
        abs(
            (int(r["gpu_tokens"]) + int(r["cpu_tokens"])) / max(int(r["prompt_tokens"]), 1)
            - int(r["measured_cached_tokens"]) / max(int(r["measured_prompt_tokens"]), 1)
        )
        for r in prof
    )
    return gpu, cpu, 100 * error


def overlapping_restores(export: Path, rows: list[dict[str, str]]) -> dict[int, list[float]]:
    """A timed replay's restores of resumed turns, by how many other restores overlapped their measured wait for the
    first token (6 for six or more): the residuals, simulated less measured, in ms."""
    reqs = []
    for rec, row in zip(records(export / FILES[0]), rows, strict=True):
        if not row["ttft_ms"]:
            continue
        start = rec["start_ms"]
        reqs.append(
            (
                start,
                start + float(row["measured_ttft_ms"]),
                int(row["cpu_tokens"]),
                float(row["ttft_ms"]) - float(row["measured_ttft_ms"]),
                row["warmup"] == "false" and int(row["turn"]) > 0,
            )
        )
    out: dict[int, list[float]] = {}
    for start, end, cpu, residual, resumed in reqs:
        if not (resumed and cpu):
            continue
        others = sum(1 for s, e, c, _, _ in reqs if c and s < end and e > start) - 1
        out.setdefault(min(others, 6), []).append(residual)
    return out


def report(exports: Path, corpus: Path, binary: Path, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    sim = Sim(binary, corpus, out)
    meas = {run: measured(exports / run) for run in RUNS}
    lines = ["# The simulator against the measured runs", ""]

    # The step model, fitted on the three points, and each point left out in turn.
    logs = {run: sim.replay_log(f"{run}-replay-log", exports / run) for _, _, run in POINTS}
    pairs = {run: (exports / run / FILES[0], out / f"{run}-replay-log.csv") for _, _, run in POINTS}
    steps, fitted = fit(list(pairs.values()), out / "steps.json")
    left_out = {
        run: fit([pairs[o] for o in pairs if o != run], out / f"steps-without-{run}.json")[0]
        for run in pairs
    }

    lines += [
        "## Calibration against the GPU operating points",
        "",
        "| Point | | Requests | Resume TTFT p50 / p90 / p99 (ms) | GPU / CPU / computed (%) |",
        "|---|---|---|---|---|",
    ]
    contention = {}
    for label, sessions, run in POINTS:
        m = meas[run]
        lines.append(
            f"| {label} | measured | {n(m['requests'])} | {' / '.join(map(n, m['resume']))} | {shares(m['shares'])} |"
        )
        for model, options in (
            ("idle (v0)", ["--service", "idle"]),
            ("contention", []),
            ("leave-one-out", ["--step-costs", str(left_out[run])]),
        ):
            s = sim.run(f"{run}-{model.split()[0]}", "--concurrency", str(sessions), *options)
            requests, resume, sim_shares = sim_row(s)
            count = (
                n(requests)
                if model == "leave-one-out"
                else f"{n(requests)} ({signed(100 * (requests / m['requests'] - 1), 1)}%)"
            )
            lines.append(
                f"| | {model} | {count} | {' / '.join(map(n, resume))} ({errors(resume, m['resume'])}%) | {shares(sim_shares)} |"
            )
            if model == "contention":
                contention[sessions] = resume
    lines.append("")

    # The repeats against the fresh points and the contention model.
    s16 = [meas[run]["resume"] for run in (POINTS[1][2], WARM[0][2], W0)]
    ranges = [(min(v[i] for v in s16), max(v[i] for v in s16)) for i in range(3)]
    model_errors = [
        (100 * (contention[16][i] / lo - 1), 100 * (contention[16][i] / hi - 1))
        for i, (lo, hi) in enumerate(ranges)
    ]
    warm24 = meas[WARM[1][2]]["resume"]
    lines += [
        "**Repeats.** Three 16-session runs (fresh, after the 8-session run, and W0), resume TTFT: "
        + ", ".join(f"{n(lo)}{DASH}{n(hi)} ms" for lo, hi in ranges)
        + " at p50 / p90 / p99; the contention model is "
        + ", ".join(f"{signed(a)} to {signed(b)}%" for a, b in model_errors)
        + ". The 24-session run after the "
        f"16-session one: {' / '.join(map(n, warm24))} ms ({errors(contention[24], warm24)}%).",
        "",
    ]

    lines.append(
        "**The cache model** on each point's own timeline (`replay-log`), against the engine's counters:"
    )
    for label, _, run in POINTS:
        gpu, cpu, error = cache_model(logs[run])
        lines.append(
            f"- {label}: GPU share {gpu:.2f}% against {meas[run]['shares'][0]:.2f}% ({signed(gpu - meas[run]['shares'][0], 2)} "
            f"points), per-request cached share within {error:.2f} points on average"
        )
    lines.append("")

    lines.append(
        "**Open loop** (`replay-log --timed`), resume TTFT p50 / p90 / p99 against the measured, and the median "
        "residual (simulated less measured) by how the engine model found the prompt:"
    )
    timed = {}
    for label, _, run in POINTS:
        rows = timed[run] = sim.replay_log(f"{run}-timed", exports / run, "--timed")
        resumed = [r for r in profiling(rows) if int(r["turn"]) > 0 and r["ttft_ms"]]
        simulated, observed = (
            p3([float(r["ttft_ms"]) for r in resumed]),
            p3([float(r["measured_ttft_ms"]) for r in resumed]),
        )
        residuals = {
            k: pct(
                [
                    float(r["ttft_ms"]) - float(r["measured_ttft_ms"])
                    for r in resumed
                    if resume_kind(r) == k
                ],
                50,
            )
            for k in ("hit", "computed", "restore")
        }
        lines.append(
            f"- {label}: {errors(simulated, observed)}%; hits {signed(residuals['hit'])}, computed "
            f"{signed(residuals['computed'])}, restores {signed(residuals['restore'])} ms"
        )
    by_overlap = overlapping_restores(exports / POINTS[2][2], timed[POINTS[2][2]])
    lines += [
        f"- {POINTS[2][0]}, restores by the other restores overlapping their measured wait: "
        + ", ".join(
            f"{k}{'+' if k == 6 else ''}: {signed(pct(v, 50))} ms ({len(v)})"
            for k, v in sorted(by_overlap.items())
        ),
        "",
    ]

    lines += [
        "**The fit** on the three points (`fit_step_cost.py`), which `StepCost::frozen_box()` holds:",
        "",
        "```",
        steps.read_text().strip(),
        fitted.strip(),
        "```",
        "",
    ]

    # W0: the engine's split, the engine model on W0's timeline, and the full simulation under the idle service.
    w0 = meas[W0]
    log = sim.replay_log(f"{W0}-replay-log", exports / W0)
    prof = profiling(log)
    corpus_prompt = sum(int(r["prompt_tokens"]) for r in prof)
    gpu, cpu, error = cache_model(log)
    idle = sim.run("w0-idle", "--service", "idle")
    idle_shares = [100 * idle["shares"][k] for k in ("gpu", "cpu", "computed")]
    model = [gpu, cpu, 100 - gpu - cpu]
    replay = idle["replay"]
    lines += [
        "## Validation against W0",
        "",
        "| | W0, measured | Engine model on W0's timeline | Full simulation |",
        "|---|---|---|---|",
        f"| Profiling requests | {w0['requests']} | {len(prof)} | {idle['requests']} |",
        f"| Session plays, warmup requests | {w0['plays']}, {w0['warmup']} | — | {replay['sessions_started']}, {replay['warmup_requests']} |",
        f"| Prompt tokens | {n(w0['prompt'])} | {n(corpus_prompt)} | {n(idle['prompt_tokens'])} |",
    ]
    for i, row in enumerate(("From the GPU pool", "Restored from the CPU tier", "Computed")):
        lines.append(
            f"| {row} | {w0['shares'][i]:.2f}% | {model[i]:.2f}% ({signed(model[i] - w0['shares'][i], 2)}) | "
            f"{idle_shares[i]:.2f}% ({signed(idle_shares[i] - w0['shares'][i], 2)}) |"
        )
    t = idle["ttft_ms"]
    lines += [
        f"| TTFT p50 / p90 / p99 | {' / '.join(map(n, w0['ttft']))} ms | — | {' / '.join(n(t[k]) for k in ('p50', 'p90', 'p99'))} ms |",
        "",
    ]
    service = statistics.mean(
        (float(r["done_s"]) - float(r["issued_s"])) * 1e3 for r in idle["rows"]
    )
    lines += [
        f"- The engine's cached share is {w0['shares'][0] + w0['shares'][1]:.2f}%, its per-request `cached_tokens` "
        f"{w0['cached_share']:.2f}%. On W0's timeline the engine model's per-request cached share is within "
        f"{error:.2f} points on average.",
        f"- Mean time to first token and service time: {n(idle['ttft_ms']['mean'])} and {n(service)} ms in the full "
        f"simulation, {n(w0['mean_ttft'])} and {n(w0['mean_latency'])} ms measured.",
        f"- The engine counted {100 * (w0['prompt'] / corpus_prompt - 1):.1f}% more prompt tokens than the corpus holds.",
    ]
    spread = [
        sim_row(sim.run(f"w0-idle-seed{seed}", "--service", "idle", "--seed", str(seed)))
        for seed in SEEDS
    ]
    ranges = [(min(s[2][i] for s in spread), max(s[2][i] for s in spread)) for i in range(3)]
    lines.append(
        f"- Seeds {SEEDS[0]}{DASH}{SEEDS[-1]}: {min(s[0] for s in spread)}{DASH}{max(s[0] for s in spread)} requests, "
        + " / ".join(f"{lo:.1f}{DASH}{hi:.1f}%" for lo, hi in ranges)
        + " (GPU / CPU / computed)."
    )
    subset = sim.run("subset-idle", "--service", "idle", corpus=SUBSET)
    sub_shares = [100 * subset["shares"][k] for k in ("gpu", "cpu", "computed")]
    lines += [
        f"- The 16-trace subset: {subset['requests']} requests, {subset['prompt_tokens'] / 1e6:.2f} M prompt tokens, KV "
        f"pressure {subset['regime']['kv_pressure_mean']:.1f} against W0's {idle['regime']['kv_pressure_mean']:.1f}; "
        f"{shares(sub_shares)}%.",
        "",
    ]

    lines += [
        "## W2's operating points (docs/BENCHMARKS.md)",
        "",
        "| Session trees | Requests | TTFT p50 / p90 / p99 | ITL p50 | Prompt tokens: GPU / CPU tier / computed | Preemptions |",
        "|---|---|---|---|---|---|",
    ]
    for _, sessions, run in POINTS:
        m = meas[run]
        lines.append(
            f"| {sessions} | {n(m['requests'])} | {' / '.join(map(n, m['ttft']))} ms | {m['itl50']:.1f} ms | "
            f"{shares(m['run_shares'])}% | {m['preemptions']} |"
        )
    # The repeats beside the table's fresh points, as the table measures them.
    s16 = [meas[run] for run in (POINTS[1][2], WARM[0][2], W0)]
    p90 = [m["ttft"][1] for m in s16]
    split = [m["run_shares"] for m in s16]
    apart = max(max(s[i] for s in split) - min(s[i] for s in split) for i in range(3))
    lines += [
        "",
        "**Repeatability.** Three 16-session runs (fresh, after the 8-session run, and W0): TTFT p90 "
        + " / ".join(map(n, p90))
        + f" ms ({n(min(p90))}{DASH}{n(max(p90))}); prompt tokens GPU / CPU tier / computed "
        + ", ".join(shares(s) for s in split)
        + f"%, at most {apart:.1f} points apart. The 24-session run after the 16-session one: "
        f"TTFT p90 {n(meas[WARM[1][2]]['ttft'][1])} ms, against "
        f"{n(meas[POINTS[2][2]]['ttft'][1])} on a fresh engine.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "calibration.md").write_text(text)
    return text


def export(runs_dir: Path, exports: Path) -> None:
    for run in RUNS:
        for name in FILES:
            source, target = runs_dir / run / name, exports / run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".prom"):
                target.write_text(
                    SNAPSHOT.sub(lambda m: f'"{m["org"]}/{m["name"]}"', source.read_text())
                )
            else:
                shutil.copyfile(source, target)
        print(exports / run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("report", help="the sections' tables and figures from the exports")
    r.add_argument("exports", type=Path)
    r.add_argument("corpus", type=Path)
    r.add_argument("lh_sim", type=Path)
    r.add_argument("out", type=Path)
    e = commands.add_parser("export", help="the runs' exports, for publishing")
    e.add_argument("runs_dir", type=Path)
    e.add_argument("exports", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        print(report(args.exports, args.corpus, args.lh_sim, args.out), end="")
    else:
        export(args.runs_dir, args.exports)


if __name__ == "__main__":
    main()
