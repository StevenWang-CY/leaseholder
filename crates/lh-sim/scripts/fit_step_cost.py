"""Fit the contention model's step-time model (lh-core's `StepCost`) from measured AIPerf runs.

Each run is given as its AIPerf `profile_export.jsonl` and the CSV `lh-sim replay-log --csv` writes for
it, which says where the engine model finds each prompt. The fit goes to stdout as the JSON that
`lh-sim run --step-costs` reads, and the size and residuals of each data set to stderr:

    uv run --no-project python crates/lh-sim/scripts/fit_step_cost.py \\
        s8.jsonl s8.csv s16.jsonl s16.csv [...] > steps.json

The terms, each from the requests' measured token times:

1. `step_ms`, `us_per_token`, `ns_per_pair`, `ns_per_context` and the host time `host`: least squares
   on relative error over three sets weighted equally. Decode-only steps: the gaps between a
   request's tokens while every request in flight was decoding, each batch size weighted equally.
   Requests that ran alone: host time, E0's restore term and their prefill in steps of the whole
   512-token budget. E0's GPU-hit and recompute cells (p50), with a host time of their own, as E0
   sends token ids where the runs send chats. The worst decile of the first two sets is dropped and
   the fit repeated.
2. `mixed_ms`: the median excess over the step model of the decode gaps during which one other
   request prefilled a long prompt in 256-token chunks.
3. `host_per_decode_ms`: GPU hits that arrived while only decodes ran: the slope, over the number of
   requests decoding, of their excess over host time, half a decode step (the step in flight) and
   their own step.
4. `restore_stall`: CPU restores of at least 2,000 tokens with only decodes beside them: the time a
   request decoding across the restore lost against the decode-step model, over E0's transfer time
   (median).

Requests whose cached share in the engine model is more than a point off the engine's own count are
left out of 1 and 3, and prefills with any restored tokens out of 2.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# E0 on the frozen box (lh-core's TierCost::frozen_box): p50 TTFT cells and the restore term.
E0_LENGTHS = (1024, 2048, 4096, 8192, 16384, 24576, 31744)
E0_RECOMPUTE_MS = (135.0, 198.0, 437.0, 1114.0, 3264.0, 6506.0, 10239.0)
E0_GPU_MS = (45.0, 40.0, 48.0, 64.0, 97.0, 128.0, 156.0)
RESTORE_MS, RESTORE_US = 10.3, 23.64
BUDGET, THRESHOLD, BLOCK = 512, 256, 16
# Unknowns of the least-squares fit, in order.
NAMES = (
    "step_ms",
    "us_per_token",
    "ns_per_pair",
    "ns_per_context",
    "host_ms",
    "host_us",
    "e0_ms",
    "e0_us",
)

Chunks = list[tuple[int, int]]  # (tokens computed, tokens before them) per scheduled request


@dataclass
class Request:
    warmup: bool
    start: float
    first: float
    end: float
    times: list[float]  # the first token, then each later chunk
    prompt: int  # as the engine counted it
    cached: int  # as the engine counted it
    cpu: int  # restored in the engine model, scaled to `prompt`
    computed: int  # in the engine model
    faithful: bool  # the engine model's cached share is within a point of the engine's

    def context(self, t: float) -> int:
        """Its tokens of KV at time `t` once it is decoding."""
        return self.prompt + bisect.bisect_right(self.times, t)


class Run:
    """The requests of one run, in the order they were sent."""

    def __init__(self, jsonl: Path, log: Path) -> None:
        self.reqs: list[Request] = []
        with jsonl.open() as f, log.open() as g:
            records = (json.loads(line) for line in f if line.strip())
            for rec, row in zip(records, csv.DictReader(g), strict=True):
                m, x = rec["metadata"], rec["metrics"]
                if "time_to_first_token" not in x:
                    continue
                start = m["request_start_ns"] / 1e6
                times = [start + x["time_to_first_token"]["value"]]
                for gap in x.get("inter_chunk_latency", {}).get("value") or []:
                    times.append(times[-1] + gap)
                prompt, cached = (
                    int(row["measured_prompt_tokens"]),
                    int(row["measured_cached_tokens"]),
                )
                corpus = max(1, int(row["prompt_tokens"]))
                computed = int(row["computed_tokens"])
                modeled = (corpus - computed) / corpus
                self.reqs.append(
                    Request(
                        warmup=m["benchmark_phase"] == "warmup",
                        start=start,
                        first=times[0],
                        end=max(m["request_end_ns"] / 1e6, times[-1]),
                        times=times,
                        prompt=prompt,
                        cached=cached,
                        cpu=round(prompt * int(row["cpu_tokens"]) / corpus),
                        computed=computed,
                        faithful=abs(modeled - cached / max(1, prompt)) <= 0.01,
                    )
                )
        self.reqs.sort(key=lambda r: r.start)
        self.starts = [r.start for r in self.reqs]
        self.longest = max(r.end - r.start for r in self.reqs)

    def overlapping(self, a: float, b: float, me: Request) -> list[Request]:
        """The requests other than `me` in flight at some time in (a, b)."""
        lo = bisect.bisect_left(self.starts, a - self.longest)
        hi = bisect.bisect_left(self.starts, b)
        return [q for q in self.reqs[lo:hi] if q is not me and q.end > a]

    def gaps(self):
        """Each request's gaps between tokens, with the requests in flight beside it."""
        for r in self.reqs:
            for a, b in pairwise(r.times):
                if b - a >= 2.0:  # shorter: chunks the client received together
                    yield r, a, b, self.overlapping(a, b, r)


def restore_ms(tokens: int) -> float:
    return RESTORE_MS + RESTORE_US * tokens / 1e3 if tokens > 0 else 0.0


def features(chunks: Chunks) -> list[float]:
    tokens = sum(n for n, _ in chunks)
    pairs = sum(n * c + n * (n + 1) / 2 for n, c in chunks)
    context = sum(c + n for n, c in chunks)
    return [1.0, tokens / 1e3, pairs / 1e6, context / 1e6]


def prefill_features(length: int, computed: int) -> list[float]:
    """A lone prefill of `computed` tokens after `length - computed`, in budget-sized steps."""
    f = [0.0] * 4
    cached, left = length - computed, max(1, computed)
    while left > 0:
        n = min(BUDGET, left)
        f = [u + v for u, v in zip(f, features([(n, cached)]), strict=True)]
        cached += n
        left -= n
    return f


def solve(rows: list[list[float]], ys: list[float], ws: list[float]) -> list[float]:
    """Weighted least squares by Gauss-Jordan elimination of the normal equations."""
    k = len(rows[0])
    a = [
        [sum(w * r[i] * r[j] for r, w in zip(rows, ws, strict=True)) for j in range(k)]
        for i in range(k)
    ]
    b = [sum(w * r[i] * y for r, y, w in zip(rows, ys, ws, strict=True)) for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda q: abs(a[q][c]))
        a[c], a[p] = a[p], a[c]
        b[c], b[p] = b[p], b[c]
        for q in range(k):
            if q != c:
                f = a[q][c] / a[c][c]
                a[q] = [u - f * v for u, v in zip(a[q], a[c], strict=True)]
                b[q] -= f * b[c]
    return [b[i] / a[i][i] for i in range(k)]


def fit_steps(runs: list[Run]) -> dict[str, float]:
    # Per set: (features, fixed ms, measured ms, decode batch size).
    sets: dict[str, list[tuple[list[float], float, float, int]]] = {
        "decode steps": [],
        "alone": [],
        "E0 cells": [],
    }
    for run in runs:
        for r, a, b, others in run.gaps():
            if all(q.first <= a for q in others):
                chunks = [(1, q.context(a)) for q in [r, *others]]
                f = [*features(chunks), 0.0, 0.0, 0.0, 0.0]
                sets["decode steps"].append((f, 0.0, b - a, len(chunks)))
        for r in run.reqs:
            if r.warmup or not r.faithful or run.overlapping(r.start, r.first, r):
                continue
            f = [*prefill_features(r.prompt, r.prompt - r.cached), 1.0, r.prompt / 1e3, 0.0, 0.0]
            sets["alone"].append((f, restore_ms(r.cpu), r.first - r.start, 0))
    for length, recompute, gpu in zip(E0_LENGTHS, E0_RECOMPUTE_MS, E0_GPU_MS, strict=True):
        for computed, ms in ((length, recompute), (BLOCK, gpu)):
            f = [*prefill_features(length, computed), 0.0, 0.0, 1.0, length / 1e3]
            sets["E0 cells"].append((f, 0.0, ms, 0))

    def errors(items, theta):
        return [
            (fixed + sum(u * v for u, v in zip(f, theta, strict=True)) - y) / y
            for f, fixed, y, _ in items
        ]

    theta: list[float] = []
    for trim in (False, True):
        if trim:
            for name in ("decode steps", "alone"):
                err = [abs(e) for e in errors(sets[name], theta)]
                cut = sorted(err)[int(0.9 * len(err))]
                sets[name] = [it for it, e in zip(sets[name], err, strict=True) if e <= cut]
        rows, ys, ws = [], [], []
        for name, items in sets.items():
            sizes: dict[int, int] = {}
            for *_, batch in items:
                sizes[min(batch, 8)] = sizes.get(min(batch, 8), 0) + 1
            for f, fixed, y, batch in items:
                rows.append([v / y for v in f])
                ys.append((y - fixed) / y)
                share = (
                    1.0 / (len(sizes) * sizes[min(batch, 8)])
                    if name == "decode steps"
                    else 1.0 / len(items)
                )
                ws.append(share / len(sets))
        theta = solve(rows, ys, ws)
    for name, items in sets.items():
        err = sorted(errors(items, theta))
        q = [100 * err[int(p * (len(err) - 1))] for p in (0.1, 0.5, 0.9)]
        print(
            f"{name}: {len(err)}, relative error p10/p50/p90 {q[0]:+.1f}/{q[1]:+.1f}/{q[2]:+.1f}%",
            file=sys.stderr,
        )
    return dict(zip(NAMES, theta, strict=True))


class Model:
    def __init__(self, fit: dict[str, float]) -> None:
        self.fit = fit

    def step_ms(self, chunks: Chunks) -> float:
        coefficients = [self.fit[n] for n in NAMES[:4]]
        return sum(u * v for u, v in zip(coefficients, features(chunks), strict=True))

    def host_ms(self, prompt: int) -> float:
        return self.fit["host_ms"] + self.fit["host_us"] * prompt / 1e3


def mixed_excess(runs: list[Run], model: Model) -> list[float]:
    """Excess over the step model of decode gaps beside one long prefill in 256-token chunks."""
    out = []
    for run in runs:
        for r, a, b, others in run.gaps():
            prefills = [q for q in others if q.first > a]
            if len(prefills) != 1:
                continue
            p = prefills[0]
            begin = p.start + model.host_ms(p.prompt)
            if p.cpu or not p.faithful or p.computed <= 3 * THRESHOLD or a < begin + 60.0:
                continue
            # Where its prefill is, assuming it runs at a steady rate.
            done = min(1.0, (a - begin) / max(1.0, p.first - begin))
            context = p.prompt - p.computed + round(done * p.computed)
            decodes = [(1, q.context(a)) for q in [r, *others] if q is not p]
            out.append(b - a - model.step_ms([*decodes, (THRESHOLD, context)]))
    return out


def hit_excess(runs: list[Run], model: Model, mixed_ms: float) -> list[tuple[int, float]]:
    """(requests decoding, excess) of each GPU hit that arrived while only decodes ran."""
    out = []
    for run in runs:
        for r in run.reqs:
            computed = r.prompt - r.cached
            if r.warmup or r.cpu or computed > 2 * BLOCK or not r.faithful:
                continue
            others = run.overlapping(r.start, r.first, r)
            if any(q.first > r.start for q in others):
                continue
            decodes = [(1, q.context(r.start)) for q in others]
            own = model.step_ms([*decodes, (computed, r.cached)]) + (mixed_ms if decodes else 0.0)
            in_flight = model.step_ms(decodes) / 2 if decodes else 0.0
            out.append(
                (len(decodes), r.first - r.start - model.host_ms(r.prompt) - in_flight - own)
            )
    return out


def restore_stalls(runs: list[Run], model: Model) -> list[float]:
    """Decode time lost over each restore with only decodes beside it, over E0's transfer time."""
    out = []
    for run in runs:
        for r in run.reqs:
            if r.cpu < 2000 or r.cpu < 0.8 * r.prompt or r.computed > 64:
                continue
            a, b = r.start + model.host_ms(r.prompt), r.first
            others = run.overlapping(a, b, r)
            across = [q for q in others if q.end >= b]
            if not across or any(q.first > a for q in others):
                continue
            times = [t for t in across[0].times if a <= t <= b]
            if len(times) < 2:
                continue
            step = model.step_ms([(1, q.context(a)) for q in others])
            out.append((times[-1] - times[0] - (len(times) - 1) * step) / restore_ms(r.cpu))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "runs", nargs="+", type=Path, help="profile_export.jsonl and replay-log CSV, per run"
    )
    args = parser.parse_args()
    if len(args.runs) % 2:
        parser.error("give each run's profile_export.jsonl and its replay-log CSV")
    runs = [Run(j, c) for j, c in zip(args.runs[::2], args.runs[1::2], strict=True)]
    fit = fit_steps(runs)
    model = Model(fit)
    mixed = mixed_excess(runs, model)
    mixed_ms = statistics.median(mixed)
    hits = hit_excess(runs, model, mixed_ms)
    beside = [(n, e) for n, e in hits if n > 0]
    per_decode = sum(n * e for n, e in beside) / sum(n * n for n, _ in beside)
    alone = [e for n, e in hits if n == 0]
    stalls = restore_stalls(runs, model)
    stall = statistics.median(stalls)
    q = statistics.quantiles(stalls, n=4)
    print(f"mixed steps: {len(mixed)}, median excess {mixed_ms:.1f} ms", file=sys.stderr)
    print(
        f"hits: {len(beside)} beside decodes, {per_decode:.1f} ms per decode; {len(alone)} alone, median excess "
        f"{statistics.median(alone):+.1f} ms",
        file=sys.stderr,
    )
    print(
        f"restores: {len(stalls)}, stall over transfer p25/p50/p75 {q[0]:.2f}/{stall:.2f}/{q[2]:.2f}",
        file=sys.stderr,
    )
    cost = {
        "step_ms": round(fit["step_ms"], 2),
        "us_per_token": round(fit["us_per_token"], 2),
        "ns_per_pair": round(fit["ns_per_pair"], 2),
        "ns_per_context": round(fit["ns_per_context"], 1),
        "mixed_ms": round(mixed_ms, 1),
        "host": {"ms": round(fit["host_ms"], 1), "us_per_token": round(fit["host_us"], 3)},
        "host_per_decode_ms": round(per_decode, 1),
        "restore_stall": round(stall, 2),
    }
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
