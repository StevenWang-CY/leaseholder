"""Tier cost of resuming a session (``lh-bench tiers``): the E0 cost model.

A sample sends one prompt of L random token ids three times, each asking for one token, so each
latency is a time to first token:

1. ``recompute``: the first send; nothing is cached and every prompt token is computed;
2. ``gpu``: the same prompt at once, served from the GPU prefix cache;
3. ``cpu``: the same prompt after other prompts pushed it out of the GPU pool. vLLM's
   OffloadingConnector wrote its blocks through to host RAM, so it is restored from the CPU tier.

The engine's prefix-cache counters (``vllm:prefix_cache_hits`` for the GPU pool and
``vllm:external_prefix_cache_hits`` for the connector), read before and after each request,
attribute its prompt tokens to the GPU, the CPU tier or computation. A sample whose eviction or
restore did not happen as intended is marked invalid instead of being mislabeled. With
``--background`` the same runs under contention; the load uses unique prompts, so it adds no hits.
"""

from __future__ import annotations

import asyncio
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson

from leaseholder_bench.kv_metrics import metric_samples
from leaseholder_bench.load import BackgroundLoad, LoadSpec, random_prompt, summarize
from leaseholder_bench.protocol import ENGINE_PRIORITY, SloClass

TIERS = ("recompute", "gpu", "cpu")
GPU_HITS = "vllm:prefix_cache_hits_total"
CPU_HITS = "vllm:external_prefix_cache_hits_total"
FLUSH_CHUNK = 16_384


def read_counters(exposition: str) -> dict[str, float]:
    """The two hit counters of a Prometheus exposition, summed over label sets."""
    totals = {GPU_HITS: 0.0, CPU_HITS: 0.0}
    for name, _, value in metric_samples(exposition):
        if name in totals:
            totals[name] += value
    return totals


def pool_tokens(exposition: str) -> int | None:
    """GPU KV pool size in tokens, from vLLM's ``cache_config_info`` labels."""
    for name, labels, _ in metric_samples(exposition):
        if name == "vllm:cache_config_info":
            try:
                return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
            except (KeyError, ValueError):
                return None
    return None


@dataclass(frozen=True)
class TiersConfig:
    url: str
    """Engine base URL, e.g. ``http://127.0.0.1:18000``."""
    model: str
    lengths: Sequence[int]
    reps: int = 5
    background: LoadSpec | None = None
    decode_tokens: int = 1024
    priority: int | None = 1
    flush_factor: float = 1.5
    seed: int = 0
    timeout: float = 300.0


@dataclass(frozen=True)
class Sample:
    rep: int
    tokens: int
    tier: str
    ms: float
    prompt_tokens: int
    gpu_hit_tokens: int
    cpu_hit_tokens: int
    valid: bool

    @property
    def computed_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.gpu_hit_tokens - self.cpu_hit_tokens)


def is_valid(tier: str, prompt_tokens: int, gpu_hit: int, cpu_hit: int) -> bool:
    """Whether the request was served from the tier it was meant to measure."""
    most, little = 0.9 * prompt_tokens, 0.1 * prompt_tokens
    if tier == "recompute":
        return gpu_hit + cpu_hit <= little
    if tier == "gpu":
        return gpu_hit >= most and cpu_hit <= little
    return cpu_hit >= most and gpu_hit <= little


class TierProbe:
    def __init__(self, client: httpx.AsyncClient, config: TiersConfig) -> None:
        self.client = client
        self.config = config
        self.rng = random.Random(config.seed)
        self.pool = 0

    async def metrics(self) -> str:
        response = await self.client.get("/metrics")
        response.raise_for_status()
        return response.text

    async def complete(self, prompt: list[int]) -> tuple[float, dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
        if self.config.priority is not None:
            body["priority"] = self.config.priority
        started = time.perf_counter()
        response = await self.client.post("/v1/completions", json=body)
        ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return ms, response.json()["usage"]

    async def measure(self, rep: int, tier: str, prompt: list[int]) -> Sample:
        before = read_counters(await self.metrics())
        ms, usage = await self.complete(prompt)
        after = read_counters(await self.metrics())
        gpu_hit = round(after[GPU_HITS] - before[GPU_HITS])
        cpu_hit = round(after[CPU_HITS] - before[CPU_HITS])
        prompt_tokens = int(usage["prompt_tokens"])
        valid = is_valid(tier, prompt_tokens, gpu_hit, cpu_hit)
        return Sample(rep, len(prompt), tier, round(ms, 2), prompt_tokens, gpu_hit, cpu_hit, valid)

    async def flush(self) -> None:
        """Push everything out of the GPU pool with unique prompts (LRU evicts the oldest first)."""
        remaining = math.ceil(self.pool * self.config.flush_factor)
        while remaining > 0:
            chunk = min(FLUSH_CHUNK, remaining)
            await self.complete(random_prompt(self.rng, chunk))
            remaining -= chunk

    async def sample(self, rep: int, tokens: int) -> list[Sample]:
        prompt = random_prompt(self.rng, tokens)
        rows = [await self.measure(rep, "recompute", prompt)]
        rows.append(await self.measure(rep, "gpu", prompt))
        await self.flush()
        rows.append(await self.measure(rep, "cpu", prompt))
        return rows


def _line(tokens: Sequence[float], ms: Sequence[float]) -> dict[str, float]:
    """The least-squares line through (tokens, ms): an intercept in ms and a slope per token."""
    fit = statistics.linear_regression(tokens, ms)
    return {"ms": round(fit.intercept, 2), "us_per_token": round(fit.slope * 1000, 3)}


def _quadratic(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares a + b x + c x^2, by the normal equations (three unknowns, Gaussian elimination)."""
    s = [sum(x**k for x in xs) for k in range(5)]
    t = [sum(y * x**k for x, y in zip(xs, ys, strict=True)) for k in range(3)]
    m = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col], strict=True)]
    a, b, c = (m[i][3] / m[i][i] if m[i][i] else 0.0 for i in range(3))
    return a, b, c


def summarize_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    """Per tier and length: latency percentiles of valid samples, and fits of the medians (linear
    for the GPU and CPU tiers, quadratic for recompute)."""
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    invalid: dict[str, int] = defaultdict(int)
    for s in samples:
        if s.valid:
            by_cell[s.tier, s.tokens].append(s.ms)
        else:
            invalid[s.tier] += 1
    lengths = sorted({s.tokens for s in samples})
    cells = {
        tier: {n: summarize(by_cell[tier, n]) for n in lengths if by_cell[tier, n]}
        for tier in TIERS
    }
    p50 = {tier: {n: cell["p50"] for n, cell in cells[tier].items()} for tier in TIERS}
    fits: dict[str, Any] = {}
    for tier in ("gpu", "cpu"):
        if len(p50[tier]) >= 2:
            fits[tier] = _line(list(p50[tier]), list(p50[tier].values()))
    common = sorted(set(p50["gpu"]) & set(p50["cpu"]))
    if len(common) >= 2:
        fits["cpu_over_gpu"] = _line(common, [p50["cpu"][n] - p50["gpu"][n] for n in common])
    if len(p50["recompute"]) >= 3:
        a, b, c = _quadratic(list(p50["recompute"]), list(p50["recompute"].values()))
        fits["recompute"] = {
            "ms": round(a, 2),
            "us_per_token": round(b * 1000, 3),
            "ns_per_token_squared": round(c * 1e6, 4),
        }
    beats = [n for n in common if n in p50["recompute"] and p50["cpu"][n] < p50["recompute"][n]]
    return {
        "p50_ms": {tier: {str(n): v for n, v in p50[tier].items()} for tier in TIERS},
        "cells": {tier: {str(n): v for n, v in cells[tier].items()} for tier in TIERS},
        "invalid": dict(invalid),
        "fits": fits,
        "cpu_restore_beats_recompute_from": beats[0] if beats else None,
    }


def render(summary: dict[str, Any]) -> str:
    rows = ["tokens  recompute_ms    gpu_ms    cpu_ms"]
    p50 = summary["p50_ms"]
    for n in sorted({int(k) for tier in TIERS for k in p50[tier]}):
        cells = [p50[tier].get(str(n)) for tier in TIERS]
        rows.append(f"{n:>6}  " + "  ".join("-" if c is None else f"{c:>8.1f}" for c in cells))
    for name, fit in summary["fits"].items():
        rows.append(f"fit {name}: {fit}")
    if summary["invalid"]:
        rows.append(f"invalid samples (excluded): {summary['invalid']}")
    return "\n".join(rows)


async def run_tiers(config: TiersConfig, out_dir: Path) -> dict[str, Any]:
    """Measure every length ``reps`` times (lengths shuffled per rep) and write the run directory:
    ``run.json`` (configuration, host, engine, times), ``samples.jsonl`` and ``summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    async with httpx.AsyncClient(base_url=config.url, timeout=config.timeout) as client:
        probe = TierProbe(client, config)
        exposition = await probe.metrics()
        probe.pool = pool_tokens(exposition) or 0
        if not probe.pool:
            raise RuntimeError("the engine's /metrics has no vllm:cache_config_info pool size")
        version = (await client.get("/version")).json().get("version")
        samples: list[Sample] = []
        load_summary = None
        order = random.Random(config.seed + 1)
        with (out_dir / "samples.jsonl").open("wb") as log:

            async def measure_all() -> None:
                for rep in range(config.reps):
                    lengths = list(config.lengths)
                    order.shuffle(lengths)
                    for n in lengths:
                        for s in await probe.sample(rep, n):
                            samples.append(s)
                            log.write(orjson.dumps(asdict(s) | {"computed": s.computed_tokens}))
                            log.write(b"\n")
                            log.flush()

            if config.background is None:
                await measure_all()
            else:
                priority = None if config.priority is None else ENGINE_PRIORITY[SloClass.BACKGROUND]
                async with BackgroundLoad(
                    client,
                    "/v1/completions",
                    config.model,
                    config.background,
                    priority=priority,
                    decode_tokens=config.decode_tokens,
                    seed=config.seed + 2,
                ) as load:
                    await asyncio.sleep(2.0)  # let the load reach steady state
                    await measure_all()
                load_summary = load.summary()
    summary = summarize_samples(samples)
    run = {
        "kind": "tiers",
        "config": asdict(config) | {"background": str(config.background or "none")},
        "host": platform.node(),
        "engine_version": version,
        "pool_tokens": probe.pool,
        "started": started,
        "finished": time.time(),
        "background": load_summary,
    }
    (out_dir / "run.json").write_bytes(orjson.dumps(run, option=orjson.OPT_INDENT_2))
    (out_dir / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return summary
