"""Background engine load for latency probes, and the helpers the probes share.

A :class:`LoadSpec` such as ``prefill:8192x2`` keeps two requests of 8,192 random token ids in
flight, each asking for one token: back-to-back cold prefills, the agent work that delays a voice
turn. ``decode:1024x8`` keeps eight 1,024-token prompts decoding (``ignore_eos``), a full decode
batch. Random token ids never repeat, so the load never hits the prefix cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import httpx
import orjson

# Qwen3's regular tokens; ids at and above 151,643 are special tokens.
_TOKEN_LOW, _TOKEN_HIGH = 1_000, 150_000


def random_prompt(rng: random.Random, tokens: int) -> list[int]:
    """A prompt of ``tokens`` random token ids, sent to ``/v1/completions`` as is."""
    return [rng.randrange(_TOKEN_LOW, _TOKEN_HIGH) for _ in range(tokens)]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Count, mean and p10/p50/p90/p99 of a sample (inclusive interpolation)."""
    xs = sorted(values)
    if not xs:
        return {"n": 0}
    qs = [xs[0]] * 99 if len(xs) == 1 else statistics.quantiles(xs, n=100, method="inclusive")
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 2),
        "p10": round(qs[9], 2),
        "p50": round(qs[49], 2),
        "p90": round(qs[89], 2),
        "p99": round(qs[98], 2),
    }


@dataclass(frozen=True)
class LoadSpec:
    kind: Literal["prefill", "decode"]
    tokens: int
    concurrency: int

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse ``<prefill|decode>:<prompt tokens>x<concurrency>``."""
        kind, sep, rest = text.partition(":")
        tokens, cross, concurrency = rest.partition("x")
        if not sep or not cross or kind not in ("prefill", "decode"):
            raise ValueError(f"expected <prefill|decode>:<tokens>x<concurrency>, got {text!r}")
        spec = cls(kind, int(tokens), int(concurrency))  # type: ignore[arg-type]
        if spec.tokens < 1 or spec.concurrency < 1:
            raise ValueError(f"tokens and concurrency must be positive in {text!r}")
        return spec

    def __str__(self) -> str:
        return f"{self.kind}:{self.tokens}x{self.concurrency}"


@dataclass
class LoadStats:
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class BackgroundLoad:
    """Runs a :class:`LoadSpec` against an engine's ``/v1/completions`` while the context is open.

    On exit the workers are cancelled, which closes their connections; vLLM aborts a request
    whose client disconnects, so no work outlives the probe. Only finished requests count.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        model: str,
        spec: LoadSpec,
        *,
        priority: int | None = 2,
        decode_tokens: int = 1024,
        seed: int = 0,
    ) -> None:
        self.client = client
        self.url = url
        self.model = model
        self.spec = spec
        self.priority = priority
        self.decode_tokens = decode_tokens
        self.stats = LoadStats()
        self._rng = random.Random(seed)
        self._tasks: list[asyncio.Task[None]] = []
        self._started = 0.0
        self._elapsed = 0.0

    async def __aenter__(self) -> Self:
        self._started = time.perf_counter()
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.spec.concurrency)]
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._elapsed = time.perf_counter() - self._started
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": random_prompt(self._rng, self.spec.tokens),
            "temperature": 0,
        }
        if self.spec.kind == "prefill":
            body["max_tokens"] = 1
        else:
            body.update(
                max_tokens=self.decode_tokens,
                ignore_eos=True,
                stream=True,
                stream_options={"include_usage": True},
            )
        if self.priority is not None:
            body["priority"] = self.priority
        return body

    async def _worker(self) -> None:
        while True:
            body = self._body()
            started = time.perf_counter()
            try:
                usage = await (self._decode(body) if body.get("stream") else self._prefill(body))
            except httpx.HTTPError:
                self.stats.errors += 1
                await asyncio.sleep(0.5)
                continue
            self.stats.requests += 1
            self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
            self.stats.completion_tokens += usage.get("completion_tokens", 0)
            self.stats.latencies_ms.append((time.perf_counter() - started) * 1000)

    async def _prefill(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(self.url, json=body)
        response.raise_for_status()
        return response.json().get("usage") or {}

    async def _decode(self, body: dict[str, Any]) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        async with self.client.stream("POST", self.url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: {") and '"usage"' in line:
                    usage = orjson.loads(line[6:]).get("usage") or usage
        return usage

    def summary(self) -> dict[str, Any]:
        """Throughput over the time the load ran: prompt tokens/s for prefill, output for decode."""
        elapsed = self._elapsed or (time.perf_counter() - self._started)
        tokens = (
            self.stats.prompt_tokens
            if self.spec.kind == "prefill"
            else self.stats.completion_tokens
        )
        return {
            "spec": str(self.spec),
            "seconds": round(elapsed, 1),
            "requests": self.stats.requests,
            "errors": self.stats.errors,
            "tokens_per_s": round(tokens / elapsed, 1) if elapsed > 0 else math.nan,
            "latency_ms": summarize(self.stats.latencies_ms),
        }


def parse_lengths(text: str) -> Sequence[int]:
    """Parse a comma-separated list of positive token counts, such as ``1024,4096,16384``."""
    lengths = [int(part) for part in text.split(",") if part.strip()]
    if not lengths or min(lengths) < 1:
        raise ValueError(f"expected comma-separated positive token counts, got {text!r}")
    return lengths
