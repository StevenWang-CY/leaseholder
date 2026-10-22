"""KV block lifecycle histograms from vLLM's ``/metrics``, compared across a run (``lh-bench kv-metrics``).

With ``--kv-cache-metrics``, vLLM 0.32 samples GPU KV blocks and exports:
- ``vllm:kv_block_reuse_gap_seconds``: time between consecutive accesses of a block;
- ``vllm:kv_block_idle_before_evict_seconds``: how long an evicted block sat unused before eviction;
- ``vllm:kv_block_lifetime_seconds``: allocation to eviction.

The comparison is a cheap first signal of retention headroom. If evicted blocks were idle for less
time than blocks typically wait between reuses, LRU is evicting KV that would have been reused.
"""

from __future__ import annotations

import math
import re
import urllib.request
from dataclasses import dataclass

REUSE_GAP = "vllm:kv_block_reuse_gap_seconds"
IDLE_BEFORE_EVICT = "vllm:kv_block_idle_before_evict_seconds"
LIFETIME = "vllm:kv_block_lifetime_seconds"
HISTOGRAMS = (REUSE_GAP, IDLE_BEFORE_EVICT, LIFETIME)

_BUCKET = re.compile(r"^(?P<name>[\w:]+)_bucket\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)$")
_LE = re.compile(r'le="([^"]+)"')


def fetch(url: str, timeout: float = 10.0) -> str:
    """The Prometheus text exposition at ``url`` (an engine's ``/metrics``)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


@dataclass(frozen=True)
class Histogram:
    """Cumulative bucket counts, ordered by upper bound; the last bound is +Inf."""

    bounds: tuple[float, ...]
    counts: tuple[float, ...]

    @classmethod
    def parse(cls, text: str, name: str) -> Histogram:
        """Read one histogram from an exposition, summing its label sets."""
        totals: dict[float, float] = {}
        for line in text.splitlines():
            m = _BUCKET.match(line)
            if m is None or m["name"] != name:
                continue
            le = _LE.search(m["labels"])
            if le is None:
                continue
            bound = math.inf if le[1] == "+Inf" else float(le[1])
            totals[bound] = totals.get(bound, 0.0) + float(m["value"])
        bounds = tuple(sorted(totals))
        return cls(bounds, tuple(totals[b] for b in bounds))

    def __sub__(self, earlier: Histogram) -> Histogram:
        before = dict(zip(earlier.bounds, earlier.counts, strict=True))
        return Histogram(
            self.bounds,
            tuple(c - before.get(b, 0.0) for b, c in zip(self.bounds, self.counts, strict=True)),
        )

    @property
    def total(self) -> float:
        return self.counts[-1] if self.counts else 0.0

    def quantile(self, q: float) -> float | None:
        """Linear interpolation inside the bucket that crosses quantile ``q``."""
        if self.total <= 0:
            return None
        target = q * self.total
        lower, below = 0.0, 0.0
        for bound, count in zip(self.bounds, self.counts, strict=True):
            if count >= target:
                if math.isinf(bound):
                    return lower  # beyond the largest finite bucket: report its bound
                return lower + (bound - lower) * (target - below) / max(count - below, 1e-12)
            lower, below = bound, count
        return lower

    def share_at_most(self, x: float) -> float | None:
        """The share of observations at or below ``x``, interpolated inside its bucket."""
        if self.total <= 0:
            return None
        lower, below = 0.0, 0.0
        for bound, count in zip(self.bounds, self.counts, strict=True):
            if x <= bound:
                if math.isinf(bound):
                    return below / self.total
                share = below + (count - below) * (x - lower) / max(bound - lower, 1e-12)
                return share / self.total
            lower, below = bound, count
        return 1.0


def compare(before: str, after: str) -> str:
    """A short report of the three histograms over the interval between two expositions."""
    hists = {
        name: Histogram.parse(after, name) - Histogram.parse(before, name) for name in HISTOGRAMS
    }
    lines = []
    for name, hist in hists.items():
        qs = [(q, hist.quantile(q)) for q in (0.1, 0.5, 0.9)]
        rendered = " ".join(f"p{round(q * 100)}={v:.3g}s" for q, v in qs if v is not None)
        lines.append(f"{name}: n={hist.total:.0f} {rendered}".rstrip())
    reuse, idle = hists[REUSE_GAP], hists[IDLE_BEFORE_EVICT]
    for q in (0.5, 0.9):
        gap = reuse.quantile(q)
        share = idle.share_at_most(gap) if gap is not None else None
        if share is not None:
            lines.append(
                f"evicted blocks idle no longer than the reuse-gap p{round(q * 100)} "
                f"({gap:.3g}s): {share:.1%}"
            )
    return "\n".join(lines)
