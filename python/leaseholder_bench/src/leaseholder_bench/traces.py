"""Block scaling of Weka-format agent traces (``lh-bench scale-traces``).

The SemiAnalysis Claude Code corpus (``semianalysisai/cc-traces-weka-062126``) records every
request as a list of 64-token prefix hash blocks. Unscaled it cannot run against a small context:
the median main-agent prompt is 254k tokens. Shrinking the tokens per block by a factor shrinks
every request by that factor and leaves the hash ids, and with them the prefix structure, the
subagent forks and the cache hit rate, exactly as recorded:

    block_size  ->  block_size / factor
    in          ->  len(hash_ids) * new block_size
    out         ->  ceil(out / factor)

Outputs shrink by the same factor, so a cache miss keeps its cost relative to decode work, and
because ceil(ceil(out / f) / (b / f)) == ceil(out / b), AIPerf's split of each new turn into
assistant and user blocks is unchanged. There is no length cap: at 32x every request of the corpus
fits a 32k context, at 64x a 16k one. Sessions, subagent entries and timing (``t``, ``api_time``,
``think_time``) are copied unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

Trace = dict[str, Any]

_TRACE_ID = re.compile(r"[A-Za-z0-9._-]+")


def common_prefix_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Length of the longest common prefix of two lists or strings.

    Binary search over slice comparisons, so the element loop runs in C even for long prompts.
    """
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo] and a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def iter_traces(src: Path) -> Iterator[Trace]:
    """Yield traces from a Weka JSONL file (one trace per line) or a directory of ``*.json``."""
    if src.is_dir():
        for path in sorted(src.glob("*.json")):
            yield orjson.loads(path.read_bytes())
        return
    with src.open("rb") as fh:
        for line in fh:
            if line.strip():
                yield orjson.loads(line)


def model_requests(trace: Trace) -> Iterator[dict[str, Any]]:
    """Every model request of a trace: main-agent turns and subagent inner requests."""
    for entry in trace["requests"]:
        if entry["type"] == "subagent":
            yield from entry["requests"]
        else:
            yield entry


def scale_trace(trace: Trace, factor: int) -> Trace:
    """Shrink ``trace`` in place by ``factor`` tokens per hash block and return it."""
    block_size = trace["block_size"]
    if factor < 1 or block_size % factor:
        raise ValueError(
            f"trace {trace['id']}: block_size {block_size} is not divisible by factor {factor}"
        )
    new_block_size = block_size // factor
    trace["block_size"] = new_block_size
    for entry in trace["requests"]:
        kind = entry["type"]
        if kind in ("n", "s"):
            _scale_request(entry, factor, new_block_size)
        elif kind == "subagent":
            for request in entry["requests"]:
                _scale_request(request, factor, new_block_size)
            if entry.get("total_tokens") is not None:
                entry["total_tokens"] = sum(r["in"] + r["out"] for r in entry["requests"])
            _scale_prefix_tokens(entry, factor)
        else:
            raise ValueError(f"trace {trace['id']}: unknown request type {kind!r}")
    _scale_prefix_tokens(trace, factor)
    # An optional summary that AIPerf treats as opaque; its token counts would be stale.
    trace.pop("totals", None)
    return trace


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _scale_request(request: dict[str, Any], factor: int, block_size: int) -> None:
    hash_ids = request.get("hash_ids")
    if hash_ids:
        request["in"] = len(hash_ids) * block_size
    else:
        request["in"] = _ceil_div(request["in"], factor)
    request["out"] = _ceil_div(request["out"], factor)


def _scale_prefix_tokens(obj: dict[str, Any], factor: int) -> None:
    for key in ("tool_tokens", "system_tokens"):
        if obj.get(key):
            obj[key] = _ceil_div(obj[key], factor)


@dataclass
class ScaleSummary:
    traces: int = 0
    requests: int = 0
    peak_tokens: int = 0
    """Largest ``in + out`` of any request: the context length the scaled corpus needs."""


def scale_corpus(
    src: Path,
    out_dir: Path,
    factor: int,
    *,
    ids: set[str] | None = None,
    limit: int | None = None,
) -> ScaleSummary:
    """Scale every trace in ``src`` and write one ``<trace id>.json`` per trace to ``out_dir``.

    ``ids`` keeps only the listed traces and ``limit`` stops after that many; both are meant for
    small iteration subsets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(ids) if ids is not None else None
    summary = ScaleSummary()
    for trace in iter_traces(src):
        trace_id = trace["id"]
        if remaining is not None and trace_id not in remaining:
            continue
        if not _TRACE_ID.fullmatch(trace_id):
            raise ValueError(f"trace id {trace_id!r} is not usable as a file name")
        scale_trace(trace, factor)
        sizes = [r["in"] + r["out"] for r in model_requests(trace)]
        summary.traces += 1
        summary.requests += len(sizes)
        summary.peak_tokens = max(summary.peak_tokens, *sizes, 0)
        (out_dir / f"{trace_id}.json").write_bytes(orjson.dumps(trace))
        if remaining is not None:
            remaining.discard(trace_id)
            if not remaining:
                break
        if limit is not None and summary.traces >= limit:
            break
    return summary
