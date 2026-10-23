"""Derive lh-sim's TraceLab pause table from the TraceLab coding-agent trace.

TraceLab (https://github.com/uw-syfi/TraceLab, v0.0.2, CC BY 4.0) records coding-agent rounds with
their tool calls. A round that called tools pauses the session from the end of its response (its
last tool call's emission) until its last tool result; the tool whose result came last names the
pause. The table has one row per tool with at least 1,000 such rounds, and `other` for the rest:
the rounds and the 0th to 100th percentiles of the pause in ms. `lh-sim run --pauses tracelab`
draws a tool by its share of rounds and a pause from its percentiles.

    uv run --no-project python crates/lh-sim/scripts/tracelab_pauses.py \\
        syfi_coding_trace.jsonl.gz crates/lh-sim/data/tracelab-pauses.tsv
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MIN_ROUNDS = 1000
HEADER = """\
# Pauses between a coding agent's requests: per tool, the wait from the end of a round's response
# (its last tool call's emission) to its last tool result, in ms, as the 0th to 100th percentiles.
# Derived by crates/lh-sim/scripts/tracelab_pauses.py from the TraceLab public coding-agent trace
# dataset v0.0.2 (https://github.com/uw-syfi/TraceLab), licensed under CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/); only these percentiles are kept.
"""


def epoch_ms(stamp: str) -> float:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1e3


def pause(tools: list[dict]) -> tuple[str, float]:
    """The tool whose result came last, and the round's pause in ms."""
    try:
        emitted = max(epoch_ms(t["emitted_at"]) for t in tools)
        last = max(tools, key=lambda t: epoch_ms(t["result_at"]))
        return last["tool_name"], max(0.0, epoch_ms(last["result_at"]) - emitted)
    except (KeyError, TypeError, ValueError):
        # No timestamps: the slowest call.
        slowest = max(tools, key=lambda t: t["tool_wall_latency_ms"])
        return slowest["tool_name"], float(slowest["tool_wall_latency_ms"])


def percentiles(values: list[float]) -> list[float]:
    """The 0th to 100th percentiles, linearly interpolated."""
    v = sorted(values)
    out = []
    for p in range(101):
        rank = p / 100 * (len(v) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(v) - 1)
        out.append(v[lo] + (rank - lo) * (v[hi] - v[lo]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path, help="syfi_coding_trace.jsonl.gz")
    parser.add_argument("out", type=Path, help="the TSV to write")
    args = parser.parse_args()
    by_tool: dict[str, list[float]] = defaultdict(list)
    with gzip.open(args.trace, "rt") as f:
        for line in f:
            tools = [
                t
                for t in json.loads(line).get("tools") or []
                if t.get("tool_wall_latency_ms") is not None
            ]
            if tools:
                name, ms = pause(tools)
                by_tool[name].append(ms)
    rows = sorted(
        ((n, v) for n, v in by_tool.items() if len(v) >= MIN_ROUNDS), key=lambda r: -len(r[1])
    )
    other = [ms for n, v in by_tool.items() if len(v) < MIN_ROUNDS for ms in v]
    if other:
        rows.append(("other", other))
    with args.out.open("w") as w:
        w.write(HEADER)
        w.write("tool\trounds\t" + "\t".join(f"p{p}" for p in range(101)) + "\n")
        for name, v in rows:
            w.write(f"{name}\t{len(v)}\t" + "\t".join(f"{q:.0f}" for q in percentiles(v)) + "\n")
    print(f"{sum(len(v) for _, v in rows)} rounds, {len(rows)} rows")


if __name__ == "__main__":
    main()
