"""CPU time and memory of the agentic-api process per benchmark phase, from server-usage.txt and campaign.log."""

import re
import sys
from datetime import datetime
from pathlib import Path

run = Path(sys.argv[1]).expanduser()
samples = [
    tuple(map(float, line.split()))
    for line in (run / "server-usage.txt").read_text().splitlines()
    if len(line.split()) == 3
]
phases, start = [], {}
for line in (run / "campaign.log").read_text().splitlines():
    m = re.match(r"\[(\S+)\] (start (\S+)|(\S+) rc=)", line)
    if not m:
        continue
    t = datetime.fromisoformat(m[1].replace("Z", "+00:00")).timestamp()
    if m[3]:
        start[m[3]] = t
    elif m[4] in start:
        phases.append((m[4], start[m[4]], t))
hz = 100.0
for name, t0, t1 in phases:
    inside = [s for s in samples if t0 <= s[0] <= t1]
    if len(inside) < 2:
        continue
    cpu_s = (inside[-1][1] - inside[0][1]) / hz
    secs = inside[-1][0] - inside[0][0]
    rss = max(s[2] for s in inside) / 1024
    print(
        f"{name:<14} {secs:6.0f} s  agentic-api CPU {cpu_s:6.2f} s ({100 * cpu_s / secs:5.1f}% of one core)  peak RSS {rss:6.1f} MiB"
    )
