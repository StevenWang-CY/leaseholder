"""long_traces.py CORPUS OUT N: link the N traces of a scaled Weka corpus whose requests carry the largest contexts
(the median `in` of the main agent's requests; subagent entries nest their own) into OUT. E2's discriminating scenario replays them:

    python benchmarks/e2/long_traces.py ~/leaseholder/data/weka-x32 ~/leaseholder/data/weka-x32-long 48
"""

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    corpus, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    medians = []
    for path in sorted(corpus.glob("*.json")):
        inputs = [r["in"] for r in json.loads(path.read_text())["requests"] if "in" in r]
        if inputs:
            medians.append((statistics.median(inputs), len(inputs), path))
    medians.sort(key=lambda row: row[0], reverse=True)
    chosen = medians[:n]
    out.mkdir(parents=True, exist_ok=True)
    for _, _, path in chosen:
        link = out / path.name
        if not link.exists():
            link.symlink_to(path.resolve())
    print(
        f"{len(medians)} traces, median input {statistics.median(m for m, _, _ in medians):,.0f}; "
        f"the {len(chosen)} largest: median input {statistics.median(m for m, _, _ in chosen):,.0f}, "
        f"{sum(c for _, c, _ in chosen):,} requests"
    )


if __name__ == "__main__":
    main()
