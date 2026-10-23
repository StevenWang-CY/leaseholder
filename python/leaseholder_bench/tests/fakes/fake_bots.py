"""A stand-in for ``lh-voice bots`` in the e2 tests: takes its options, waits ``FAKE_BOTS_SECONDS``
(0.3 s) and writes the callers' run directory. Its pid goes to ``bots.pid`` beside the run
directory. ``FAKE_BOTS`` picks an outcome: ``crash`` (exit 1, no summary) or ``hang``."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
for option in ("--corpus", "--callers", "--turns", "--barge-in", "--seed", "--out"):
    parser.add_argument(option, required=True)
parser.add_argument("--save-audio", action="store_true")
args = parser.parse_args()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
(out.parent / "bots.pid").write_text(str(os.getpid()))
(out.parent / "bots.args").write_text(json.dumps(sys.argv[1:]))
mode = os.environ.get("FAKE_BOTS", "")
time.sleep(60 if mode == "hang" else float(os.environ.get("FAKE_BOTS_SECONDS", "0.3")))
if mode == "crash":
    sys.exit(1)
out.mkdir(exist_ok=True)
turns = int(args.callers) * int(args.turns)
summary = {
    "turns": turns,
    "errors": 0,
    "v2v_ms": {"n": turns, "p50": 1600.0, "p95": 1800.0, "p99": 1900.0},
    "barge_in_to_silence_ms": {"n": 2, "p50": 600.0, "p95": 650.0, "p99": 660.0},
    "barge_ins": 2,
    "barge_ins_after_the_reply": 0,
    "caller_max_lateness_ms": 0.0,
}
(out / "summary.json").write_text(json.dumps(summary))
(out / "turns.jsonl").write_text("")
