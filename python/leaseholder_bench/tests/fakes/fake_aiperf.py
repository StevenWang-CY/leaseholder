"""A stand-in for ``aiperf profile`` in the e2 tests.

It logs AIPerf's profiling line, then writes one profiling record every 20 ms to
``$LH_BENCH_ARTIFACTS/profile_export.jsonl`` until SIGINT, and saves the LH_BENCH_ settings it was
started with to ``env.json``. ``FAKE_AIPERF`` picks a failure: ``exit-early`` (before profiling),
``exit-later`` (0.5 s into profiling), ``stubborn`` (ignores SIGINT, as does a child it starts,
whose pid goes to ``child.pid``) or ``orphan`` (starts a child in a session of its own, as AIPerf
starts some services, and leaves it behind when it exits).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

artifacts = Path(os.environ["LH_BENCH_ARTIFACTS"])
artifacts.mkdir(parents=True, exist_ok=True)
settings = {k: v for k, v in os.environ.items() if k.startswith(("LH_BENCH_", "AIPERF_"))}
(artifacts / "env.json").write_text(json.dumps(settings))
mode = os.environ.get("FAKE_AIPERF", "")
print("Building the dataset", flush=True)
if mode == "exit-early":
    sys.exit(3)

stopping = False


def stop(*_: object) -> None:
    global stopping
    stopping = True


IGNORE_SIGINT = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
if mode == "stubborn":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", IGNORE_SIGINT])
    (artifacts / "child.pid").write_text(str(child.pid))
else:
    signal.signal(signal.SIGINT, stop)
if mode == "orphan":
    child = subprocess.Popen([sys.executable, "-c", IGNORE_SIGINT], start_new_session=True)
    (artifacts / "child.pid").write_text(str(child.pid))
print("Credit phase start: profiling", flush=True)
started = time.monotonic()
with (artifacts / "profile_export.jsonl").open("w") as records:
    n = 0
    while not stopping:
        if mode == "exit-later" and time.monotonic() - started > 0.5:
            sys.exit(4)
        metrics = {
            "time_to_first_token": {"value": 100.0 + n % 10, "unit": "ms"},
            "usage_prompt_tokens": {"value": 1000, "unit": "tokens"},
            "usage_completion_tokens": {"value": 10, "unit": "tokens"},
        }
        meta = {"benchmark_phase": "profiling", "request_end_ns": time.time_ns()}
        records.write(json.dumps({"metadata": meta, "metrics": metrics}) + "\n")
        records.flush()
        n += 1
        time.sleep(0.02)
print("Exporting all records", flush=True)
