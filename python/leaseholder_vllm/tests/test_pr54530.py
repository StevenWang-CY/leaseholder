"""The pr54530 engine variant's build (deploy/phastform/variants/pr54530/build.py) against vLLM v0.32.0's scheduler
source: its anchors still match, the module compiles, and it changes nothing but the PR's lines."""

import difflib
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[3] / "deploy/phastform/variants/pr54530/build.py"
VLLM_SRC = os.environ.get("VLLM_SRC")

pytestmark = pytest.mark.skipif(not VLLM_SRC, reason="needs VLLM_SRC, a vLLM v0.32.0 checkout")


def test_build_applies_the_pr_and_nothing_else(tmp_path: Path) -> None:
    # A clean interpreter: build.py finds vLLM with find_spec, without the stand-ins conftest installs.
    env = os.environ | {"PYTHONPATH": str(VLLM_SRC)}
    subprocess.run(
        [sys.executable, str(BUILD), str(tmp_path)], env=env, check=True, capture_output=True
    )
    module = (tmp_path / "lh_pr54530_scheduler.py").read_text()
    compile(module, "lh_pr54530_scheduler.py", "exec")
    assert module.count("\nclass Scheduler(") == 1
    assert module.count("\nclass AsyncScheduler(Scheduler):") == 1  # over the patched Scheduler
    assert "from vllm.v1.core.sched.scheduler import Scheduler" not in module

    sched = Path(str(VLLM_SRC)) / "vllm" / "v1" / "core" / "sched"
    stock = (sched / "scheduler.py").read_text() + (sched / "async_scheduler.py").read_text()
    diff = list(difflib.ndiff(stock.splitlines(), module.splitlines()))
    removed = [line[2:].strip() for line in diff if line.startswith("- ")]
    added = "\n".join(line[2:] for line in diff if line.startswith("+ "))
    assert removed == [
        "if num_running >= self.max_num_active_reqs:",
        "from vllm.v1.core.sched.scheduler import Scheduler",
    ]
    assert "if queue_full and self.policy != SchedulingPolicy.PRIORITY:" in added
    assert "if not _preempt_lowest_priority_runner(request):" in added
