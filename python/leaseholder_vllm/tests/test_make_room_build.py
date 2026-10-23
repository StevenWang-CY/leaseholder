"""The make-room engine variant's build (deploy/phastform/variants/make-room/build.py) against vLLM v0.32.0's
scheduler source: its anchors still match, the module compiles, and it only adds the change's lines.
tests/test_make_room.py checks its behavior on the real scheduler."""

import difflib
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[3] / "deploy/phastform/variants/make-room/build.py"
VLLM_SRC = os.environ.get("VLLM_SRC")

pytestmark = pytest.mark.skipif(not VLLM_SRC, reason="needs VLLM_SRC, a vLLM v0.32.0 checkout")


def test_build_adds_the_change_and_nothing_else(tmp_path: Path) -> None:
    # A clean interpreter: build.py finds vLLM with find_spec, without the stand-ins conftest installs.
    env = os.environ | {"PYTHONPATH": str(VLLM_SRC)}
    subprocess.run(
        [sys.executable, str(BUILD), str(tmp_path)], env=env, check=True, capture_output=True
    )
    module = (tmp_path / "lh_make_room_scheduler.py").read_text()
    compile(module, "lh_make_room_scheduler.py", "exec")
    assert module.count("\nclass Scheduler(") == 1
    assert module.count("\nclass AsyncScheduler(Scheduler):") == 1  # over the patched Scheduler
    assert "from vllm.v1.core.sched.scheduler import Scheduler" not in module

    sched = Path(str(VLLM_SRC)) / "vllm" / "v1" / "core" / "sched"
    stock = (sched / "scheduler.py").read_text() + (sched / "async_scheduler.py").read_text()
    diff = list(difflib.ndiff(stock.splitlines(), module.splitlines()))
    removed = [line[2:].strip() for line in diff if line.startswith("- ")]
    added = "\n".join(line[2:] for line in diff if line.startswith("+ "))
    assert removed == ["from vllm.v1.core.sched.scheduler import Scheduler"]
    for line in (
        "make_room = (",
        "make_room_victims = self._make_room(waiting_req, scheduled_timestamp)",
        "if request in make_room_victims:",
        "num_new_tokens, max(token_budget - num_waiting_tokens, 0)",
        "and self._num_blocks_short(request) > 0",
        "def _make_room(self, request: Request, timestamp: float) -> set[Request]:",
        "victim, timestamp, drop_stale_output=self.requires_kv_delivery",
    ):
        assert line in added
