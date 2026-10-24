"""The make-room engine variant's build (deploy/phastform/variants/make-room/build.py) against vLLM v0.32.0's
scheduler source: its anchors still match, the module compiles, it only adds the change's lines, and it refuses a
vLLM other than 0.32.0. tests/test_make_room.py checks its behavior on the real scheduler."""

import difflib
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[3] / "deploy/phastform/variants/make-room/build.py"
VLLM_SRC = os.environ.get("VLLM_SRC")

pytestmark = pytest.mark.skipif(not VLLM_SRC, reason="needs VLLM_SRC, a vLLM v0.32.0 checkout")


def build(out: Path, *flags: str, path: tuple[Path, ...] = ()) -> subprocess.CompletedProcess:
    """build.py in a clean interpreter that finds vLLM, with find_spec, in VLLM_SRC alone, after ``path``."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join([*map(str, path), str(VLLM_SRC)])}
    return subprocess.run(
        [sys.executable, str(BUILD), *flags, str(out)], env=env, capture_output=True, text=True
    )


def installed(tmp_path: Path, version: str) -> Path:
    """A directory on which importlib.metadata finds an installed vLLM of ``version``."""
    dist_info = tmp_path / "site" / f"vllm-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n")
    return dist_info.parent


def test_build_adds_the_change_and_nothing_else(tmp_path: Path) -> None:
    assert build(tmp_path, "--source-tree").returncode == 0
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
    # async_scheduler.py's import of the stock Scheduler, and its logger, which would rebind the module's.
    assert sorted(removed) == [
        "from vllm.v1.core.sched.scheduler import Scheduler",
        "logger = init_logger(__name__)",
    ]
    for line in (
        'logger = init_logger("vllm.v1.core.sched.scheduler")',
        "make_room = self._make_room_switches(",
        "waiting_req = self._make_room_waiting_req()",
        "make_room_victims = self._make_room(",
        "first_tokens_priority = self._make_room_first_tokens_priority(",
        "if request in make_room_victims:",
        "self._make_room_count_hold(num_new_tokens, 0)",
        "if request_queue is self.waiting and self._make_room_holds_back(",
        "self._make_room_untried.discard(request)",
        "def _make_room(",
        "victim, timestamp, drop_stale_output=self.requires_kv_delivery",
    ):
        assert line in added


def test_build_refuses_another_installed_vllm(tmp_path: Path) -> None:
    """Its edits match 0.32.0's source: another installed version fails the build, even if every anchor still
    matches, as they do in VLLM_SRC here; 0.32.0 builds."""
    result = build(tmp_path / "out", path=(installed(tmp_path / "other", "0.31.0"),))
    assert result.returncode != 0
    assert "written against vLLM 0.32.0, and vLLM 0.31.0 is installed" in result.stderr
    assert not (tmp_path / "out").exists()
    assert build(tmp_path / "ok", path=(installed(tmp_path / "same", "0.32.0"),)).returncode == 0


def test_build_needs_source_tree_for_a_checkout(tmp_path: Path) -> None:
    """A source checkout has no package metadata to check: only --source-tree builds from one."""
    result = build(tmp_path / "out")
    assert result.returncode != 0 and "--source-tree builds from a source checkout" in result.stderr
    assert build(tmp_path / "out", "--source-tree").returncode == 0
