"""Build vLLM 0.32.0's scheduler with vLLM PR #54530 applied, as an importable module for ``--scheduler-cls``.

``python build.py OUT_DIR`` reads the installed vLLM's ``vllm/v1/core/sched/scheduler.py`` and ``async_scheduler.py``,
applies the PR's change, and writes ``OUT_DIR/lh_pr54530_scheduler.py``. The engine then runs with
``PYTHONPATH=OUT_DIR`` and ``--scheduler-cls lh_pr54530_scheduler.AsyncScheduler``; the installed vLLM is never
modified. The variant's lh-engine.conf runs this at every engine start.

PR #54530 (https://github.com/vllm-project/vllm/pull/54530, open): under priority scheduling, when the running queue
is full (``max_num_seqs``; ``max_num_active_reqs`` in 0.32.0), a waiting request may preempt the lowest-priority
running request if it has strictly higher priority. The PR targets a newer ``main``; 0.32.0 differs in the waiting
loop, so the change is applied by matching three anchors, each required to occur exactly once. A vLLM upgrade that
moves them fails here, loudly.

The PR changes ``Scheduler.schedule``, which vLLM's ``AsyncScheduler`` inherits, and the frozen box schedules
asynchronously (vLLM's default for this configuration). So the module also defines ``AsyncScheduler``, the stock
class over the patched ``Scheduler``, and the variant loads that one: with and without the PR differ only by the PR.

The generated module is vLLM's code (Apache-2.0, Copyright contributors to the vLLM project; its SPDX lines are
kept) with the PR's change; this script, like the rest of the repository, is Apache-2.0. Finding vLLM through
``importlib.util.find_spec`` does not import it, so this runs without torch or a GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CLOSURE = '''
        def _preempt_lowest_priority_runner(request: Request) -> bool:
            """Evict the lowest-priority running request so that a higher-priority waiting request can take its slot
            once the running queue is full (vLLM PR #54530). Strict priority comparison: equal priorities never preempt
            each other, which would ping-pong."""
            nonlocal token_budget, input_budget, encoder_compute_budget
            if not self.running:
                return False
            victim = max(self.running, key=lambda r: (r.priority, r.arrival_time))
            if request.priority >= victim.priority:
                return False
            self.running.remove(victim)
            for scheduled in (scheduled_running_reqs, scheduled_new_reqs, scheduled_resumed_reqs):
                if victim in scheduled:
                    scheduled.remove(victim)
                    restored = num_scheduled_tokens.pop(victim.request_id)
                    token_budget += restored
                    input_budget += restored + draft_slots
                    req_to_new_blocks.pop(victim.request_id)
                    scheduled_spec_decode_tokens.pop(victim.request_id, None)
                    encoder_inputs = scheduled_encoder_inputs.pop(victim.request_id, None)
                    if encoder_inputs:
                        encoder_compute_budget += sum(victim.get_num_encoder_embeds(i) for i in encoder_inputs)
                    if self.lora_config and victim.lora_request:
                        lora_id = victim.lora_request.lora_int_id
                        if not any(
                            r.lora_request is not None and r.lora_request.lora_int_id == lora_id
                            for r in itertools.chain(scheduled_running_reqs, scheduled_new_reqs, scheduled_resumed_reqs)
                        ):
                            scheduled_loras.discard(lora_id)
                    break
            self._preempt_request(victim, scheduled_timestamp, drop_stale_output=self.requires_kv_delivery)
            preempted_reqs.append(victim)
            return True

'''

SCHEDULER_EDITS = [
    (
        "        # Next, schedule the WAITING requests.\n",
        CLOSURE + "        # Next, schedule the WAITING requests.\n",
    ),
    (
        "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
        "                if num_running >= self.max_num_active_reqs:\n"
        "                    break\n",
        "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
        "                queue_full = num_running >= self.max_num_active_reqs\n"
        "                if queue_full and self.policy != SchedulingPolicy.PRIORITY:\n"
        "                    break\n",
    ),
    (
        "                num_external_computed_tokens = 0\n                load_kv_async = False\n",
        "                if queue_full:\n"
        "                    # Running queue full: evict a strictly lower-priority runner, then retry this request.\n"
        "                    if not _preempt_lowest_priority_runner(request):\n"
        "                        break\n"
        "                    continue\n"
        "\n"
        "                num_external_computed_tokens = 0\n                load_kv_async = False\n",
    ),
]
# The stock AsyncScheduler, appended after the patched Scheduler, extends it instead of importing the stock one.
ASYNC_EDITS = [("from vllm.v1.core.sched.scheduler import Scheduler\n", "")]


def sched_dir() -> Path:
    """The installed vLLM's scheduler package, found without importing vLLM (which needs a GPU stack)."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("build.py: vllm is not installed in this interpreter")
    return Path(next(iter(spec.submodule_search_locations))) / "v1" / "core" / "sched"


def patched(path: Path, edits: list[tuple[str, str]]) -> str:
    source = path.read_text()
    for anchor, replacement in edits:
        count = source.count(anchor)
        if count != 1:
            sys.exit(f"build.py: anchor found {count} times in {path}, expected once:\n{anchor}")
        source = source.replace(anchor, replacement, 1)
    return source


def build(out_dir: Path) -> Path:
    sched = sched_dir()
    header = (
        '"""vLLM v1 Scheduler and AsyncScheduler with PR #54530 applied, generated by Leaseholder\'s '
        'deploy/phastform/variants/pr54530/build.py. Do not edit."""\n'
    )
    source = (
        header
        + patched(sched / "scheduler.py", SCHEDULER_EDITS)
        + "\n\n"
        + patched(sched / "async_scheduler.py", ASYNC_EDITS)
    )
    compile(source, "lh_pr54530_scheduler.py", "exec")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "lh_pr54530_scheduler.py"
    target.write_text(source)
    return target


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
