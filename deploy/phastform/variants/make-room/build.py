"""Build vLLM 0.32.0's scheduler with Leaseholder's "make room" change, as an importable module for
``--scheduler-cls``.

``python build.py OUT_DIR`` reads the installed vLLM's ``vllm/v1/core/sched/scheduler.py`` and ``async_scheduler.py``,
applies the change, and writes ``OUT_DIR/lh_make_room_scheduler.py``. The engine then runs with ``PYTHONPATH=OUT_DIR``
and ``--scheduler-cls lh_make_room_scheduler.AsyncScheduler``; the installed vLLM is never modified. The variant's
lh-engine.conf runs this at every engine start.

The change is the KV case of vLLM issue #40004, which PR #54530 leaves open: under priority scheduling, vLLM 0.32
preempts only when a running request cannot grow, so a waiting request of a higher priority waits behind running
requests of a lower priority for KV and for step budget. Each part is a switch:

1. ``max_preemptions_per_step``: before the running requests are scheduled in a step, the waiting request of the
   highest priority that lacks free KV blocks for its whole remaining sequence preempts running requests of a
   strictly lower priority, the lowest priority first and, within it, the one whose sequence takes the most blocks
   (the most KV, unless shared), the latest arrival among equals, through the stock ``_preempt_request``. As few as needed, at most this many, and none if that many would not make
   it fit. Preempting before the running loop means no victim is un-scheduled (vLLM #59907), and a victim is not
   resumed in the step that preempted it. Blocks whose free is deferred (async scheduling with a KV-consumer
   connector, as on the frozen box) count as free: they, like the victims', are free by the next step, when the
   request is admitted; blocks a connector's pending transfers will free are waited for, as the running loop waits.
2. ``budget_cap``: a running prefill of a lower priority than that waiting request leaves the waiting request's first
   chunk in the step's token budget. Decodes are never cut.
3. ``reserve_tokens``: a waiting request of a positive priority value is admitted only if its whole remaining
   sequence, the blocks in-flight prefills still need and this reserve all fit in the free blocks, so a request of
   priority 0 (Leaseholder's realtime class) usually finds room without preempting anyone. With
   ``max_preemptions_per_step`` on, the same check, with a reserve of 0, holds back a request while a prefill of a
   higher priority is in flight: a victim, back in the waiting queue with its blocks still cached, would otherwise
   take them back between the chunks of the request it made room for.

The switches are keys of ``--additional-config``'s ``make_room`` table, read once at scheduler init and ignored unless
the policy is priority; with none set, the module schedules exactly as stock. The edits match anchors in 0.32.0's
source, each required to occur exactly once, so a vLLM upgrade that moves one fails here, loudly.

The change targets ``Scheduler.schedule``, which vLLM's ``AsyncScheduler`` inherits, and the frozen box schedules
asynchronously (vLLM's default for this configuration). So the module also defines ``AsyncScheduler``, the stock
class over the patched ``Scheduler``, and the variant loads that one, as the pr54530 variant does.

The generated module is vLLM's code (Apache-2.0, Copyright contributors to the vLLM project; its SPDX lines are
kept) with the change; this script, like the rest of the repository, is Apache-2.0. Finding vLLM through
``importlib.util.find_spec`` does not import it, so this runs without torch or a GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The inserted code is written as vLLM's own (88 columns, ruff format), so that it lifts into scheduler.py as is.
SWITCHES = """        # Make room for waiting requests of a higher priority: preempt running
        # requests of a lower priority for their KV, cap lower-priority prefill
        # chunks for their first chunk, and keep a KV reserve that requests of a
        # positive priority value leave free.
        make_room = (
            vllm_config.additional_config.get("make_room", {})
            if self.policy == SchedulingPolicy.PRIORITY
            else {}
        )
        self.max_preemptions_per_step = make_room.get("max_preemptions_per_step", 0)
        self.prefill_budget_cap = make_room.get("budget_cap", False)
        self.reserve_blocks = cdiv(make_room.get("reserve_tokens", 0), block_size)
        num_max_len_blocks = cdiv(self.max_model_len, block_size)
        if self.reserve_blocks and (
            self.reserve_blocks + num_max_len_blocks >= num_gpu_blocks
        ):
            raise ValueError(
                "make_room.reserve_tokens leaves no room for a request of "
                "max_model_len tokens"
            )
"""

MAKE_ROOM = """        # Make room for the waiting request of the highest priority before the
        # running requests are scheduled, so that no victim is un-scheduled.
        waiting_req = self._make_room_waiting_req()
        make_room_victims: set[Request] = set()
        if waiting_req is not None and self.max_preemptions_per_step:
            make_room_victims = self._make_room(waiting_req, scheduled_timestamp)

"""

NOT_RESUMED = """                if request in make_room_victims:
                    # Preempted in this step to make room: resuming it now would
                    # take back the blocks its preemption has just freed.
                    break
"""

BUDGET_CAP = """            if (
                self.prefill_budget_cap
                and waiting_req is not None
                and request.is_prefill_chunk
                and request.priority > waiting_req.priority
            ):
                # Leave the waiting request its first chunk; decodes are not cut.
                num_waiting_tokens = (
                    waiting_req.num_tokens - waiting_req.num_computed_tokens
                )
                if 0 < long_prefill_token_threshold < num_waiting_tokens:
                    num_waiting_tokens = long_prefill_token_threshold
                num_new_tokens = min(
                    num_new_tokens, max(token_budget - num_waiting_tokens, 0)
                )
"""

RESERVE = """                if (
                    request_queue is self.waiting
                    and (
                        (self.reserve_blocks and request.priority > 0)
                        or (
                            self.max_preemptions_per_step
                            and any(
                                req.priority < request.priority
                                for req in self._inflight_prefills
                            )
                        )
                    )
                    and self._num_blocks_short(request) > 0
                ):
                    # A positive priority value leaves the KV reserve free, and a
                    # priority below an in-flight prefill's leaves free what
                    # in-flight prefills still need: otherwise a request preempted
                    # to make room could take back its cached blocks between the
                    # chunks of the request it made room for.
                    break

"""

HELPERS = '''
    def _make_room_waiting_req(self) -> Request | None:
        """The waiting request of the highest priority, if make-room is on and
        it is ready to be scheduled."""
        if (
            not (self.max_preemptions_per_step or self.prefill_budget_cap)
            or self._pause_state != PauseState.UNPAUSED
        ):
            return None
        queues = (self.kv_holding_waiting, self.waiting)
        request = min((queue.peek_request() for queue in queues if queue), default=None)
        if request is None or self._is_blocked_waiting_status(request.status):
            return None
        return request

    def _num_blocks_short(self, request: Request) -> int:
        """Free blocks `request` lacks to be admitted with its whole remaining
        sequence beside what in-flight prefills still need, as async KV loads
        are admitted; with a positive priority value, the reserve too."""
        num_blocks = self._inflight_prefill_reserved_blocks()
        if request not in self._inflight_prefills:
            num_blocks += self._request_remaining_blocks(request)
        if self.running:
            # As allocate_slots applies the watermark.
            num_blocks += self.kv_cache_manager.watermark_blocks
        if request.priority > 0:
            num_blocks += self.reserve_blocks
        return num_blocks - self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _make_room(self, request: Request, timestamp: float) -> set[Request]:
        """Preempt running requests of a lower priority than `request` until it
        is no block short, and return them: the lowest priority first and,
        within it, the one whose sequence takes the most blocks (the most KV,
        unless shared), so that as few as possible are preempted, and the latest
        arrival among equals, as vLLM's own preemption picks; none if
        `max_preemptions_per_step` of them would not do. Deferred frees count as
        free: like the victims', they land by the next step."""
        if self.connector is not None and self.connector.has_pending_block_frees():
            # Pending transfers free blocks instead, as the running loop waits.
            return set()
        candidates = [req for req in self.running if req.priority > request.priority]
        if not candidates:
            return set()
        num_short = self._num_blocks_short(request) - self._num_freeable_blocks(
            *(blocks for _, blocks in self.deferred_frees)
        )
        if num_short <= 0:
            return set()
        candidates.sort(
            key=lambda req: (
                req.priority,
                cdiv(req.num_tokens, self.block_size),
                req.arrival_time,
            ),
            reverse=True,
        )
        victims: list[Request] = []
        for victim in candidates[: self.max_preemptions_per_step]:
            victims.append(victim)
            num_short -= self._num_released_blocks(victim)
            if num_short <= 0:
                break
        if num_short > 0:
            return set()
        for victim in victims:
            self.running.remove(victim)
            self._preempt_request(
                victim, timestamp, drop_stale_output=self.requires_kv_delivery
            )
        return set(victims)

    def _num_released_blocks(self, request: Request) -> int:
        """Blocks preempting `request` gives back: those it holds alone, and
        those it still reserves as an in-flight prefill."""
        blocks = self.kv_cache_manager.get_blocks(request.request_id).blocks
        num_blocks = self._num_freeable_blocks(*blocks)
        if request in self._inflight_prefills:
            num_blocks += self._request_remaining_blocks(request)
        return num_blocks

    def _num_freeable_blocks(self, *blocks: list[KVCacheBlock]) -> int:
        """Blocks that go back to the free pool once released: not shared with
        another holder, not the null block and not of another pool."""
        block_pool = self.kv_cache_manager.block_pool
        return sum(
            block.ref_cnt == 1 and block.pool is block_pool and not block.is_null
            for group in blocks
            for block in group
        )
'''


def before(anchor: str, code: str) -> tuple[str, str]:
    return anchor, code + anchor


def after(anchor: str, code: str) -> tuple[str, str]:
    return anchor, anchor + code


# Each edit inserts code before or after its anchor; no line of vLLM's changes.
SCHEDULER_EDITS = [
    after(
        "        self.scheduler_reserve_full_isl = (\n"
        "            self.scheduler_config.scheduler_reserve_full_isl\n"
        "        )\n",
        SWITCHES,
    ),
    before("        # First, schedule the RUNNING requests.\n", MAKE_ROOM),
    after(
        "                request = request_queue.peek_request()\n"
        "                request_id = request.request_id\n",
        NOT_RESUMED,
    ),
    after(
        "            num_new_tokens = min(\n"
        "                num_new_tokens, token_budget, input_budget - draft_slots\n"
        "            )\n",
        BUDGET_CAP,
    ),
    before(
        "                num_external_computed_tokens = 0\n                load_kv_async = False\n",
        RESERVE,
    ),
    after(
        "        return sum(\n"
        "            self._request_remaining_blocks(req) for req in self._inflight_prefills\n"
        "        )\n",
        HELPERS,
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
        '"""vLLM v1 Scheduler and AsyncScheduler with Leaseholder\'s make-room change, generated by '
        'deploy/phastform/variants/make-room/build.py. Do not edit."""\n'
    )
    source = (
        header
        + patched(sched / "scheduler.py", SCHEDULER_EDITS)
        + "\n\n"
        + patched(sched / "async_scheduler.py", ASYNC_EDITS)
    )
    compile(source, "lh_make_room_scheduler.py", "exec")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "lh_make_room_scheduler.py"
    target.write_text(source)
    return target


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
