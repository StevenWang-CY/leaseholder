"""Build vLLM 0.32.0's scheduler with Leaseholder's "make room" change, as an importable module for
``--scheduler-cls``.

``python build.py [--source-tree] OUT_DIR`` reads the installed vLLM's ``vllm/v1/core/sched/scheduler.py`` and
``async_scheduler.py``, applies the change, and writes ``OUT_DIR/lh_make_room_scheduler.py``. The engine then runs with
``PYTHONPATH=OUT_DIR`` and ``--scheduler-cls lh_make_room_scheduler.AsyncScheduler``; the installed vLLM is never
modified. The variant's lh-engine.conf runs this at every engine start. The edits match 0.32.0's source, so the build
refuses any other installed version; ``--source-tree`` lets the tests build from a v0.32.0 source checkout, which has
no package metadata to check.

The change is the KV case of vLLM issue #40004, which PR #54530 leaves open: under priority scheduling, vLLM 0.32
preempts only when a running request cannot grow, so a waiting request of a higher priority waits behind running
requests of a lower priority for KV and for step budget. It acts for the waiting request of the highest priority
that the waiting loop can schedule in the step: not one loading its KV (until the load is done), compiling its
grammar, paused for streaming input or awaiting output from before its preemption (in flight under pipeline
parallelism), nor one whose LoRA adapter could not join the running requests' (max_loras; it counts every running
request's adapter, as the waiting loop counts the scheduled ones'). Each part is a switch:

1. ``max_preemptions_per_step``: before the running requests are scheduled in a step, if that request lacks free KV
   blocks for its whole remaining sequence (net of its prefix-cache hit), it preempts running requests of a strictly
   lower priority through the stock ``_preempt_request``, in the order vLLM's own preemption takes them (the highest
   priority value, then the latest arrival), as few as make it fit, at most this many, and none if that many would
   not. A victim releases the blocks all of whose holders are victims, except those of the request's prefix-cache hit,
   which it takes as hits; the room counts the blocks the running decodes take until the victims' frees land (one
   step, or two with deferred frees). Preempting before the running loop means no victim is un-scheduled (vLLM
   #59907), and a victim is not resumed in the step that preempted it. If the waiting loop then does not try the
   request's allocation in that step (it skipped it), no room is made for it again until it has. Blocks whose free is
   deferred (async scheduling with a KV-consumer connector, as on the frozen box) count as free: they, like the
   victims', are free by the next step, when the request is admitted; blocks a connector's pending transfers will free
   are waited for, as the running loop waits.
2. ``budget_cap``: prefills of a lower priority, running or waiting (a waiting one that holds its KV is skipped, the
   others wait), sit out the steps of a request's first tokens: from the step in which the waiting request fits
   (deferred frees counted), or, for a request loading its KV from a connector, in which its load is done, while it
   prefills, until it has its first ``budget_cap_tokens`` output tokens (16 by default). Those steps carry its tokens
   and the decodes alone; decodes are never cut. A load in flight holds too, but only for its first
   ``budget_cap_tokens`` steps, so a stuck load costs lower-priority prefill no more than that. A waiting request that
   does not fit holds nothing, so prefills that will free its blocks go on; nor does any request while every running
   slot is taken, since it could not be promoted. The scheduler counts the steps it holds, the prefill tokens and admissions
   it holds back in them (a request held back in several steps counts in each) and make-room's preemptions, and logs
   them at most every 10 seconds.
3. ``reserve_tokens``: a waiting request of a positive priority value is admitted only if its whole remaining
   sequence (net of its prefix-cache hit), the blocks in-flight prefills still need and this reserve all fit in the
   free blocks, so a request of priority 0 (Leaseholder's realtime class) usually finds room without preempting
   anyone. With ``max_preemptions_per_step`` on, the same check, with a reserve of 0, holds back a waiting request
   while a prefill of a higher priority is in flight: a victim, back in the waiting queue with its blocks still
   cached, would otherwise take them back between the chunks of the request it made room for. Neither check holds
   back a request that holds its KV.

For an encoder-decoder model, a request's remaining sequence includes its cross-attention blocks. The switches are
keys of ``--additional-config``'s ``make_room`` table, read once at scheduler init. An unknown key, a bad value or a
switch under a policy other than priority fails the engine start, naming it. With none on, the module schedules
exactly as stock. The edits match anchors in 0.32.0's source, each required to occur exactly once.

The change targets ``Scheduler.schedule``, which vLLM's ``AsyncScheduler`` inherits, and the frozen box schedules
asynchronously (vLLM's default for this configuration). So the module also defines ``AsyncScheduler``, the stock
class over the patched ``Scheduler``, and the variant loads that one, as the pr54530 variant does.

The generated module is vLLM's code (Apache-2.0, Copyright contributors to the vLLM project; its SPDX lines are
kept) with the change; this script, like the rest of the repository, is Apache-2.0. Finding vLLM through
``importlib.util.find_spec`` does not import it, so this runs without torch or a GPU.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

VLLM_VERSION = "0.32.0"

# The generated module's own name is not one vLLM's logging config covers, so the module logs under the name of
# vLLM's scheduler; ASYNC_EDITS drops async_scheduler.py's own logger, which would rebind the module's after it.
LOGGER = """# Log as vLLM's scheduler does: vLLM's logging covers its loggers only.
logger = init_logger("vllm.v1.core.sched.scheduler")
"""

# The inserted code is written as vLLM's own (88 columns, ruff format), so that it lifts into scheduler.py as is.
SWITCHES = """        # Make room for waiting requests of a higher priority: preempt running
        # requests of a lower priority for their KV, hold lower-priority prefill
        # back for a request's first tokens, and keep a KV reserve that requests
        # of a positive priority value leave free.
        make_room = self._make_room_switches(
            vllm_config.additional_config.get("make_room", {})
        )
        self.max_preemptions_per_step = make_room["max_preemptions_per_step"]
        self.prefill_budget_cap = make_room["budget_cap"]
        self.prefill_budget_cap_tokens = make_room["budget_cap_tokens"]
        self.reserve_blocks = cdiv(make_room["reserve_tokens"], block_size)
        num_max_len_blocks = cdiv(self.max_model_len, block_size)
        if self.reserve_blocks and (
            self.reserve_blocks + num_max_len_blocks >= num_gpu_blocks
        ):
            raise ValueError(
                "make_room.reserve_tokens leaves no room for a request of "
                "max_model_len tokens"
            )
        # Requests make-room preempted for whose allocation the waiting loop has
        # not tried since: no room is made for them again until it has.
        self._make_room_untried: set[Request] = set()
        # The step in which make-room first saw each KV load in flight: a load
        # holds lower-priority prefill back for its first budget_cap_tokens steps.
        self._make_room_load_start: dict[Request, int] = {}
        # What make-room did, for the operator (see _make_room_log).
        self.make_room_held_steps = 0
        self.make_room_held_tokens = 0
        self.make_room_held_admissions = 0
        self.make_room_preemptions = 0
        self._make_room_held_step = -1
        self._make_room_logged = (0, 0, 0, 0)
        self._make_room_log_time = time.monotonic()
"""

MAKE_ROOM = """        # Make room for the waiting request of the highest priority that can be
        # scheduled in this step, before the running requests are scheduled, so
        # that no victim is un-scheduled.
        make_room_victims: set[Request] = set()
        first_tokens_priority = None
        if self.max_preemptions_per_step or self.prefill_budget_cap:
            self._make_room_log()
            waiting_req = self._make_room_waiting_req()
            hit = None if waiting_req is None else self._make_room_hit(waiting_req)
            hold_priority = self._make_room_hold_priority()
            if (
                waiting_req is not None
                and self.max_preemptions_per_step
                and waiting_req not in self._make_room_untried
                # Nor while the budget cap holds it back.
                and (hold_priority is None or waiting_req.priority <= hold_priority)
            ):
                make_room_victims = self._make_room(
                    waiting_req, hit, scheduled_timestamp
                )
            # Prefills of a lower priority sit out the steps of a request's first
            # tokens.
            first_tokens_priority = self._make_room_first_tokens_priority(
                waiting_req, hit, hold_priority
            )

"""

NOT_RESUMED = """                if request in make_room_victims:
                    # Preempted in this step to make room: resuming it now would
                    # take back the blocks its preemption has just freed.
                    break
"""

BUDGET_CAP = """            if (
                first_tokens_priority is not None
                and request.is_prefill_chunk
                and request.priority > first_tokens_priority
            ):
                # A request of a higher priority awaits its first tokens: this
                # step is theirs and the decodes'.
                self._make_room_count_hold(num_new_tokens, 0)
                num_new_tokens = 0
"""

RESERVE = """                if (
                    first_tokens_priority is not None
                    and request.priority > first_tokens_priority
                ):
                    # Nor does a prefill of a lower priority start meanwhile.
                    self._make_room_count_hold(0, 1)
                    if request_queue is self.kv_holding_waiting:
                        # Requests holding KV are drained first, so the request
                        # whose first tokens are due may be behind this one.
                        skip_request(request_queue)
                        continue
                    break
                if request_queue is self.waiting and self._make_room_holds_back(
                    request
                ):
                    # Nor one the reserve, or a prefill of a higher priority in
                    # flight, keeps waiting.
                    break

"""

TRIED = """                # The waiting loop tries the request's allocation: make-room may
                # make room for it again.
                self._make_room_untried.discard(request)
"""

HELPERS = '''
    def _make_room_switches(self, make_room: Any) -> dict[str, Any]:
        """make_room's switches, from --additional-config's make_room table,
        validated at engine start: a bad key or value fails it, named."""
        switches: dict[str, Any] = {
            "max_preemptions_per_step": 0,
            "budget_cap": False,
            "budget_cap_tokens": 16,
            "reserve_tokens": 0,
        }
        if not isinstance(make_room, dict):
            raise ValueError(f"make_room must be a table, not {make_room!r}")
        for key, value in make_room.items():
            if key == "budget_cap":
                flag = {"true": True, "1": True, "false": False, "0": False}.get(
                    str(value).lower()
                )
                if flag is None:
                    raise ValueError(
                        f"make_room.budget_cap must be true or false, not {value!r}"
                    )
                switches[key] = flag
            elif key in switches:
                least = 1 if key == "budget_cap_tokens" else 0
                if isinstance(value, bool) or not isinstance(value, int) or (
                    value < least
                ):
                    raise ValueError(
                        f"make_room.{key} must be an integer of at least {least}, "
                        f"not {value!r}"
                    )
                switches[key] = value
            else:
                raise ValueError(
                    f"make_room has no key {key!r}; its keys are "
                    + ", ".join(switches)
                )
        if self.policy != SchedulingPolicy.PRIORITY and (
            switches["max_preemptions_per_step"]
            or switches["budget_cap"]
            or switches["reserve_tokens"]
        ):
            raise ValueError("make_room needs --scheduling-policy priority")
        return switches

    def _make_room_log(self) -> None:
        """Log make-room's totals, at most every 10 s and when they moved: the
        steps in which the budget cap held lower-priority prefill back, the
        prefill tokens and admissions it held back in them, summed over the
        steps, and the requests make-room preempted, apart from the running
        loop's preemptions."""
        now = time.monotonic()
        if now - self._make_room_log_time < 10:
            return
        self._make_room_log_time = now
        totals = (
            self.make_room_held_steps,
            self.make_room_held_tokens,
            self.make_room_held_admissions,
            self.make_room_preemptions,
        )
        if totals != self._make_room_logged:
            self._make_room_logged = totals
            logger.info(
                "make-room: budget cap held lower-priority prefill back in %d "
                "steps (%d prefill tokens and %d admissions, summed over the "
                "steps); %d preemptions to make room",
                *totals,
            )

    def _make_room_count_hold(self, num_tokens: int, num_admissions: int) -> None:
        """Count a hold of the budget cap: the step, and the lower-priority
        prefill tokens and admissions it holds back in it. A request held back
        in several steps counts in each."""
        if not (num_tokens or num_admissions):
            return
        if self._make_room_held_step != self.current_step:
            self._make_room_held_step = self.current_step
            self.make_room_held_steps += 1
        self.make_room_held_tokens += num_tokens
        self.make_room_held_admissions += num_admissions

    def _make_room_waiting_req(self) -> Request | None:
        """The waiting request of the highest priority that the waiting loop can
        schedule in this step: not one loading its KV (until the load is done),
        compiling its grammar, paused for streaming input or awaiting output from
        before its preemption, nor one whose LoRA adapter could not join the
        running requests' (max_loras). It counts every running request's
        adapter, where the waiting loop counts the scheduled ones': one the budget
        cap holds back is not scheduled, so this may make no room for a request
        the loop would admit."""
        if self._pause_state != PauseState.UNPAUSED:
            return None
        if self._make_room_untried:
            # Forget the requests that are gone.
            self._make_room_untried = {
                req
                for req in self._make_room_untried
                if self.requests.get(req.request_id) is req
            }
        loras: set[int] = set()
        if self.lora_config:
            loras = {
                req.lora_request.lora_int_id
                for req in self.running
                if req.lora_request and req.lora_request.lora_int_id > 0
            }
        # Each queue iterates in priority order.
        heads = (
            next((req for req in queue if self._make_room_ready(req, loras)), None)
            for queue in (self.kv_holding_waiting, self.waiting)
        )
        return min((req for req in heads if req is not None), default=None)

    def _make_room_ready(self, request: Request, loras: set[int]) -> bool:
        """Whether the waiting loop can schedule `request` in this step, as far
        as its status and its LoRA adapter beside the running requests' tell."""
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
        elif request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            struct_output_req = request.structured_output_request
            if not struct_output_req or struct_output_req.grammar is None:
                return False
            if isinstance(struct_output_req.grammar, Exception):
                return False
        elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            return False
        if request.num_stale_output_tokens and not request.drop_stale_output:
            # Output from before its preemption is still in flight (pipeline
            # parallelism): it resumes once that is delivered.
            return False
        return not (
            self.lora_config
            and request.lora_request
            and len(loras) >= self.lora_config.max_loras
            and request.lora_request.lora_int_id not in loras
        )

    def _make_room_hit(
        self, request: Request
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | None:
        """`request`'s local prefix-cache hit before its first step, its blocks
        by group and its tokens, as get_computed_blocks finds it but without its
        KV events; None if it looks up none."""
        if (
            request.num_computed_tokens
            or not self.kv_cache_manager.prefix_cache_lookup_enabled(request)
        ):
            return None
        hit_blocks, num_hit_tokens, _ = (
            self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
        )
        return hit_blocks, num_hit_tokens

    def _make_room_hold_priority(self) -> int | None:
        """The highest priority among the requests whose first tokens are due,
        if the budget cap is on: the running ones with fewer than
        `budget_cap_tokens` output tokens and, while a running slot is free, the
        ones whose KV load is done, which the waiting loop promotes in this step
        but only on budget the running loop leaves, and the ones whose load is
        in flight, for its first `budget_cap_tokens` steps: a stuck load holds
        nothing after those."""
        if not self.prefill_budget_cap:
            return None
        # Each load in flight and the step make-room first saw it in; the loads
        # that are done or gone drop out.
        self._make_room_load_start = {
            req: self._make_room_load_start.get(req, self.current_step)
            for req in self._inflight_prefills
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and req.request_id not in self.finished_recving_kv_req_ids
        }
        priorities = [
            req.priority
            for req in self.running
            if req.num_output_tokens < self.prefill_budget_cap_tokens
        ]
        if self._make_room_slot_free():
            priorities.extend(
                self.requests[req_id].priority
                for req_id in self.finished_recving_kv_req_ids
                if req_id in self.requests
            )
            priorities.extend(
                req.priority
                for req, start in self._make_room_load_start.items()
                if self.current_step - start < self.prefill_budget_cap_tokens
            )
        return min(priorities, default=None)

    def _make_room_first_tokens_priority(
        self,
        waiting_req: Request | None,
        hit: tuple[tuple[list[KVCacheBlock], ...], int] | None,
        hold_priority: int | None,
    ) -> int | None:
        """The highest priority whose first tokens hold lower-priority prefill
        back in this step, if the budget cap is on: `hold_priority`, and that of
        `waiting_req` (whose prefix-cache hit is `hit`) once it fits, deferred
        frees counted, while a running slot is free."""
        if not self.prefill_budget_cap:
            return None
        priorities = [] if hold_priority is None else [hold_priority]
        if (
            waiting_req is not None
            and self._make_room_slot_free()
            and self._num_blocks_short(waiting_req, hit)
            <= self._num_released_blocks([], hit)
        ):
            priorities.append(waiting_req.priority)
        return min(priorities, default=None)

    def _make_room_slot_free(self) -> bool:
        """Whether a running slot is free: a request admitted with its KV loading
        from a connector, or waiting, needs one to be promoted, and with none
        free, holding prefill back would only stall the running requests."""
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        return num_active < self.max_num_active_reqs

    def _make_room_holds_back(self, request: Request) -> bool:
        """Whether the reserve keeps waiting `request`, which is short of free
        blocks: a positive priority value leaves the KV reserve free, and a
        priority below an in-flight prefill's leaves free what in-flight
        prefills still need, so that a request preempted to make room cannot
        take back its cached blocks between the chunks of the request it made
        room for."""
        held = (self.reserve_blocks and request.priority > 0) or (
            self.max_preemptions_per_step
            and any(req.priority < request.priority for req in self._inflight_prefills)
        )
        return (
            bool(held)
            and self._num_blocks_short(request, self._make_room_hit(request)) > 0
        )

    def _num_blocks_short(
        self,
        request: Request,
        hit: tuple[tuple[list[KVCacheBlock], ...], int] | None,
    ) -> int:
        """Free blocks `request` lacks to be admitted with its whole remaining
        sequence, net of its prefix-cache hit `hit`, beside what in-flight
        prefills still need, as async KV loads are admitted; with a positive
        priority value, the reserve too."""
        num_blocks = self._inflight_prefill_reserved_blocks()
        if request not in self._inflight_prefills:
            num_blocks += self._make_room_remaining_blocks(request, hit)
        if self.running:
            # As allocate_slots applies the watermark.
            num_blocks += self.kv_cache_manager.watermark_blocks
        if request.priority > 0:
            num_blocks += self.reserve_blocks
        return num_blocks - self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _make_room_remaining_blocks(
        self,
        request: Request,
        hit: tuple[tuple[list[KVCacheBlock], ...], int] | None,
    ) -> int:
        """Blocks `request` still needs for its whole remaining sequence, as
        allocate_slots admits it: net of its prefix-cache hit `hit`, whose blocks
        other requests hold take none and whose blocks in the free queue take
        one each, and with an encoder-decoder model's cross-attention blocks."""
        hit_blocks, num_computed_tokens = hit or (
            self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            request.num_computed_tokens,
        )
        num_encoder_tokens = 0
        if self.is_encoder_decoder:
            num_encoder_tokens = sum(
                request.get_num_encoder_embeds(i)
                for i in range(request.num_encoder_inputs)
            )
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_tokens = full_num_tokens
        if self.num_spec_tokens:
            num_tokens += 1 + self.num_spec_tokens + self.num_lookahead_tokens
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens,
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_computed_tokens,
            num_local_computed_tokens=num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )

    def _num_growth_blocks(self, request: Request, num_steps: int) -> int:
        """New blocks the running loop allocates `request` in the next
        `num_steps` steps: a decode's tokens (and lookahead) may start blocks.
        An in-flight prefill's are among the blocks it still reserves."""
        if request in self._inflight_prefills:
            return 0
        num_tokens_main_model = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            + (num_steps - 1) * (1 + self.num_spec_tokens)
        )
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(
                num_tokens_main_model + self.num_lookahead_tokens, self.max_model_len
            ),
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

    def _make_room(
        self,
        request: Request,
        hit: tuple[tuple[list[KVCacheBlock], ...], int] | None,
        timestamp: float,
    ) -> set[Request]:
        """Preempt running requests of a lower priority than `request` until it
        is no block short, and return them. They go in the order vLLM's own
        preemption takes them, the highest priority value first and the latest
        arrival among equals, which spares older work; as few as make it fit, at
        most `max_preemptions_per_step`, and none if that many would not. The
        room counts the blocks the running decodes take until the victims'
        frees land, and a block as released once all its holders go, unless
        `request` takes it as a prefix-cache hit (`hit`)."""
        if self.connector is not None and self.connector.has_pending_block_frees():
            # Pending transfers free blocks instead, as the running loop waits.
            return set()
        candidates = sorted(
            (req for req in self.running if req.priority > request.priority),
            key=lambda req: (req.priority, req.arrival_time),
            reverse=True,
        )
        if not candidates:
            return set()
        num_short = self._num_blocks_short(request, hit)
        if num_short + sum(self._num_growth_blocks(req, 1) for req in self.running) <= 0:
            # It fits in this step, after the running loop.
            return set()
        # Otherwise it is admitted once the victims' frees land: in this step,
        # or with deferred frees in the next, after the running loop has grown
        # its decodes once more.
        num_steps = 2 if self.defer_block_free else 1
        num_short += sum(self._num_growth_blocks(req, num_steps) for req in self.running)
        if num_short <= self._num_released_blocks([], hit):
            return set()
        victims: list[Request] = []
        for victim in candidates[: self.max_preemptions_per_step]:
            victims.append(victim)
            num_short -= self._num_growth_blocks(victim, num_steps)
            if num_short <= self._num_released_blocks(victims, hit):
                break
        else:
            return set()
        for victim in victims:
            self.running.remove(victim)
            self._preempt_request(
                victim, timestamp, drop_stale_output=self.requires_kv_delivery
            )
        self._make_room_untried.add(request)
        self.make_room_preemptions += len(victims)
        return set(victims)

    def _num_released_blocks(
        self,
        victims: list[Request],
        hit: tuple[tuple[list[KVCacheBlock], ...], int] | None,
    ) -> int:
        """Blocks back in the free pool by the next step once `victims` are
        preempted: those whose every holder is a victim or a deferred free,
        which lands by then, but not the blocks of the prefix-cache hit `hit`,
        which its request takes as hits; and the blocks the in-flight prefills
        among `victims` still reserve."""
        hit_blocks = set() if hit is None else {id(b) for g in hit[0] for b in g}
        holders: dict[int, int] = defaultdict(int)
        blocks: dict[int, KVCacheBlock] = {}
        groups = [freed for _, freed in self.deferred_frees]
        for victim in victims:
            groups.extend(self.kv_cache_manager.get_blocks(victim.request_id).blocks)
        for group in groups:
            for block in group:
                holders[id(block)] += 1
                blocks[id(block)] = block
        block_pool = self.kv_cache_manager.block_pool
        num_blocks = sum(
            holders[key] == block.ref_cnt
            and key not in hit_blocks
            and block.pool is block_pool
            and not block.is_null
            for key, block in blocks.items()
        )
        return num_blocks + sum(
            self._request_remaining_blocks(req)
            for req in victims
            if req in self._inflight_prefills
        )
'''


def before(anchor: str, code: str) -> tuple[str, str]:
    return anchor, code + anchor


def after(anchor: str, code: str) -> tuple[str, str]:
    return anchor, anchor + code


# Each edit inserts code before or after its anchor; no line of vLLM's changes.
SCHEDULER_EDITS = [
    after("logger = init_logger(__name__)\n", LOGGER),
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
    before(
        "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
        "                    request,\n"
        "                    num_tokens_past_hit,\n",
        TRIED,
    ),
    after(
        "        return sum(\n"
        "            self._request_remaining_blocks(req) for req in self._inflight_prefills\n"
        "        )\n",
        HELPERS,
    ),
]
# The stock AsyncScheduler, appended after the patched Scheduler, extends it instead of importing the stock one.
ASYNC_EDITS = [
    ("from vllm.v1.core.sched.scheduler import Scheduler\n", ""),
    ("logger = init_logger(__name__)\n", ""),
]


def sched_dir(source_tree: bool) -> Path:
    """The scheduler package of the vLLM this interpreter finds, without importing vLLM (which needs a GPU stack): an
    installed vLLM 0.32.0, or with ``source_tree``, a source checkout, which has no package metadata to check."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("build.py: vllm is not installed in this interpreter")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        if not source_tree:
            sys.exit(
                "build.py: vllm has no package metadata; --source-tree builds from a source checkout"
            )
    else:
        if version != VLLM_VERSION:
            sys.exit(
                f"build.py: the change is written against vLLM {VLLM_VERSION}, and vLLM {version} is installed"
            )
    return Path(next(iter(spec.submodule_search_locations))) / "v1" / "core" / "sched"


def patched(path: Path, edits: list[tuple[str, str]]) -> str:
    source = path.read_text()
    for anchor, replacement in edits:
        count = source.count(anchor)
        if count != 1:
            sys.exit(f"build.py: anchor found {count} times in {path}, expected once:\n{anchor}")
        source = source.replace(anchor, replacement, 1)
    return source


def build(out_dir: Path, source_tree: bool = False) -> Path:
    sched = sched_dir(source_tree)
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
    args = sys.argv[1:]
    source_tree = "--source-tree" in args
    paths = [arg for arg in args if arg != "--source-tree"]
    if len(paths) != 1:
        sys.exit("usage: build.py [--source-tree] OUT_DIR")
    print(build(Path(paths[0]), source_tree))
