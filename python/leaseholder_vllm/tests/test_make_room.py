"""The make-room engine variant (deploy/phastform/variants/make-room) on vLLM's real v1 scheduler, on CPU: hand-made
requests, no model and no GPU, as vLLM's own tests/v1/core build a scheduler.

The module under test is built from the installed vLLM's source, as the variant builds it at engine start, so these
tests need vLLM with torch: the engine environment on a GPU node (python/leaseholder_vllm/README.md says how to run
them there). They skip under VLLM_SRC, where conftest stands torch in.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get("VLLM_SRC"):
    pytest.skip("needs the installed vLLM with torch, not VLLM_SRC", allow_module_level=True)
pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.sched.scheduler")

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler as StockScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BUILD = Path(__file__).resolve().parents[3] / "deploy/phastform/variants/make-room/build.py"
BLOCK = 16
TOKEN = 7  # every sampled token; requests ignore EOS

init_none_hash(sha256)


@pytest.fixture(scope="session")
def module(tmp_path_factory):
    """The generated lh_make_room_scheduler module."""
    out = tmp_path_factory.mktemp("make-room")
    subprocess.run([sys.executable, str(BUILD), str(out)], check=True, capture_output=True)
    sys.path.insert(0, str(out))
    try:
        import lh_make_room_scheduler
    finally:
        sys.path.remove(str(out))
    return lh_make_room_scheduler


@pytest.fixture(scope="session")
def model(tmp_path_factory) -> str:
    """A model directory with only a config: the scheduler reads the model's configuration, never its weights."""
    path = tmp_path_factory.mktemp("model")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def scheduler(
    cls,
    model: str,
    num_blocks: int,
    make_room: dict | None = None,
    policy: str = "priority",
    max_num_batched_tokens: int = 512,
    long_prefill_token_threshold: int = 0,
    max_model_len: int = 24 * BLOCK,
    watermark: float = 0.0,
    prefix_caching: bool = False,
):
    """A scheduler over a KV pool of ``num_blocks`` blocks of 16 tokens (one of them vLLM's null block)."""
    model_config = ModelConfig(
        model=model, dtype="float16", seed=42, skip_tokenizer_init=True, max_model_len=max_model_len
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        watermark=watermark,
        policy=policy,
    )
    cache_config = CacheConfig(
        block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=prefix_caching
    )
    cache_config.num_gpu_blocks = num_blocks
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        additional_config={"make_room": make_room} if make_room is not None else {},
    )
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
    )


def request(
    req_id: str,
    num_tokens: int,
    priority: int,
    arrival: float,
    max_tokens: int = 64,
    token: int = TOKEN,
) -> Request:
    """A request of ``num_tokens`` prompt tokens, all ``token``: with prefix caching on, requests of different tokens
    share no block."""
    return Request(
        request_id=req_id,
        prompt_token_ids=[token] * num_tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        arrival_time=arrival,
        priority=priority,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def blocks(num_blocks: int) -> int:
    """A prompt that fills ``num_blocks`` blocks and leaves room for 8 decoded tokens in the last one."""
    return num_blocks * BLOCK - 8


def output(sched, scheduler_output) -> ModelRunnerOutput:
    """What the model runner returns for a step: a token for each request whose prefill it finishes."""
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[
            [] if sched.requests[req_id].is_prefill_chunk else [TOKEN] for req_id in req_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def step(sched):
    """Schedule one step and feed back its output, as a synchronous engine does."""
    scheduler_output = sched.schedule()
    sched.update_from_output(scheduler_output, output(sched, scheduler_output))
    return scheduler_output


def run(sched, *requests: Request) -> None:
    """Add ``requests`` and step until every one of them decodes."""
    for req in requests:
        sched.add_request(req)
    for _ in range(20):
        if all(
            req.status == RequestStatus.RUNNING and not req.is_prefill_chunk for req in requests
        ):
            return
        step(sched)
    raise AssertionError(f"not all of {[req.request_id for req in requests]} decode after 20 steps")


def free_blocks(sched) -> int:
    return sched.kv_cache_manager.block_pool.get_num_free_blocks()


def scheduled(scheduler_output) -> set[str]:
    return set(scheduler_output.num_scheduled_tokens)


# KV make-room


def test_a_victim_takes_the_stock_preemption_path(module, model):
    """A make-room victim is preempted by Scheduler._preempt_request, as stock preemptions are: it is in that step's
    preempted_req_ids (which a KV connector reads to flush its pending stores), back in the waiting queue, and holds
    no blocks."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    victim = request("bg", blocks(8), priority=2, arrival=1)
    run(sched, request("rt", blocks(22), priority=0, arrival=0), victim)
    assert free_blocks(sched) == 2

    voice = request("voice", 100, priority=0, arrival=2)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"bg"}
    assert victim.status == RequestStatus.PREEMPTED and victim in sched.waiting
    assert victim.num_preemptions == 1
    assert not any(sched.kv_cache_manager.get_blocks("bg").blocks)
    assert "voice" in scheduled(out) and voice.status == RequestStatus.RUNNING


def test_preempts_the_lowest_priority_first_and_only_as_many_as_needed(module, model):
    """The priority-0 request needs 7 blocks with 2 free: one priority-2 runner is enough, the priority-1 one holds
    more but goes last, and the second priority-2 runner keeps running."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 4})
    run(
        sched,
        request("p1", blocks(10), priority=1, arrival=0),
        request("big", blocks(8), priority=2, arrival=1),
        request("small", blocks(6), priority=2, arrival=2),
        request("other", blocks(6), priority=2, arrival=3),
    )
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", 100, priority=0, arrival=4))
    out = step(sched)

    assert out.preempted_req_ids == {"big"}
    assert {"voice", "p1", "small", "other"} <= scheduled(out)


@pytest.mark.parametrize("cap, victims", [(1, set()), (2, {"big", "other"})])
def test_the_per_step_cap(module, model, cap, victims):
    """15 blocks for the priority-0 request with 2 free take two victims: under a cap of one, it preempts none."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": cap})
    run(
        sched,
        request("p1", blocks(10), priority=1, arrival=0),
        request("big", blocks(8), priority=2, arrival=1),
        request("small", blocks(6), priority=2, arrival=2),
        request("other", blocks(6), priority=2, arrival=3),
    )

    voice = request("voice", 15 * BLOCK - 8, priority=0, arrival=4)
    sched.add_request(voice)
    out = step(sched)

    assert (out.preempted_req_ids or set()) == victims
    assert ("voice" in scheduled(out)) == bool(victims)


def test_victim_choice_most_blocks_then_latest_arrival(module, model):
    """Among runners of the same priority, the one whose sequence takes the most blocks goes first; between equals,
    the latest arrival, even one a few tokens shorter within the same block count."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    run(
        sched,
        request("rt", blocks(12), priority=0, arrival=0),
        request("early", blocks(6) + 4, priority=2, arrival=1),  # 4 tokens longer, as many blocks
        request("large", blocks(7), priority=2, arrival=2),
        request("late", blocks(6), priority=2, arrival=3),
    )
    sched.add_request(request("voice-1", blocks(8), priority=0, arrival=4))
    assert step(sched).preempted_req_ids == {"large"}

    sched.add_request(request("voice-2", blocks(6), priority=0, arrival=5))
    assert step(sched).preempted_req_ids == {"late"}


def test_an_inflight_prefill_gives_back_what_it_still_reserves(module, model):
    """The priority-0 request needs 7 blocks and an in-flight prefill still needs 8, with 8 free: it is 7 short. The
    prefill holds 4 blocks and reserves 8, so preempting it gives back 12, more than the decode's 8: it goes alone."""
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, max_num_batched_tokens=64
    )
    run(
        sched,
        request("rt", blocks(12), priority=0, arrival=0),
        request("decode", blocks(8), priority=2, arrival=1),
    )
    prefill = request("prefill", 12 * BLOCK, priority=2, arrival=2)
    sched.add_request(prefill)
    step(sched)  # admitted, as its 12 blocks fit; 62 of its 192 tokens beside the two decodes
    assert prefill.is_prefill_chunk and free_blocks(sched) == 8

    sched.add_request(request("voice", 7 * BLOCK, priority=0, arrival=3))
    assert step(sched).preempted_req_ids == {"prefill"}


def test_a_victim_is_not_resumed_in_the_step_that_preempted_it(module, model):
    """The room counts the 8 blocks an in-flight prefill still needs: the priority-0 request of 4 blocks, with 10 free,
    preempts the decode of 3. The prefill takes 4 blocks in the running loop and the request 4, so 5 stay free, enough
    to resume the decode in the waiting loop; it waits for the next step."""
    sched = scheduler(
        module.Scheduler,
        model,
        33,
        {"max_preemptions_per_step": 2},
        long_prefill_token_threshold=64,
    )
    run(
        sched,
        request("rt", blocks(15), priority=0, arrival=0),
        request("bg", blocks(3), priority=2, arrival=1),
    )
    prefill = request("prefill", 12 * BLOCK, priority=0, arrival=2)
    sched.add_request(prefill)
    step(sched)
    assert prefill.is_prefill_chunk and free_blocks(sched) == 10

    sched.add_request(request("voice", blocks(4), priority=0, arrival=3))
    out = step(sched)
    assert out.preempted_req_ids == {"bg"}
    assert scheduled(out) == {"rt", "prefill", "voice"}
    assert free_blocks(sched) == 5


def test_the_room_includes_the_watermark(module, model):
    """With a watermark of 2 blocks, the priority-0 request of 7 blocks needs 9 free: with 2 free, preempting the
    runner of 6 is not enough, so the runner of 2 goes too, and the request runs."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, watermark=0.07)
    assert sched.kv_cache_manager.watermark_blocks == 2
    run(
        sched,
        request("rt", blocks(22), priority=0, arrival=0),
        request("bg", blocks(6), priority=2, arrival=1),
        request("bg2", blocks(2), priority=2, arrival=2),
    )
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", 100, priority=0, arrival=3))
    out = step(sched)
    assert out.preempted_req_ids == {"bg", "bg2"}
    assert "voice" in scheduled(out)


def test_no_preemption_for_an_equal_priority(module, model):
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 4})
    run(
        sched,
        request("a", blocks(15), priority=2, arrival=0),
        request("b", blocks(15), priority=2, arrival=1),
    )
    sched.add_request(request("c", 100, priority=2, arrival=2))
    for _ in range(3):
        out = step(sched)
        assert not out.preempted_req_ids and "c" not in scheduled(out)


def test_no_preemption_when_it_would_not_make_room(module, model):
    """Preempting the only lower-priority runner frees 6 blocks; with 2 free, a request of 10 still would not fit."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 4})
    run(
        sched,
        request("rt", blocks(24), priority=0, arrival=0),
        request("bg", blocks(6), priority=2, arrival=1),
    )
    sched.add_request(request("voice", 10 * BLOCK - 8, priority=0, arrival=2))
    for _ in range(3):
        out = step(sched)
        assert not out.preempted_req_ids and "voice" not in scheduled(out)


def test_with_deferred_frees_the_request_runs_one_step_later(module, model):
    """With async scheduling and a KV-consumer connector (the frozen box), a victim's blocks are freed once the step
    in flight is processed. Make-room preempts once and the request runs in the next step; the deferred blocks count
    as free, so nothing else is preempted meanwhile."""
    sched = scheduler(module.AsyncScheduler, model, 33, {"max_preemptions_per_step": 4})
    sched.defer_block_free = True  # as vLLM's tests force it: the gate also needs a P/D connector
    in_flight = []

    def schedule():
        scheduler_output = sched.schedule()
        in_flight.append((scheduler_output, output(sched, scheduler_output)))
        return scheduler_output

    def process():
        sched.update_from_output(*in_flight.pop(0))

    sched.add_request(request("rt", blocks(16), priority=0, arrival=0))
    sched.add_request(request("bg", blocks(8), priority=2, arrival=1))
    sched.add_request(request("bg2", blocks(6), priority=2, arrival=2))
    schedule()
    for _ in range(3):  # one step stays in flight, as in the engine
        schedule()
        process()
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", 100, priority=0, arrival=3))
    out = schedule()
    assert out.preempted_req_ids == {"bg"} and "voice" not in scheduled(out)
    out = schedule()  # before the step in flight is processed
    assert not out.preempted_req_ids and "voice" not in scheduled(out)
    process()
    process()
    out = schedule()
    assert not out.preempted_req_ids and {"voice", "bg2"} <= scheduled(out)


def test_a_victim_with_cached_blocks_waits_for_the_prefill_it_made_room_for(module, model):
    """With prefix caching, a victim's freed blocks stay cached, and resuming it takes them all back. Once the deferred
    frees land, the voice turn is admitted with its first chunk of three; the victim must then wait for the turn's
    other chunks, which in-flight prefills reserve, or the turn would find no free block for its next chunk."""
    sched = scheduler(
        module.AsyncScheduler,
        model,
        33,
        {"max_preemptions_per_step": 2},
        long_prefill_token_threshold=2 * BLOCK,
        prefix_caching=True,
    )
    sched.defer_block_free = True  # as vLLM's tests force it: the gate also needs a P/D connector
    in_flight = []

    def schedule():
        scheduler_output = sched.schedule()
        in_flight.append((scheduler_output, output(sched, scheduler_output)))
        return scheduler_output

    def process():
        sched.update_from_output(*in_flight.pop(0))

    rt = request("rt", blocks(14), priority=0, arrival=0, token=1)
    bg = request("bg", blocks(16), priority=2, arrival=1, token=2)
    sched.add_request(rt)
    sched.add_request(bg)
    schedule()
    for _ in range(30):  # one step stays in flight, as in the engine
        if all(r.status == RequestStatus.RUNNING and not r.is_prefill_chunk for r in (rt, bg)):
            break
        schedule()
        process()
    assert free_blocks(sched) == 2

    voice = request("voice", 6 * BLOCK, priority=0, arrival=2, token=3)
    sched.add_request(voice)
    out = schedule()
    assert out.preempted_req_ids == {"bg"} and "voice" not in scheduled(out)
    computed = []
    for _ in range(5):
        process()
        out = schedule()
        assert "bg" not in scheduled(out) or not voice.is_prefill_chunk
        computed.append(voice.num_computed_tokens)
    # Three chunks of 32 tokens in consecutive steps, then the turn decodes, and only then may the victim resume.
    assert computed[:3] == [32, 64, 96], computed


# The budget cap


@pytest.mark.parametrize("threshold", [0, 256])
def test_the_budget_cap_leaves_the_waiting_request_its_first_chunk(module, model, threshold):
    """A running priority-2 prefill leaves a waiting priority-0 request its first chunk (300 tokens, or the
    long-prefill threshold) in the step's 512 tokens; the priority-2 decode still gets its token."""
    first_chunk = threshold or 300

    def voice_step(make_room):
        sched = scheduler(
            module.Scheduler,
            model,
            400,
            make_room,
            long_prefill_token_threshold=threshold,
            max_model_len=4096,
        )
        run(sched, request("decode", 40, priority=2, arrival=0))
        sched.add_request(request("prefill", 3000, priority=2, arrival=1))
        step(sched)
        sched.add_request(request("voice", 300, priority=0, arrival=2))
        return step(sched).num_scheduled_tokens

    stock = voice_step({})  # the prefill takes the budget first
    assert stock["decode"] == 1 and stock.get("voice", 0) < first_chunk

    capped = voice_step({"budget_cap": True})
    assert capped["decode"] == 1 and capped["voice"] == first_chunk
    assert capped["prefill"] == min(threshold or 511, 511 - first_chunk)


def test_the_budget_cap_never_cuts_a_decode(module, model):
    """A priority-2 decode scheduled after the capped prefill keeps its token, though it takes one of the waiting
    request's 256."""
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        {"budget_cap": True},
        long_prefill_token_threshold=256,
        max_model_len=4096,
    )
    sched.add_request(request("prefill", 3000, priority=2, arrival=0))
    step(sched)
    run(sched, request("decode", 40, priority=2, arrival=1))  # runs after the prefill
    sched.add_request(request("voice", 300, priority=0, arrival=2))
    tokens = step(sched).num_scheduled_tokens
    assert (tokens["prefill"], tokens["decode"], tokens["voice"]) == (256, 1, 255)


# The reserve


def test_the_reserve_keeps_room_for_priority_0_and_never_blocks_it(module, model):
    """With a reserve of 8 blocks, the third priority-2 request waits, and a priority-0 request of 10 blocks fits at
    once, below the reserve."""
    sched = scheduler(module.Scheduler, model, 33, {"reserve_tokens": 8 * BLOCK})
    background = [request(f"bg{i}", blocks(10), priority=2, arrival=i) for i in range(3)]
    for req in background:
        sched.add_request(req)
    for _ in range(3):
        out = step(sched)
    assert [req.status.name for req in background] == ["RUNNING", "RUNNING", "WAITING"]
    assert free_blocks(sched) == 12

    voice = request("voice", blocks(10), priority=0, arrival=3)
    sched.add_request(voice)
    out = step(sched)
    assert "voice" in scheduled(out) and not out.preempted_req_ids
    assert background[2].status == RequestStatus.WAITING


def test_the_reserve_never_holds_back_priority_0(module, model):
    """A priority-0 request that fits now is admitted as stock admits it, though an in-flight prefill still needs
    11 of the 20 free blocks and it takes 15."""
    sched = scheduler(
        module.Scheduler,
        model,
        33,
        {"reserve_tokens": 8 * BLOCK},
        max_num_batched_tokens=128,
        long_prefill_token_threshold=64,
    )
    prefill = request("prefill", 23 * BLOCK, priority=0, arrival=0)
    sched.add_request(prefill)
    step(sched)  # 128 of its 368 tokens: alone, it is not held to the threshold
    sched.add_request(request("voice", blocks(15), priority=0, arrival=1))
    out = step(sched)
    assert prefill.is_prefill_chunk and out.num_scheduled_tokens["voice"] == 64


def test_the_reserve_must_leave_room_for_a_full_length_request(module, model):
    """A pool of 32 blocks holds a request of max_model_len (24 blocks) beside a reserve of 8, not of 9."""
    scheduler(module.Scheduler, model, 33, {"reserve_tokens": 8 * BLOCK})
    with pytest.raises(ValueError, match="reserve_tokens"):
        scheduler(module.Scheduler, model, 33, {"reserve_tokens": 9 * BLOCK})


# Stock behavior


def workload() -> list[tuple[int, Request]]:
    """(step, request): 24 requests of mixed priorities and sizes, one every other step, more than a pool
    of 48 blocks holds."""
    lengths = [(37 * i) % 200 + 20 for i in range(24)]
    return [
        (2 * i, request(f"r{i}", n, priority=i % 3, arrival=i, max_tokens=(11 * i) % 40 + 4))
        for i, n in enumerate(lengths)
    ]


def trace(sched, steps: int = 160) -> list[tuple]:
    """Each step's scheduled tokens by request, preempted requests and new requests."""
    pending = workload()
    seen = []
    for n in range(steps):
        while pending and pending[0][0] == n:
            sched.add_request(pending.pop(0)[1])
        out = step(sched)
        seen.append(
            (
                dict(out.num_scheduled_tokens),
                sorted(out.preempted_req_ids or ()),
                sorted(req.req_id for req in out.scheduled_new_reqs),
            )
        )
    return seen


@pytest.mark.parametrize(
    "policy, make_room",
    [
        ("priority", None),
        ("priority", {"max_preemptions_per_step": 0, "budget_cap": False, "reserve_tokens": 0}),
        ("fcfs", {"max_preemptions_per_step": 4, "budget_cap": True, "reserve_tokens": 8 * BLOCK}),
    ],
)
def test_switched_off_or_under_fcfs_it_schedules_as_stock(module, model, policy, make_room):
    """With every switch off, or under FCFS (which ignores them), the module schedules exactly as stock vLLM, here
    through running-loop preemptions and chunked prefills."""
    options = {"policy": policy, "max_num_batched_tokens": 128, "long_prefill_token_threshold": 64}
    stock = trace(scheduler(StockScheduler, model, 48, **options))
    patched = trace(scheduler(module.Scheduler, model, 48, make_room, **options))
    assert any(preempted for _, preempted, _ in stock)  # the workload exercises preemption
    assert patched == stock


def test_the_workload_exercises_make_room(module, model):
    """A control for the test above: with the switches on, the same workload schedules differently."""
    options = {"max_num_batched_tokens": 128, "long_prefill_token_threshold": 64}
    stock = trace(scheduler(StockScheduler, model, 48, **options))
    switches = {"max_preemptions_per_step": 4, "budget_cap": True, "reserve_tokens": 8 * BLOCK}
    assert trace(scheduler(module.Scheduler, model, 48, switches, **options)) != stock
