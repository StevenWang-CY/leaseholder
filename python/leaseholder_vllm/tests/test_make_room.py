"""The make-room engine variant (deploy/phastform/variants/make-room) on vLLM's real v1 scheduler, on CPU: hand-made
requests, no model and no GPU, as vLLM's own tests/v1/core build a scheduler.

The module under test is built from the installed vLLM's source, as the variant builds it at engine start, so these
tests need vLLM with torch: the engine environment on a GPU node (python/leaseholder_vllm/README.md says how to run
them there). Without them, as in CI, they skip.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm.v1.core.sched.scheduler")

import torch
from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderDecoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler as StockScheduler
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
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
    subprocess.run(
        [sys.executable, str(BUILD), "--source-tree", str(out)], check=True, capture_output=True
    )
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


class Connector:
    """A KV connector's scheduler side, as far as the scheduler calls it: it finds ``matched[req_id]`` tokens of a
    request outside the GPU and loads them asynchronously, reported done by ``step(finished_recving=...)``; for a
    request in ``delayed`` it needs more time and answers None, as the CPU tier's connector answers for a request
    whose earlier transfers are still in flight. While ``pending_frees`` is set, its transfers have blocks to free.
    """

    supports_divergent_local_hybrid_hits = False

    def __init__(self, matched: dict[str, int] | None = None):
        self.matched = dict(matched or {})
        self.delayed: set[str] = set()
        self.pending_frees = False

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.request_id in self.delayed:
            return None, False
        num_tokens = self.matched.get(request.request_id, 0)
        return num_tokens, num_tokens > 0

    def has_pending_block_frees(self):
        return self.pending_frees

    def has_pending_push_work(self):
        return False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def get_loaded_kv_cache_group_ids(self, request):
        return ()

    def build_connector_meta(self, scheduler_output):
        return None

    def update_connector_output(self, connector_output):
        pass

    def get_kv_connector_stats(self):
        return None

    def take_events(self):
        return ()

    def on_new_request(self, request):
        pass

    def request_finished(self, request, block_ids):
        return False, None


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
    max_num_seqs: int = 16,
    lora_config: LoRAConfig | None = None,
    connector: Connector | None = None,
    encoder_decoder: bool = False,
):
    """A scheduler over a KV pool of ``num_blocks`` blocks of 16 tokens (one of them vLLM's null block). An
    ``encoder_decoder`` one has a cross-attention group too and schedules as for such a model, patched in after
    construction as vLLM's own tests do: a real one would need the model's multimodal processor."""
    model_config = ModelConfig(
        model=model, dtype="float16", seed=42, skip_tokenizer_init=True, max_model_len=max_model_len
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=True,
        is_encoder_decoder=encoder_decoder,
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
        lora_config=lora_config,
        additional_config={"make_room": make_room} if make_room is not None else {},
    )
    shape = {"block_size": BLOCK, "num_kv_heads": 1, "head_size": 1, "dtype": torch.float32}
    groups = [KVCacheGroupSpec(["layer"], FullAttentionSpec(**shape))]
    if encoder_decoder:
        groups.append(KVCacheGroupSpec(["cross_layer"], CrossAttentionSpec(**shape)))
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    sched = cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK,
    )
    sched.connector = connector  # vLLM's tests attach a mock one likewise
    if encoder_decoder:
        sched.is_encoder_decoder = True
        sched.max_num_encoder_input_tokens = max_num_batched_tokens
        sched.encoder_cache_manager = EncoderDecoderCacheManager(cache_size=max_num_batched_tokens)
    return sched


def request(
    req_id: str,
    num_tokens: int,
    priority: int,
    arrival: float,
    max_tokens: int = 64,
    token: int = TOKEN,
    prompt: list[int] | None = None,
    lora: int = 0,
    resumable: bool = False,
    encoder_tokens: int = 0,
) -> Request:
    """A request of ``num_tokens`` prompt tokens, all ``token``, or of ``prompt``: with prefix caching on, requests
    of different tokens share no block. ``lora`` names a LoRA adapter; ``encoder_tokens`` sizes an encoder input,
    for an encoder-decoder model."""
    encoder_inputs = None
    if encoder_tokens:
        encoder_inputs = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=PlaceholderRange(offset=0, length=encoder_tokens),
                identifier=f"audio-{req_id}",
                modality="audio",
            )
        ]
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt if prompt is not None else [token] * num_tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        arrival_time=arrival,
        priority=priority,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
        lora_request=LoRARequest(f"lora{lora}", lora, f"/lora{lora}") if lora else None,
        resumable=resumable,
        mm_features=encoder_inputs,
    )


def blocks(num_blocks: int) -> int:
    """A prompt that fills ``num_blocks`` blocks and leaves room for 8 decoded tokens in the last one."""
    return num_blocks * BLOCK - 8


def output(sched, scheduler_output, finished_recving: set[str] | None = None) -> ModelRunnerOutput:
    """What the model runner returns for a step: a token for each request whose prefill it finishes, and the
    connector's finished loads."""
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
        kv_connector_output=KVConnectorOutput(finished_recving=finished_recving)
        if finished_recving
        else None,
    )


def step(sched, finished_recving: set[str] | None = None):
    """Schedule one step and feed back its output, as a synchronous engine does."""
    scheduler_output = sched.schedule()
    sched.update_from_output(scheduler_output, output(sched, scheduler_output, finished_recving))
    return scheduler_output


class Pipelined:
    """An AsyncScheduler as the engine runs it on the frozen box: deferred block frees (vLLM's tests force them, as
    the gate also needs a P/D connector) and one step in flight, whose output is processed after the next step is
    scheduled."""

    def __init__(self, sched):
        self.sched = sched
        sched.defer_block_free = True
        self.in_flight = []

    def step(self):
        scheduler_output = self.sched.schedule()
        self.in_flight.append((scheduler_output, output(self.sched, scheduler_output)))
        if len(self.in_flight) > 1:
            self.process()
        return scheduler_output

    def process(self):
        self.sched.update_from_output(*self.in_flight.pop(0))


def run(sched, *requests: Request) -> None:
    """Add ``requests`` and step until every one of them decodes."""
    for req in requests:
        sched.add_request(req)
    for _ in range(40):
        if all(
            req.status == RequestStatus.RUNNING and not req.is_prefill_chunk for req in requests
        ):
            return
        step(sched)
    raise AssertionError(f"not all of {[req.request_id for req in requests]} decode after 40 steps")


def free_blocks(sched) -> int:
    return sched.kv_cache_manager.block_pool.get_num_free_blocks()


def scheduled(scheduler_output) -> set[str]:
    return set(scheduler_output.num_scheduled_tokens)


def preempted(scheduler_output) -> set[str]:
    return set(scheduler_output.preempted_req_ids or ())


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
    assert sched.make_room_preemptions == 1  # the operator's count, apart from the running loop's


def test_preempts_the_lowest_priority_first_and_only_as_many_as_needed(module, model):
    """The priority-0 request needs 7 blocks with 2 free: one priority-2 runner is enough, the latest arrival among
    them, as vLLM's own preemption picks; the priority-1 runner, which holds more, would go last, and the older
    priority-2 runners keep running."""
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

    assert out.preempted_req_ids == {"other"}
    assert {"voice", "p1", "big", "small"} <= scheduled(out)


@pytest.mark.parametrize("cap, victims", [(2, set()), (3, {"other", "small", "big"})])
def test_the_per_step_cap(module, model, cap, victims):
    """15 blocks for the priority-0 request with 2 free take three victims in vLLM's order (6, 6 and 8 blocks): under
    a cap of two, it preempts none."""
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

    assert preempted(out) == victims
    assert ("voice" in scheduled(out)) == bool(victims)


def test_victims_go_in_vllms_preemption_order(module, model):
    """Victims go in the order vLLM's own preemption takes them, the latest arrival first among the lowest priority,
    which spares older work, rather than the one that holds the most: the 7-block runner, which arrived last, makes
    room for the first request; the 9-block one then goes for the second."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    run(
        sched,
        request("rt", blocks(14), priority=0, arrival=0),
        request("large", blocks(9), priority=2, arrival=1),
        request("late", blocks(7), priority=2, arrival=2),
    )
    assert free_blocks(sched) == 2

    sched.add_request(request("voice-1", blocks(8), priority=0, arrival=3))
    out = step(sched)
    assert out.preempted_req_ids == {"late"} and "voice-1" in scheduled(out)

    sched.add_request(request("voice-2", blocks(9), priority=0, arrival=4))
    out = step(sched)
    assert out.preempted_req_ids == {"large"} and "voice-2" in scheduled(out)


def test_an_inflight_prefill_gives_back_what_it_still_reserves(module, model):
    """The priority-0 request needs 7 blocks and an in-flight prefill still needs 8, with 8 free: it is 7 short. The
    prefill, the latest arrival, holds 4 blocks and reserves 8, so preempting it gives back 12: it goes alone."""
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
    preempts the decode of 14, the latest arrival. The prefill takes 4 blocks in the running loop and the request 4,
    so 16 stay free, enough to resume the decode in the waiting loop; it waits for the next step. The prefill is of
    the decode's priority, so nothing else holds the decode back."""
    sched = scheduler(
        module.Scheduler,
        model,
        33,
        {"max_preemptions_per_step": 2},
        long_prefill_token_threshold=64,
    )
    run(
        sched,
        request("rt", blocks(4), priority=0, arrival=0),
        request("bg", blocks(14), priority=2, arrival=2),
    )
    prefill = request("prefill", 12 * BLOCK, priority=2, arrival=1)
    sched.add_request(prefill)
    step(sched)
    assert prefill.is_prefill_chunk and free_blocks(sched) == 10

    sched.add_request(request("voice", blocks(4), priority=0, arrival=3))
    out = step(sched)
    assert out.preempted_req_ids == {"bg"}
    assert scheduled(out) == {"rt", "prefill", "voice"}
    assert free_blocks(sched) == 16


def test_the_room_includes_the_watermark(module, model):
    """With a watermark of 2 blocks, the priority-0 request of 7 blocks needs 9 free: with 2 free, preempting the
    runner of 6, the latest arrival, is not enough, so the runner of 2 goes too, and the request runs."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, watermark=0.07)
    assert sched.kv_cache_manager.watermark_blocks == 2
    run(
        sched,
        request("rt", blocks(22), priority=0, arrival=0),
        request("bg", blocks(6), priority=2, arrival=2),
        request("bg2", blocks(2), priority=2, arrival=1),
    )
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", 100, priority=0, arrival=3))
    out = step(sched)
    assert out.preempted_req_ids == {"bg", "bg2"}
    assert "voice" in scheduled(out)


def test_the_room_includes_the_decodes_next_blocks(module, model):
    """The running loop allocates the decodes' next blocks before the waiting loop admits anyone. Here rt's next token
    starts a block, so the priority-0 request, which needs the 4 free blocks and 6 more, takes both runners in the
    same step and runs at once, rather than a block short until the next step."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    # rt's 4th decoded token starts a block.
    rt = request("rt", 20 * BLOCK - 3, priority=0, arrival=0)
    run(
        sched,
        rt,
        request("bg", 6 * BLOCK - 12, priority=2, arrival=1),
        request("bg2", 2 * BLOCK - 12, priority=2, arrival=2),
    )
    for _ in range(4):
        if rt.num_tokens % BLOCK == 1:
            break
        step(sched)
    assert rt.num_tokens % BLOCK == 1 and free_blocks(sched) == 4

    voice = request("voice", 10 * BLOCK, priority=0, arrival=3)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"bg", "bg2"}
    assert "voice" in scheduled(out)


def test_the_room_includes_an_encoder_decoder_requests_cross_attention_blocks(module, model):
    """For an encoder-decoder model, a request is admitted with blocks for its encoder output too, which
    cross-attention reads: the priority-0 request needs 2 blocks for its prompt and 15 for its encoder output, 17 with
    11 free. The latest priority-2 runner gives back its 4 and its 5, which is enough, and the request runs at once."""
    sched = scheduler(
        module.Scheduler, model, 65, {"max_preemptions_per_step": 2}, encoder_decoder=True
    )
    run(
        sched,
        request("rt", blocks(4), priority=0, arrival=0, encoder_tokens=10 * BLOCK),
        request("bg", blocks(10), priority=2, arrival=1, encoder_tokens=20 * BLOCK),
        request("bg2", blocks(4), priority=2, arrival=2, encoder_tokens=5 * BLOCK),
    )
    assert [len(group) for group in sched.kv_cache_manager.get_blocks("bg2").blocks] == [4, 5]
    assert free_blocks(sched) == 11

    voice = request("voice", blocks(2), priority=0, arrival=3, encoder_tokens=15 * BLOCK)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"bg2"}
    assert "voice" in scheduled(out) and voice.status == RequestStatus.RUNNING
    assert free_blocks(sched) == 3


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


def test_the_room_is_net_of_the_requests_prefix_cache_hit(module, model):
    """The priority-0 request shares an 18-block prefix that a priority-2 runner holds: while the runner holds them,
    those blocks cost it nothing, so it lacks only 3 of the 5 blocks of its own (2 free). The other runner, the
    latest arrival, gives them, and the request is admitted beside the runner that holds its prefix."""
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, prefix_caching=True
    )
    prefix = [5] * (18 * BLOCK)
    holder = request("holder", 0, priority=2, arrival=1, prompt=prefix + [6] * 8)
    run(
        sched,
        request("rt", blocks(5), priority=0, arrival=0, token=1),
        holder,
        request("bg", blocks(6), priority=2, arrival=2, token=2),
    )
    assert free_blocks(sched) == 2

    voice = request("voice", 0, priority=0, arrival=3, prompt=prefix + [8] * 72)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"bg"}
    assert "voice" in scheduled(out) and voice.num_computed_tokens == len(prefix) + 72
    assert holder.status == RequestStatus.RUNNING


def test_a_victim_holding_the_requests_prefix_releases_none_of_it(module, model):
    """The priority-0 request shares an 18-block prefix that a priority-2 runner holds: those blocks cost it nothing
    while the runner holds them, and preempting the runner only puts them in the free queue, which the request takes
    them from as hits, a free block each. So the runner, the latest arrival and first to go, releases only its own
    block toward the 3 the request lacks (5 of its own, 2 free), and the other runner goes too."""
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, prefix_caching=True
    )
    prefix = [5] * (18 * BLOCK)
    run(
        sched,
        request("rt", blocks(5), priority=0, arrival=0, token=1),
        request("bg", blocks(6), priority=2, arrival=1, token=2),
        request("holder", 0, priority=2, arrival=2, prompt=prefix + [6] * 8),
    )
    assert free_blocks(sched) == 2

    voice = request("voice", 0, priority=0, arrival=3, prompt=prefix + [8] * 72)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"holder", "bg"}
    assert "voice" in scheduled(out) and voice.num_computed_tokens == len(prefix) + 72


def test_victims_sharing_a_prefix_release_it_together(module, model):
    """Two priority-2 runners share a 16-block prefix and hold 2 blocks each of their own: either alone releases its
    2, both together 20, as the shared blocks are released once all their holders go. The priority-0 request of 12
    blocks, with 2 free, takes both."""
    sched = scheduler(
        module.Scheduler,
        model,
        33,
        {"max_preemptions_per_step": 2},
        prefix_caching=True,
        max_model_len=32 * BLOCK,
    )
    prefix = [5] * (16 * BLOCK)
    run(
        sched,
        request("rt", blocks(10), priority=0, arrival=0, token=1),
        request("a", 0, priority=2, arrival=1, prompt=prefix + [6] * 24),
        request("b", 0, priority=2, arrival=2, prompt=prefix + [8] * 24),
    )
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", blocks(12), priority=0, arrival=3, token=9))
    out = step(sched)

    assert out.preempted_req_ids == {"a", "b"}
    assert "voice" in scheduled(out)


def test_with_deferred_frees_the_request_runs_one_step_later(module, model):
    """With async scheduling and a KV-consumer connector (the frozen box), a victim's blocks are freed once the step
    in flight is processed. Make-room preempts once and the request runs in the next step; the deferred blocks count
    as free, so nothing else is preempted meanwhile."""
    sched = scheduler(module.AsyncScheduler, model, 33, {"max_preemptions_per_step": 4})
    engine = Pipelined(sched)
    sched.add_request(request("rt", blocks(16), priority=0, arrival=0))
    sched.add_request(request("bg", blocks(8), priority=2, arrival=1))
    sched.add_request(request("bg2", blocks(6), priority=2, arrival=2))
    for _ in range(4):
        engine.step()
    assert free_blocks(sched) == 2

    sched.add_request(request("voice", 100, priority=0, arrival=3))
    out = engine.step()
    assert out.preempted_req_ids == {"bg2"} and "voice" not in scheduled(out)
    out = engine.step()
    assert not out.preempted_req_ids and {"voice", "bg"} <= scheduled(out)


def test_with_deferred_frees_the_room_counts_the_decodes_growth_until_they_land(module, model):
    """With deferred frees the request is admitted a step after its victims are preempted, so the room counts the
    running decodes' blocks for both steps. Here rt's next block starts in the second: the latest arrival's 6 blocks
    cover the 6 the request lacks now but not the 7 it lacks then, so it takes both runners in one round and runs in
    the next step, rather than a block short then, which a second round would fix a step later still."""
    sched = scheduler(module.AsyncScheduler, model, 33, {"max_preemptions_per_step": 2})
    engine = Pipelined(sched)
    rt = request("rt", 20 * BLOCK - 8, priority=0, arrival=0, max_tokens=400)
    for req in (
        rt,
        request("bg", 6 * BLOCK - 15, priority=2, arrival=2, max_tokens=400),
        request("bg2", 2 * BLOCK - 15, priority=2, arrival=1, max_tokens=400),
    ):
        sched.add_request(req)
    for _ in range(8):
        engine.step()
    # The position rt's next token takes: the one after starts a block.
    assert (rt.num_tokens + rt.num_output_placeholders) % BLOCK == 0 and free_blocks(sched) == 4

    sched.add_request(request("voice", 10 * BLOCK, priority=0, arrival=3))
    rounds, admitted = [], None
    for n in range(4):
        out = engine.step()
        if out.preempted_req_ids:
            rounds.append((n, preempted(out)))
        if "voice" in scheduled(out):
            admitted = n
            break
    assert rounds == [(0, {"bg", "bg2"})] and admitted == 1


def test_with_deferred_frees_a_request_that_fits_now_takes_no_room(module, model):
    """The room counts the decodes' growth until the victims' frees land only for a request that lacks blocks now:
    one that fits in this step, after the running loop, runs at once, though rt's next block starts in the next step
    and it would then lack one."""
    sched = scheduler(module.AsyncScheduler, model, 33, {"max_preemptions_per_step": 2})
    engine = Pipelined(sched)
    rt = request("rt", 20 * BLOCK - 8, priority=0, arrival=0, max_tokens=400)
    for req in (
        rt,
        request("bg", 6 * BLOCK - 15, priority=2, arrival=1, max_tokens=400),
        request("bg2", 2 * BLOCK - 15, priority=2, arrival=2, max_tokens=400),
    ):
        sched.add_request(req)
    for _ in range(8):
        engine.step()
    assert (rt.num_tokens + rt.num_output_placeholders) % BLOCK == 0 and free_blocks(sched) == 4

    sched.add_request(request("voice", 4 * BLOCK, priority=0, arrival=3))
    out = engine.step()
    assert not out.preempted_req_ids and "voice" in scheduled(out)


def test_with_deferred_frees_the_blocks_of_an_aborted_request_make_room(module, model):
    """A request aborted while its step is in flight frees its blocks once that step is processed, as a victim's
    are: they count as free. The priority-0 request, which lacks 3 blocks and finds 4 on their way, preempts
    nothing and runs in the next step."""
    sched = scheduler(module.AsyncScheduler, model, 33, {"max_preemptions_per_step": 2})
    engine = Pipelined(sched)
    for req in (
        request("rt", 20 * BLOCK - 15, priority=0, arrival=0, max_tokens=400),
        request("bg", 6 * BLOCK - 15, priority=2, arrival=1, max_tokens=400),
        request("gone", 4 * BLOCK - 15, priority=1, arrival=2, max_tokens=400),
    ):
        sched.add_request(req)
    for _ in range(3):
        engine.step()
    sched.finish_requests("gone", RequestStatus.FINISHED_ABORTED)  # its step in flight
    assert sched.deferred_frees and free_blocks(sched) == 2

    voice = request("voice", 5 * BLOCK, priority=0, arrival=3)
    sched.add_request(voice)
    out = engine.step()
    assert not out.preempted_req_ids and "voice" not in scheduled(out)
    out = engine.step()
    assert not out.preempted_req_ids and "voice" in scheduled(out)


def test_no_room_while_the_connector_has_block_frees_pending(module, model):
    """While the KV connector's transfers have blocks to free, make-room preempts nothing, as the running loop waits
    for those blocks rather than preempt; once none are pending, it makes room."""
    connector = Connector()
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, connector=connector
    )
    run(
        sched,
        request("rt", blocks(22), priority=0, arrival=0),
        request("bg", blocks(8), priority=2, arrival=1),
    )
    connector.pending_frees = True
    sched.add_request(request("voice", 100, priority=0, arrival=2))
    for _ in range(2):
        out = step(sched)
        assert not out.preempted_req_ids and "voice" not in scheduled(out)

    connector.pending_frees = False
    out = step(sched)
    assert out.preempted_req_ids == {"bg"} and "voice" in scheduled(out)


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
    engine = Pipelined(sched)
    rt = request("rt", blocks(14), priority=0, arrival=0, token=1)
    bg = request("bg", blocks(16), priority=2, arrival=1, token=2)
    sched.add_request(rt)
    sched.add_request(bg)
    for _ in range(30):  # one step stays in flight, as in the engine
        if all(r.status == RequestStatus.RUNNING and not r.is_prefill_chunk for r in (rt, bg)):
            break
        engine.step()
    assert free_blocks(sched) == 2

    voice = request("voice", 6 * BLOCK, priority=0, arrival=2, token=3)
    sched.add_request(voice)
    out = engine.step()
    assert out.preempted_req_ids == {"bg"} and "voice" not in scheduled(out)
    computed = []
    for _ in range(5):
        out = engine.step()
        assert "bg" not in scheduled(out) or not voice.is_prefill_chunk
        computed.append(voice.num_computed_tokens)
    # Three chunks of 32 tokens in consecutive steps, then the turn decodes, and only then may the victim resume.
    assert computed[:3] == [32, 64, 96], computed


def test_no_room_for_a_request_whose_lora_adapter_cannot_join_the_running_ones(module, model):
    """Under max_loras 2, with two priority-0 requests decoding on adapters 1 and 2, the waiting loop skips a
    priority-0 request on adapter 3, as stock vLLM does. Make-room makes no room for it: preempting the priority-2
    runner would only see it resume in the next step and be preempted again in the one after, for as long as the
    adapters stay taken. The runner keeps its tokens, as on stock vLLM."""

    def run_tokens(cls, make_room):
        sched = scheduler(
            cls,
            model,
            33,
            make_room,
            lora_config=LoRAConfig(max_loras=2, max_lora_rank=8),
            max_model_len=32 * BLOCK,
        )
        run(
            sched,
            request("rt1", blocks(4), priority=0, arrival=0, token=1, max_tokens=400, lora=1),
            request("rt2", blocks(4), priority=0, arrival=1, token=2, max_tokens=400, lora=2),
            request("bg", blocks(20), priority=2, arrival=2, token=3, max_tokens=400),
        )
        assert free_blocks(sched) == 4
        sched.add_request(request("voice", blocks(6), priority=0, arrival=3, token=4, lora=3))
        outs = [step(sched) for _ in range(12)]
        assert not any("voice" in scheduled(out) for out in outs)
        return [preempted(out) for out in outs], sum(out.num_scheduled_tokens["bg"] for out in outs)

    victims, bg_tokens = run_tokens(module.Scheduler, {"max_preemptions_per_step": 2})
    assert not any(victims)
    assert bg_tokens == run_tokens(StockScheduler, None)[1] == 12


def test_a_paused_streaming_session_does_not_mask_the_requests_behind_it(module, model):
    """A priority-0 streaming session paused for its next input waits ahead of a priority-0 request that arrives
    later: the waiting loop cannot schedule the session, so make-room makes room for the request behind it, as it
    would without the session."""
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2, "budget_cap": True}
    )
    run(
        sched,
        request("rt", blocks(4), priority=0, arrival=0, token=1, max_tokens=300),
        request("bg", blocks(20), priority=2, arrival=1, token=3, max_tokens=300),
    )
    session = request(
        "session", 30, priority=0, arrival=1.5, token=11, max_tokens=2, resumable=True
    )
    sched.add_request(session)
    for _ in range(4):
        step(sched)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert session in sched.kv_holding_waiting and free_blocks(sched) == 5

    voice = request("voice", blocks(8), priority=0, arrival=3, token=4)
    sched.add_request(voice)
    out = step(sched)

    assert out.preempted_req_ids == {"bg"}
    assert "voice" in scheduled(out)


def test_a_request_compiling_its_grammar_does_not_mask_the_requests_behind_it(module, model):
    """A priority-0 request whose structured-output grammar is still compiling waits ahead of a later priority-0
    request: the waiting loop cannot schedule it yet, so make-room makes room for the request behind it, though the
    first would fit."""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    run(
        sched,
        request("rt", blocks(22), priority=0, arrival=0),
        request("bg", blocks(8), priority=2, arrival=1),
    )
    compiling = request("compiling", BLOCK, priority=0, arrival=2)
    # Blocked as vLLM's own tests block one.
    compiling.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    sched.add_request(compiling)
    sched.add_request(request("voice", 100, priority=0, arrival=3))
    out = step(sched)

    assert out.preempted_req_ids == {"bg"} and "voice" in scheduled(out)
    assert compiling.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def test_no_room_for_a_request_awaiting_output_from_before_its_preemption(module, model):
    """Under pipeline parallelism, a preempted request's output can be in flight for steps after its preemption, and
    the waiting loop skips it until that output is delivered: make-room makes no room for it meanwhile. (Only
    several steps in flight leave that output pending, so this sets it as the scheduler's preemption does.)"""
    sched = scheduler(module.Scheduler, model, 33, {"max_preemptions_per_step": 2})
    run(
        sched,
        request("rt", blocks(22), priority=0, arrival=0),
        request("bg", blocks(8), priority=2, arrival=1),
    )
    voice = request("voice", 100, priority=0, arrival=2)
    voice.num_stale_output_tokens, voice.drop_stale_output = 1, False
    sched.add_request(voice)
    for _ in range(2):
        out = step(sched)
        assert not out.preempted_req_ids and "voice" not in scheduled(out)

    voice.num_stale_output_tokens = 0  # delivered
    out = step(sched)
    assert out.preempted_req_ids == {"bg"} and "voice" in scheduled(out)


def test_no_second_round_for_a_request_the_waiting_loop_skipped(module, model):
    """Make-room preempts a runner for the priority-0 request, and the waiting loop then skips the request: here its
    connector needs more time, as the CPU tier's does while the request's earlier transfers are in flight. The victim
    resumes in the next step, and no room is made for the request again until the waiting loop has tried it: so the
    runner is preempted once while the request is skipped, rather than every other step, and room is made again
    once the request, tried, lacks it (by then for the decodes' growth too)."""
    connector = Connector()
    sched = scheduler(
        module.Scheduler, model, 33, {"max_preemptions_per_step": 2}, connector=connector
    )
    run(
        sched,
        request("rt", blocks(10), priority=0, arrival=0, token=1, max_tokens=400),
        request("big", blocks(14), priority=2, arrival=1, token=2, max_tokens=400),
        request("bg", blocks(6), priority=2, arrival=2, token=3, max_tokens=400),
    )
    assert free_blocks(sched) == 2

    voice = request("voice", blocks(7), priority=0, arrival=3, token=4)
    connector.delayed.add("voice")
    sched.add_request(voice)
    outs = [step(sched) for _ in range(8)]
    assert [preempted(out) for out in outs] == [{"bg"}] + [set()] * 7
    assert not any("voice" in scheduled(out) for out in outs)
    assert all("bg" in scheduled(out) for out in outs[1:])  # it resumed in the next step

    connector.delayed.clear()
    out = step(sched)  # tried, and short
    assert not out.preempted_req_ids and "voice" not in scheduled(out)
    out = step(sched)
    assert "bg" in preempted(out) and "voice" in scheduled(out)


# The budget cap


@pytest.mark.parametrize("threshold", [0, 256])
def test_the_budget_cap_gives_a_waiting_request_that_fits_the_step(module, model, threshold):
    """A waiting priority-0 request that fits takes its first chunk (300 tokens, or the long-prefill threshold) in a
    step that a running priority-2 prefill sits out; the priority-2 decode still gets its token. Stock vLLM (the
    control) gives the prefill the budget first."""
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

    stock = voice_step({})
    assert stock["decode"] == 1 and stock.get("voice", 0) < first_chunk

    assert voice_step({"budget_cap": True}) == {"decode": 1, "voice": first_chunk}


def test_the_budget_cap_never_cuts_a_decode(module, model):
    """A priority-2 decode scheduled after the prefill that sits out the step keeps its token."""
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
    assert step(sched).num_scheduled_tokens == {"decode": 1, "voice": 256}


def test_a_waiting_request_that_does_not_fit_holds_nothing_back(module, model):
    """Without KV make-room, a priority-0 request of 10 blocks does not fit beside the priority-2 prefill in flight,
    which still needs 10 of the 12 free blocks: the prefill, whose blocks it waits for, goes on."""
    sched = scheduler(module.Scheduler, model, 33, {"budget_cap": True}, max_num_batched_tokens=64)
    run(sched, request("decode", blocks(16), priority=2, arrival=0))
    prefill = request("prefill", 14 * BLOCK, priority=2, arrival=1)
    sched.add_request(prefill)
    step(sched)  # admitted, as its 14 blocks fit; 63 of its 224 tokens beside the decode
    assert free_blocks(sched) == 12
    sched.add_request(request("voice", blocks(10), priority=0, arrival=2))
    for _ in range(2):
        tokens = step(sched).num_scheduled_tokens
        assert tokens["prefill"] == 63 and "voice" not in tokens


@pytest.mark.parametrize("make_room", [{"budget_cap": True, "budget_cap_tokens": 4}, {}])
def test_lower_priority_prefill_sits_out_a_requests_first_tokens(module, model, make_room):
    """Once the priority-0 request runs, the running priority-2 prefill and a waiting priority-2 request sit out the
    steps until it has its first 4 tokens, while the priority-2 decode keeps its token; then both go on. A request's
    first tokens hold back only lower priorities: before the voice turn, the decode's first tokens hold back no
    prefill of its own priority. Stock vLLM (the control) schedules the prefill in every step."""
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        make_room,
        long_prefill_token_threshold=256,
        max_model_len=4096,
    )
    run(sched, request("decode", 40, priority=2, arrival=0))
    sched.add_request(request("prefill", 3000, priority=2, arrival=1))
    assert step(sched).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    voice = request("voice", 100, priority=0, arrival=2)
    sched.add_request(voice)
    tokens = step(sched).num_scheduled_tokens  # its whole prompt
    assert tokens == {"decode": 1, "voice": 100} or not make_room
    assert voice.num_output_tokens == 1
    sched.add_request(request("late", 300, priority=2, arrival=3))
    steps = [step(sched).num_scheduled_tokens for _ in range(4)]

    if make_room:
        assert steps[:3] == [{"decode": 1, "voice": 1}] * 3
        assert steps[3]["prefill"] == 256 and steps[3]["late"] > 0
    else:
        assert all(tokens["prefill"] == 256 for tokens in steps)
    assert all(tokens["decode"] == tokens["voice"] == 1 for tokens in steps)


@pytest.mark.parametrize("make_room", [{"budget_cap": True}, {}])
def test_a_lower_priority_request_holding_kv_sits_out_a_requests_first_tokens(
    module, model, make_room
):
    """The waiting loop drains first the requests that hold KV, as one does whose load from the CPU tier is done. A
    priority-2 one sits out the priority-0 request's first tokens like any lower-priority prefill, and it is skipped
    rather than waited on, so the loop goes on to the priority-0 request behind it, which runs in the same step; once
    its first 16 tokens are out, the priority-2 one prefills. Stock vLLM (the control) prefills it at once."""
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        make_room,
        long_prefill_token_threshold=256,
        max_model_len=4096,
    )
    run(sched, request("decode", 40, priority=2, arrival=0))
    late = request("late", 3000, priority=2, arrival=1)
    assert sched.kv_cache_manager.allocate_slots(late, 256) is not None  # 256 tokens loaded
    late.num_computed_tokens = 256
    sched.add_request(late)
    assert late in sched.kv_holding_waiting
    sched.add_request(request("voice", 100, priority=0, arrival=2))

    steps = [step(sched).num_scheduled_tokens for _ in range(17)]

    if make_room:
        # The prompt's step gives the first token, the next 15 the rest of the first 16.
        assert steps[:16] == [{"decode": 1, "voice": 100}] + [{"decode": 1, "voice": 1}] * 15
        assert steps[16] == {"decode": 1, "voice": 1, "late": 256}
    else:
        assert steps[0] == {"decode": 1, "late": 256, "voice": 100}


def loading(module, model, make_room, max_num_seqs: int = 16):
    """A priority-2 decode and prefill run when a priority-0 request arrives whose first 256 tokens its connector
    loads asynchronously: the waiting loop admits it to load them (WAITING_FOR_REMOTE_KVS)."""
    connector = Connector({"voice": 256})
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        make_room,
        long_prefill_token_threshold=256,
        max_model_len=4096,
        max_num_seqs=max_num_seqs,
        connector=connector,
    )
    run(sched, request("decode", 40, priority=2, arrival=0, max_tokens=400))
    sched.add_request(request("prefill", 3000, priority=2, arrival=1))
    step(sched)
    voice = request("voice", 300, priority=0, arrival=2)
    sched.add_request(voice)
    step(sched)
    assert (
        voice.status == RequestStatus.WAITING_FOR_REMOTE_KVS and voice in sched._inflight_prefills
    )
    return sched, voice


def test_a_load_counts_the_blocks_it_still_needs_once(module, model):
    """A request loading its KV reserves the blocks it still needs among the in-flight prefills' reservations, so
    its own shortfall counts them once."""
    sched, voice = loading(module, model, {"max_preemptions_per_step": 2})
    reserved = sched._inflight_prefill_reserved_blocks()
    # Its 44 tokens, and the prefill's.
    assert sched._request_remaining_blocks(voice) == 3 < reserved
    assert sched._num_blocks_short(voice, None) == reserved - free_blocks(sched)


@pytest.mark.parametrize("make_room", [{"budget_cap": True, "budget_cap_tokens": 4}, {}])
def test_a_load_holds_lower_priority_prefill_back_for_its_first_steps(module, model, make_room):
    """While its connector loads a request's KV, lower-priority prefill sits out the load's first budget_cap_tokens
    steps, here 4: the priority-2 prefill gets no chunk and a new priority-2 request is not admitted. Stock vLLM (the
    control) runs the prefill throughout."""
    sched, voice = loading(module, model, make_room)
    sched.add_request(request("late", 300, priority=2, arrival=3))
    steps = [step(sched).num_scheduled_tokens for _ in range(4)]

    if make_room:
        assert steps == [{"decode": 1}] * 4
    else:
        assert all(tokens["prefill"] == 256 for tokens in steps)
    assert voice.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_stuck_load_holds_nothing_after_its_first_steps(module, model):
    """A load that never completes holds lower-priority prefill back for its first budget_cap_tokens steps only, here
    4: then the priority-2 prefill keeps its chunk and a new priority-2 request is admitted, as on stock vLLM."""
    sched, voice = loading(module, model, {"budget_cap": True, "budget_cap_tokens": 4})
    sched.add_request(request("late", 300, priority=2, arrival=3))
    steps = [step(sched).num_scheduled_tokens for _ in range(12)]
    assert steps[:4] == [{"decode": 1}] * 4
    assert all(tokens["prefill"] == 256 for tokens in steps[4:])
    assert any("late" in tokens for tokens in steps[4:])
    assert voice.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("make_room", [{"budget_cap": True, "budget_cap_tokens": 4}, {}])
def test_a_load_holds_again_from_the_step_it_completes(module, model, make_room):
    """A load past its first 4 steps holds nothing, but once the connector reports it done, the waiting loop
    promotes the request in the next step, on budget the running loop leaves: from that step the priority-2 prefill
    sits out again, until the request's first 4 tokens. Stock vLLM (the control) runs the prefill's chunk beside the
    request's."""
    sched, _ = loading(module, model, make_room)
    for _ in range(4):
        step(sched)  # the load's first steps
    assert step(sched).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    step(sched, finished_recving={"voice"})  # the step in which the load is reported done
    tokens = step(sched).num_scheduled_tokens

    if make_room:
        assert tokens == {"decode": 1, "voice": 44}  # the 44 tokens it did not load
        steps = [step(sched).num_scheduled_tokens for _ in range(3)]
        assert steps == [{"decode": 1, "voice": 1}] * 3
        assert step(sched).num_scheduled_tokens["prefill"] == 256
    else:
        assert tokens == {"decode": 1, "prefill": 256, "voice": 44}


def test_a_completed_load_holds_prefill_back_while_a_request_that_does_not_fit_waits(module, model):
    """A request whose load is done holds lower-priority prefill back from the step the waiting loop promotes it,
    whichever waiting request make-room serves: here a priority-0 request of 38 blocks, with fewer free, waits ahead
    of the priority-1 one whose load is done, past its first 4 steps, and the priority-2 prefill sits out the
    promotion's step again."""
    sched = scheduler(
        module.Scheduler,
        model,
        60,
        {"budget_cap": True, "budget_cap_tokens": 4},
        long_prefill_token_threshold=4 * BLOCK,
        max_model_len=40 * BLOCK,
        connector=Connector({"load": 256}),
    )
    run(sched, request("decode", 40, priority=2, arrival=0, max_tokens=400))
    sched.add_request(request("prefill", 30 * BLOCK, priority=2, arrival=1))
    step(sched)
    load = request("load", 300, priority=1, arrival=2)
    sched.add_request(load)
    step(sched)
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    big = request("big", 38 * BLOCK, priority=0, arrival=3)
    sched.add_request(big)
    steps = [step(sched).num_scheduled_tokens for _ in range(5)]
    assert steps == [{"decode": 1}] * 4 + [{"decode": 1, "prefill": 4 * BLOCK}]
    step(sched, finished_recving={"load"})  # the step in which the load is reported done

    assert step(sched).num_scheduled_tokens == {"decode": 1, "load": 44}
    assert big.status == RequestStatus.WAITING and free_blocks(sched) < 38


def test_with_every_running_slot_taken_a_completed_load_holds_nothing_back(module, model):
    """A request whose load is done needs a running slot to be promoted. With both slots taken by priority-2
    requests, admitted while it loaded (past the load's first 4 steps), it holds nothing back: otherwise the running
    prefill would sit out every step until the decode finished, and with no decode running, forever."""
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        {"budget_cap": True, "budget_cap_tokens": 4},
        long_prefill_token_threshold=256,
        max_model_len=4096,
        max_num_seqs=2,
        connector=Connector({"voice": 256}),
    )
    run(sched, request("decode", 40, priority=2, arrival=0, max_tokens=400))
    voice = request("voice", 300, priority=0, arrival=1)
    sched.add_request(voice)
    step(sched)
    assert voice.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.add_request(request("prefill", 3000, priority=2, arrival=2))
    for _ in range(4):
        step(sched)  # the load's first steps: the prefill waits
    step(sched, finished_recving={"voice"})
    assert len(sched.running) == 2
    assert step(sched).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    assert voice.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_the_module_logs_as_vllms_scheduler(module):
    """vLLM's logging config covers its own loggers only: under the generated module's name, the scheduler's info
    lines, make-room's among them, would be dropped."""
    assert module.logger.name == "vllm.v1.core.sched.scheduler"


def test_make_room_counts_and_logs_what_it_holds_back(module, model, monkeypatch):
    """The scheduler counts the steps in which the budget cap holds lower-priority prefill back, the prefill tokens
    and admissions it holds back, and the requests make-room preempts, and logs the totals at most every 10 s, when
    they moved."""
    sched = scheduler(
        module.Scheduler,
        model,
        400,
        {"budget_cap": True, "budget_cap_tokens": 4, "max_preemptions_per_step": 2},
        long_prefill_token_threshold=256,
        max_model_len=4096,
    )
    run(sched, request("decode", 40, priority=2, arrival=0))
    sched.add_request(request("prefill", 3000, priority=2, arrival=1))
    step(sched)
    sched.add_request(request("voice", 100, priority=0, arrival=2))
    step(sched)  # held: the prefill's chunk of 256
    sched.add_request(request("late", 300, priority=2, arrival=3))
    for _ in range(3):
        step(sched)  # held: the prefill's chunk of 256, and the late request's admission
    assert (sched.make_room_held_steps, sched.make_room_held_tokens) == (4, 4 * 256)
    assert (sched.make_room_held_admissions, sched.make_room_preemptions) == (3, 0)

    lines = []
    monkeypatch.setattr(
        module, "logger", SimpleNamespace(info=lambda msg, *a: lines.append(msg % a))
    )
    step(sched)  # within 10 s of the scheduler's start: nothing yet
    sched._make_room_log_time -= 10
    step(sched)
    sched._make_room_log_time -= 10
    step(sched)  # the totals have not moved
    assert lines == [
        "make-room: budget cap held lower-priority prefill back in 4 steps (1024 prefill "
        "tokens and 3 admissions, summed over the steps); 0 preemptions to make room"
    ]


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


def test_a_shared_prefix_counts_once_against_the_reserve(module, model):
    """A request's prefix-cache hit takes no new blocks. A running priority-2 request holds 19 of 32 blocks, 18 of
    them a prefix its child shares: the child needs 1 new block, which fits beside a reserve of 8 blocks in the 13
    free, as stock vLLM admits it."""

    def admitted(cls, make_room) -> bool:
        sched = scheduler(cls, model, 33, make_room, prefix_caching=True)
        prefix = [5] * (18 * BLOCK)
        run(sched, request("parent", 0, priority=2, arrival=0, prompt=prefix + [1] * 8))
        assert free_blocks(sched) == 13
        child = request("child", 0, priority=2, arrival=1, prompt=prefix + [2] * 8)
        sched.add_request(child)
        step(sched)
        return child.status == RequestStatus.RUNNING

    assert admitted(StockScheduler, None)
    assert admitted(module.Scheduler, {"reserve_tokens": 8 * BLOCK})


def test_the_inflight_gate_never_holds_back_a_request_holding_kv(module, model):
    """KV make-room holds back a waiting request short of blocks while a prefill of a higher priority is in flight,
    but not one that holds its KV, as vLLM PR #63350's test of the same name has it: here a priority-2 streaming
    session's next chunk, which needs 6 more blocks with 16 free while a priority-0 prefill in flight still needs
    12."""

    def session(num_tokens, max_tokens):
        return Request(
            request_id="session",
            prompt_token_ids=[1] * num_tokens,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            arrival_time=0,
            priority=2,
            block_hasher=get_request_block_hasher(BLOCK, sha256),
            resumable=True,
        )

    sched = scheduler(
        module.Scheduler,
        model,
        33,
        {"max_preemptions_per_step": 2},
        long_prefill_token_threshold=64,
    )
    sched.add_request(session(blocks(4), 1))
    step(sched)
    assert sched.requests["session"].status == RequestStatus.WAITING_FOR_STREAMING_REQ
    run(sched, request("rt", blocks(8), priority=0, arrival=1, token=0))
    prefill = request("prefill", 16 * BLOCK, priority=0, arrival=2, token=0)
    sched.add_request(prefill)
    step(sched)
    assert prefill.is_prefill_chunk and free_blocks(sched) == 16

    sched.add_request(session(100, 16))
    assert step(sched).num_scheduled_tokens == {"rt": 1, "prefill": 64, "session": 64}


def test_the_reserve_must_leave_room_for_a_full_length_request(module, model):
    """A pool of 32 blocks holds a request of max_model_len (24 blocks) beside a reserve of 8, not of 9."""
    scheduler(module.Scheduler, model, 33, {"reserve_tokens": 8 * BLOCK})
    with pytest.raises(ValueError, match="reserve_tokens"):
        scheduler(module.Scheduler, model, 33, {"reserve_tokens": 9 * BLOCK})


# The switches


@pytest.mark.parametrize(
    "make_room, message",
    [
        ({"max_preemptions": 2}, "make_room has no key 'max_preemptions'"),
        ({"budget_cap": "off"}, "make_room.budget_cap must be true or false, not 'off'"),
        ({"budget_cap": 2}, "make_room.budget_cap must be true or false, not 2"),
        (
            {"max_preemptions_per_step": 2.5},
            "make_room.max_preemptions_per_step must be an integer",
        ),
        ({"max_preemptions_per_step": -1}, "of at least 0, not -1"),
        ({"max_preemptions_per_step": "2"}, "not '2'"),
        ({"max_preemptions_per_step": True}, "not True"),
        (
            {"budget_cap_tokens": 0},
            "make_room.budget_cap_tokens must be an integer of at least 1, not 0",
        ),
        (
            {"reserve_tokens": -16},
            "make_room.reserve_tokens must be an integer of at least 0, not -16",
        ),
    ],
)
def test_a_bad_switch_fails_the_engine_start_naming_it(module, model, make_room, message):
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        scheduler(module.Scheduler, model, 33, make_room)


@pytest.mark.parametrize(
    "value, on", [(True, True), ("TRUE", True), (1, True), ("False", False), (0, False)]
)
def test_budget_cap_takes_true_or_false_in_any_case(module, model, value, on):
    """As vLLM's CLI parses --additional-config.make_room.budget_cap: false and true become booleans, 0 and 1 integers,
    and other spellings stay strings."""
    assert scheduler(module.Scheduler, model, 33, {"budget_cap": value}).prefill_budget_cap is on


def test_the_switches_need_priority_scheduling(module, model):
    """Under FCFS, which would ignore them, a switch that is on fails the start; switches that are off are fine."""
    with pytest.raises(ValueError, match="make_room needs --scheduling-policy priority"):
        scheduler(module.Scheduler, model, 33, {"budget_cap": True}, policy="fcfs")
    scheduler(
        module.Scheduler, model, 33, {"budget_cap": False, "reserve_tokens": 0}, policy="fcfs"
    )


# Stock behavior


def workload() -> list[tuple[int, Request]]:
    """(step, request): 24 requests of mixed priorities and sizes, one every other step, more than a pool
    of 48 blocks holds. With prefix caching, every third shares a 3-block prefix."""
    lengths = [(37 * i) % 200 + 20 for i in range(24)]
    return [
        (
            2 * i,
            request(
                f"r{i}",
                0,
                priority=i % 3,
                arrival=i,
                max_tokens=(11 * i) % 40 + 4,
                prompt=([3] * 3 * BLOCK if i % 3 == 0 else []) + [i] * n,
            ),
        )
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


@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize(
    "policy, make_room",
    [
        ("priority", None),
        ("priority", {"max_preemptions_per_step": 0, "budget_cap": False, "reserve_tokens": 0}),
        ("fcfs", None),
        ("fcfs", {"max_preemptions_per_step": 0, "budget_cap": False, "budget_cap_tokens": 48}),
    ],
)
def test_switched_off_it_schedules_as_stock(module, model, policy, make_room, prefix_caching):
    """With every switch off, the module schedules exactly as stock vLLM, here through running-loop preemptions,
    chunked prefills and, with prefix caching, shared prefixes."""
    options = {
        "policy": policy,
        "max_num_batched_tokens": 128,
        "long_prefill_token_threshold": 64,
        "prefix_caching": prefix_caching,
    }
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
