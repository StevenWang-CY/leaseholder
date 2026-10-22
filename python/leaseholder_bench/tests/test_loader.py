"""The weka_trace_scaled loader reproduces a scaled trace's prefix structure on the wire.

AIPerf's own reconstruction runs offline here: a fake prompt generator (see conftest) decodes
every token to one word, so whitespace splitting recovers the exact tokens of each prompt.
"""

from __future__ import annotations

import itertools

import pytest
from aiperf.common import environment as env_mod
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from helpers import model_endpoint, request, trace, write_traces

from leaseholder_bench.aiperf_plugin import LeaseholderChatEndpoint, WekaTraceScaledLoader
from leaseholder_bench.traces import common_prefix_len

MAIN = list(range(1, 26))
COMPACTED = [*MAIN[:15], *range(100, 111)]
SUB = list(range(300, 312))

SCALED = trace(
    "trace-a",
    [
        request(0.0, MAIN[:10], out=3),
        request(1.0, MAIN[:18], out=5, think=0.5),
        request(2.0, MAIN[:25], out=2, think=0.5),
        {"t": 3.0, "type": "subagent", "agent_id": "subagent_001", "subagent_type": "Explore",
         "duration_ms": 900, "status": "completed", "models": ["m"],
         "requests": [request(3.0, SUB[:8], out=3, think=0.0),
                      request(3.5, SUB, out=1, think=0.0)]},
        request(4.0, COMPACTED, out=4, think=0.5),
        request(5.0, [*COMPACTED, 111, 112, 113], out=1, think=0.5),
    ],
)  # fmt: skip
SENT = 7  # model requests in SCALED


@pytest.fixture(autouse=True)
def _serial_reconstruction(monkeypatch):
    monkeypatch.setattr(env_mod.Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)


def make_run(directory, format_name: str = "weka_trace_scaled") -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://127.0.0.1:8000/v1/chat/completions"]},
            "datasets": [{"name": "default", "type": "file", "path": str(directory),
                          "format": format_name}],
            "phases": [{"name": "profiling", "type": "concurrency", "requests": 1,
                        "concurrency": 1}],
            "tokenizer": {"name": "fake"},
            "runtime": {"ui": "simple"},
        }
    )  # fmt: skip
    return BenchmarkRun(benchmark_id="t", cfg=cfg, artifact_dir=cfg.artifacts.dir, random_seed=0)


def load(format_name: str, directory, prompt_generator):
    """Build the loader the way AIPerf's composer does, registry default included."""
    kwargs = {"prompt_generator": prompt_generator}
    default_block_size = plugins.get_dataset_loader_metadata(format_name).default_block_size
    if default_block_size is not None:
        kwargs["default_block_size"] = default_block_size
    loader_class = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, format_name)
    loader = loader_class(filename=str(directory), run=make_run(directory, format_name), **kwargs)
    loader._tokenizer_name = "fake"
    return loader.convert_to_conversations(loader.load_dataset())


def source_request(turn):
    entry = SCALED["requests"][turn.source_outer_idx]
    return entry if turn.source_inner_idx is None else entry["requests"][turn.source_inner_idx]


def wire_checks(conversations):
    """Shared-prefix pairs (observed, expected content tokens) and ISL pairs (sent, recorded).

    A prompt is the endpoint's message list for the turn, as the worker accumulates it, with one
    marker token per message so that a changed role layout also breaks the shared prefix.
    """
    endpoint = LeaseholderChatEndpoint(model_endpoint=model_endpoint())

    def tokens(conversation, k):
        out = []
        for message in endpoint.build_messages(conversation.turns[: k + 1]):
            out.append(f"<{message['role']}>")
            out.extend(message["content"].split())
        return out

    def content(toks):
        return sum(1 for t in toks if not t.startswith("<"))

    pairs, isl = [], []
    for conversation in conversations:
        sent = [k for k, turn in enumerate(conversation.turns) if not turn.no_request]
        for k in sent:
            isl.append(
                (content(tokens(conversation, k)), source_request(conversation.turns[k])["in"])
            )
        for a, b in itertools.pairwise(sent):
            prev, curr = tokens(conversation, a), tokens(conversation, b)
            blocks = common_prefix_len(
                source_request(conversation.turns[a])["hash_ids"],
                source_request(conversation.turns[b])["hash_ids"],
            )
            pairs.append((content(prev[: common_prefix_len(prev, curr)]), blocks * 2))
    return pairs, isl


def test_scaled_loader_reproduces_prefix_structure(tmp_path, fake_prompt_generator):
    conversations = load("weka_trace_scaled", write_traces(tmp_path, SCALED), fake_prompt_generator)
    pairs, isl = wire_checks(conversations)

    assert len(isl) == SENT
    assert all(sent == recorded for sent, recorded in isl), isl
    assert len(pairs) >= 4
    assert all(observed == expected for observed, expected in pairs), pairs

    # A turn opens with the previous reply's share of the new blocks, as an assistant message.
    main = next(c for c in conversations if c.session_id == "trace-a")
    second_turn = main.turns[1].raw_messages
    assert [m["role"] for m in second_turn] == ["assistant", "user"]
    assert len(second_turn[0]["content"].split()) == 2 * -(-3 // 2)


def test_stock_loader_does_not(tmp_path, fake_prompt_generator):
    # The registry default of 64 tokens per block overrides the trace's block_size of 2.
    conversations = load("weka_trace", write_traces(tmp_path, SCALED), fake_prompt_generator)
    pairs, isl = wire_checks(conversations)
    assert any(observed != expected for observed, expected in pairs)
    assert any(sent != recorded for sent, recorded in isl)
    roles = {m["role"] for c in conversations for t in c.turns for m in t.raw_messages or []}
    assert "assistant" not in roles


def test_scaled_loader_drops_the_block_size_override(tmp_path, fake_prompt_generator):
    directory = write_traces(tmp_path, SCALED)
    loader = WekaTraceScaledLoader(
        filename=str(directory),
        run=make_run(directory),
        prompt_generator=fake_prompt_generator,
        default_block_size=64,
    )
    assert loader._user_block_size_override is None
