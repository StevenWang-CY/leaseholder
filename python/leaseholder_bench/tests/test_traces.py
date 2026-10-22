from __future__ import annotations

import copy
import json
import math

import orjson
import pytest
from aiperf.dataset.loader.weka_trace_models import WekaTrace

from leaseholder_bench.cli import main
from leaseholder_bench.traces import common_prefix_len, iter_traces, scale_corpus, scale_trace

RAW = {
    "id": "trace-a",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "s", "model": "m", "in": 320, "out": 100, "hash_ids": [1, 2, 3, 4, 5],
         "api_time": 2.0, "ttft": 1.0},
        {"t": 10.0, "type": "n", "model": "m", "in": 512, "out": 1,
         "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8], "api_time": 1.0, "think_time": 3.5},
        {"t": 12.0, "type": "subagent", "agent_id": "subagent_001", "subagent_type": "Explore",
         "duration_ms": 5000, "total_tokens": 999, "tool_use_count": 2, "status": "completed",
         "models": ["m"], "requests": [
             {"t": 12.0, "type": "n", "model": "m", "in": 192, "out": 64,
              "hash_ids": [20, 21, 22], "think_time": 0.0},
             {"t": 14.0, "type": "n", "model": "m", "in": 256, "out": 0,
              "hash_ids": [20, 21, 22, 23], "think_time": 0.0},
         ]},
        {"t": 20.0, "type": "n", "model": "m", "in": 768, "out": 2000,
         "hash_ids": list(range(1, 13)), "api_time": 4.0, "think_time": 1.0},
    ],
}  # fmt: skip


def requests_of(trace):
    for entry in trace["requests"]:
        yield from entry["requests"] if entry["type"] == "subagent" else [entry]


def test_scaling_shrinks_tokens_per_block_and_keeps_structure():
    scaled = scale_trace(copy.deepcopy(RAW), 32)

    assert scaled["block_size"] == 2
    for before, after in zip(requests_of(RAW), requests_of(scaled), strict=True):
        assert after["hash_ids"] == before["hash_ids"]
        assert after["in"] == 2 * len(after["hash_ids"])
        assert after["out"] == math.ceil(before["out"] / 32)
        for key in ("t", "type", "model", "think_time", "api_time", "ttft"):
            assert after.get(key) == before.get(key)
    subagent = scaled["requests"][2]
    assert subagent["total_tokens"] == sum(r["in"] + r["out"] for r in subagent["requests"])
    assert (subagent["agent_id"], subagent["duration_ms"]) == ("subagent_001", 5000)


def test_64x_gives_one_token_per_block():
    scaled = scale_trace(copy.deepcopy(RAW), 64)
    assert scaled["block_size"] == 1
    assert [r["in"] for r in requests_of(scaled)] == [5, 8, 3, 4, 12]
    assert [r["out"] for r in requests_of(scaled)] == [2, 1, 1, 0, 32]


@pytest.mark.parametrize("factor", [2, 4, 8, 16, 32, 64])
def test_assistant_block_count_is_unchanged(factor):
    # AIPerf labels ceil(previous out / block_size) new blocks of each turn as the assistant's.
    block_size = 64 // factor
    for out in range(5000):
        assert math.ceil(math.ceil(out / factor) / block_size) == math.ceil(out / 64)


def test_scaled_trace_is_valid_for_aiperf():
    WekaTrace.model_validate(scale_trace(copy.deepcopy(RAW), 32))


def test_prefix_tokens_are_scaled():
    raw = copy.deepcopy(RAW) | {"tool_tokens": 6400, "system_tokens": 100, "totals": {"in": 1}}
    scaled = scale_trace(raw, 32)
    assert (scaled["tool_tokens"], scaled["system_tokens"]) == (200, 4)
    assert "totals" not in scaled


def test_rejects_factor_that_does_not_divide_block_size():
    with pytest.raises(ValueError, match="not divisible"):
        scale_trace(copy.deepcopy(RAW), 3)


def test_rejects_unknown_request_type():
    raw = copy.deepcopy(RAW)
    raw["requests"].append({"t": 30.0, "type": "x"})
    with pytest.raises(ValueError, match="unknown request type"):
        scale_trace(raw, 32)


def test_corpus_from_jsonl_matches_corpus_from_directory(tmp_path):
    other = copy.deepcopy(RAW) | {"id": "trace-b"}
    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_bytes(orjson.dumps(RAW) + b"\n" + orjson.dumps(other) + b"\n")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for t in (RAW, other):
        (src_dir / f"{t['id']}.json").write_text(json.dumps(t))

    a = scale_corpus(jsonl, tmp_path / "a", 32)
    b = scale_corpus(src_dir, tmp_path / "b", 32)

    assert (a.traces, a.requests, a.peak_tokens) == (2, 10, 24 + 63)
    assert a == b
    for name in ("trace-a.json", "trace-b.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert [t["id"] for t in iter_traces(tmp_path / "a")] == ["trace-a", "trace-b"]


def test_corpus_subset_by_ids_and_limit(tmp_path):
    jsonl = tmp_path / "traces.jsonl"
    rows = [copy.deepcopy(RAW) | {"id": f"trace-{i}"} for i in range(5)]
    jsonl.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))

    picked = scale_corpus(jsonl, tmp_path / "ids", 32, ids={"trace-1", "trace-3"})
    assert picked.traces == 2
    assert sorted(p.name for p in (tmp_path / "ids").iterdir()) == ["trace-1.json", "trace-3.json"]
    assert scale_corpus(jsonl, tmp_path / "limit", 32, limit=3).traces == 3


def test_rejects_trace_id_that_is_not_a_file_name(tmp_path):
    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_bytes(orjson.dumps(copy.deepcopy(RAW) | {"id": "../escape"}) + b"\n")
    with pytest.raises(ValueError, match="file name"):
        scale_corpus(jsonl, tmp_path / "out", 32)


def test_cli_scale_traces(tmp_path, capsys):
    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_bytes(orjson.dumps(RAW) + b"\n")
    assert main(["scale-traces", str(jsonl), str(tmp_path / "out"), "--factor", "64"]) == 0
    assert "wrote 1 traces (5 requests)" in capsys.readouterr().out
    assert json.loads((tmp_path / "out" / "trace-a.json").read_text())["block_size"] == 1


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [([], [], 0), ([1, 2, 3], [1, 2, 3, 4], 3), ([1, 2, 3], [1, 9, 3], 1), ([5], [6], 0),
     (list(range(1000)), [*range(637), -1], 637), ("abcdef", "abcxyz", 3)],
)  # fmt: skip
def test_common_prefix_len(a, b, expected):
    assert common_prefix_len(a, b) == expected
    assert common_prefix_len(b, a) == expected
