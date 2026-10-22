"""lh-bench replay against an in-process server (httpx.MockTransport, no sockets)."""

from __future__ import annotations

import asyncio
import io
import json
import time

import httpx
import pytest
from helpers import request, trace, write_traces

from leaseholder_bench.mock_server import Recorder, sse_chunks
from leaseholder_bench.protocol import SessionTagger
from leaseholder_bench.replay import ReplayConfig, Replayer, Session, replay

H = list(range(1, 26))


class FakeServer:
    """Streams the mock server's replies and keeps every request it saw."""

    def __init__(self, fail: set[int] | None = None) -> None:
        self.recorder = Recorder(max_tokens=64)
        self.fail = fail or set()
        self.seen: list[dict] = []

    def __call__(self, http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        exchange = self.recorder.receive(http_request.headers, body)
        self.seen.append(
            {"headers": http_request.headers, "body": body, "record": exchange.record,
             "reply": "".join(exchange.tokens)}
        )  # fmt: skip
        if len(self.seen) - 1 in self.fail:
            return httpx.Response(400, json={"error": "bad request"})
        chunks = sse_chunks("m", f"r{len(self.seen)}", exchange.tokens, exchange.usage)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks)
        )

    def messages(self, n: int) -> list[dict]:
        return self.seen[n]["body"]["messages"]


def play(t, server, *, tagger=None, sleep=None, **config):
    async def run():
        sink = io.BytesIO()
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            replayer = Replayer(
                ReplayConfig(url="http://bench.test/v1/chat/completions", model="m", **config),
                tagger or SessionTagger(),
                client,
                sink,
            )
            if sleep is not None:
                replayer.sleep = sleep
            session = await replayer.play(t)
        return session, [json.loads(line) for line in sink.getvalue().splitlines()]

    return asyncio.run(run())


def words(message: dict) -> list[str]:
    return message["content"].split()


def test_live_reply_is_fed_back_into_the_next_turn():
    server = FakeServer()
    t = trace("trace-a", [request(0, H[:10], 3), request(1, H[:18], 5, 0.5), request(2, H, 2, 0.5)])
    session, records = play(t, server)

    first, second, third = (server.messages(n) for n in range(3))
    assert [m["role"] for m in first] == ["system", "user"]
    assert len(words(first[1])) == 20  # the trace's `in`: 10 blocks of 2 tokens
    # Turn 2 resends turn 1, then the model's reply, then the blocks after the reply's share.
    assert second[: len(first)] == first
    assert second[len(first)] == {"role": "assistant", "content": server.seen[0]["reply"]}
    reply_blocks = -(-3 // 2)
    assert second[len(first) + 1] == {
        "role": "user",
        "content": session.text(H[10 + reply_blocks : 18]),
    }
    assert third[: len(second)] == second
    assert third[len(second)]["content"] == server.seen[1]["reply"]
    assert [s["record"]["echoed_replies"] for s in server.seen] == [0, 1, 2]
    assert [r["chain"] for r in records] == [0, 0, 0]
    assert [r["turn"] for r in records] == [0, 1, 2]


def test_requests_carry_the_session_metadata():
    server = FakeServer()
    session, records = play(
        trace("trace-a", [request(0, H[:10], 3), request(1, H[:18], 0)]), server
    )

    for seen, expected_max_tokens in zip(server.seen, (3, 1), strict=True):
        assert seen["headers"]["X-Session-ID"] == session.id
        assert seen["headers"]["X-LH-Class"] == "background"
        body = seen["body"]
        assert body["prompt_cache_key"] == session.id
        assert body["priority"] == 2
        assert body["max_completion_tokens"] == expected_max_tokens
        assert body["ignore_eos"] is True
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert body["messages"][0] == {"role": "system", "content": f"Session {session.id}."}
    record = records[1]
    assert record["session_id"] == session.id and record["trace_id"] == "trace-a"
    assert record["class"] == "background" and record["status"] == 200
    assert record["cached_tokens"] == server.seen[1]["record"]["cached_tokens"] > 0
    assert record["prompt_tokens"] == server.seen[1]["record"]["prompt_tokens"]
    assert record["ttft_ms"] is not None and record["latency_ms"] >= record["ttft_ms"]
    assert (record["trace_in"], record["trace_out"]) == (36, 0)


def test_priority_and_ignore_eos_can_be_left_out():
    server = FakeServer()
    play(trace("t", [request(0, H[:4], 2)]), server, tagger=SessionTagger(send_priority=False),
         ignore_eos=False)  # fmt: skip
    assert "priority" not in server.seen[0]["body"]
    assert "ignore_eos" not in server.seen[0]["body"]


def test_compaction_keeps_the_messages_inside_the_shared_prefix():
    server = FakeServer()
    keeps_reply = [*H[:14], 70, 71]  # cuts inside the second user message, after the reply
    drops_all = [*H[:8], 80, 81]  # cuts inside the first user message
    t = trace("t", [request(0, H[:10], 3), request(1, H[:18], 5), request(2, keeps_reply, 1),
                    request(3, drops_all, 1)])  # fmt: skip
    session, _ = play(t, server)

    first, _, third, fourth = (server.messages(n) for n in range(4))
    assert third[:3] == [*first, {"role": "assistant", "content": server.seen[0]["reply"]}]
    assert third[3:] == [{"role": "user", "content": session.text(keeps_reply[12:])}]
    assert server.seen[2]["record"]["echoed_replies"] == 1
    # Nothing survives the second cut but the session line; the shared words still lead.
    assert fourth == [first[0], {"role": "user", "content": session.text(drops_all)}]
    assert words(fourth[1])[:16] == words(first[1])[:16]


def test_interleaved_side_chains_keep_their_own_history():
    server = FakeServer()
    t = trace("t", [request(0, H[:10], 3), request(1, [500, 501, 502], 2), request(2, H[:18], 5)])
    _, records = play(t, server)
    assert [r["chain"] for r in records] == [0, 1, 0]
    third = server.messages(2)
    assert {"role": "assistant", "content": server.seen[0]["reply"]} in third
    assert {"role": "assistant", "content": server.seen[1]["reply"]} not in third


def test_subagents_share_the_session_and_are_joined_as_recorded():
    server = FakeServer()
    joined = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "sub-a",
        "subagent_type": "Explore",
        "duration_ms": 1000,
        "status": "completed",
        "models": ["m"],
        "requests": [request(1.0, [200, 201], 1, 0.0), request(1.5, [200, 201, 202], 1, 0.0)],
    }
    parallel = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "sub-b",
        "subagent_type": "Explore",
        "duration_ms": 100_000,
        "status": "completed",
        "models": ["m"],
        "requests": [request(1.0, [300, 301], 1, 0.0), request(1.5, [300, 301, 302], 1, 0.2)],
    }
    t = trace("t", [request(0.0, H[:10], 3), joined, parallel, request(3.0, H[:18], 2, 0.0),
                    request(200.0, H, 2, 0.0)])  # fmt: skip
    session, records = play(t, server)

    order = [(r["agent_id"], r["turn"]) for r in records]
    position = {key: i for i, key in enumerate(order)}
    assert len(order) == 7
    # The parent's next request waits for sub-a, which the trace shows ended first ...
    assert position[("sub-a", 1)] < position[(None, 1)]
    # ... but not for sub-b, which ran past it; the request after sub-b's end waits for it.
    assert position[(None, 1)] < position[("sub-b", 1)] < position[(None, 2)]
    assert {s["headers"]["X-Session-ID"] for s in server.seen} == {session.id}
    assert {s["headers"]["X-LH-Class"] for s in server.seen} == {"background"}


def test_think_time_is_waited_out_up_to_the_cap():
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    t = trace("t", [request(0, H[:4], 1), request(1, H[:6], 1, 5.0), request(2, H[:8], 1, 2000.0)])
    play(t, FakeServer(), sleep=sleep, think_cap=600.0, duration=3600.0)
    assert waits == [5.0, 600.0]


def test_think_time_never_runs_past_the_deadline():
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    t = trace("t", [request(0, H[:4], 1), request(1, H[:6], 1, 500.0)])
    play(t, FakeServer(), sleep=sleep, duration=30.0)
    assert len(waits) == 1 and 29.0 < waits[0] <= 30.0


def test_failed_request_is_recorded_and_its_reply_not_fed_back():
    server = FakeServer(fail={1})
    t = trace("t", [request(0, H[:10], 3), request(1, H[:18], 5), request(2, H, 2)])
    session, records = play(t, server)
    assert records[1]["status"] == 400 and "bad request" in records[1]["error"]
    second, third = server.messages(1), server.messages(2)
    assert third[: len(second)] == second
    assert third[len(second) :] == [{"role": "user", "content": session.text(H[18:])}]


def test_error_inside_a_stream_counts_as_a_failure():
    def server(_: httpx.Request) -> httpx.Response:
        body = b'data: {"error": {"message": "engine died", "code": 500}}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    _, records = play(trace("t", [request(0, H[:4], 1)]), server)
    assert records[0]["status"] == 200 and "engine died" in records[0]["error"]


def test_block_text_is_deterministic_per_trace():
    a = Session(None, trace("trace-a", []))
    again = Session(None, trace("trace-a", []))
    other = Session(None, trace("trace-b", []))
    assert a.text([1, 2, 3]) == again.text([1, 2, 3])
    assert a.text([1, 2, 3]) != other.text([1, 2, 3])
    assert len(a.text([1, 2, 3]).split()) == 6
    assert a.id != again.id


def test_run_cycles_traces_until_the_deadline(tmp_path):
    server = FakeServer()
    traces = write_traces(
        tmp_path / "traces",
        trace("trace-a", [request(0, H[:10], 3), request(1, H[:18], 5, 0.01)]),
        trace("trace-b", [request(0, [900, 901], 1), request(1, [900, 901, 902], 1, 0.01)]),
    )
    out = tmp_path / "out" / "replay.jsonl"
    config = ReplayConfig(url="http://bench.test/v1/chat/completions", sessions=2,
                          duration=0.3, grace=2.0)  # fmt: skip
    started = time.monotonic()
    replayer = asyncio.run(
        replay(config, SessionTagger(), traces, out, transport=httpx.MockTransport(server))
    )
    assert time.monotonic() - started < 2.0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == len(replayer.stats) > 4
    assert {r["trace_id"] for r in records} == {"trace-a", "trace-b"}
    assert all(r["status"] == 200 and r["error"] is None for r in records)
    assert "requests:" in replayer.summary() and "cached prompt tokens" in replayer.summary()


def test_empty_trace_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no trace files"):
        asyncio.run(replay(ReplayConfig(), SessionTagger(), tmp_path, tmp_path / "out.jsonl"))
