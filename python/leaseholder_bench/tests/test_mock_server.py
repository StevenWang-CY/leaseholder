from __future__ import annotations

import json

import orjson
from multidict import CIMultiDict, CIMultiDictProxy

from leaseholder_bench.mock_server import Recorder, sse_chunks


def chat(session: str, *messages: tuple[str, str], max_tokens: int = 4) -> tuple[dict, dict]:
    headers = {"X-Session-ID": session, "X-LH-Class": "background"}
    body = {
        "model": "m",
        "messages": [{"role": role, "content": content} for role, content in messages],
        "max_completion_tokens": max_tokens,
        "priority": 2,
    }
    return headers, body


def test_records_metadata_and_the_cached_prefix():
    recorder = Recorder(max_tokens=64)
    first = recorder.receive(*chat("s1", ("system", "Session s1."), ("user", " a b c d")))
    reply = "".join(first.tokens)
    assert len(first.tokens) == 4
    assert first.record["cached_tokens"] == 0
    assert first.record["headers"]["X-Session-ID"] == "s1"
    assert first.record["body"] == {"model": "m", "max_completion_tokens": 4, "priority": 2}
    assert first.record["messages"] == [["system", 11], ["user", 8]]

    second = recorder.receive(
        *chat("s1", ("system", "Session s1."), ("user", " a b c d"), ("assistant", reply),
              ("user", " e f"))
    )  # fmt: skip
    # The whole first prompt is a prefix of the second: "system:", "Session", "s1.", "user:" + 4.
    assert second.record["cached_tokens"] == 8
    assert second.usage["prompt_tokens_details"] == {"cached_tokens": 8}
    assert second.usage["completion_tokens"] == 4

    # Sessions do not share a cache.
    other = recorder.receive(*chat("s2", ("system", "Session s1."), ("user", " a b c d")))
    assert other.record["cached_tokens"] == 0


def test_records_serialize_with_aiohttp_headers():
    # aiohttp header names are a str subclass, which orjson refuses as dict keys.
    headers = CIMultiDictProxy(CIMultiDict({"X-Session-ID": "s1", "x-lh-class": "background"}))
    exchange = Recorder().receive(headers, chat("s1", ("user", "hi"))[1])
    assert orjson.loads(orjson.dumps(exchange.record))["headers"] == {
        "X-Session-ID": "s1",
        "x-lh-class": "background",
    }


def test_reply_length_follows_the_request_up_to_the_cap():
    recorder = Recorder(max_tokens=3)
    assert len(recorder.receive(*chat("s", ("user", "hi"), max_tokens=10)).tokens) == 3
    assert len(recorder.receive(*chat("s", ("user", "hi"), max_tokens=0)).tokens) == 1


def test_sse_stream_carries_tokens_then_usage():
    usage = {"prompt_tokens": 5, "completion_tokens": 2}
    events = [c.decode() for c in sse_chunks("m", "r1", [" t0", " t1"], usage)]
    assert all(e.startswith("data: ") and e.endswith("\n\n") for e in events)
    assert events[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(e[len("data: ") :]) for e in events[:-1]]
    content = [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"]]
    assert content == [" t0", " t1", None]
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1] == {"id": "r1", "object": "chat.completion.chunk", "model": "m",
                          "choices": [], "usage": usage}  # fmt: skip
