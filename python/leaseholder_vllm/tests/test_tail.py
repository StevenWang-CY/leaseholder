"""The make-room variant's load check (deploy/phastform/variants/make-room/tail.py): a failed call turn counts at the
timeout, as E2's report censors a reply that never came, and the turns that failed and the agents that stopped early
are printed. It needs aiohttp, which the engine environment has; without it, as in CI, these tests skip."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

aiohttp = pytest.importorskip("aiohttp")

TAIL = Path(__file__).resolve().parents[3] / "deploy/phastform/variants/make-room/tail.py"


@pytest.fixture(scope="module")
def tail():
    spec = importlib.util.spec_from_file_location("lh_tail", TAIL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    """An aiohttp response as tail.py reads it: a status, and a stream of server-sent lines."""

    def __init__(
        self, status: int, lines: list[bytes] | None = None, error: Exception | None = None
    ):
        self.status, self.lines, self.error = status, lines or [], error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def read(self) -> bytes:
        return b""

    @property
    def content(self):
        async def lines():
            for line in self.lines:
                yield line

        return lines()


class Session:
    def __init__(self, *responses: Response):
        self.responses = list(responses)

    def post(self, url, json=None, timeout=None):
        return self.responses.pop(0)


def streamed(*deltas: str) -> list[bytes]:
    chunks = [{"choices": [{"delta": {"content": delta}}]} for delta in deltas]
    return [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]


def test_a_call_turn_fails_on_an_error_status_an_exception_or_the_timeout(tail):
    messages = [{"role": "user", "content": "hello"}]

    def turn(response):
        return asyncio.run(tail.turn(Session(response), "http://engine", messages))

    ttft, reply = turn(Response(200, streamed("Sure", ", done.")))
    assert ttft is not None and reply == "Sure, done."
    assert turn(Response(500)) == (None, "")
    assert turn(Response(200, error=aiohttp.ClientConnectionError())) == (None, "")
    assert turn(Response(200, error=TimeoutError())) == (None, "")
    assert turn(Response(200, [b"data: [DONE]\n"])) == (None, "")  # no token at all


def test_an_agent_stops_at_its_first_failed_request(tail):
    for response in (Response(503), Response(200, error=aiohttp.ClientConnectionError())):
        session = Session(response)
        ran = asyncio.run(tail.agent(session, "http://engine", 100, stop=float("inf"), fresh=True))
        assert ran is False and not session.responses


def test_the_summary_counts_failed_turns_at_the_timeout_and_stopped_agents(tail):
    ttfts = [100.0 + i for i in range(18)] + [None, None]
    assert tail.summary(ttfts, [True] * 11 + [False]) == (
        "calls: 20 turns, 2 failed (counted at 30,000 ms), TTFT p50 110 p90 27012 p95 ≥30000 max ≥30000 ms; "
        "agents: 1 of 12 stopped early"
    )
    assert tail.summary([100.0 + i for i in range(20)], [True]) == (
        "calls: 20 turns, 0 failed (counted at 30,000 ms), TTFT p50 110 p90 118 p95 119 max 119 ms; "
        "agents: 0 of 1 stopped early"
    )
