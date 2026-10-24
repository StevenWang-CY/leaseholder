"""Recording OpenAI-compatible mock for CPU-only dry runs (``lh-bench mock-server``).

It binds 127.0.0.1 only and streams chat completions with a fixed time to first token and
inter-token gap. Each request appends one JSON line to the log: arrival and end times, every
header, every body field except ``messages``, a role/size digest of the messages, and
``cached_tokens``: the longest word-level prefix the prompt shares with one of the session's recent
prompts, a stand-in for a per-session prefix cache. It is also returned as
``usage.prompt_tokens_details.cached_tokens``.

Sessions are keyed by the ``X-Session-ID`` header. Token counts are whitespace words.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import IO, Any

import orjson
from aiohttp import web

from leaseholder_bench.protocol import SESSION_HEADER
from leaseholder_bench.traces import common_prefix_len

HOST = "127.0.0.1"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


@dataclass
class Exchange:
    serial: int
    record: dict[str, Any]
    tokens: list[str]
    usage: dict[str, Any]


class Recorder:
    """Builds replies and request records; keeps a bounded per-session history."""

    def __init__(self, max_tokens: int = 64, history: int = 4, max_sessions: int = 1024) -> None:
        self.max_tokens = max_tokens
        self.history = history
        self.max_sessions = max_sessions
        self._serial = itertools.count()
        self._sessions: OrderedDict[str, deque[str]] = OrderedDict()

    def _session(self, key: str) -> deque[str]:
        prompts = self._sessions.pop(key, None)
        if prompts is None:
            prompts = deque(maxlen=self.history)
        self._sessions[key] = prompts
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return prompts

    def receive(self, headers: Mapping[str, str], body: dict[str, Any]) -> Exchange:
        messages = body.get("messages") or []
        texts = [_text(m.get("content")) for m in messages]
        prompt = "\n".join(f"{m.get('role')}: {t}" for m, t in zip(messages, texts, strict=True))
        prompts = self._session(headers.get(SESSION_HEADER, ""))
        shared = max((common_prefix_len(p, prompt) for p in prompts), default=0)
        cached = len(prompt[:shared].split())
        serial = next(self._serial)
        requested = body.get("max_completion_tokens", body.get("max_tokens"))
        count = max(1, min(16 if requested is None else int(requested), self.max_tokens))
        tokens = [f" r{serial}t{i}" for i in range(count)]
        prompts.append(prompt)
        prompt_tokens = len(prompt.split())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        record = {
            "headers": {str(name): str(value) for name, value in headers.items()},
            "body": {k: v for k, v in body.items() if k != "messages"},
            "messages": [[m.get("role"), len(t)] for m, t in zip(messages, texts, strict=True)],
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached,
        }
        return Exchange(serial, record, tokens, usage)


def sse_chunks(
    model: str, request_id: str, tokens: list[str], usage: dict[str, Any]
) -> list[bytes]:
    """The streamed response: one content chunk per token, a finish chunk, a usage chunk, done."""

    def event(choices: list[dict[str, Any]], **extra: Any) -> bytes:
        chunk = {"id": request_id, "object": "chat.completion.chunk", "model": model}
        chunk.update(choices=choices, **extra)
        return b"data: " + orjson.dumps(chunk) + b"\n\n"

    chunks = [
        event([{"index": 0, "delta": {"content": token}, "finish_reason": None}])
        for token in tokens
    ]
    chunks.append(event([{"index": 0, "delta": {}, "finish_reason": "length"}]))
    chunks.append(event([], usage=usage))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def build_app(
    recorder: Recorder,
    log: IO[bytes] | None = None,
    *,
    model: str = "mock",
    ttft_ms: float = 20.0,
    itl_ms: float = 2.0,
) -> web.Application:
    async def chat(request: web.Request) -> web.StreamResponse:
        arrival = time.time()
        body = await request.json()
        exchange = recorder.receive(request.headers, body)
        request_id = f"chatcmpl-{exchange.serial}"
        await asyncio.sleep(ttft_ms / 1000)
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for i, chunk in enumerate(
                sse_chunks(model, request_id, exchange.tokens, exchange.usage)
            ):
                if 0 < i < len(exchange.tokens):
                    await asyncio.sleep(itl_ms / 1000)
                await response.write(chunk)
            await response.write_eof()
        else:
            await asyncio.sleep(itl_ms * (len(exchange.tokens) - 1) / 1000)
            message = {"role": "assistant", "content": "".join(exchange.tokens)}
            choice = {"index": 0, "message": message, "finish_reason": "length"}
            response = web.json_response(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "model": model,
                    "choices": [choice],
                    "usage": exchange.usage,
                }
            )
        if log is not None:
            exchange.record.update(t_arrival=arrival, t_end=time.time())
            log.write(orjson.dumps(exchange.record) + b"\n")
            log.flush()
        return response

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", models)
    return app


def serve(
    port: int,
    log: IO[bytes] | None,
    *,
    model: str = "mock",
    ttft_ms: float = 20.0,
    itl_ms: float = 2.0,
    max_tokens: int = 64,
) -> None:
    """Run the mock on 127.0.0.1:``port`` until interrupted."""
    app = build_app(Recorder(max_tokens), log, model=model, ttft_ms=ttft_ms, itl_ms=itl_ms)
    web.run_app(app, host=HOST, port=port)
