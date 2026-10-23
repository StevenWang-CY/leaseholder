"""OpenAI-compatible STT, LLM and TTS mock servers on Unix sockets.

The servers run on their own event loop in a background thread, like services outside
the voice plane, and time what they see with ``time.monotonic()``, the clock that every
process on the host shares.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import web

TRANSCRIPT = "Tell me a long story."
FIRST_SENTENCE = "Once upon a time, a lighthouse keeper kept a log of every ship."
STORY = "Night after night the keeper climbed the stairs and lit the lamp."
TTS_SAMPLE_RATE = 24_000


@dataclass
class ChatStream:
    """One streamed chat completion, as the mock LLM saw it."""

    headers: dict[str, str]  # lower-case names
    body: dict[str, Any]
    opened_at: float
    chunks_sent: int = 0
    completed_at: float | None = None
    disconnected_at: float | None = None

    @property
    def closed_at(self) -> float | None:
        return self.completed_at if self.completed_at is not None else self.disconnected_at

    def open_at(self, t: float) -> bool:
        return self.opened_at <= t and (self.closed_at is None or self.closed_at > t)


@dataclass
class Transcription:
    received_at: float
    fields: dict[str, str]
    audio_bytes: int


@dataclass
class Speech:
    received_at: float
    body: dict[str, Any]


class OpenAIMocks:
    """Mock STT, LLM and TTS endpoints, each on its own Unix socket under ``socket_dir``.

    The LLM streams the story one token per ``token_interval`` for ``stream_seconds``
    and records when the client disconnects. The TTS answers every request with
    ``speech_seconds`` of a 440 Hz tone as raw 24 kHz PCM. The STT always returns
    ``TRANSCRIPT``. Like the control plane's socket, the LLM's also takes lifecycle
    events (``POST /v1/events``), and records them.
    """

    def __init__(
        self,
        socket_dir: Path,
        *,
        token_interval: float = 0.1,
        stream_seconds: float = 20.0,
        speech_seconds: float = 2.0,
    ) -> None:
        self.stt_uds = str(socket_dir / "stt.sock")
        self.llm_uds = str(socket_dir / "llm.sock")
        self.tts_uds = str(socket_dir / "tts.sock")
        self.token_interval = token_interval
        self.stream_tokens = max(1, round(stream_seconds / token_interval))
        self.speech_seconds = speech_seconds
        self.speech_pcm = _tone(speech_seconds)
        self._lock = threading.Lock()
        self._chat_streams: list[ChatStream] = []
        self._transcriptions: list[Transcription] = []
        self._speech_requests: list[Speech] = []
        self._lifecycle_events: list[dict[str, Any]] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="openai-mocks")
        self._runners: list[web.AppRunner] = []

    def __enter__(self) -> OpenAIMocks:
        self._thread.start()
        self._call(self._serve())
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._call(self._shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
            self._loop.close()

    def chat_streams(self) -> list[ChatStream]:
        with self._lock:
            return [dataclasses.replace(s) for s in self._chat_streams]

    def transcriptions(self) -> list[Transcription]:
        with self._lock:
            return list(self._transcriptions)

    def speech_requests(self) -> list[Speech]:
        with self._lock:
            return list(self._speech_requests)

    def lifecycle_events(self) -> list[dict[str, Any]]:
        """The events posted, in arrival order, each with its batch's ``received_at``."""
        with self._lock:
            return list(self._lifecycle_events)

    def _call(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    async def _serve(self) -> None:
        apps = {
            self.stt_uds: [web.post("/v1/audio/transcriptions", self._transcribe)],
            self.llm_uds: [
                web.get("/v1/models", self._models),
                web.post("/v1/chat/completions", self._chat_completions),
                web.post("/v1/events", self._events),
            ],
            self.tts_uds: [web.post("/v1/audio/speech", self._synthesize)],
        }
        for path, routes in apps.items():
            app = web.Application()
            app.add_routes(routes)
            # handler_cancellation: a client disconnect cancels the handler at once, so
            # the LLM records the moment the plugin closes its stream
            runner = web.AppRunner(
                app, handler_cancellation=True, shutdown_timeout=1.0, access_log=None
            )
            await runner.setup()
            await web.UnixSite(runner, path).start()
            self._runners.append(runner)

    async def _shutdown(self) -> None:
        for runner in self._runners:
            await runner.cleanup()

    async def _models(self, request: web.Request) -> web.Response:
        model = {"id": "mock", "object": "model", "created": 0, "owned_by": "leaseholder"}
        return web.json_response({"object": "list", "data": [model]})

    async def _chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        stream = ChatStream(
            headers={name.lower(): value for name, value in request.headers.items()},
            body=body,
            opened_at=time.monotonic(),
        )
        with self._lock:
            self._chat_streams.append(stream)
            completion_id = f"chatcmpl-mock-{len(self._chat_streams)}"
        model = body.get("model", "mock")
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        try:
            await response.prepare(request)
            await response.write(_sse(_chunk(completion_id, model, {"role": "assistant"})))
            for token in self._story():
                await response.write(_sse(_chunk(completion_id, model, {"content": token})))
                with self._lock:
                    stream.chunks_sent += 1
                await asyncio.sleep(self.token_interval)
            await response.write(_sse(_chunk(completion_id, model, {}, finish_reason="stop")))
            usage = {
                "prompt_tokens": 16,
                "completion_tokens": self.stream_tokens,
                "total_tokens": 16 + self.stream_tokens,
            }
            await response.write(_sse({**_chunk(completion_id, model), "usage": usage}))
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except asyncio.CancelledError:  # the client closed the connection
            self._mark_disconnected(stream)
            raise
        except ConnectionResetError:  # a write found the connection closed
            self._mark_disconnected(stream)
            return response
        with self._lock:
            stream.completed_at = time.monotonic()
        return response

    def _mark_disconnected(self, stream: ChatStream) -> None:
        with self._lock:
            if stream.disconnected_at is None:
                stream.disconnected_at = time.monotonic()

    def _story(self) -> list[str]:
        words = STORY.split()
        tokens = [FIRST_SENTENCE]
        tokens += [f" {words[i % len(words)]}" for i in range(self.stream_tokens - 1)]
        return tokens

    async def _events(self, request: web.Request) -> web.Response:
        body = await request.json()
        batch = body if isinstance(body, list) else [body]
        received_at = time.monotonic()
        with self._lock:
            self._lifecycle_events += [event | {"received_at": received_at} for event in batch]
        return web.json_response({"accepted": len(batch), "duplicates": 0}, status=202)

    async def _transcribe(self, request: web.Request) -> web.Response:
        form = await request.post()
        audio = form.get("file")
        size = len(audio.file.read()) if isinstance(audio, web.FileField) else 0
        fields = {name: value for name, value in form.items() if isinstance(value, str)}
        with self._lock:
            self._transcriptions.append(Transcription(time.monotonic(), fields, size))
        return web.json_response({"text": TRANSCRIPT})

    async def _synthesize(self, request: web.Request) -> web.Response:
        body = await request.json()
        with self._lock:
            self._speech_requests.append(Speech(time.monotonic(), body))
        return web.Response(
            body=self.speech_pcm,
            content_type="audio/pcm",
            headers={"x-request-id": f"speech-mock-{len(self._speech_requests)}"},
        )


def _chunk(
    completion_id: str,
    model: str,
    delta: dict[str, str] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    choices = (
        [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _tone(seconds: float) -> bytes:
    """A 440 Hz tone at a quarter of full scale with 10 ms fades, as 16-bit PCM."""
    t = np.arange(round(seconds * TTS_SAMPLE_RATE)) / TTS_SAMPLE_RATE
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    ramp = min(len(signal) // 2, TTS_SAMPLE_RATE // 100)
    signal[:ramp] *= np.linspace(0.0, 1.0, ramp)
    signal[len(signal) - ramp :] *= np.linspace(1.0, 0.0, ramp)
    return (signal * 32767).astype("<i2").tobytes()
