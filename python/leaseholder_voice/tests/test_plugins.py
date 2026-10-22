"""The plugin factory against the mock servers, over Unix sockets (no network).

The base URL's host is in the reserved ``.invalid`` domain, which never resolves, so
every request that succeeds went through the socket the factory was given.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import llm
from livekit.plugins import openai as lk_openai

from leaseholder_voice import plugins

from .mocks import FIRST_SENTENCE, TRANSCRIPT, OpenAIMocks

BASE_URL = "http://lh.invalid/v1"


@pytest.fixture(scope="module")
def mocks() -> Iterator[OpenAIMocks]:
    # a short directory: socket paths are limited to 104 bytes on macOS, 108 on Linux
    with (
        tempfile.TemporaryDirectory(prefix="lh-voice-") as sockets,
        OpenAIMocks(
            Path(sockets), token_interval=0.01, stream_seconds=0.1, speech_seconds=0.5
        ) as m,
    ):
        yield m


def test_llm_streams_with_session_headers_over_uds(mocks: OpenAIMocks) -> None:
    model = plugins.make_llm(BASE_URL, "voice-RM_test", uds=mocks.llm_uds)
    assert isinstance(model, lk_openai.LLM)

    async def chat() -> str:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content=TRANSCRIPT)
        text = []
        async with model.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text.append(chunk.delta.content)
        await model.aclose()
        return "".join(text)

    assert asyncio.run(chat()).startswith(FIRST_SENTENCE)
    [stream] = mocks.chat_streams()
    assert stream.headers["x-session-id"] == "voice-RM_test"
    assert stream.headers["x-lh-class"] == "realtime"
    assert stream.body["model"] == plugins.LLM_MODEL
    assert stream.body["stream"] is True
    assert stream.completed_at is not None
    assert model._client.is_closed(), "aclose() must close the client the factory created"


def test_stt_transcribes_over_uds(mocks: OpenAIMocks) -> None:
    stt = plugins.make_stt(BASE_URL, uds=mocks.stt_uds)
    assert isinstance(stt, lk_openai.STT)
    half_second = rtc.AudioFrame(np.zeros(8000, np.int16).tobytes(), 16000, 1, 8000)

    event = asyncio.run(stt.recognize([half_second]))

    assert event.alternatives[0].text == TRANSCRIPT
    [request] = mocks.transcriptions()
    assert request.fields["model"] == plugins.STT_MODEL
    assert request.fields["language"] == "en"
    assert request.audio_bytes == 44 + 16000  # WAV header + 0.5 s of 16-bit mono


def test_tts_returns_raw_pcm_over_uds(mocks: OpenAIMocks) -> None:
    tts = plugins.make_tts(BASE_URL, uds=mocks.tts_uds)
    assert isinstance(tts, lk_openai.TTS)

    async def synthesize() -> list[rtc.AudioFrame]:
        async with tts.synthesize("Hello there.") as stream:
            frames = [audio.frame async for audio in stream]
        await tts.aclose()
        return frames

    frames = asyncio.run(synthesize())

    assert {(frame.sample_rate, frame.num_channels) for frame in frames} == {(24000, 1)}
    pcm = b"".join(bytes(frame.data) for frame in frames)
    # the PCM passes through untouched; the emitter ends with at most 10 ms of silence
    assert pcm[: len(mocks.speech_pcm)] == mocks.speech_pcm
    assert pcm[len(mocks.speech_pcm) :] == bytes(len(pcm) - len(mocks.speech_pcm))
    assert len(pcm) - len(mocks.speech_pcm) <= 24000 * 2 // 100
    [request] = mocks.speech_requests()
    assert request.body["response_format"] == "pcm"
    assert request.body["model"] == plugins.TTS_MODEL
    assert request.body["voice"] == plugins.TTS_VOICE
    assert request.body["input"] == "Hello there."
    assert tts._client.is_closed(), "aclose() must close the client the factory created"
