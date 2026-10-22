from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import orjson
import pytest
from aiohttp import web

from leaseholder_bench.load import LoadSpec, stream_chunks, summarize
from leaseholder_bench.tiers import (
    Sample,
    TiersConfig,
    is_valid,
    pool_tokens,
    read_counters,
    run_tiers,
    summarize_samples,
)
from leaseholder_bench.voice_turn import VoiceTurnConfig, first_sentence, run_voice_turns


def test_load_spec_round_trips_and_rejects_malformed_specs():
    spec = LoadSpec.parse("prefill:8192x2")
    assert (spec.kind, spec.tokens, spec.concurrency) == ("prefill", 8192, 2)
    assert str(LoadSpec.parse("decode:1024x8")) == "decode:1024x8"
    for bad in ("prefill", "prefill:8192", "burst:10x2", "decode:0x1", "decode:10x0"):
        with pytest.raises(ValueError):
            LoadSpec.parse(bad)


def test_summarize_reports_percentiles():
    cell = summarize(float(x) for x in range(1, 101))
    assert cell["n"] == 100 and cell["p50"] == 50.5 and cell["p90"] == 90.1
    assert cell["p95"] == 95.05  # the SLO percentile
    single = {"n": 1, "mean": 7.0, "p10": 7.0, "p50": 7.0, "p90": 7.0, "p95": 7.0, "p99": 7.0}
    assert summarize([7.0]) == single
    assert summarize([]) == {"n": 0}


def test_stream_chunks_reads_the_json_chunks_and_the_whole_stream():
    drained: list[bool] = []

    async def body() -> AsyncIterator[bytes]:
        yield b'data: {"choices": []}\n\n'
        yield b": keep-alive\n\n"
        yield b'data:{"usage": {"prompt_tokens": 3}}\n\n'
        yield b"data: [DONE]\n\n"
        yield b"\n"
        drained.append(True)  # the body was read past [DONE], so the connection can be reused

    async def main() -> list[dict]:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body()))
        async with (
            httpx.AsyncClient(transport=transport) as client,
            client.stream("POST", "http://engine.test/v1/chat/completions") as response,
        ):
            return [chunk async for chunk in stream_chunks(response)]

    assert asyncio.run(main()) == [{"choices": []}, {"usage": {"prompt_tokens": 3}}]
    assert drained == [True]


EXPOSITION = """\
# HELP vllm:prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 1200.0
vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.7e9
vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 300.0
vllm:cache_config_info{block_size="16",cache_dtype="auto",num_gpu_blocks="2275"} 1.0
"""


def test_reads_hit_counters_and_pool_size():
    assert read_counters(EXPOSITION) == {
        "vllm:prefix_cache_hits_total": 1200.0,
        "vllm:external_prefix_cache_hits_total": 300.0,
    }
    assert pool_tokens(EXPOSITION) == 36_400
    assert pool_tokens("vllm:num_requests_running 0\n") is None


def test_validity_requires_the_intended_tier():
    assert is_valid("recompute", 1000, 0, 0)
    assert not is_valid("recompute", 1000, 992, 0)
    assert is_valid("gpu", 1000, 992, 0)
    assert not is_valid("gpu", 1000, 0, 992)  # a CPU restore is not a GPU hit
    assert is_valid("cpu", 1000, 0, 992)
    assert not is_valid("cpu", 1000, 992, 0)  # the flush did not evict the prompt


def test_fits_recover_the_tier_cost_model():
    samples = []
    for n in (1024, 4096, 16384, 32768):
        gpu = 50 + 0.0072 * n
        cpu = gpu + 8 + 0.0236 * n
        recompute = 20 + 0.2 * n + 1e-6 * n * n
        for tier, ms in (("recompute", recompute), ("gpu", gpu), ("cpu", cpu)):
            samples.append(Sample(0, n, tier, ms, n, 0, 0, True))
    samples.append(Sample(1, 1024, "cpu", 9999.0, 1024, 1000, 0, False))  # excluded
    summary = summarize_samples(samples)
    fits = summary["fits"]
    assert fits["gpu"] == {"ms": 50.0, "us_per_token": 7.2}
    assert fits["cpu_over_gpu"]["ms"] == pytest.approx(8.0, abs=0.02)  # medians are rounded
    assert fits["cpu_over_gpu"]["us_per_token"] == pytest.approx(23.6, abs=0.01)
    assert fits["recompute"]["us_per_token"] == pytest.approx(200.0, abs=0.01)
    assert fits["recompute"]["ns_per_token_squared"] == pytest.approx(1.0, abs=0.001)
    assert summary["invalid"] == {"cpu": 1}
    assert summary["cpu_restore_beats_recompute_from"] == 1024


class FakeEngine:
    """A GPU prefix cache (LRU over whole prompts, ``capacity`` tokens) over a write-through CPU
    tier, exposing vLLM's hit counters, pool size and completions API."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self.gpu: OrderedDict[tuple[int, ...], None] = OrderedDict()
        self.cpu: set[tuple[int, ...]] = set()
        self.gpu_hits = self.cpu_hits = 0

    def complete(self, prompt: tuple[int, ...]) -> None:
        reusable = len(prompt) - 1  # the last token is always computed
        if prompt in self.gpu:
            self.gpu_hits += reusable
            self.gpu.move_to_end(prompt)
        elif prompt in self.cpu:
            self.cpu_hits += reusable
        self.cpu.add(prompt)
        self.gpu[prompt] = None
        while sum(map(len, self.gpu)) > self.capacity:
            self.gpu.popitem(last=False)

    def app(self) -> web.Application:
        async def completions(request: web.Request) -> web.Response:
            body = await request.json()
            prompt = tuple(body["prompt"])
            self.complete(prompt)
            usage = {"prompt_tokens": len(prompt), "completion_tokens": 1}
            return web.json_response({"choices": [{"text": "x"}], "usage": usage})

        async def metrics(_: web.Request) -> web.Response:
            return web.Response(
                text=f'vllm:prefix_cache_hits_total{{engine="0"}} {self.gpu_hits}\n'
                f'vllm:external_prefix_cache_hits_total{{engine="0"}} {self.cpu_hits}\n'
                f'vllm:cache_config_info{{block_size="16",num_gpu_blocks="{self.capacity // 16}"}} 1\n'
            )

        async def version(_: web.Request) -> web.Response:
            return web.json_response({"version": "fake"})

        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        app.router.add_get("/metrics", metrics)
        app.router.add_get("/version", version)
        return app


@asynccontextmanager
async def serving(app: web.Application) -> AsyncIterator[str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def run(coro: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(coro())


def test_tiers_measures_each_tier_against_a_fake_engine(tmp_path: Path):
    engine = FakeEngine(capacity=64)

    async def main() -> None:
        async with serving(engine.app()) as url:
            config = TiersConfig(url=url, model="m", lengths=[16, 32], reps=2)
            summary = await run_tiers(config, tmp_path)
        assert summary["invalid"] == {}
        assert set(summary["p50_ms"]["cpu"]) == {"16", "32"}

    run(main)
    samples = [orjson.loads(line) for line in (tmp_path / "samples.jsonl").read_bytes().split()]
    assert [s["tier"] for s in samples[:3]] == ["recompute", "gpu", "cpu"]
    assert all(s["valid"] for s in samples) and len(samples) == 12
    cpu = [s for s in samples if s["tier"] == "cpu"]
    assert all(s["cpu_hit_tokens"] == s["tokens"] - 1 and s["gpu_hit_tokens"] == 0 for s in cpu)
    run_record = orjson.loads((tmp_path / "run.json").read_bytes())
    assert run_record["pool_tokens"] == 64 and run_record["engine_version"] == "fake"


def voice_stack(seen: dict[str, list]) -> web.Application:
    async def transcribe(request: web.Request) -> web.Response:
        form = await request.post()
        seen["stt"].append(form["model"])
        return web.json_response({"text": " What time do you close tonight? "})

    async def chat(request: web.Request) -> web.StreamResponse:
        seen["chat"].append((request.headers.get("X-LH-Class"), await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for piece in ("We close", " at ten tonight.", " See you", " soon!"):
            chunk = {"choices": [{"delta": {"content": piece}}]}
            await response.write(b"data: " + orjson.dumps(chunk) + b"\n\n")
            await asyncio.sleep(0.01)
        await response.write(b"data: [DONE]\n\n")
        return response

    async def speech(request: web.Request) -> web.StreamResponse:
        seen["tts"].append((await request.json())["input"])
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"\x00\x01" * 2400)
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/audio/speech", speech)
    return app


def test_voice_turn_times_every_stage_and_speaks_the_first_sentence(tmp_path: Path):
    seen: dict[str, list] = {"stt": [], "chat": [], "tts": []}
    wav = tmp_path / "turn.wav"
    wav.write_bytes(b"RIFF....WAVE")

    async def main() -> None:
        async with serving(voice_stack(seen)) as url:
            config = VoiceTurnConfig(
                stt_url=f"{url}/v1", llm_url=f"{url}/v1", tts_url=f"{url}/v1",
                stt_model="whisper", llm_model="m", tts_model="kokoro", voice="af_heart",
                audio=[wav], reps=3, warm=1, concurrency=2, gap_s=0.0,
            )  # fmt: skip
            summary = await run_voice_turns(config, tmp_path / "run")
        assert summary["turn_ms"]["n"] == 6
        assert summary["llm_first_sentence_ms"]["p50"] >= summary["llm_ttft_ms"]["p50"]

    run(main)
    assert seen["tts"][0] == "We close at ten tonight."
    header, body = seen["chat"][0]
    assert header == "realtime" and body["priority"] == 0 and body["stream"] is True
    assert body["messages"][1] == {"role": "user", "content": "What time do you close tonight?"}
    assert len(seen["stt"]) == 8  # (1 warm-up + 3 measured) x 2 callers


def test_first_sentence_holds_short_sentences_back_like_livekit():
    assert first_sentence("Sure. We close at ten tonight.") == "Sure. We close at ten tonight."
    assert first_sentence("Sure. We close at ten") is None
    assert first_sentence('The waiter said "yes, sir." Then left') == 'The waiter said "yes, sir."'
    assert first_sentence("We close at ten tonight.") == "We close at ten tonight."
