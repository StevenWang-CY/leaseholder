"""probe.py [URL]: the make-room variant's KV check. Two priority-2 requests of 18,000 tokens fill the
engine's KV pool, leaving 24 blocks free, and keep decoding; then a priority-0 chat turn of about 700 tokens
(44 blocks) arrives. Without make-room it waits until the two have grown into the free blocks and vLLM
preempts one of them so that the other can grow (a few seconds); with it, it preempts one at once. The probe
sends the voice turn once the engine reports both running and the pool full, and prints its time to
first token and the engine's state before and after it.

The reserve (make_room.reserve_tokens) keeps the second priority-2 request out of a full pool, so the KV
case is run with it off. On the GPU node, with the engine's environment, which has aiohttp:
``~/leaseholder/venvs/engine/bin/python probe.py http://127.0.0.1:8000`` on the frozen box, then after
``deploy/phastform/variant.sh use make-room --additional-config.make_room.reserve_tokens 0``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
import sys
import time

import aiohttp

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
BACKGROUND = 2
FREE_BLOCKS = 24  # left free by the priority-2 prompts; their decodes grow into them
# The priority-2 requests decode until the probe ends. With a short reply the first would finish while the second
# still prefills, and the voice turn would find half the pool free.
DECODE_TOKENS = 4096
CALL = (
    "the call is about a late order and the customer wants it moved to friday so check the address"
    " then confirm the time window and read back the total with tax before you say goodbye"
)


async def metrics(session: aiohttp.ClientSession) -> str:
    async with session.get(f"{URL}/metrics") as response:
        return await response.text()


async def engine_state(session: aiohttp.ClientSession, num_blocks: int) -> dict[str, int]:
    """The engine's running and waiting requests and free KV blocks, from its metrics."""
    exposition = await metrics(session)
    usage = sample(exposition, "vllm:kv_cache_usage_perc")
    return {
        "running": round(sample(exposition, "vllm:num_requests_running")),
        "waiting": round(sample(exposition, "vllm:num_requests_waiting")),
        "free": round((1 - usage) * num_blocks),
    }


def sample(exposition: str, name: str) -> float:
    """A metric's value in a Prometheus exposition, summed over its label sets."""
    line = re.compile("^" + re.escape(name) + r"(?:\{[^}]*\})? (\S+)$", re.M)
    return sum(float(value) for value in line.findall(exposition))


async def stream(
    session: aiohttp.ClientSession, path: str, body: dict, first: asyncio.Event | None = None
) -> float:
    """Seconds to the first streamed token. With ``first``, set it then and read the stream to its end:
    closing the response would make vLLM abort the request."""
    started = time.perf_counter()
    ttft = None
    body = body | {"model": MODEL, "stream": True}
    async with session.post(f"{URL}{path}", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {await response.text()}")
        async for line in response.content:
            if ttft is None and line.startswith(b"data: {") and json.loads(line[6:]).get("choices"):
                ttft = time.perf_counter() - started
                if first is None:
                    return ttft
                first.set()
    if ttft is None:
        raise RuntimeError("the stream ended before its first token")
    return ttft


async def main() -> None:
    rng = random.Random()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        info = re.search(r"^vllm:cache_config_info\{([^}]*)\}", await metrics(session), re.M)
        labels = dict(re.findall(r'(\w+)="([^"]*)"', info.group(1)))
        block_size = int(labels["block_size"])
        num_blocks = int(labels["num_gpu_blocks"]) - 1  # less vLLM's null block
        # A prompt one token short of k blocks fills them with its first decoded token.
        prompt_tokens = (num_blocks - FREE_BLOCKS) // BACKGROUND * block_size - 1
        bodies = [
            {
                "prompt": [rng.randrange(1000, 100_000) for _ in range(prompt_tokens)],
                "max_tokens": DECODE_TOKENS,
                "ignore_eos": True,
                "priority": 2,
            }
            for _ in range(BACKGROUND)
        ]
        started = [asyncio.Event() for _ in bodies]
        background = [
            asyncio.create_task(stream(session, "/v1/completions", body, event))
            for body, event in zip(bodies, started, strict=True)
        ]
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 120)
        if not all(event.is_set() for event in started):
            sys.exit("probe: the priority-2 requests did not start decoding within 120 s")

        # A voice session's history that the engine has never seen, so nothing of it is cached,
        # ending on the caller's turn.
        words = CALL.split()
        system = "You are a voice agent on a phone call. Answer briefly."
        messages = [{"role": "system", "content": system}] + [
            {"role": role, "content": " ".join(rng.choices(words, k=120)) + f" ({rng.random()})"}
            for role in ("user", "assistant", "user", "assistant", "user")
        ]
        tokenize = {"model": MODEL, "messages": messages}
        async with session.post(f"{URL}/tokenize", json=tokenize) as response:
            voice_tokens = (await response.json())["count"]

        # Send the voice turn while both priority-2 requests run and the pool is full: the engine
        # reports it, so the probe does not rely on timing. Once they have grown into the free
        # blocks, vLLM moves one of them back to waiting, and a voice turn would find room.
        while True:
            state = await engine_state(session, num_blocks)
            if state["running"] == BACKGROUND and state["free"] <= FREE_BLOCKS:
                break
            if state["running"] < BACKGROUND:
                sys.exit(
                    f"probe: a priority-2 request stopped running before the pool filled ({state})"
                )
            await asyncio.sleep(0.05)
        body = {"messages": messages, "max_tokens": 4, "priority": 0}
        ttft = await stream(session, "/v1/chat/completions", body)
        after = await engine_state(session, num_blocks)
        print(
            f"priority-0 TTFT {ttft * 1000:.0f} ms for {voice_tokens} prompt tokens, sent with"
            f" {state['running']} priority-2 requests of {prompt_tokens} tokens running and"
            f" {state['free']} of {num_blocks} blocks free; after it, {after['running']} running and"
            f" {after['waiting']} waiting"
        )
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)


asyncio.run(main())
