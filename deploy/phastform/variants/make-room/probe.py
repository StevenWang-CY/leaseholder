"""probe.py [URL]: the make-room variant's KV check. Two priority-2 requests of 18,000 tokens fill the
engine's KV pool and decode, leaving 24 blocks free and growing into them only; then a priority-0 chat turn
of about 700 tokens (44 blocks) arrives. Without make-room it waits until a priority-2 request finishes
(192 decoded tokens); with it, it preempts one of them. The probe prints its time to first token, the free
blocks it found and the preemptions.

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
FREE_BLOCKS = 24  # left free by the priority-2 prompts; their decodes grow into them, no further
CALL = (
    "the call is about a late order and the customer wants it moved to friday so check the address"
    " then confirm the time window and read back the total with tax before you say goodbye"
)


async def metrics(session: aiohttp.ClientSession) -> str:
    async with session.get(f"{URL}/metrics") as response:
        return await response.text()


def sample(exposition: str, name: str) -> float:
    """A metric's value in a Prometheus exposition, summed over its label sets."""
    line = re.compile("^" + re.escape(name) + r"(?:\{[^}]*\})? (\S+)$", re.M)
    return sum(float(value) for value in line.findall(exposition))


async def stream(
    session: aiohttp.ClientSession, path: str, body: dict, first: asyncio.Event | None = None
) -> float:
    """Seconds to the first streamed token; with ``first``, set it then and read on to the end."""
    started = time.perf_counter()
    body = body | {"model": MODEL, "stream": True}
    async with session.post(f"{URL}{path}", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {await response.text()}")
        async for line in response.content:
            if line.startswith(b"data: {") and json.loads(line[6:]).get("choices"):
                ttft = time.perf_counter() - started
                if first is None:
                    return ttft
                first.set()
                first = None
    raise RuntimeError("the stream ended before its first token")


async def main() -> None:
    rng = random.Random()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        info = re.search(r"^vllm:cache_config_info\{([^}]*)\}", await metrics(session), re.M)
        labels = dict(re.findall(r'(\w+)="([^"]*)"', info.group(1)))
        block_size = int(labels["block_size"])
        num_blocks = int(labels["num_gpu_blocks"]) - 1  # less vLLM's null block
        # A prompt one token short of k blocks fills them with its first decoded token; each request
        # then decodes into FREE_BLOCKS / BACKGROUND blocks and stops there.
        prompt_tokens = (num_blocks - FREE_BLOCKS) // BACKGROUND * block_size - 1
        decode_tokens = FREE_BLOCKS // BACKGROUND * block_size
        bodies = [
            {
                "prompt": [rng.randrange(1000, 100_000) for _ in range(prompt_tokens)],
                "max_tokens": decode_tokens,
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
        decoding = sum(event.is_set() for event in started)

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
        before = await metrics(session)
        free = round((1 - sample(before, "vllm:kv_cache_usage_perc")) * num_blocks)
        body = {"messages": messages, "max_tokens": 4, "priority": 0}
        ttft = await stream(session, "/v1/chat/completions", body)
        after = await metrics(session)
        preemptions = sample(after, "vllm:num_preemptions_total") - sample(
            before, "vllm:num_preemptions_total"
        )
        print(
            f"priority-0 TTFT {ttft * 1000:.0f} ms for {voice_tokens} prompt tokens, with {decoding}"
            f" of {BACKGROUND} priority-2 requests of {prompt_tokens} tokens decoding and {free}"
            f" of {num_blocks} blocks free; preemptions: {preemptions:.0f}"
        )
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)


asyncio.run(main())
