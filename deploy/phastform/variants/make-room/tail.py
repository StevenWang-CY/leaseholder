"""tail.py URL SECONDS [AGENTS] [fresh|grow]: the make-room variant's load check, the calls' first-token tail.

AGENTS (12) closed-loop agent sessions at priority 2 beside two calls at priority 0, against one engine for SECONDS:
- agents: 48 output tokens and 0-3 s of thought a turn; with ``fresh``, a new prompt of 6-30k tokens each turn, as
  E2's discriminating replay computes 98% of its prompt tokens; with ``grow`` (the default), the prompt grows by a
  tool result of 1-4k tokens each turn from 6-14k up to 30k, so that it reuses the CPU tier;
- calls: a conversation that grows by about 30 tokens a turn, a turn every 4-8 s, 16 output tokens.
It prints each call turn's time to first token, then their distribution. On the GPU node, with the engine's
environment, which has aiohttp: ``~/leaseholder/venvs/engine/bin/python tail.py http://127.0.0.1:8000 240 12 fresh``.
"""

import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp

URL, SECONDS = sys.argv[1], float(sys.argv[2])
AGENTS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
FRESH = len(sys.argv) > 4 and sys.argv[4] == "fresh"
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
CALL = (
    "the order is late and the customer wants it moved to friday so check the address then confirm"
)


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 100_000) for _ in range(n)]


async def agent(session: aiohttp.ClientSession, seed: int, stop: float) -> None:
    rng = random.Random(seed)
    context = tokens(rng, rng.randrange(6000, 14000))
    while time.time() < stop:
        body = {
            "model": MODEL,
            "prompt": context,
            "max_tokens": 48,
            "ignore_eos": True,
            "priority": 2,
        }
        try:
            async with session.post(f"{URL}/v1/completions", json=body) as response:
                await response.read()
        except aiohttp.ClientError:
            return
        if FRESH:  # E2's discriminating replay computes 98% of its prompt tokens
            context = tokens(rng, rng.randrange(6000, 30_000))
        else:
            context = context + tokens(rng, rng.randrange(1000, 4000))
            if len(context) > 30_000:
                context = tokens(rng, rng.randrange(6000, 14000))
        await asyncio.sleep(rng.uniform(0, 3))


async def call(session: aiohttp.ClientSession, seed: int, stop: float, ttfts: list) -> None:
    rng = random.Random(seed)
    messages = [
        {
            "role": "system",
            "content": f"You are a phone agent ({seed}). Answer in one short sentence.",
        }
    ]
    await asyncio.sleep(rng.uniform(0, 4))
    while time.time() < stop:
        messages.append({"role": "user", "content": " ".join(rng.choices(CALL.split(), k=20))})
        body = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 16,
            "priority": 0,
            "stream": True,
        }
        started = time.time()
        first, reply = None, []
        async with session.post(f"{URL}/v1/chat/completions", json=body) as response:
            async for line in response.content:
                if line.startswith(b"data: {"):
                    chunk = json.loads(line[6:])
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        first = first or time.time()
                        reply.append(delta)
        ttft = (first - started) * 1000 if first else float("nan")
        ttfts.append(ttft)
        print(
            f"{time.strftime('%H:%M:%S', time.gmtime(started))}.{int(started % 1 * 1000):03d} call {seed} ttft {ttft:6.0f} ms",
            flush=True,
        )
        messages.append({"role": "assistant", "content": "".join(reply)})
        await asyncio.sleep(rng.uniform(4, 8))


async def main() -> None:
    stop = time.time() + SECONDS
    ttfts: list[float] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        await asyncio.gather(
            *(agent(session, 100 + i, stop) for i in range(AGENTS)),
            *(call(session, i, stop, ttfts) for i in (1, 2)),
        )
    ok = sorted(t for t in ttfts if t == t)
    q = statistics.quantiles(ok, n=20)
    print(
        f"calls: {len(ok)} turns, TTFT p50 {statistics.median(ok):.0f} p90 {q[17]:.0f} p95 {q[18]:.0f} max {ok[-1]:.0f} ms"
    )


asyncio.run(main())
