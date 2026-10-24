"""tail.py URL SECONDS [AGENTS] [fresh|grow]: the make-room variant's load check, the calls' first-token tail.

AGENTS (12) closed-loop agent sessions at priority 2 beside two calls at priority 0, against one engine for SECONDS:
- agents: 48 output tokens and 0-3 s of thought a turn; with ``fresh``, a new prompt of 6-30k tokens each turn, as
  E2's discriminating replay computes about 95% of its prompt tokens; with ``grow`` (the default), the prompt grows
  by a tool result of 1-4k tokens each turn from 6-14k up to 30k, so that it reuses the CPU tier; an agent stops at
  its first failed request;
- calls: a conversation that grows by about 30 tokens a turn, a turn every 4-8 s, 16 output tokens.
It prints each call turn's time to first token, then their distribution, with each failed turn (an error, or no first
token within TIMEOUT_MS) counted at TIMEOUT_MS, as E2's report censors a reply that never came: a percentile it sets
is marked ≥. It also prints how many turns failed and how many agents stopped early. On the GPU node, with the
engine's environment, which has aiohttp: ``~/leaseholder/venvs/engine/bin/python tail.py http://127.0.0.1:8000 240 12
fresh``.
"""

import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp

MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
CALL = (
    "the order is late and the customer wants it moved to friday so check the address then confirm"
)
TIMEOUT_MS = 30_000.0  # E2's REPLY_TIMEOUT_MS


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 100_000) for _ in range(n)]


async def agent(
    session: aiohttp.ClientSession, url: str, seed: int, stop: float, fresh: bool
) -> bool:
    """Run an agent session until ``stop``; whether it got there, rather than stopping at a failed request."""
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
            async with session.post(f"{url}/v1/completions", json=body) as response:
                await response.read()
                if response.status != 200:
                    return False
        except aiohttp.ClientError:
            return False
        if fresh:  # E2's discriminating replay computes about 95% of its prompt tokens
            context = tokens(rng, rng.randrange(6000, 30_000))
        else:
            context = context + tokens(rng, rng.randrange(1000, 4000))
            if len(context) > 30_000:
                context = tokens(rng, rng.randrange(6000, 14000))
        await asyncio.sleep(rng.uniform(0, 3))
    return True


async def turn(
    session: aiohttp.ClientSession, url: str, messages: list[dict]
) -> tuple[float | None, str]:
    """A call turn: its time to first token in ms, or None if it failed (an error, or no first token within
    TIMEOUT_MS), and its reply."""
    body = {"model": MODEL, "messages": messages, "max_tokens": 16, "priority": 0, "stream": True}
    started = time.time()
    first, reply = None, []
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT_MS / 1000),
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line.startswith(b"data: {"):
                        chunk = json.loads(line[6:])
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                        if delta:
                            first = first or time.time()
                            reply.append(delta)
    except (TimeoutError, aiohttp.ClientError):
        pass
    return ((first - started) * 1000 if first else None), "".join(reply)


async def call(
    session: aiohttp.ClientSession, url: str, seed: int, stop: float, ttfts: list[float | None]
) -> None:
    """Run a call until ``stop``, appending each turn's time to first token in ms, or None for a failed turn."""
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
        started = time.time()
        ttft, reply = await turn(session, url, messages)
        ttfts.append(ttft)
        at = f"{time.strftime('%H:%M:%S', time.gmtime(started))}.{int(started % 1 * 1000):03d}"
        print(
            f"{at} call {seed} " + (f"ttft {ttft:6.0f} ms" if ttft is not None else "failed"),
            flush=True,
        )
        messages.append({"role": "assistant", "content": reply})
        await asyncio.sleep(rng.uniform(4, 8))


def summary(ttfts: list[float | None], agents_ran: list[bool]) -> str:
    """The calls' first-token tail, a failed turn counted at TIMEOUT_MS (≥ marks a percentile one sets), and how
    many turns failed and agents stopped early."""
    samples = sorted(TIMEOUT_MS if ttft is None else ttft for ttft in ttfts)
    failed = sum(ttft is None for ttft in ttfts)
    stopped = sum(not ran for ran in agents_ran)
    agents = f"agents: {stopped} of {len(agents_ran)} stopped early"
    if len(samples) < 2:
        return f"calls: {len(samples)} turns, {failed} failed; {agents}"

    def ms(value: float) -> str:
        return ("≥" if value >= TIMEOUT_MS else "") + f"{value:.0f}"

    q = statistics.quantiles(samples, n=20)
    return (
        f"calls: {len(samples)} turns, {failed} failed (counted at {TIMEOUT_MS:,.0f} ms), TTFT p50 "
        f"{ms(statistics.median(samples))} p90 {ms(q[17])} p95 {ms(q[18])} max {ms(samples[-1])} ms; {agents}"
    )


async def main(url: str, seconds: float, num_agents: int, fresh: bool) -> None:
    stop = time.time() + seconds
    ttfts: list[float | None] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        agents_ran = await asyncio.gather(
            *(agent(session, url, 100 + i, stop, fresh) for i in range(num_agents)),
            *(call(session, url, i, stop, ttfts) for i in (1, 2)),
        )
    print(summary(ttfts, list(agents_ran[:num_agents])))


if __name__ == "__main__":
    asyncio.run(
        main(
            sys.argv[1],
            float(sys.argv[2]),
            int(sys.argv[3]) if len(sys.argv) > 3 else 12,
            len(sys.argv) > 4 and sys.argv[4] == "fresh",
        )
    )
