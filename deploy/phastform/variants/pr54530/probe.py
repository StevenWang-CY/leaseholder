"""probe.py [URL]: the pr54530 variant's check. Fill the engine's 12 running slots (``--max-num-seqs 12``) with
priority-2 decodes, then time a priority-0 request. With the PR it preempts one of them; without it, it waits for a
slot. Measured on the frozen box on 2026-10-22, with the port's synchronous Scheduler before the variant loaded the
stock AsyncScheduler over it: 81 ms with the PR, 8,900 ms without; not measured since.

Run it on the GPU node with the engine's environment, which has aiohttp:
``~/leaseholder/venvs/engine/bin/python probe.py http://127.0.0.1:8000``
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import aiohttp

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
SLOTS = 12


async def chat(session: aiohttp.ClientSession, priority: int, max_tokens: int, tag: str) -> float:
    """Seconds to the first streamed token of one chat completion."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Count slowly from one. ({tag})"}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "priority": priority,
    }
    started = time.perf_counter()
    async with session.post(f"{URL}/v1/chat/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {await response.text()}")
        async for line in response.content:
            if line.startswith(b"data: {") and json.loads(line[6:]).get("choices"):
                return time.perf_counter() - started
    raise RuntimeError("the stream ended before its first token")


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        background = [asyncio.create_task(chat(session, 2, 800, f"bg{i}")) for i in range(SLOTS)]
        await asyncio.sleep(3)  # every slot is decoding
        ttft = await chat(session, 0, 4, "voice")
        print(f"priority-0 TTFT {ttft * 1000:.0f} ms with {SLOTS} priority-2 decodes running")
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)


asyncio.run(main())
