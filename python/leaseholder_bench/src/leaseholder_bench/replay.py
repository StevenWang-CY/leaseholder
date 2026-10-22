"""Live-reply replay of block-scaled agent traces (``lh-bench replay``).

AIPerf replays a synthesized assistant history, so the KV the model writes while decoding turn k is
never matched by turn k+1. This driver keeps a trace's structure (context growth, think time,
subagents, interleaved side chains) but appends the model's actual reply to the history before
the next turn, the way an agent resends its own output.

Prompt text: each hash block becomes ``block_size`` words from :data:`WORDS`, seeded by trace and
hash id, so equal hash ids give equal text and block-level prefix identity holds on the wire.
Every word is a single Qwen3 token, so prompts track the trace's ``in``. A per-play system line
keeps two plays of one trace from sharing KV.

History: a request continues the chain (one message history) of its agent whose last prompt shares
the longest hash prefix with it, or starts a new chain, as AIPerf does for interleaved hidden agents.
If the request extends that prompt, the live reply stands for the first ``ceil(out / block_size)``
new blocks and the remaining new blocks become one user message. If it keeps only part of it (a
compaction or an edit), the messages inside the shared prefix stay and the rest is resent as one
user message.

Subagents run as concurrent tasks under the parent's session id. A parent waits for a subagent
only where the trace shows it did: the subagent ended before the parent's next request.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx
import orjson

from leaseholder_bench.protocol import SessionTagger
from leaseholder_bench.traces import common_prefix_len

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"

_WORD_LIST = """
have has had can will one all new more time about which their there were been would other
what when your only also some into than its use most over such these after first two then
them way any many like may before through back where much should well each just those people
work part year years between both under same last long make made great still own while found
here take life place right down data state system file code test value type name line list
error table field model user server client event order group point level plan rule note page
form view step task item case mode path port node key open close start stop read write send
call load save find show build check update create delete return print parse match merge
split sort count index query cache store queue token block batch frame chunk string number
object array class method module package import export config option setting input output
result source target default local remote global public private static final const simple
basic common main core base full small large high low fast slow early late next previous
above below inside outside left top bottom front north south east west water light sound
power energy force speed space earth world city house room door window chair paper book
story music movie game sport team player coach school student teacher lesson study market
price cost money bank trade sale stock share growth profit budget income credit account
health doctor patient hospital body
"""
WORDS = tuple(_WORD_LIST.split())
"""256 words that Qwen3's tokenizer encodes as one token each, with a leading space."""


@dataclass
class ReplayConfig:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    sessions: int = 4
    """Session trees replayed at once; each lane starts the next trace when one finishes."""
    duration: float = 60.0
    """Seconds after which no new request starts."""
    grace: float = 30.0
    """Seconds after ``duration`` before in-flight requests are cancelled."""
    think_cap: float = 600.0
    """Cap on each recorded think time, in seconds."""
    timeout: float = 300.0
    ignore_eos: bool = True
    """Decode exactly the trace's ``out`` tokens, so contexts grow as recorded."""
    seed: int = 0


@dataclass
class Segment:
    message: dict[str, str]
    blocks: int
    """Trace blocks the message stands for."""


@dataclass
class Chain:
    """One message history of one agent."""

    index: int
    segments: list[Segment]
    last: list[int] = field(default_factory=list)
    """Hash ids of the previous prompt."""
    covered: int = 0
    """Trace blocks the segments stand for."""


@dataclass
class Stat:
    ok: bool
    ttft_ms: float | None
    prompt_tokens: int
    cached_tokens: int


class Session:
    """One play of one trace. The main agent and its subagents share :attr:`id`."""

    def __init__(self, replayer: Replayer, trace: dict[str, Any]) -> None:
        self.replayer = replayer
        self.trace = trace
        self.id = str(uuid.uuid4())
        self.trace_id: str = trace["id"]
        self.block_size: int = trace["block_size"]
        self._texts: dict[int, str] = {}
        self._tasks: asyncio.TaskGroup | None = None

    def text(self, hash_ids: list[int]) -> str:
        parts = []
        for hash_id in hash_ids:
            block = self._texts.get(hash_id)
            if block is None:
                rng = random.Random(f"{self.trace_id}/{hash_id}")
                block = "".join(" " + w for w in rng.choices(WORDS, k=self.block_size))
                self._texts[hash_id] = block
            parts.append(block)
        return "".join(parts)

    def new_chain(self, index: int) -> Chain:
        marker = {"role": "system", "content": f"Session {self.id}."}
        return Chain(index=index, segments=[Segment(marker, 0)])

    def prompt(self, chain: Chain, hash_ids: list[int]) -> list[dict[str, str]]:
        """Messages for the next request on ``chain``; updates the chain."""
        shared = common_prefix_len(chain.last, hash_ids)
        if shared < len(chain.last):
            kept, covered = [], 0
            for segment in chain.segments:
                if covered + segment.blocks > shared:
                    break
                kept.append(segment)
                covered += segment.blocks
            chain.segments, chain.covered = kept, covered
        new = hash_ids[chain.covered :]
        if new:
            chain.segments.append(Segment({"role": "user", "content": self.text(new)}, len(new)))
        chain.last, chain.covered = hash_ids, len(hash_ids)
        return [segment.message for segment in chain.segments]

    def add_reply(self, chain: Chain, reply: str, out_tokens: int) -> None:
        blocks = -(-out_tokens // self.block_size)
        chain.segments.append(Segment({"role": "assistant", "content": reply}, blocks))
        chain.covered += blocks

    async def play(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            self._tasks = tasks
            await self.run_agent(self.trace["requests"], agent_id=None)

    async def run_agent(self, requests: list[dict[str, Any]], agent_id: str | None) -> None:
        assert self._tasks is not None
        chains: list[Chain] = []
        running: list[tuple[float, asyncio.Task[None]]] = []
        turn = 0
        for entry in requests:
            if entry["type"] == "subagent":
                task = self._tasks.create_task(self.run_agent(entry["requests"], entry["agent_id"]))
                duration_ms = entry.get("duration_ms")
                if duration_ms is not None and entry.get("status") != "async_launched":
                    running.append((entry["t"] + duration_ms / 1000, task))
                continue
            finished = [task for end, task in running if end <= entry["t"]]
            running = [(end, task) for end, task in running if end > entry["t"]]
            if finished:
                await asyncio.gather(*finished)
            if not await self.replayer.pause(entry.get("think_time")):
                return
            hash_ids = entry.get("hash_ids") or []
            chain = self.route(chains, hash_ids)
            messages = self.prompt(chain, hash_ids)
            reply = await self.replayer.send(self, agent_id, turn, chain.index, messages, entry)
            if reply:
                self.add_reply(chain, reply, entry["out"])
            turn += 1

    def route(self, chains: list[Chain], hash_ids: list[int]) -> Chain:
        """The chain whose last prompt shares the longest hash prefix, else a new chain."""
        best, best_shared = None, 0
        for chain in chains:
            shared = common_prefix_len(chain.last, hash_ids)
            if shared > best_shared:
                best, best_shared = chain, shared
        if best is None:
            best = self.new_chain(len(chains))
            chains.append(best)
        return best


class Replayer:
    def __init__(
        self,
        config: ReplayConfig,
        tagger: SessionTagger,
        client: httpx.AsyncClient,
        sink: IO[bytes],
    ) -> None:
        self.config = config
        self.tagger = tagger
        self.client = client
        self.sink = sink
        self.stats: list[Stat] = []
        self.sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._deadline = 0.0

    def _left(self) -> float:
        return self._deadline - asyncio.get_running_loop().time()

    async def run(self, trace_files: list[Path]) -> None:
        self._deadline = asyncio.get_running_loop().time() + self.config.duration
        order = list(trace_files)
        random.Random(self.config.seed).shuffle(order)
        source = itertools.cycle(order)
        lanes = [asyncio.create_task(self._lane(source)) for _ in range(self.config.sessions)]
        _, pending = await asyncio.wait(
            lanes,
            timeout=self.config.duration + self.config.grace,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for lane in pending:
            lane.cancel()
        for result in await asyncio.gather(*lanes, return_exceptions=True):
            if isinstance(result, Exception):
                raise result

    async def _lane(self, source: Iterator[Path]) -> None:
        while self._left() > 0:
            await self.play(await asyncio.to_thread(_load_trace, next(source)))

    async def play(self, trace: dict[str, Any]) -> Session:
        """Replay one trace as one session; starts the run clock if :meth:`run` has not."""
        if not self._deadline:
            self._deadline = asyncio.get_running_loop().time() + self.config.duration
        session = Session(self, trace)
        await session.play()
        return session

    async def pause(self, think_time: float | None) -> bool:
        """Wait out a recorded think time (capped); False once no new request may start."""
        delay = min(think_time or 0.0, self.config.think_cap)
        if delay > 0 and self._left() > 0:
            await self.sleep(min(delay, self._left()))
        return self._left() > 0

    async def send(
        self,
        session: Session,
        agent_id: str | None,
        turn: int,
        chain: int,
        messages: list[dict[str, str]],
        request: dict[str, Any],
    ) -> str | None:
        """Stream one request; write its record and return the reply text, or None on failure."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": max(1, request["out"]),
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.tagger.body(session.id, session.trace_id),
        }
        if self.config.ignore_eos:
            body["ignore_eos"] = True
        headers = {"Content-Type": "application/json"}
        headers.update(self.tagger.headers(session.id, session.trace_id))

        wall, started = time.time(), time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        status: int | None = None
        error: str | None = None
        try:
            async with self.client.stream(
                "POST", self.config.url, content=orjson.dumps(body), headers=headers
            ) as response:
                status = response.status_code
                if status != 200:
                    error = (await response.aread()).decode(errors="replace")[:300]
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = orjson.loads(data)
                        if "error" in chunk:  # vLLM reports a failure inside a 200 stream
                            error = str(chunk["error"])[:300]
                            break
                        usage = chunk.get("usage") or usage
                        for choice in chunk.get("choices") or ():
                            piece = (choice.get("delta") or {}).get("content")
                            if piece:
                                if ttft is None:
                                    ttft = time.perf_counter() - started
                                pieces.append(piece)
        except (httpx.HTTPError, orjson.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started

        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        record = {
            "start": wall,
            "session_id": session.id,
            "trace_id": session.trace_id,
            "agent_id": agent_id,
            "chain": chain,
            "turn": turn,
            "class": self.tagger.class_for(session.trace_id).value,
            "status": status,
            "error": error,
            "trace_in": request["in"],
            "trace_out": request["out"],
            "messages": len(messages),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached,
            "completion_tokens": usage.get("completion_tokens"),
            "ttft_ms": None if ttft is None else round(ttft * 1000, 3),
            "latency_ms": round(latency * 1000, 3),
        }
        self.sink.write(orjson.dumps(record) + b"\n")
        self.sink.flush()
        self.stats.append(
            Stat(error is None, record["ttft_ms"], usage.get("prompt_tokens") or 0, cached or 0)
        )
        return "".join(pieces) if error is None and pieces else None

    def summary(self) -> str:
        ok = [s for s in self.stats if s.ok]
        ttfts = sorted(s.ttft_ms for s in ok if s.ttft_ms is not None)
        prompt = sum(s.prompt_tokens for s in ok)
        cached = sum(s.cached_tokens for s in ok)
        lines = [f"requests: {len(self.stats)} ok, {len(self.stats) - len(ok)} failed"]
        if ttfts:
            p50, p99 = (ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))] for q in (0.5, 0.99))
            lines.append(f"ttft_ms: p50 {p50:.1f}, p99 {p99:.1f}")
        if prompt:
            lines.append(f"cached prompt tokens: {cached}/{prompt} ({cached / prompt:.1%})")
        return "\n".join(lines)


def _load_trace(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


async def replay(
    config: ReplayConfig,
    tagger: SessionTagger,
    traces: Path,
    out: Path,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Replayer:
    """Replay the scaled traces in directory ``traces``; one JSON line per request goes to ``out``."""
    trace_files = sorted(traces.glob("*.json"))
    if not trace_files:
        raise ValueError(f"no trace files (*.json) in {traces}")
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(config.timeout, connect=10.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, transport=transport) as client:
        with out.open("wb") as sink:
            replayer = Replayer(config, tagger, client, sink)
            await replayer.run(trace_files)
    return replayer
