"""One server-side voice turn, timed per stage (``lh-bench voice-turn``).

The cascade a LiveKit voice agent runs, against OpenAI-compatible STT, LLM and TTS servers:

1. ``stt_ms``: the utterance WAV to ``/v1/audio/transcriptions``;
2. ``llm_ttft_ms`` and ``llm_first_sentence_ms``: a streamed chat completion of the transcript;
3. ``tts_first_audio_ms``: as soon as the first sentence is complete, it goes to
   ``/v1/audio/speech`` while the rest of the reply is still streaming.

``turn_ms`` runs from sending the audio to the first byte of TTS audio: the server-side share of
voice-to-voice latency, without VAD, end-of-turn detection, network or playout. With
``--background`` the same turns run while agent load keeps the LLM engine busy.
"""

from __future__ import annotations

import asyncio
import platform
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson

from leaseholder_bench.load import BackgroundLoad, LoadSpec, summarize
from leaseholder_bench.protocol import CLASS_HEADER, ENGINE_PRIORITY, SloClass

SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?:\s|$)")
MIN_SENTENCE_CHARS = 20
"""LiveKit Agents' sentence tokenizer default (``min_sentence_len``): shorter sentences are held
back and spoken together with the next one."""
STAGES = (
    "stt_ms",
    "llm_ttft_ms",
    "llm_first_sentence_ms",
    "tts_first_audio_ms",
    "turn_ms",
    "llm_total_ms",
)
DEFAULT_SYSTEM = (
    "You are Ava, the voice assistant of a small restaurant, speaking on the phone. Answer in one "
    "or two short, natural sentences, with no lists or markdown."
)


def first_sentence(text: str) -> str | None:
    """The text up to the first sentence boundary that leaves at least ``MIN_SENTENCE_CHARS``
    characters to speak, or None while the reply is shorter."""
    for m in SENTENCE_END.finditer(text):
        if len(head := text[: m.end()].strip()) >= MIN_SENTENCE_CHARS:
            return head
    return None


@dataclass(frozen=True)
class VoiceTurnConfig:
    stt_url: str
    """STT base URL, e.g. ``http://127.0.0.1:18100/v1``."""
    llm_url: str
    tts_url: str
    stt_model: str
    llm_model: str
    tts_model: str
    voice: str
    audio: Sequence[Path]
    reps: int = 20
    warm: int = 2
    concurrency: int = 1
    max_tokens: int = 60
    system: str = DEFAULT_SYSTEM
    slo_class: SloClass = SloClass.REALTIME
    send_priority: bool = True
    background: LoadSpec | None = None
    decode_tokens: int = 1024
    gap_s: float = 0.5
    timeout: float = 120.0


class VoiceTurn:
    def __init__(self, config: VoiceTurnConfig, stt: httpx.AsyncClient, llm: httpx.AsyncClient,
                 tts: httpx.AsyncClient) -> None:  # fmt: skip
        self.config = config
        self.stt, self.llm, self.tts = stt, llm, tts

    async def transcribe(self, wav: bytes) -> str:
        response = await self.stt.post(
            "/audio/transcriptions",
            files={"file": ("turn.wav", wav, "audio/wav")},
            data={"model": self.config.stt_model, "language": "en", "response_format": "json"},
        )
        response.raise_for_status()
        return response.json()["text"].strip()

    async def first_audio(self, text: str) -> float:
        """Milliseconds from the TTS request to its first audio byte."""
        payload = {
            "model": self.config.tts_model,
            "input": text,
            "voice": self.config.voice,
            "response_format": "pcm",
            "stream": True,
        }
        started = time.perf_counter()
        async with self.tts.stream("POST", "/audio/speech", json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_raw():
                if chunk:
                    return (time.perf_counter() - started) * 1000
        raise RuntimeError("the TTS server returned no audio")

    def chat_request(self, transcript: str) -> tuple[dict[str, str], dict[str, Any]]:
        config = self.config
        body: dict[str, Any] = {
            "model": config.llm_model,
            "stream": True,
            "max_tokens": config.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": config.system},
                {"role": "user", "content": transcript},
            ],
        }
        if config.send_priority:
            body["priority"] = ENGINE_PRIORITY[config.slo_class]
        return {CLASS_HEADER: config.slo_class.value}, body

    async def run(self, wav: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        transcript = await self.transcribe(wav)
        t_stt = time.perf_counter()
        headers, body = self.chat_request(transcript)
        reply = ""
        sentence: str | None = None
        t_first = t_sentence = None
        tts: asyncio.Task[float] | None = None
        async with self.llm.stream("POST", "/chat/completions", headers=headers, json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: {"):
                    continue
                choices = orjson.loads(line[6:]).get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta and t_first is None:
                    t_first = time.perf_counter()
                reply += delta
                if sentence is None and (sentence := first_sentence(reply)):
                    t_sentence = time.perf_counter()
                    tts = asyncio.create_task(self.first_audio(sentence))
        t_done = time.perf_counter()
        if tts is None:  # no sentence boundary within max_tokens: speak the whole reply
            sentence, t_sentence = reply.strip() or "Sorry?", t_done
            tts = asyncio.create_task(self.first_audio(sentence))
        tts_ms = await tts
        t_first = t_first or t_done

        def ms(a: float, b: float) -> float:
            return round((a - b) * 1000, 1)

        return {
            "transcript": transcript,
            "first_sentence": sentence,
            "stt_ms": ms(t_stt, t0),
            "llm_ttft_ms": ms(t_first, t_stt),
            "llm_first_sentence_ms": ms(t_sentence, t_stt),
            "llm_total_ms": ms(t_done, t_stt),
            "tts_first_audio_ms": round(tts_ms, 1),
            "turn_ms": round(ms(t_sentence, t0) + tts_ms, 1),
        }


async def run_voice_turns(config: VoiceTurnConfig, out_dir: Path) -> dict[str, Any]:
    """Run ``warm + reps`` turns per caller (``concurrency`` callers at once) and write the run
    directory: ``run.json``, ``turns.jsonl`` (measured turns only) and ``summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [path.read_bytes() for path in config.audio]
    started = time.time()
    timeout = httpx.Timeout(config.timeout)
    rows: list[dict[str, Any]] = []
    load_summary = None
    async with (
        httpx.AsyncClient(base_url=config.stt_url, timeout=timeout) as stt,
        httpx.AsyncClient(base_url=config.llm_url, timeout=timeout) as llm,
        httpx.AsyncClient(base_url=config.tts_url, timeout=timeout) as tts,
    ):
        turn = VoiceTurn(config, stt, llm, tts)
        with (out_dir / "turns.jsonl").open("wb") as log:

            async def caller(index: int) -> None:
                for i in range(config.warm + config.reps):
                    row = await turn.run(clips[(index + i) % len(clips)])
                    if i >= config.warm:
                        row = {"caller": index, "turn": i - config.warm} | row
                        rows.append(row)
                        log.write(orjson.dumps(row) + b"\n")
                        log.flush()
                    await asyncio.sleep(config.gap_s)

            async def callers() -> None:
                await asyncio.gather(*(caller(i) for i in range(config.concurrency)))

            if config.background is None:
                await callers()
            else:
                async with BackgroundLoad(
                    llm,
                    "/completions",
                    config.llm_model,
                    config.background,
                    priority=ENGINE_PRIORITY[SloClass.BACKGROUND] if config.send_priority else None,
                    decode_tokens=config.decode_tokens,
                ) as load:
                    await asyncio.sleep(2.0)  # let the load reach steady state
                    await callers()
                load_summary = load.summary()
    summary = {stage: summarize(row[stage] for row in rows) for stage in STAGES}
    summary["background"] = load_summary
    run = {
        "kind": "voice-turn",
        "config": asdict(config)
        | {"audio": [str(p) for p in config.audio], "background": str(config.background or "none")},
        "host": platform.node(),
        "started": started,
        "finished": time.time(),
    }
    (out_dir / "run.json").write_bytes(orjson.dumps(run, option=orjson.OPT_INDENT_2))
    (out_dir / "summary.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return summary


def render(summary: dict[str, Any]) -> str:
    rows = ["stage                    n     p50     p90     p99"]
    for stage in STAGES:
        cell = summary[stage]
        if cell.get("n"):
            rows.append(
                f"{stage:<22} {cell['n']:>3} {cell['p50']:>7.1f} {cell['p90']:>7.1f} {cell['p99']:>7.1f}"
            )
    if summary.get("background"):
        rows.append(f"background: {summary['background']}")
    return "\n".join(rows)
