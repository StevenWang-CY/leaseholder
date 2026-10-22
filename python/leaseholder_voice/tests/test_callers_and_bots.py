from __future__ import annotations

import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np

from leaseholder_voice.audio import Clip
from leaseholder_voice.bots import Corpus, summarize
from leaseholder_voice.callers import (
    BARGE_IN_LINES,
    TURN_LINES,
    VOICES,
    render,
    resample_24k_to_16k,
)


def tone(hz: float, seconds: float = 1.0, rate: int = 24_000) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * 10_000).astype(np.int16)


def test_resampling_keeps_speech_band_and_rejects_aliases():
    speech = resample_24k_to_16k(tone(440))
    assert len(speech) == 16_000
    assert abs(int(np.abs(speech[2000:14000]).max()) - 10_000) < 100
    alias = resample_24k_to_16k(tone(11_000))  # above the new 8 kHz Nyquist
    assert int(np.abs(alias[2000:14000]).max()) < 30


class FakeTTS(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeTTS.requests.append(body)
        pcm = tone(300, 0.5).tobytes()
        self.send_response(200)
        self.send_header("Content-Length", str(len(pcm)))
        self.end_headers()
        self.wfile.write(pcm)

    def log_message(self, *args: object) -> None:
        pass


def test_render_writes_16k_clips_and_a_manifest(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = render(f"http://127.0.0.1:{server.server_port}/v1", tmp_path)
    finally:
        server.shutdown()
    assert len(manifest) == len(TURN_LINES) + len(BARGE_IN_LINES)
    assert {r["voice"] for r in FakeTTS.requests} == set(VOICES)
    assert all(r["response_format"] == "pcm" and r["model"] == "kokoro" for r in FakeTTS.requests)
    first = manifest["turn-00"]
    assert first["kind"] == "turn" and first["text"] == TURN_LINES[0]
    assert first["duration_s"] == 0.9  # 0.2 s of padding on each side of 0.5 s of speech
    with wave.open(str(tmp_path / "turn-00.wav")) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)
    clip = Clip.load("turn-00", str(tmp_path / "turn-00.wav"))
    assert (clip.onset_frame, clip.offset_frame) == (20, 70)  # speech between the paddings
    corpus = Corpus(tmp_path)
    assert len(corpus.turns) == len(TURN_LINES) and len(corpus.barge_ins) == len(BARGE_IN_LINES)


def test_summary_pools_v2v_samples_and_counts_missed_barge_ins():
    rows = [
        {"caller": 0, "turn": 0, "v2v_ms": 800.0, "barge_in": False},
        {"caller": 0, "turn": 1, "v2v_ms": 900.0, "barge_in": True, "interrupted_speech": True,
         "barge_in_to_silence_ms": 620.0, "v2v_after_barge_in_ms": 1000.0},
        {"caller": 1, "turn": 0, "v2v_ms": 1100.0, "barge_in": True, "interrupted_speech": False,
         "v2v_after_barge_in_ms": 950.0},
        {"caller": 1, "turn": 1, "error": "the agent did not start speaking"},
    ]  # fmt: skip
    summary = summarize(rows, [0.004, 0.012])
    assert summary["turns"] == 4 and summary["errors"] == 1
    assert summary["v2v_ms"]["n"] == 5 and summary["v2v_ms"]["p50"] == 950.0
    assert summary["barge_in_to_silence_ms"] == {"n": 1, "p50": 620.0}
    assert summary["barge_ins"] == 2 and summary["barge_ins_after_the_reply"] == 1
    assert summary["caller_max_lateness_ms"] == 12.0
