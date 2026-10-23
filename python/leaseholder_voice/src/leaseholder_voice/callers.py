"""The caller corpus for the measurement bots (``lh-voice render-callers``).

Scripted caller lines for a restaurant's phone line, spoken by six stock Kokoro voices (none of
them the agent's), are rendered once through an OpenAI-compatible ``/audio/speech`` server and
written as 16 kHz mono WAVs, the bots' microphone format. ``manifest.json`` lists each clip's
text, voice, kind (``turn`` or ``barge_in``) and duration. Each clip's end of speech is read
from its waveform, which is what makes voice-to-voice latency exact.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import httpx2 as httpx
import numpy as np

from .audio import resample

TURN_LINES = (
    "Hi, do you have a table for four at seven tonight?",
    "What time do you close on Sundays?",
    "Do you have any vegetarian options on the menu?",
    "Can I book a table for two this Friday evening?",
    "Is there parking near the restaurant?",
    "How long is the wait right now for a table?",
    "Do you take reservations for large groups, like twelve people?",
    "What's the soup of the day?",
    "Are you open for lunch on weekdays?",
    "Can I bring my own bottle of wine?",
    "Is the patio open this time of year?",
    "Do you have gluten free pasta?",
    "I'd like to cancel my reservation for tomorrow night.",
    "Can you move my booking from seven to eight thirty?",
    "Do you have a kids menu?",
    "How far are you from the train station?",
    "Is there a dress code?",
    "What desserts do you have tonight?",
    "Can I order takeout over the phone?",
    "Do you deliver to the north side of town?",
    "Are dogs allowed on the patio?",
    "How spicy is the curry?",
    "Do you have anything without nuts? My son has an allergy.",
    "Is it quiet enough for a business dinner?",
    "Can we get a table by the window?",
    "What are your hours on public holidays?",
    "Do you have a private room for a birthday party?",
    "How much is the tasting menu?",
    "Do you serve brunch on Saturdays?",
    "Is there a corkage fee?",
    "Can I pay by card at the table?",
    "Do you have vegan desserts?",
    "We're running ten minutes late, can you hold our table?",
    "Is the kitchen still open?",
    "What's in the house salad?",
    "Do you have high chairs for babies?",
    "Can I make a reservation under the name Taylor?",
    "Is your fish fresh or frozen?",
    "Do you offer a lunch special?",
    "Thanks, that's all I needed. Have a good night.",
)
BARGE_IN_LINES = (
    "Sorry, wait, make that six people.",
    "Actually, hold on a second.",
    "No, no, I meant tomorrow.",
    "Wait, sorry, can you repeat that?",
    "Hang on, one more question.",
    "Oh, actually, never mind that.",
    "Sorry to interrupt, is that for tonight?",
    "Wait, what about Saturday instead?",
)
VOICES = ("af_bella", "am_michael", "bf_emma", "am_adam", "af_nicole", "bm_george")
"""Kokoro voices for callers; the agent speaks with ``af_heart``."""

RATE = 16_000
PAD_S = 0.2
"""Silence before and after each line, so every clip has a clean onset and end of speech."""


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(samples.tobytes())


def render(tts_url: str, out_dir: Path, *, model: str = "kokoro", uds: str | None = None) -> dict:
    """Render every line to ``out_dir`` and write ``manifest.json``; return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [("turn", text) for text in TURN_LINES] + [("barge_in", t) for t in BARGE_IN_LINES]
    manifest: dict[str, dict] = {}
    pad = np.zeros(int(PAD_S * RATE), dtype=np.int16)
    transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(base_url=tts_url, transport=transport, timeout=60.0) as client:
        for index, (kind, text) in enumerate(lines):
            voice = VOICES[index % len(VOICES)]
            response = client.post(
                "/audio/speech",
                json={"model": model, "input": text, "voice": voice, "response_format": "pcm"},
            )
            response.raise_for_status()
            pcm = np.frombuffer(response.content, dtype=np.int16)  # 24 kHz
            samples = np.concatenate([pad, resample(pcm, 2, 3), pad])
            name = f"{kind}-{index:02d}"
            write_wav(out_dir / f"{name}.wav", samples)
            manifest[name] = {
                "kind": kind,
                "text": text,
                "voice": voice,
                "duration_s": round(len(samples) / RATE, 3),
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
