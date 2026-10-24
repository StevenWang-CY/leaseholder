"""The ``lh-voice`` command line: render the caller corpus and run measurement callers.

The voice agent itself is ``lh-voice-worker`` (LiveKit's worker CLI, ``lh-voice-worker start``).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path


def _render(args: argparse.Namespace) -> int:
    from .callers import render

    manifest = render(args.tts_url, args.out, model=args.model, uds=args.uds)
    print(f"wrote {len(manifest)} clips to {args.out}")
    return 0


def _bots(args: argparse.Namespace) -> int:
    from .bots import BotsConfig, run_bots

    config = BotsConfig(
        url=args.url,
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        corpus=args.corpus,
        callers=args.callers,
        turns=args.turns,
        barge_in=args.barge_in,
        seed=args.seed,
        warmup_turns=args.warmup_turns,
        save_audio=args.save_audio,
    )
    summary = asyncio.run(run_bots(config, args.out))
    # Release the LiveKit handles while the native library is still loaded; left to interpreter
    # shutdown, each one prints a failed assertion.
    gc.collect()
    print(json.dumps(summary, indent=1))
    return 0 if not summary["errors"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lh-voice", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser(
        "render-callers",
        help="render the scripted caller lines to 16 kHz WAVs with a TTS server",
        description="Render the caller corpus once through an OpenAI-compatible TTS server: "
        "16 kHz mono WAVs and manifest.json.",
    )
    render.add_argument("out", type=Path, help="corpus directory")
    render.add_argument(
        "--tts-url", default="http://127.0.0.1:8880/v1", help="TTS base URL (default %(default)s)"
    )
    render.add_argument("--uds", help="reach the TTS server through this Unix socket")
    render.add_argument("--model", default="kokoro", help="TTS model (default %(default)s)")
    render.set_defaults(func=_render)

    bots = commands.add_parser(
        "bots",
        help="call the voice agent and measure voice-to-voice latency and barge-in",
        description="Run callers against the voice agent through LiveKit (inside the voice-plane "
        "namespace). LIVEKIT_API_KEY and LIVEKIT_API_SECRET come from the environment.",
    )
    bots.add_argument(
        "--url",
        default=os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:17880"),
        help="LiveKit signalling URL (env LIVEKIT_URL; default %(default)s)",
    )
    bots.add_argument("--corpus", type=Path, required=True, help="directory from render-callers")
    bots.add_argument(
        "--callers", type=int, default=2, help="concurrent calls (default %(default)s)"
    )
    bots.add_argument("--turns", type=int, default=20, help="turns per call (default %(default)s)")
    bots.add_argument(
        "--barge-in",
        type=float,
        default=0.25,
        help="share of turns with an interruption (default %(default)s)",
    )
    bots.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the lines, interruptions and pauses (default %(default)s)",
    )
    bots.add_argument(
        "--warmup-turns",
        type=int,
        default=0,
        help="turns that open each call unmeasured: a line and its reply, no interruption, "
        'marked "warmup" in turns.jsonl and left out of the summary (default %(default)s)',
    )
    bots.add_argument(
        "--save-audio",
        action="store_true",
        help="also write each call's audio and its turns' times, once the calls have ended "
        "(call-<i>.wav, call-<i>.json)",
    )
    bots.add_argument("--out", type=Path, required=True, help="run directory")
    bots.set_defaults(func=_bots)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
