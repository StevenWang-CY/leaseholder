"""The ``lh-bench`` command line: ``scale-traces``, ``replay`` and ``mock-server``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from pathlib import Path

from leaseholder_bench.protocol import SessionTagger, SloClass, load_class_map


def _scale(args: argparse.Namespace) -> int:
    from leaseholder_bench.traces import scale_corpus

    ids = None
    if args.ids is not None:
        ids = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
    summary = scale_corpus(args.src, args.out_dir, args.factor, ids=ids, limit=args.limit)
    print(
        f"wrote {summary.traces} traces ({summary.requests} requests) to {args.out_dir}; "
        f"largest in+out is {summary.peak_tokens} tokens"
    )
    if ids is not None and summary.traces < len(ids):
        print(
            f"warning: {len(ids) - summary.traces} listed trace ids were not found", file=sys.stderr
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    from leaseholder_bench.replay import ReplayConfig, replay

    tagger = SessionTagger.from_env()
    if args.slo_class is not None:
        tagger = dataclasses.replace(tagger, default_class=args.slo_class)
    if args.class_map is not None:
        tagger = dataclasses.replace(tagger, trace_classes=load_class_map(args.class_map))
    if args.no_priority:
        tagger = dataclasses.replace(tagger, send_priority=False)
    config = ReplayConfig(
        url=args.url,
        model=args.model,
        sessions=args.sessions,
        duration=args.duration,
        grace=args.grace,
        think_cap=args.think_cap,
        timeout=args.timeout,
        ignore_eos=not args.no_ignore_eos,
        seed=args.seed,
    )
    replayer = asyncio.run(replay(config, tagger, args.traces, args.out))
    print(replayer.summary())
    return 0 if replayer.stats else 1


def _mock(args: argparse.Namespace) -> int:
    from leaseholder_bench.mock_server import serve

    options = {
        "model": args.model,
        "ttft_ms": args.ttft_ms,
        "itl_ms": args.itl_ms,
        "max_tokens": args.max_tokens,
    }
    if args.log is None:
        serve(args.port, None, **options)
        return 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab") as log:
        serve(args.port, log, **options)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from leaseholder_bench.replay import DEFAULT_MODEL, DEFAULT_URL

    parser = argparse.ArgumentParser(prog="lh-bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scale = commands.add_parser(
        "scale-traces",
        help="shrink Weka traces by a factor of tokens per hash block",
        description="Shrink every request of a Weka trace corpus by the same factor, keeping "
        "hash ids, sessions, subagents and think times. Writes one <trace id>.json per trace.",
    )
    scale.add_argument("src", type=Path, help="Weka traces: a .jsonl file or a directory of *.json")
    scale.add_argument("out_dir", type=Path)
    scale.add_argument(
        "--factor",
        type=int,
        default=32,
        help="tokens-per-block shrink factor: 32 fits a 32k context, 64 a 16k one (default 32)",
    )
    scale.add_argument("--ids", type=Path, help="keep only the trace ids listed in this file")
    scale.add_argument("--limit", type=int, help="stop after this many traces")
    scale.set_defaults(func=_scale)

    replay = commands.add_parser(
        "replay",
        help="replay scaled traces, feeding each live reply into the next turn",
        description="Closed-loop replay of scaled traces with recorded think times and "
        "subagents. Each turn's history carries the model's actual reply. Writes one JSON "
        "line per request.",
    )
    replay.add_argument("traces", type=Path, help="directory of scaled trace *.json files")
    replay.add_argument("--out", type=Path, required=True, help="per-request JSONL file")
    replay.add_argument(
        "--url",
        default=os.environ.get("LH_BENCH_URL", DEFAULT_URL),
        help="chat completions URL (env LH_BENCH_URL; default %(default)s)",
    )
    replay.add_argument(
        "--model",
        default=os.environ.get("LH_BENCH_MODEL", DEFAULT_MODEL),
        help="model name sent in each request (env LH_BENCH_MODEL; default %(default)s)",
    )
    replay.add_argument("--sessions", type=int, default=4, help="concurrent session trees")
    replay.add_argument("--duration", type=float, default=60.0, help="seconds to send requests")
    replay.add_argument(
        "--grace", type=float, default=30.0, help="seconds to let in-flight requests finish"
    )
    replay.add_argument(
        "--think-cap", type=float, default=600.0, help="cap on each think time in seconds"
    )
    replay.add_argument("--timeout", type=float, default=300.0, help="per-request timeout")
    replay.add_argument(
        "--class",
        dest="slo_class",
        type=SloClass,
        choices=list(SloClass),
        help="SLO class of every session (env LH_BENCH_CLASS; default background)",
    )
    replay.add_argument(
        "--class-map", type=Path, help="JSON object of trace id -> class (env LH_BENCH_CLASS_MAP)"
    )
    replay.add_argument(
        "--no-priority",
        action="store_true",
        help="omit the vLLM priority field (needed against a server without priority scheduling)",
    )
    replay.add_argument(
        "--no-ignore-eos", action="store_true", help="let the model stop before the trace's out"
    )
    replay.add_argument("--seed", type=int, default=0, help="seed for the trace order")
    replay.set_defaults(func=_replay)

    mock = commands.add_parser(
        "mock-server",
        help="run a recording OpenAI-compatible mock on 127.0.0.1",
        description="Streaming chat-completions mock bound to 127.0.0.1 that logs every "
        "request's headers and body fields.",
    )
    mock.add_argument("--port", type=int, default=8000)
    mock.add_argument("--log", type=Path, help="append one JSON line per request to this file")
    mock.add_argument("--model", default="mock", help="model id listed by /v1/models")
    mock.add_argument("--ttft-ms", type=float, default=20.0)
    mock.add_argument("--itl-ms", type=float, default=2.0)
    mock.add_argument("--max-tokens", type=int, default=64, help="cap on streamed tokens")
    mock.set_defaults(func=_mock)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
