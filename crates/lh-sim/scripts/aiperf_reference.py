"""Dump AIPerf 0.13.0's agentic_replay structure for a directory of Weka traces.

Runs AIPerf's own Weka loader planning (chain detection, think-time delays, branches, joins, replay
predecessors) and its TrajectorySource (each lane's start point and snapshot), with prompt synthesis
stubbed out, and writes one JSON document:

    {"conversations": [...], "lanes": [...], "recycle": [...]}

`lh-sim aiperf-plan` prints the same document from the simulator's port, so the two can be compared;
`testdata/aiperf-0.13.0-sub16.json` is this script's output for `testdata/weka-x32-sub16`. Run it in
the benchmark package's environment, which pins AIPerf:

    uv run --project python/leaseholder_bench python crates/lh-sim/scripts/aiperf_reference.py \\
        crates/lh-sim/testdata/weka-x32-sub16 crates/lh-sim/testdata/aiperf-0.13.0-sub16.json
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import orjson
from aiperf.common.models import DatasetMetadata
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.dataset.loader import weka_synth_buf as wsb
from aiperf.dataset.loader import weka_trace as wt
from aiperf.dataset.loader._delay_cap import DelayCapTracker
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.trajectory_source import TrajectorySource


class _StubReconstructor:
    """Stands in for prompt synthesis, which the replay structure does not depend on."""

    def __init__(self, **_: object) -> None:
        pass

    def init_turn_0(self, **_: object) -> None:
        pass

    def advance_turn(self, **_: object) -> None:
        pass

    def turn_delta(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            delta_messages=[{"role": "user", "content": "x"}], reset_context=False
        )


wsb.ConversationReconstructor = _StubReconstructor
wsb.compute_asst_block_caps = lambda pairs, block_size: [0] * len(pairs)


def build_loader(cap_seconds: float) -> wt.WekaTraceLoader:
    loader = wt.WekaTraceLoader.__new__(wt.WekaTraceLoader)
    loader.prompt_generator = types.SimpleNamespace(
        _cache={},
        _hash_id_corpus_rng=types.SimpleNamespace(set_trace_id=lambda _scope: None),
        _bpe_stable_terminator_tokens=(),
        tokenizer=None,
    )
    loader._block_size = 64
    loader._user_block_size_override = None
    loader._max_isl = None
    loader._max_osl = None
    loader._fixed_schedule_start_offset = None
    loader._fixed_schedule_end_offset = None
    loader._delay_cap_tracker = DelayCapTracker(cap_seconds=cap_seconds)
    loader._tool_shaped_messages = False
    return loader


def conversation(c: object) -> dict:
    return {
        "id": c.session_id,
        "is_root": c.is_root,
        "scope": c.replay_scope_id,
        "turns": [
            {
                "outer": t.source_outer_idx,
                "inner": t.source_inner_idx,
                "ts_ms": t.timestamp,
                "delay_ms": t.delay,
                "api_ms": t.api_time_ms,
                "max_tokens": t.max_tokens,
                "branches": list(t.branch_ids),
                "joins": [p.branch_id for p in t.prerequisites],
                "preds": [[p.conversation_id, p.turn_index] for p in t.replay_predecessors],
            }
            for t in c.turns
        ],
        "branches": [
            {
                "id": b.branch_id,
                "children": list(b.child_conversation_ids),
                "background": b.is_background,
                "start_ms": b.start_timestamp_ms,
            }
            for b in c.branches
        ],
    }


def lane(t: object) -> dict:
    snap = t.snapshot
    return {
        "trace": t.conversation_id,
        "t_star_ms": snap.t_star_ms if snap else None,
        "states": [
            {
                "conv": s.conversation_id,
                "next": s.next_turn_index,
                "offset_ms": s.next_dispatch_offset_ms,
                "depth": s.agent_depth,
                "waiting": s.waiting_on_children,
                "join": s.join_target_turn_index,
                "branch": s.branch_id,
            }
            for s in (snap.states if snap else ())
        ],
        "boundaries": [
            [b.conversation_id, b.next_turn_index]
            for b in (snap.replay_resume_boundaries if snap else ())
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", type=Path, help="a directory of <trace id>.json")
    ap.add_argument("out", type=Path)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cap", type=float, default=120.0, help="interTurnDelayCapSeconds")
    ap.add_argument("--min-ratio", type=float, default=0.0)
    ap.add_argument("--max-ratio", type=float, default=1.0)
    args = ap.parse_args()

    data = {}
    for path in sorted(args.traces.glob("*.json")):
        trace = WekaTrace.model_validate(orjson.loads(path.read_bytes()))
        data[trace.id] = [trace]

    loader = build_loader(args.cap)
    plans = loader._build_reconstruction_plans(data)
    metric = loader._build_shared_metric_values(
        plans.parent_plans, plans.child_plans, plans.flat_plans
    )
    convs = loader._reconstruct_serial(
        parent_plans=plans.parent_plans,
        child_plans=plans.child_plans,
        data=data,
        dropped_per_trace={},
        ignore_delays=False,
        think_time_only=True,
        cap_seconds=args.cap,
        t_start=0.0,
        model_map_per_trace={},
        trace_idle_timing_by_trace={},
        metric_values_by_trace=metric,
        flat_plans=plans.flat_plans,
    )
    wt._install_replay_dependencies(convs)

    metadata = DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    roots = [c.conversation_id for c in metadata.conversations if c.is_root]
    source = TrajectorySource(
        dataset_metadata=metadata,
        dataset_sampler=SequentialSampler(roots),
        concurrency=args.concurrency,
        random_seed=args.seed,
        start_min_ratio=args.min_ratio,
        start_max_ratio=args.max_ratio,
        cache_bust_enabled=True,
    )
    doc = {
        "conversations": [conversation(c) for c in convs],
        "lanes": [lane(t) for t in source.trajectories],
        # The sampler's order after the initial lanes: the traces recycled lanes start.
        "recycle": [source.next_recycle_conversation_id() for _ in range(32)],
    }
    args.out.write_text(json.dumps(doc))
    print(f"{len(doc['conversations'])} conversations, {len(doc['lanes'])} lanes -> {args.out}")


if __name__ == "__main__":
    main()
