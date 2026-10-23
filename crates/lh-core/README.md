# lh-core

The deterministic policy core the simulator and the gateway share. Pure code: no I/O and no clock. Callers pass token counts and events, and get durations and decisions back.

| Module | What it holds | Used by |
|---|---|---|
| `cost` | What serving a prompt costs from each tier, the E0 model fitted on the frozen box ([BENCHMARKS.md](../../docs/BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)), and decode time | `lh-sim` |
| `step` | What one engine step costs from what it carries (step tokens, attention pairs and context, a prefill beside decodes), fitted by `lh-sim`'s `scripts/fit_step_cost.py` on three operating points of the frozen box | `lh-sim`'s contention model |
| `admission` | Whether a request goes upstream now, waits or is refused: the E2 baselines (static cap, token reservation, flow control, preempt-resubmit), driven by arrivals, first bytes and departures, and the token estimate they share | `lh-gateway`'s admission point |
| `geometry` | KV bytes per token, the engine block and tier capacities in blocks | both |

Leaseholder's own policies (leases, turn-imminent headroom, prefetch, the ledger's attribution) and the `Policy` trait the gateway is to run them through arrive from W3 ([ARCHITECTURE.md](../../docs/ARCHITECTURE.md#41-crates)).

`cargo run --release -p lh-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...` measures the admission token estimate's error against the prompt tokens an engine reported in an AIPerf replay.
