# lh-core

The deterministic policy core the simulator and the gateway share. Pure code: no I/O and no clock. Callers pass token counts and events, and get durations and decisions back.

| Module | What it holds | Used by |
|---|---|---|
| `cost` | What serving a prompt costs from each tier, the E0 model fitted on the frozen box ([BENCHMARKS.md](../../docs/BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)), and decode time | `lh-sim` |
| `step` | What one engine step costs from what it carries (step tokens, attention pairs and context, a prefill beside decodes), fitted by `lh-sim`'s `scripts/fit_step_cost.py` on three operating points of the frozen box | `lh-sim`'s contention model; `admission`'s prefill price |
| `admission` | Whether a request goes upstream now, waits or is refused: the E2 baselines (static cap, token reservation, flow control, preempt-resubmit) and Leaseholder's own policy (`leaseholder`, below), driven by arrivals, first bytes and departures (and the realtime sessions' turns), and the token estimate they share | `lh-gateway`'s admission point |
| `geometry` | KV bytes per token, the engine block and tier capacities in blocks | both |

**Policy `leaseholder`** (E2 arm 9) is a variant of `admission::Policy` beside the baselines, pure and without a clock like them. Realtime requests go upstream at once, never queued and never refused ([ADR-0003](../../docs/adr/0003-realtime-is-never-shed.md)). Background requests yield to the realtime sessions' turns, which the caller tracks from lifecycle events and reports as two counts (`Turns`: sessions whose user is speaking, and sessions in their critical window, from the end of the user's turn to the reply's first audio). Each mechanism can be switched off for the ablations:
- *headroom*: while sessions are armed (from speech onset, or only in the critical window), a background request goes upstream only if the estimated tokens of the background requests in flight and its own leave `headroom_tokens` of the pool free for each armed session;
- *throttle*: the background requests admitted in a critical window share a budget of predicted prefill time, each priced by the `PrefillCost` the admission point is made with (`Admission::with_prefill_cost`): a `step` model of the engine and GPU, and the engine's step budget (`--max-num-batched-tokens`), in whose steps a lone prefill of the request's new tokens runs after its session's last prompt. On the frozen box's fit, 512 tokens after a 30,000-token context take 269 ms, after none 38 ms. The policy carries no fit of its own: a different engine or GPU needs its own, from `lh-sim`'s `scripts/fit_step_cost.py`;
- *degrade*: a realtime request that arrives while the background requests in flight leave less than one turn's headroom free has its reply capped;
- *aging*: a request the caller has let wait long enough is admitted regardless (`Admission::age`), and is `oversize` if its estimate alone is over what the headroom leaves.

`Admission::held_back` gives the estimated background tokens the headroom holds back at the moment, which the gateway integrates over time to price false alarms. Leaseholder's leases, prefetch and the ledger's attribution are to follow ([ARCHITECTURE.md](../../docs/ARCHITECTURE.md#41-crates)).

`cargo run --release -p lh-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...` measures the admission token estimate's error against the prompt tokens an engine reported in an AIPerf replay.
