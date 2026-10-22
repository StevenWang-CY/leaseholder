# Experiments

Four questions decide whether Leaseholder earns its place in front of stock vLLM. Each experiment compares against tuned stock vLLM, and against what engines and routers already ship, on the same frozen box (same model, precision, memory split and SLO), and every plot is regenerated from scripted runs.

| # | Experiment | Question |
|---|---|---|
| E2 | **Voice under agent load** (the headline) | How much background goodput can one GPU harvest while its realtime tier holds a voice SLO, and when does lifecycle timing beat static reservation and engine priority? |
| E1 | **Parking** | How much of the gap between LRU plus CPU offload and a clairvoyant reference does each level of lifecycle information close, on resume latency? |
| E3 | **Ledger and contracts** | Can every recomputed token be attributed to a cause, and can Anthropic `cache_control` TTLs be honored? |
| — | **Reliability drills** | Node loss, an engine killed mid-stream, CPU-tier exhaustion, control-plane restart: each produces a recovery time |

E0 comes first. It re-measures the tier cost model on the frozen box and checks that `lh-sim` reproduces 2–3 GPU operating points within 10–15% on resume TTFT and recompute.
- **Cost model: done in W1.** The table and fits are in [BENCHMARKS.md](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0), and `lh-core` carries them.
- **Simulator: in progress.** Driven by the measured request times, its engine model matches the W0 replay's GPU and CPU shares ([`lh-sim` README](../crates/lh-sim/README.md#validation-against-w0)). Its own replay still runs at the idle box's speed, so the TTFT check waits for a contention model (W2).

## E2: voice under agent load

- **Workload.**
  - 2–4 voice callers through LiveKit (`lh-voice bots`): scripted lines with known end-of-speech times, barging in on 0% or 25% of turns; then a calls-per-GPU sweep to the knee.
  - 0–24 background agent sessions, replayed closed-loop from the Claude Code traces.
- **Placements.** Audio co-located on the 16 GB card (the edge pattern, as in NVIDIA's single-GPU voice-agent profile).

| # | Arm | Represents |
|---|---|---|
| 1 | Voice alone | The floor |
| 2 | Stock defaults (FCFS, 2,048-token chunks) | Out of the box |
| 3 | Stock + priority scheduling, voice high | The first thing an operator tries |
| 4 | Arm 3 + `--long-prefill-token-threshold` ∈ {256, 512, 1024} × `--max-num-batched-tokens` ∈ {512, 1024}; `prefill_schedule_interval` ∈ {2, 4} on a build with #54627 (v0.32.0 applies it only to the data-parallel MoE engine) | Tuned engine |
| 5 | Arm 4 + `watermark` ∈ {0.05, 0.1, 0.2} and `max_num_queued_tokens` | vLLM's own headroom and TTFT knobs |
| 6 | Arm 5 + an equal-size static reservation; separately, a static background-concurrency cap | The simple operator answers |
| 7 | vLLM with #54530 applied; SGLang priority preemption on the same model | What the engines ship or are shipping |
| 8 | llm-d flow control (priority bands, holdback, SLO-deadline ordering, in-flight eviction), emulated and labeled; SMG's priority scheduler, as shipped; abort-and-resubmit of background work on a voice turn | What routers ship |
| 9 | Leaseholder: headroom timed by lifecycle events and freed by parking, the critical-window prefill throttle, class-aware admission (never shed), the engine "make room" patch; each also alone | This project |
| 10 | MPS with an active-thread split; audio off the GPU | Equal-hardware partitioning |

**Factors.** LiveKit preemptive generation on and off (on by default in 1.8.8); the headroom trigger (speech onset, end of turn, the preemptive-generation transcript); MPS vs time-slicing. VAD, turn detection and interruption handling are otherwise fixed. The stock pause-then-interrupt handling keeps the LLM stream open through a barge-in, where interrupting on speech cuts the reply after 0.5 s of speech ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)); the difference is reported as wasted tokens.

**Metrics.**
- Voice V2V p50, p95 and p99 from waveforms, server-side and end-to-end (including endpointing).
- Barge-in to silence; audio gaps of 100 ms or more.
- LLM time to first sentence and the stage waterfall, separating compute interference (step time) from KV admission blocking.
- Abort latency and wasted tokens, including discarded preemptive generations.
- Harvested background goodput, counting shed and aborted work as lost; goodput per GPU-hour and $/M tokens at the SLO.
- The false-alarm cost of headroom held without a following turn (block-seconds).

**Plot.** Harvested background goodput on x, voice V2V p95 on y, one frontier per arm, with the regimes where tuned stock or engine preemption is already enough marked.

**Discriminating scenario.** Background agents with large contexts fill the KV pool while voice turns keep arriving. Tuned stock vLLM should still spike there, because a waiting high-priority request cannot preempt running work.

**Volume.** 500–1,000 turns per arm; at least five repeats in the discriminating scenario, reported per run.

**Other hardware.** E2 runs on the frozen box, and a second RTX 5060 Ti in booked windows adds dedicate vs pool. What transfers is stated in hardware-free terms: each arm's behavior against KV pressure, π and ρ, and the engine logic (a waiting request cannot preempt running work under KV pressure on any GPU). `lh-sim` places L4 and H100 configurations on the same map from cost fits built from published specifications, labeled as modeled.

## E1: parking

Parking KV during a tool call and prefetching it before the return is published (see [PRIOR-ART.md](PRIOR-ART.md)), so E1 is a measurement: how much of the gap between LRU plus CPU offload and a clairvoyant reference each level of lifecycle information closes (none, session history, an oracle), per regime, with resume latency as the objective.

- **Workload.**
  - Block-scaled Claude Code traces, replayed as AIPerf's `agentic_replay` sends them, with recorded think time; gaps are capped at 10–30 minutes.
  - Session count is swept to reach KV pressure of 1–8×, with the CPU tier at 2×, 4× and 8× the GPU pool.
- **Order.** Simulator first: `lh-sim` runs every arm and the clairvoyant reference. The GPU runs the arms the simulator says matter.
- **Arms.**
  - GPU-only LRU; LRU + CPU tier; S3-FIFO + CPU tier (these three are in `lh-sim` v0);
  - a fixed-TTL hold (Continuum);
  - LRU + CPU with prefetch timed from session history; the same with prefetch timed by an oracle;
  - predictive upload (Tokencake), idleness ranking (MORI) and min-waste preserve, discard or swap (InferCept);
  - lease-aware CPU eviction;
  - active-session admission (against stock `max_num_active_seqs`);
  - the clairvoyant reference (simulated only; in `lh-sim` v0): Belady's MIN in each tier, replayed on the LRU + CPU timeline. MIN is optimal only for one tier's hit count over unit-size blocks; with prefix hits across two tiers it is a reference, not a bound.
- **Metrics.** Resume TTFT p50, p90 and p99 by source tier, and stalls; avoidable recompute; PCIe bytes; sessions within SLO; the ledger's cause breakdown.
- **Plot.** A regime map over KV pressure (live working set over the pool), π (median pause over restore time) and ρ (restore over recompute time at the median prompt). Each cell shows the share of the gap to the clairvoyant reference that each information level closes.
- **Gate (G-sim).** If the clairvoyant reference's headroom over LRU + CPU is under about 10% in the realistic regime, E1 is published as a regime map with a negative result, and the `CachePolicy` plugin stays off.

## E3: ledger

| Part | What it does |
|---|---|
| Injected-cause suite | Triggers the key causes deliberately and requires at least 95% correct attribution, with the `unknown` share reported: a timestamp that breaks the prefix, a forced eviction, a preemption, an engine restart |
| Recompute anatomy | The share of recomputed tokens by cause on E1 and E2 replays: cold first turns, prefix breaks (chat templates re-rendering tool-call JSON, context edits), subagent forks, eviction, preemption |
| Cache contracts as an SLI | Anthropic `cache_control` TTLs (renewed on each use) become lease floors. The honor rate inside the TTL ([SLO.md](SLO.md#indicators-and-targets)) is measured under a deliberately small CPU tier where honoring can fail, with usage in Anthropic's fields and a cache read reported only for a real hit |
| Real harness | Claude Code through agentic-api and Leaseholder, with ledger output |

## Reliability drills and the two-node window

| Drill | Measured outcome |
|---|---|
| vLLM killed mid-stream | A typed error to the client, epoch revocation, recovery time |
| Control-plane restart under load | Lease table rebuilt from KV-event replay, with no stale-holder routing |
| Node loss (in the Gray window) | Sessions moved, re-prefill time, SLO impact |

CPU-tier exhaustion is a test: grants degrade to `floor = NONE`, violations are counted, nothing runs out of memory.

**Dedicate vs pool.** On two GPUs of equal hardware: one voice node plus one agent node; two mixed nodes under tuned stock; two mixed nodes under Leaseholder with sticky sessions and lease-aware placement of new calls. KV never moves between nodes, so the 1 GbE link does not matter.

## How runs are made

- **One frozen box** for every experiment ([`deploy/phastform`](../deploy/phastform)). Variants are engine drop-ins ([RUNBOOK.md](../ops/RUNBOOK.md#engine-variants)). Runs are batched by engine configuration, because a restart costs minutes.
- **Repeats.** Three repeats, at least five in E2's discriminating scenario, reported per run; warm-up trimmed; the prefix cache reset between load levels with `POST /reset_prefix_cache?reset_external=true`, once in-flight stores finish.
- **Load generator health.** Every run checks that the load generator is not saturated. Admission rejections are reported separately. Agents are replayed closed-loop; open-loop arrivals come from multiple processes, to avoid coordinated omission.
- **Run logs.** Each run directory holds the configuration, host and load, start and end times, raw samples and the plot inputs, and is archived on mustard.
