# Experiments

Four questions decide whether Leaseholder earns its place in front of stock vLLM. Each experiment compares against tuned stock vLLM on the same frozen box (same model, precision, memory split and SLO), and every plot is regenerated from scripted runs.

| # | Experiment | Question |
|---|---|---|
| E2 | **Voice under agent load** (the headline) | Can one GPU hold a voice SLO while most of its capacity goes to background agents? |
| E1 | **Parking** | How much do lease-aware parking and prefetch save over LRU plus CPU offload, and how far is each policy from optimal? |
| E3 | **Ledger and contracts** | Can every recomputed token be attributed to a cause, and can Anthropic `cache_control` TTLs be honored? |
| — | **Reliability drills** | Node loss, an engine killed mid-stream, CPU-tier exhaustion, control-plane restart: each produces a recovery time |

E0 comes first. It re-measures the tier cost model on the frozen box and checks that `lh-sim` reproduces 2–3 GPU operating points within 10–15% on resume TTFT and recompute.

## E2: voice under agent load

- **Workload.**
  - 2–4 voice bots through LiveKit: synthesized callers with known end-of-speech times, who barge in on 0% or 25% of turns.
  - 0–24 background agent sessions, replayed closed-loop from the Claude Code traces.
- **Placements.** Audio is co-located on the GPU (the headline). A control arm moves audio to a CPU host to isolate LLM scheduling.

| # | Arm | Represents |
|---|---|---|
| 1 | Voice alone | The floor |
| 2 | Stock defaults (FCFS, 2,048-token chunks) | Out of the box |
| 3 | Stock + priority scheduling, voice high | The first thing an operator tries |
| 4 | Arm 3 + `--long-prefill-token-threshold` ∈ {256, 512, 1024} × `--max-num-batched-tokens` ∈ {512, 1024} | Tuned engine |
| 5 | Arm 4 + `max_num_queued_tokens` | vLLM's own TTFT knob |
| 6 | Arm 5 + a static background concurrency cap | The simple operator answer |
| 7 | llm-d turn priority, emulated (`turnNumber + 0.05 × headWaitSeconds`, shed with 429) | The current industry default |
| 8 | Leaseholder: turn-imminent headroom, speech-onset prefetch, class-aware admission (never shed) | This project |
| 9 | Partitioned GPUs (one card per workload) | The upper bound |

**Fixed across arms.** VAD, turn detection and interruption handling. The interrupt-on-speech setting closes the LLM stream 0.58 s after barge-in onset; the stock pause-then-interrupt keeps it open for about 2.2 s, a cost reported as wasted tokens.

**Metrics.**
- Voice V2V p50/p95/p99, from waveforms.
- Barge-in to silence; audio gaps of 100 ms or more.
- LLM time to first sentence, and the stage waterfall.
- Abort latency and wasted tokens.
- Harvested background goodput.

**Plot.** Harvested background goodput on x, voice V2V p95 on y, one frontier per arm.

**Discriminating scenario.** Background agents with large contexts fill the KV pool while voice turns keep arriving. Tuned stock vLLM should still spike there, because waiting high-priority work cannot preempt running work.

**Volume.** 500–1,000 turns per arm, three repeats.

## E1: parking

- **Workload.**
  - Block-scaled Claude Code traces, replayed closed-loop with recorded think time; gaps are capped at 10–30 minutes.
  - Session count is swept to reach KV pressure of 1–8×, with the CPU tier at 2×, 4× and 8× the GPU pool.
- **Order.** Simulator first: `lh-sim` runs every arm, including Belady-OPT. The GPU runs the arms the simulator says matter.
- **Arms.**
  - GPU-only LRU; LRU + CPU tier; ARC + CPU tier;
  - a fixed-TTL hold;
  - LRU + CPU with prefetch timed from session history; the same with prefetch timed by an oracle;
  - lease-aware CPU eviction;
  - active-session admission (against stock `max_num_active_seqs`);
  - Belady-OPT (simulated only).
- **Metrics.** Resume TTFT p50/p90/p99 by source tier; avoidable recompute; PCIe bytes; sessions within SLO; the ledger's cause breakdown.
- **Plot.** A regime map of KV pressure against the pause/residency ratio. Each cell shows the gain over LRU + CPU and the remaining gap to OPT.
- **Gate (G-sim).** If OPT's headroom over LRU + CPU is under about 10% in the realistic regime, E1 is published as a regime map with a negative result, and the `CachePolicy` plugin stays off.

## E3: ledger and contracts

| Part | What it does |
|---|---|
| Injected-cause suite | Triggers each cause deliberately and requires at least 95% correct attribution, with the `unknown` share reported. The causes: a timestamp that breaks the prefix, a forced eviction, a lease expiry, a wrong-node route, a preemption, an engine restart |
| Recompute anatomy | The share of recomputed tokens by cause on E1 and E2 replays: cold first turns, prefix breaks (for example chat templates re-rendering tool-call JSON), subagent forks, eviction, expiry, preemption |
| Contracts | Anthropic `cache_control` TTL → lease floor; usage fields; the violation SLI. Measured under a deliberately small CPU tier, where honoring can fail |
| TTL capacity model | Little's law: resident tokens ≈ rate of distinct new context × TTL. It predicts the RAM needed for 5-minute and 1-hour terms, then checks the prediction |
| Real harness | Claude Code through agentic-api and Leaseholder, with ledger output |

## Reliability drills

| Drill | Measured outcome |
|---|---|
| Node loss (a Gray window) | Sessions moved, re-prefill time, SLO impact |
| vLLM killed mid-stream | A typed error to the client, epoch revocation, recovery time |
| CPU tier exhausted | Grants degrade to `floor = NONE`, violations counted, no OOM |
| Control-plane restart under load | Lease table rebuilt from KV-event replay, with no stale-holder routing |

## How runs are made

- **One frozen box** for every experiment ([deploy/phastform](../deploy/phastform)). Variants are engine drop-ins (see [RUNBOOK.md](../ops/RUNBOOK.md)). Runs are batched by engine configuration, because a restart costs minutes.
- **Repeats.** Three repeats; warm-up trimmed; the prefix cache reset between load levels with `POST /reset_prefix_cache?reset_external=true`, once in-flight stores finish.
- **Load generator health.** Every run checks that the load generator is not saturated. Admission rejections are reported separately. Agents are replayed closed-loop; open-loop arrivals come from multiple processes, to avoid coordinated omission.
- **Run logs.** Each run directory holds the configuration, host and load, start and end times, raw samples and the plot inputs, and is archived on mustard.
