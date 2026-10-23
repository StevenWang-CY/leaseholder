# lh-sim

`lh-sim` replays the SemiAnalysis Claude Code traces as AIPerf 0.13.0's `agentic_replay` sends them, through a model of vLLM 0.32's GPU prefix cache and OffloadingConnector CPU tier, served by a model of vLLM's step loop on the frozen box (`lh-core`'s step-time model; E0's idle-box cost model is its zero-load limit). For each cache policy it reports where prompt tokens came from (the GPU pool, the CPU tier, or computation), what that costs in time to first token and PCIe traffic, and where the run sits in the regime coordinates E1 is plotted in. A clairvoyant reference gives the gap each policy leaves, and the E1 arms measure how much of it each level of information about a session's return closes.

## The replay

- **Traces.** A directory of `<trace id>.json` as `lh-bench scale-traces` writes it, or the original JSONL. A prompt is `len(hash_ids) × block_size` tokens; the raw `in` field overcounts.
- **AIPerf's structure, ported.** The Weka loader's chain detection splits each trace into the main agent, flattened agent chains and subagent chains, with their think-time delays (`useThinkTimeOnly`, capped at `interTurnDelayCapSeconds`), SPAWN branches and joins, and the cross-stream replay barriers. `TrajectorySource` gives each lane its start instant `t*` (SHA-256 of seed, trace and lane, then numpy's PCG64) and the streams live at `t*`. `scripts/aiperf_reference.py` dumps AIPerf's own plan with prompt synthesis stubbed out, and `lh-sim aiperf-plan` dumps the port's: they agree field for field on the full 32× corpus (9,843 conversations, 98,827 turns) for 393 lanes with seed 11 and for 450 lanes with seed 7, start ratios 0.25–0.75 and a 30 s cap. The test subset's plan is pinned in `testdata/aiperf-0.13.0-sub16.json`.
- **Dispatch.** `concurrency` lanes take traces from AIPerf's sequential sampler. Every stream live at `t*` first sends its last earlier request with `max_tokens = 1`, all at once; profiling starts when the last of these returns. Each lane then sends its earliest live turn at once and the others at their recorded offsets. A turn's successor goes its delay after the turn returns; a root turn spawns its children when it returns (when it is sent, if the recorded spawn overlapped the request), each at its recorded offset from the spawn; a joining turn waits for its children; a turn whose replay predecessors have not completed is held. A lane whose session tree has drained starts the next trace from its first turn, under a fresh KV namespace (AIPerf's cache-bust marker). No request is sent after the duration, requests still running at the end of the grace period are dropped, and when nothing is in flight the pending think-time timers are pulled forward so the engine idles at most `systemIdleGapCapSeconds`.
- **Pauses.** A turn's delay is the trace's think time (`--pauses trace`, the default), or, with `--pauses tracelab`, a TraceLab tool wait drawn in its place and capped the same way: a tool by its share of TraceLab's tool rounds, then a wait from that tool's percentiles (`data/tracelab-pauses.tsv`, below). Every arm draws the same pause for the same turn.
- **Service.** Under contention (`--service contention`, the default) a request reaches vLLM's step loop after its host time and shares every step with the requests beside it (the contention model below). `--service idle` is v0: a request's time to first token is E0's cost for its hit pattern after any wait for free blocks, then it decodes at a constant inter-token latency, with no queueing behind other requests; it reproduces v0's outputs exactly.

## The engine model

- **Blocks.** Prompts are cut into 16-token blocks. A block's identity is its exact prefix: the trace's hash-id trie node its last token falls in, its offset inside that hash block, the session play and, by default, the model. Two prompts share a block iff they share every token up to its end, which is vLLM's chained block hashing without collisions. `--shared-models` drops the model from the identity, as an AIPerf replay against one served model does.
- **GPU pool.** 2,275 blocks (36,400 tokens) on the frozen box, one of them vLLM's null block. A request takes the longest cached prefix of its full blocks, at most `⌊(L − 1) / 16⌋` blocks, because vLLM always computes the last prompt token. Running requests pin their blocks (reference counted, shared by common prefixes). A finished request frees its blocks in reverse order: cached blocks join the back of the free queue, so the end of a prefix is evicted before its start, and the partial last block joins the front. A block recomputed while a copy is cached is cached a second time, as vLLM keeps every block per hash. A hit takes a copy that running requests already pin when there is one, where vLLM takes the first copy cached; the two differ only for such recomputed blocks, and the clairvoyant pass relies on the rule (see below).
- **CPU tier.** 9,102 blocks (20 GiB at 144 KiB per token) on the frozen box, or a ratio of the pool. After the GPU hit, the tier extends it with the following blocks it holds; they are restored into new GPU blocks. When a request finishes, every full block it holds, prompt and output (`offload_prompt_only: false`), is offered to the tier: from the start of the prompt, or from past the restored blocks when it restored any (vLLM's chunk-level store index). LRU follows vLLM's `LRUCachePolicy`: blocks become most recent when stored and when a request that used them finishes, in reverse prefix order. A store never evicts a block it offers, and stores nothing when it cannot make room.
- **Output KV.** Output tokens take blocks and are written through, but they are never reused: AIPerf synthesizes each turn's history from the trace, not from what the model generated.
- **Admission.** A request needs blocks for `L + out − 1` tokens. It waits when the free blocks, less the free cached blocks its hit takes, cannot hold the blocks it must allocate; waiting requests are admitted in order as blocks free up, and the first that does not fit holds back those behind it, as in vLLM's scheduler. Each wait is a stall. A request larger than the whole pool is an error, as vLLM refuses to start with `max_model_len` above its pool.
- **Retention.** For the E1 arms the engine can hold the pool's cached start of a prompt (pin it), prefetch the rest from the CPU tier into free blocks and hold it, and demote free cached blocks to be reused first. Holds give way: a request that does not fit releases holds, oldest first, before it waits, so no arm makes a request wait for memory vLLM would give it.

## The contention model

`scheduler.rs` is vLLM 0.32's step loop on the frozen box (`Scheduler.schedule()` in `v1/core/sched`, async scheduling on):

- **Host.** A request reaches the scheduler after its host time: the API server, the tunnel, rendering, tokenization and block hashing, `29.6 ms + 7.994 µs` per prompt token, plus 6.5 ms for every request decoding when it is sent. The API server prepares one request at a time in the order they are sent (its renderer pool has `--renderer-num-workers` threads, 1 by default), so a burst of sends queues there.
- **Budget.** A step carries at most 512 tokens (`max_num_batched_tokens`). Running requests come first, in admission order: a decode takes one token, a prefill the rest of its prompt, at most 256 per step (`long_prefill_token_threshold`) when more than one request is running or waiting; alone, a request takes the whole budget. A request reached with the budget spent waits, decodes included.
- **Waiting requests** follow, first come first served (the replay sends every request with the background class's priority), after those whose restore has finished. A request is admitted into the cache model when it is first reached with budget left; one that does not fit stops the loop. A prompt found whole recomputes its last token.
- **Restores.** A request with blocks in the CPU tier waits for their load without taking budget. The load starts once the step that admitted it has run and the loads before it have finished (the connector issues loads one after another on one stream), and takes E0's restore term (`10.3 ms + 23.64 µs` per token). The step loop stalls for 0.80 of it, charged to the first step that starts during the load. The engine learns that a load has finished from the output of the step in flight and has already scheduled the next step, so the request computes from the second step that starts after its load, or at once when nothing else runs. A prefetch is a load that no request waits for.
- **Step time.** A step that computes `T` tokens takes `11.56 ms + 47.99 µs·T + 14.17 ns·P + 429.3 ns·K`, plus 7.6 ms when it carries both a prefill chunk and a decode, where `P` is the attention pairs (each computed token times the tokens it attends to) and `K` the KV tokens every scheduled request reads. `K`'s coefficient is 147,456 bytes per token read at 343 GB/s (the card's memory bandwidth is 448 GB/s); `P`'s is within 13% of twice E0's quadratic recompute term; the mixed-step extra is measured, not derived. Each output token comes out at the end of a step; a request with all its tokens finishes and frees its blocks.

`scripts/fit_step_cost.py` fits every coefficient from AIPerf exports and their `replay-log` CSVs, each term from its own data: the step terms and host time by least squares on relative error over decode-only steps (the gaps between a request's tokens while every request in flight was decoding: 46,315 at the three fresh points), requests that ran alone (517) and E0's 14 GPU-hit and recompute cells, with E0's token-id requests given a host time of their own; the mixed-step extra from 1,443 decode steps beside one long prefill (median excess); the host time per decoding request from 92 GPU hits that arrived while only decodes ran (slope); and the restore stall from 36 restores of at least 2,000 tokens with only decodes beside them (median lost decode time over the transfer time; quartiles 0.74 and 1.01). `--step-costs` takes the JSON it writes. The scheduler is synchronous otherwise: a GPU hit that arrives beside one decode waits about half a step in the measured runs, as here, not the step and a half that scheduling a step ahead would add.

## Simplifications

- **Whole footprint at admission.** A request reserves its prompt and output blocks when it is admitted, where vLLM reserves the prompt (`scheduler_reserve_full_isl`) and allocates output blocks as it decodes; so nothing is preempted (the operating points saw 0 to 3 preemptions per run).
- **Transfers.** Loads take E0's restore term each, one after another, and stall the step loop by a fixed share; stores happen when a request finishes, take no time and never delay a block's reuse. vLLM 0.32 stores as a request computes, and when the scheduler reallocates a block whose store is still pending the worker waits for that store before the step (the connector's `jobs_to_flush`); with the pool full this is likely what the model misses at 24 sessions (see the calibration).
- **Fixed step costs.** The step-time terms are fitted per step, not per kernel: CUDA-graph and chunk-shape effects are folded into the coefficients and the measured mixed-step extra.
- **Corpus tokens.** Prompts have no chat-template or cache-bust-marker tokens: the engine counted 4.7% more prompt tokens than the corpus holds for the same requests at W0.

## Arms

| Arm | GPU pool | CPU tier | While a session pauses |
|---|---|---|---|
| `gpu-lru` | vLLM's LRU free queue | none | nothing |
| `lru-cpu` | vLLM's LRU free queue | `LRUCachePolicy`, write-through: the frozen box | nothing |
| `s3fifo-cpu` | vLLM's LRU free queue | S3-FIFO: a 10% probationary FIFO, a main FIFO with reinsertion, and ghost entries | nothing |
| `clairvoyant` | Belady's MIN | Belady's MIN, with bypass | nothing |
| `ttl-5`, `ttl-30`, `ttl-120` | LRU | LRU | hold its cached blocks for the TTL |
| `history-prefetch` | LRU | LRU | prefetch it to end as the history predicts it returns |
| `oracle-prefetch` | LRU | LRU | prefetch it to end as it returns |
| `tokencake` | LRU | LRU | offload it, then prefetch it by history |
| `mori` | the free block expected back last | LRU | nothing |
| `infercept` | LRU | LRU | hold it or offload it, by minimum waste |

`run` runs the first four by default; `--arms` picks any, and `regime` runs `lru-cpu`, the E1 arms and the reference.

The **clairvoyant reference** runs in two passes. The first runs the reference arm (`lru-cpu` unless `--reference` says otherwise) and records the order in which the engine admitted and finished requests. The second replays that fixed order with farthest-next-use eviction in each tier: the pool reuses the free block whose next use is farthest, and the CPU tier keeps the blocks with the nearest next uses, declining to store a block whose next use is farther than every block it holds. A block's next use is the next request whose prompt holds it, whichever tier serves that request. Waits for free blocks, and so the whole timeline, are the reference arm's; time to first token is re-priced from the new hit pattern (under contention, the reference run's queueing stays and each request's own service, as it would take alone, is re-priced). Under any eviction policy a request takes the blocks its footprint needs minus those it shares with running requests (because a hit takes a pinned copy when one exists), so every admission of the first pass fits in the second.

It is a reference, not a bound. MIN per tier is optimal only for the hit count of one tier over a fixed sequence of unit-size blocks; here a block's hit counts only when its whole prefix hits, a CPU copy is worth nothing while the GPU holds the block, faster service would change which requests overlap, and PCIe has limits. `lh-sim mrc` reports the case where MIN is a bound.

**The E1 arms** (`retention.rs`) act on a session's KV, its latest prompt, while the session pauses between turns: from when a turn returns to when its successor, due after a delay, is admitted. Turns sent at once, or gated on children, are not pauses. The information levels are none (`lru-cpu`, and the fixed TTLs), history (an EWMA, α = 0.5, of the session's own pauses, per tool when the pauses name one, falling back to its mean over every tool, then to the mean over every session) and the oracle (the exact time the next turn is sent). Each arm simplifies its paper to this engine:

- **Continuum** ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230)) pins a paused agent's KV with a TTL it picks per tool call; here a fixed TTL is swept.
- **history-prefetch** and **oracle-prefetch** start a load of the session's blocks that the CPU tier holds so it ends 1 ms before the predicted (exact) return, and hold them until the turn is admitted; the oracle is Ask the Tool's progress-reporting level ([arXiv 2609.18849](https://arxiv.org/abs/2609.18849)).
- **Tokencake** ([arXiv 2510.18586](https://arxiv.org/abs/2510.18586)) offloads an agent's KV at the start of a function call and uploads it before the predicted completion. The CPU tier already has a write-through copy, so offloading demotes the session's free blocks; the upload is the history prefetch. Its space scheduling across agents is not modeled.
- **MORI** ([arXiv 2606.00866](https://arxiv.org/abs/2606.00866)) splits KV between HBM and DRAM by idleness; here only its ranking, as the pool's eviction order: the free block whose session is expected back last (by history) goes first. Its proactive moves are not modeled.
- **InferCept** (Abhyankar et al., ICML 2024) chooses per interception among preserving, discarding and swapping the KV by minimum waste in memory × time: preserving wastes the context for the predicted pause, swapping wastes the running batch for the step-loop stall of the restore (0.80 of E0's restore term), discarding wastes it for the recompute. Discarding never wins with a CPU tier here (ρ < 1). Preserving holds the blocks; swapping demotes them.

A hold also ends once its session is later than twice its predicted pause. A prefetch takes only free blocks and runs on the load stream like any restore: it delays the restores queued behind it and stalls the step loop as they do.

## Results per arm

- **Tokens.** Requests and prompt tokens, and the shares served from the GPU pool, restored from the CPU tier and computed. Computed tokens split into cold first turns of conversations, new context (blocks the run had not seen), and recomputed blocks (seen before, since evicted from every tier, plus the last block of a fully cached block-aligned prompt): the avoidable recompute.
- **Latency.** Time to first token p50/p90/p99, overall and for resumed turns.
- **Traffic.** PCIe bytes restored, prefetched and stored, GPU and CPU evictions, and stalls (requests that waited for free blocks, and the total wait). Under contention `engine` also counts the steps run, their tokens and their total time.
- **Regime.** KV pressure, the live working set (distinct blocks in the latest prompt of every live stream) over the pool, as a time-weighted mean and a peak; π, the median pause over the restore time of the median prompt (E0's restore extra over a GPU hit); ρ, that restore time over the recompute time of the median prompt.

Shares, latencies, restored bytes and stalls cover the profiling requests that completed, as AIPerf's results do; store, prefetch and eviction counts cover the whole run.

## Usage

```sh
cargo build --release -p lh-sim
lh-sim run DATA/weka-x32 --shared-models --json w0.json --csv w0.csv     # the W0 configuration, the stock arms
lh-sim run DATA/weka-x32 --cpu-ratio 2,4,8 --arms lru-cpu,clairvoyant    # CPU tier at 2x, 4x and 8x the pool
lh-sim run DATA/weka-x32 --shared-models --service idle                   # v0's idle-box service
lh-sim run DATA/weka-x32 --shared-models --pauses tracelab --arms lru-cpu,oracle-prefetch
lh-sim regime DATA/weka-x32 --shared-models --csv regime.csv --markdown regime.md
lh-sim mrc DATA/weka-x32 --shared-models --csv mrc.csv                    # miss-ratio curves
lh-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl      # a measured run through the engine model
lh-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl --timed --csv w0-timed.csv
```

The replay options default to `configs/agentic_replay.yaml` with the W0 settings: 16 lanes, 600 s plus a 60 s grace, a 120 s think-time cap, seed 11, start ratios 0 to 1, burst starts and a 10 s idle cap. Capacities default to the frozen box (`--pool-tokens 36400`, a 20 GiB CPU tier; `--cpu-ratio` or `--cpu-gib` change it), costs to the E0 fits (`--costs` takes the `summary.json` of `lh-bench tiers`; `--itl-ms` sets the idle service's decode rate) and the step-time model to the frozen box's fit (`--step-costs`). `run` writes a JSON summary (`--json`), one CSV row per arm and capacity (`--csv`) and one CSV row per request (`--requests`). `--token-scale 32` replays a 32× corpus at its original token counts.

`regime` runs `lru-cpu`, the E1 arms and the reference in every cell of `--sessions` (5, 8, 12, 16, 24, 32, 40), `--cpu-ratio` (2, 4, 8) and `--pause-sources` (trace, tracelab), on `--jobs` threads (default: every core). It writes one CSV row per cell and arm (`--csv`) and the regime map (`--markdown`): per cell, the coordinates, `lru-cpu`'s, the reference's and the oracle's resume TTFT, the oracle's gain, and the share of the clairvoyant gap each level closes at p90. A share is `(lru − arm) / (lru − clairvoyant)`, left empty where the reference is less than 5% faster than `lru-cpu`.

`replay-log` sends a measured run's requests at their measured start and end times through the engine model, which separates the cache model from the replay timing; with `--timed` it sends each at its measured start and serves it with the step loop, open loop, which separates the service model from the closed loop's feedback, and adds the measured and simulated TTFT to the CSV.

## Calibration against the GPU operating points

Three operating points on the frozen box, each on a freshly started engine: AIPerf `agentic_replay` at 8, 16 and 24 sessions (the 32× corpus, 120 s think cap, 600 s plus 60 s grace, seed 11, burst starts, straight to the engine as W0). The simulator runs `lh-sim run DATA/weka-x32 --shared-models --arms lru-cpu --concurrency N`. Measured shares are the engine's `prompt_tokens_cached_by_source` counters over the profiling phase (the counters' deltas less the warm-up requests' own tokens); simulated shares cover the profiling requests. Resume TTFT covers profiling requests after a conversation's first turn.

| Point | | Requests | Resume TTFT p50 / p90 / p99 (ms) | GPU / CPU / computed (%) |
|---|---|---|---|---|
| 8 sessions | measured | 408 | 83 / 400 / 1,463 | 66.6 / 29.0 / 4.4 |
| | idle (v0) | 410 (+0.5%) | 67 / 200 / 563 (−19 / −50 / −61%) | 67.6 / 28.1 / 4.3 |
| | contention | 405 (−0.7%) | 81 / 382 / 1,749 (−3 / −5 / +20%) | 66.1 / 29.5 / 4.4 |
| | leave-one-out | 405 | 83 / 400 / 1,772 (−0 / −0 / +21%) | 66.1 / 29.5 / 4.4 |
| 16 sessions | measured | 686 | 132 / 743 / 8,102 | 57.4 / 36.0 / 6.6 |
| | idle (v0) | 685 (−0.1%) | 79 / 317 / 4,287 (−40 / −57 / −47%) | 62.8 / 31.1 / 6.1 |
| | contention | 691 (+0.7%) | 127 / 685 / 7,776 (−4 / −8 / −4%) | 58.8 / 34.3 / 6.9 |
| | leave-one-out | 690 | 125 / 751 / 7,830 (−5 / +1 / −3%) | 59.1 / 33.9 / 7.0 |
| 24 sessions | measured | 1,568 | 173 / 1,345 / 15,172 | 52.9 / 34.3 / 12.9 |
| | idle (v0) | 1,756 (+12.0%) | 71 / 272 / 4,766 (−59 / −80 / −69%) | 59.5 / 28.6 / 11.9 |
| | contention | 1,649 (+5.2%) | 142 / 942 / 9,896 (−18 / −30 / −35%) | 55.9 / 32.4 / 11.7 |
| | leave-one-out | 1,646 | 137 / 826 / 9,931 (−21 / −39 / −35%) | 55.9 / 32.9 / 11.2 |

Leave-one-out refits every coefficient on the other two points and predicts the third. The repeats give the measurements' own spread: three 16-session runs (fresh, after the 8-session run, and W0) measured 132–140 ms at p50, 743–966 ms at p90 and 8,102–8,456 ms at p99, against which the model is −4 to −9%, −8 to −29% and −4 to −8%. The 24-session run after the 16-session one measured 213 / 1,798 / 18,403 ms (−33 / −48 / −46%): the starting state matters more there.

- **The cache model.** On each point's own timeline (`replay-log`) the engine model's GPU share is within 0.7, 0.3 and 1.7 points of the counters' at 8, 16 and 24 sessions, and its per-request cached share within 0.6, 1.3 and 2.0 points on average: about half of the closed loop's +3.0-point GPU share at 24 sessions is the cache model's.
- **8 and 16 sessions meet the targets** (p50 and p90 within 10%, shares within 2.1 points, requests within 1%) in sample and out of sample. p99 at 8 sessions is four requests; it moves 20% with one of them.
- **24 sessions does not**, and the evidence points at transfers. Open loop (`replay-log --timed`), the model is −17 / −47 / −56% at 24 sessions against +1 / −3 / −33% at 8 and −6 / −20 / +6% at 16. Hits match at every point (median residual 0, 0 and −5 ms) and so do computed prompts at 8 and 16 sessions (+12, +11 ms), but restores take 163 ms longer than modeled at 24 sessions (15 ms at 8, 31 ms at 16), the more so the more other restores overlap their measured wait: 41 ms with none, 102 ms with one, 163 ms with three, 1.2 s with six or more. The model gives each load E0's transfer time and never delays a block's reuse for a store; vLLM 0.32 stores as requests compute and, when it reallocates a block whose store is pending, waits for the store before the step, which happens most when the pool is full. Modeling store traffic is left for when E1 or E2 needs the 24-session regime.

## Validation against W0 (v0, the idle service)

W0 replayed the full 32× corpus against the frozen box: AIPerf's sequential sampler gave its 16 lanes the first 16 traces in name order, and its 10 recycled plays the next 10. The simulator runs the same configuration with seed 11 and shared models (one served model), here with `--service idle`. `replay-log` feeds the engine model W0's own requests at their measured start and end times, which separates the cache model from the replay timing.

| | W0, measured | Engine model on W0's timeline | Full simulation |
|---|---|---|---|
| Profiling requests | 688 | 688 | 685 |
| Session plays, warmup requests | 26, 16 | — | 26, 16 |
| Prompt tokens | 3,268,924 | 3,122,538 | 3,113,792 |
| From the GPU pool | 56.54% | 56.96% (+0.42) | 62.84% (+6.30) |
| Restored from the CPU tier | 36.90% | 37.31% (+0.41) | 31.05% (−5.85) |
| Computed | 6.56% | 5.73% (−0.83) | 6.11% (−0.45) |
| TTFT p50 / p90 / p99 | 135 / 815 / 8,501 ms | — | 79 / 322 / 4,420 ms |

The measured split is the engine's `prompt_tokens_cached_by_source` counters over the run; its cached share agrees with the per-request `cached_tokens` (93.44%).

- **The cache model.** On W0's timeline the engine model is within half a point of the measured GPU and CPU shares, and its per-request cached share is within 1.35 points on average.
- **The timeline.** At the idle box's speed the full simulation's sessions return sooner than W0's (a mean modeled time to first token of 237 ms and service time of 783 ms, where W0 measured 541 ms and 1,269 ms), so fewer blocks pass through the pool between a session's turns and more prompts still hit the GPU. The contention model closes the gap: 58.8 / 34.3 / 6.9% against W0's 56.5 / 36.9 / 6.6% (profiling phase), and resume TTFT within 10% (above).
- **Prompt length.** The engine counted chat-template and cache-bust tokens the corpus does not hold (4.7% more).
- **Spread.** The seed sets where each lane starts in its trace, so it changes the workload itself. Seeds 11–15 give 639–955 requests and 60.9–72.2% / 21.8–32.5% / 4.4–7.4% (GPU / CPU / computed): one 10-minute, 16-lane run is a small sample.
- **The 16-trace subset** in `testdata/` is not W0's trace pool. Replayed the same way it is a lighter workload (559 requests, 0.74 M prompt tokens, KV pressure 0.9 against 3.3; 72.4 / 13.3 / 14.3% with seed 11), and it serves as the end-to-end test.

## The E1 regime map

`lh-sim regime DATA/weka-x32 --shared-models` at seeds 11, 12 and 13: resume TTFT gain over `lru-cpu` (1 − arm / `lru-cpu`) at p50 / p90, the mean over the seeds, with the CPU tier at 4× the pool. ρ is 0.25–0.28 in every cell. The CSV has the 2× and 8× tiers, the other arms, avoidable recompute and PCIe bytes.

| Pauses | Sessions | Pressure | π | `oracle-prefetch` | `history-prefetch` | `tokencake` | `ttl-30` | `clairvoyant` |
|---|---|---|---|---|---|---|---|---|
| trace | 5 | 1.3 | 20.6 | +0 / +1% | −1 / +0% | −5 / −9% | −1 / +2% | +1 / +7% |
| trace | 8 | 1.8 | 19.1 | −0 / +22% | +1 / +11% | −6 / −8% | +1 / +4% | +1 / +27% |
| trace | 12 | 2.6 | 16.1 | +5 / +26% | −1 / −6% | −10 / −21% | +0 / −2% | +0 / +7% |
| trace | 16 | 3.6 | 15.9 | +13 / +8% | −11 / −27% | −27 / −22% | +2 / −5% | +2 / +8% |
| trace | 24 | 4.7 | 15.9 | −25 / −14% | −31 / −22% | −60 / −23% | +5 / −3% | +3 / +8% |
| trace | 32 | 6.2 | 16.9 | −48 / −24% | −40 / −17% | −37 / −11% | −7 / +1% | +3 / +2% |
| trace | 40 | 7.5 | 17.5 | −55 / −27% | −46 / −29% | −47 / −22% | +4 / +6% | +1 / +1% |
| tracelab | 5 | 1.2 | 2.0 | −1 / +1% | −0 / −2% | −5 / −7% | −1 / +1% | +0 / +1% |
| tracelab | 8 | 1.8 | 0.9 | −0 / +3% | −2 / −3% | −6 / −14% | +1 / +3% | +1 / +4% |
| tracelab | 12 | 2.4 | 1.0 | +1 / +6% | −3 / −9% | −13 / −18% | +0 / +0% | +0 / +7% |
| tracelab | 16 | 3.4 | 1.1 | −29 / −51% | −39 / −42% | −30 / −41% | +7 / −2% | +4 / +2% |
| tracelab | 24 | 4.7 | 1.1 | −81 / −25% | −80 / −27% | −86 / −27% | +13 / +1% | +2 / +0% |
| tracelab | 32 | 6.1 | 1.2 | −30 / −21% | −30 / −26% | −32 / −27% | −0 / −0% | +1 / +1% |
| tracelab | 40 | 7.3 | 1.1 | −11 / −4% | −20 / −3% | −14 / −4% | +0 / +0% | +1 / +1% |

With trace pauses up to 24 sessions, `mori` loses 1–15% at p50 and 10–24% at p90 against `lru-cpu` (ranking by noisy predicted returns evicts sessions that come back), and `infercept` 18–36% at p50 (its swaps turn GPU hits into restores, and the pool has no use for the memory they free).

**The realistic regime.** The frozen box with the traces' think times and the CPU tier at 4× (π about 16, against 2–4 times production's per PLAN §3.2), over 20 seeds (11–30); gain over `lru-cpu`, mean (median) [range]:

| Sessions | Pressure | `lru-cpu` resume p50 / p90 | `oracle-prefetch` p50 | p90 | `clairvoyant` p50 / p90 | Best history arm, p90 |
|---|---|---|---|---|---|---|
| 8 | 1.6 | 86 / 267 ms | +0.7% (+0.6) [−2, +3] | +11.3% (+10.2) [−4, +26] | +1.3 / +15.5% | `history-prefetch` +3.6% |
| 12 | 2.4 | 103 / 299 ms | +2.6% (+2.3) [−1, +7] | +23.3% (+22.9) [+11, +37] | +0.5 / +8.0% | `history-prefetch` −9.5% |
| 16 | 3.2 | 139 / 895 ms | +6.8% (+3.9) [−0, +23] | +7.7% (+16.6) [−43, +37] | +1.7 / +9.4% | `infercept` −9.8% |
| 24 | 4.4 | 208 / 1,645 ms | −2.3% (+6.2) [−103, +20] | −17.0% (−15.2) [−61, +14] | +2.3 / +10.9% | `infercept` −7.0% |

**G-sim.** The oracle level does not close 10% of resume TTFT across the realistic regime. At p50 it closes under 7% at every point: the median resumed turn is a GPU hit or a short restore, and a restore turned into a hit saves only its transfer and the two steps it waits after it. At p90 it closes 11% and 23% at 8 and 12 sessions (pressure 1.6–2.4), 8% on average at W0's 16 sessions with a seed range wider than the effect (−43 to +37%), and it is 17% worse at 24 sessions (pressure 4.4), where its loads queue with the demand restores on the one load stream and stall the step loop as they do. The clairvoyant gap itself is 0.5–2.3% at p50 and 8–16% at p90. The history level closes nothing: its best arm gains 3.6% at p90 at 8 sessions and loses 7–10% from 12 sessions on, its prefetches mistimed by heavy-tailed pauses. With TraceLab's tool pauses (π about 1, a pause as long as the restore) the oracle gains at most 6% (p90, 12 sessions) and loses from 16 sessions on. Caveat: the model under-predicts restores at 24 sessions (above), and a dearer restore at high pressure would raise both what prefetching saves and what its loads cost.

## Miss-ratio curves

`lh-sim mrc` computes curves for the GPU tier alone on the reference arm's admission sequence: LRU at 41 capacities from 1/16 to 64 times the pool in one pass of Mattson stack distances, and MIN at chosen capacities. Each request looks up its blocks, then its blocks become the most recent in the order vLLM frees them. Hits are token weighted, per block and per prefix (a block counts only when its whole prefix hits). MIN's per-block hit ratio is a bound: no policy hits more blocks of this sequence at that capacity. Its prefix hit ratio is a reference.

Three sequences: the 32× replay; the same admission order with every prompt cut at its original token counts (64-token hash blocks, outputs ×32, capacities ×32); and the replay run again at the original token counts. Miss ratios on the W0 configuration, with the idle service:

| Pool multiple | LRU, 32× | LRU, original tokens, same order | MIN, 32× | MIN, original tokens, same order |
|---|---|---|---|---|
| 0.25 | 0.718 | 0.718 | 0.503 | 0.503 |
| 0.5 | 0.541 | 0.540 | 0.341 | 0.340 |
| 1 | 0.356 | 0.355 | 0.229 | 0.228 |
| 2 | 0.217 | 0.216 | 0.108 | 0.107 |
| 4 | 0.058 | 0.057 | 0.041 | 0.039 |
| 8 | 0.041 | 0.039 | 0.041 | 0.039 |

- **Scaling keeps the reuse structure.** On the same order the curves agree within 0.2 points at every capacity: cutting blocks 32 times coarser loses almost no sharing.
- **Scaling changes the timing.** Replayed at original token counts with the box's E0 fit, a cold prefill of a 100k-token prompt is priced at 88 s, and the 10-minute run completes 10 requests. The 32× replay's pacing, and with it its KV pressure and π, is a property of the scaling. So E1 states its results in the regime coordinates (KV pressure, π, ρ), and other hardware is placed on that map from cost fits built from published specifications, labeled as modeled.
- **Prefixes stay whole.** LRU's per-block and per-prefix ratios are equal: tail-first eviction never leaves a hole in a cached prefix. MIN's are equal to three decimals.
- **Headroom.** At the pool's size MIN misses 22.9% of prompt tokens where LRU misses 35.6%. MIN is down to the cold misses (4.1%) at four times the pool, LRU at eight.

## Runtime

Release builds on the full corpus (393 traces, 98,827 requests, 1.7 GB of JSON), one core unless noted, on an Apple M5 Max. The full-corpus runs use `--duration 100000000 --grace 0 --max-sessions 393 --start-max 0`, so every trace plays once from its start.

| Run | Contention | Idle |
|---|---|---|
| The W0 configuration, the four stock arms | 0.23 s | 0.20 s |
| The W0 configuration, `lru-cpu`, the E1 arms and the reference | 0.29 s | — |
| Every trace through 16 recycling lanes, the four stock arms | 11.5 s | 10.5 s |
| `regime`, 42 cells × 10 arms | 10.6 s (2.1 s on 18 cores) | — |

W1 measured the idle runs on cyan (Xeon Silver 4310 at 2.1 GHz, Linux) too, with identical results: 0.53 s for the W0 configuration and 29.7 s through 16 recycling lanes.

## Tests

`cargo test -p lh-sim`: block identity and namespacing; LRU tail-first freeing, pinning, recomputed copies, CPU write-through and restore; holds, their release under pressure, prefetch, demotion and the idleness ranking; S3-FIFO; both clairvoyant tiers and MIN against exhaustive search on small sequences; Mattson stack distances against a direct LRU simulation; the step loop against hand-computed schedules (budget and cap, decodes behind prefills, head-of-line blocking, restore loads, stalls and the two-step lag, mixed steps, the serialized host); the replay scheduler on small synthetic traces (the think-time and idle caps, concurrent subagents and joins, recycling, and the TTL hold and the prefetches on a paused session); the history predictor and InferCept's choice; the TraceLab table and its draws; the regime table; AIPerf's seed derivation against numpy; and, on the 16-trace subset, the plan against AIPerf's, every stock arm end to end, and the E1 arms under TraceLab pauses, twice, with identical results.

`testdata/weka-x32-sub16` holds 16 traces of [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) (Apache-2.0), block-scaled 32× by `lh-bench scale-traces`. To regenerate the pinned AIPerf plan, see `scripts/aiperf_reference.py`. `data/tracelab-pauses.tsv` holds the percentiles of the wait between a coding agent's tool calls and their results, per tool, derived by `scripts/tracelab_pauses.py` from the [TraceLab](https://github.com/uw-syfi/TraceLab) public coding-agent trace dataset v0.0.2 (CC BY 4.0); the dataset itself is not in the repository.
