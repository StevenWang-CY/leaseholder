# lh-sim

`lh-sim` replays the SemiAnalysis Claude Code traces as AIPerf 0.13.0's `agentic_replay` sends them, through a model of vLLM 0.32's GPU prefix cache and OffloadingConnector CPU tier, and prices every request with `lh-core`'s E0 tier cost model. For each cache policy it reports where prompt tokens came from (the GPU pool, the CPU tier, or computation), what that costs in time to first token and PCIe traffic, and where the run sits in the regime coordinates E1 is plotted in. A clairvoyant reference gives the gap each policy leaves.

## The replay

- **Traces.** A directory of `<trace id>.json` as `lh-bench scale-traces` writes it, or the original JSONL. A prompt is `len(hash_ids) × block_size` tokens; the raw `in` field overcounts.
- **AIPerf's structure, ported.** The Weka loader's chain detection splits each trace into the main agent, flattened agent chains and subagent chains, with their think-time delays (`useThinkTimeOnly`, capped at `interTurnDelayCapSeconds`), SPAWN branches and joins, and the cross-stream replay barriers. `TrajectorySource` gives each lane its start instant `t*` (SHA-256 of seed, trace and lane, then numpy's PCG64) and the streams live at `t*`. `scripts/aiperf_reference.py` dumps AIPerf's own plan with prompt synthesis stubbed out, and `lh-sim aiperf-plan` dumps the port's: they agree field for field on the full 32× corpus (9,843 conversations, 98,827 turns) for 393 lanes with seed 11 and for 450 lanes with seed 7, start ratios 0.25–0.75 and a 30 s cap. The test subset's plan is pinned in `testdata/aiperf-0.13.0-sub16.json`.
- **Dispatch.** `concurrency` lanes take traces from AIPerf's sequential sampler. Every stream live at `t*` first sends its last earlier request with `max_tokens = 1`, all at once; profiling starts when the last of these returns. Each lane then sends its earliest live turn at once and the others at their recorded offsets. A turn's successor goes its delay after the turn returns; a root turn spawns its children when it returns (when it is sent, if the recorded spawn overlapped the request), each at its recorded offset from the spawn; a joining turn waits for its children; a turn whose replay predecessors have not completed is held. A lane whose session tree has drained starts the next trace from its first turn, under a fresh KV namespace (AIPerf's cache-bust marker). No request is sent after the duration, requests still running at the end of the grace period are dropped, and when nothing is in flight the pending think-time timers are pulled forward so the engine idles at most `systemIdleGapCapSeconds`.
- **Service.** A request's time to first token is `lh-core`'s cost for its hit pattern, after any wait for free blocks; it then decodes at a constant inter-token latency. Requests do not queue behind each other otherwise (see the simplifications).

## The engine model

- **Blocks.** Prompts are cut into 16-token blocks. A block's identity is its exact prefix: the trace's hash-id trie node its last token falls in, its offset inside that hash block, the session play and, by default, the model. Two prompts share a block iff they share every token up to its end, which is vLLM's chained block hashing without collisions. `--shared-models` drops the model from the identity, as an AIPerf replay against one served model does.
- **GPU pool.** 2,275 blocks (36,400 tokens) on the frozen box, one of them vLLM's null block. A request takes the longest cached prefix of its full blocks, at most `⌊(L − 1) / 16⌋` blocks, because vLLM always computes the last prompt token. Running requests pin their blocks (reference counted, shared by common prefixes). A finished request frees its blocks in reverse order: cached blocks join the back of the free queue, so the end of a prefix is evicted before its start, and the partial last block joins the front. A block recomputed while a copy is cached is cached a second time, as vLLM keeps every block per hash. A hit takes a copy that running requests already pin when there is one, where vLLM takes the first copy cached; the two differ only for such recomputed blocks, and the clairvoyant pass relies on the rule (see below).
- **CPU tier.** 9,102 blocks (20 GiB at 144 KiB per token) on the frozen box, or a ratio of the pool. After the GPU hit, the tier extends it with the following blocks it holds; they are restored into new GPU blocks. When a request finishes, every full block it holds, prompt and output (`offload_prompt_only: false`), is offered to the tier: from the start of the prompt, or from past the restored blocks when it restored any (vLLM's chunk-level store index). LRU follows vLLM's `LRUCachePolicy`: blocks become most recent when stored and when a request that used them finishes, in reverse prefix order. A store never evicts a block it offers, and stores nothing when it cannot make room.
- **Output KV.** Output tokens take blocks and are written through, but they are never reused: AIPerf synthesizes each turn's history from the trace, not from what the model generated.
- **Admission.** A request needs blocks for `L + out − 1` tokens. It waits when the free blocks, less the free cached blocks its hit takes, cannot hold the blocks it must allocate; waiting requests are admitted in order as blocks free up, and the first that does not fit holds back those behind it, as in vLLM's scheduler. Each wait is a stall. A request larger than the whole pool is an error, as vLLM refuses to start with `max_model_len` above its pool.

## Simplifications

- **No contention.** The cost model is E0's, measured on the idle box, and decode runs at the W0 replay's median inter-token latency of 15.6 ms. Batching, the 512-token prefill step cap and the queueing they cause are not modeled; this is the main source of disagreement with W0 below.
- **Whole footprint at admission.** A request reserves its prompt and output blocks when it is admitted, where vLLM allocates as it computes; so nothing is preempted (W0 saw one preemption).
- **Free transfers.** Restores take no time beyond the cost model's restore term, stores happen when a request finishes rather than while it runs, and PCIe has no bandwidth limit.
- **Corpus tokens.** Prompts have no chat-template or cache-bust-marker tokens: W0's engine counted 4.7% more prompt tokens than the corpus holds for the same requests.

## Arms

| Arm | GPU pool | CPU tier |
|---|---|---|
| `gpu-lru` | vLLM's LRU free queue | none |
| `lru-cpu` | vLLM's LRU free queue | `LRUCachePolicy`, write-through: the frozen box |
| `s3fifo-cpu` | vLLM's LRU free queue | S3-FIFO: a 10% probationary FIFO, a main FIFO with reinsertion, and ghost entries |
| `clairvoyant` | Belady's MIN | Belady's MIN, with bypass |

The **clairvoyant reference** runs in two passes. The first runs the reference arm (`lru-cpu` unless `--reference` says otherwise) and records the order in which the engine admitted and finished requests. The second replays that fixed order with farthest-next-use eviction in each tier: the pool reuses the free block whose next use is farthest, and the CPU tier keeps the blocks with the nearest next uses, declining to store a block whose next use is farther than every block it holds. A block's next use is the next request whose prompt holds it, whichever tier serves that request. Waits for free blocks, and so the whole timeline, are the reference arm's, and time to first token is re-priced from the new hit pattern. Under any eviction policy a request takes the blocks its footprint needs minus those it shares with running requests (because a hit takes a pinned copy when one exists), so every admission of the first pass fits in the second.

It is a reference, not a bound. MIN per tier is optimal only for the hit count of one tier over a fixed sequence of unit-size blocks; here a block's hit counts only when its whole prefix hits, a CPU copy is worth nothing while the GPU holds the block, faster service would change which requests overlap, and PCIe has limits. `lh-sim mrc` reports the case where MIN is a bound.

Policies are small traits, `GpuPolicy` and `CpuPolicy` in `engine/`, so later arms (TTL holds, history- and oracle-timed prefetch, predictive upload, idleness ranking) are additions.

## Results per arm

- **Tokens.** Requests and prompt tokens, and the shares served from the GPU pool, restored from the CPU tier and computed. Computed tokens split into cold first turns of conversations, new context (blocks the run had not seen), and recomputed blocks (seen before, since evicted from every tier, plus the last block of a fully cached block-aligned prompt).
- **Latency.** Modeled time to first token p50/p90/p99, overall and for resumed turns.
- **Traffic.** PCIe bytes restored and stored, GPU and CPU evictions, and stalls (requests that waited for free blocks, and the total wait).
- **Regime.** KV pressure, the live working set (distinct blocks in the latest prompt of every live stream) over the pool, as a time-weighted mean and a peak; π, the median think time over the restore time of the median prompt; ρ, the restore time over the recompute time of the median prompt.

Shares, latencies, restored bytes and stalls cover the profiling requests that completed, as AIPerf's results do; store and eviction counts cover the whole run.

## Usage

```sh
cargo build --release -p lh-sim
lh-sim run DATA/weka-x32 --shared-models --json w0.json --csv w0.csv     # the W0 configuration, every arm
lh-sim run DATA/weka-x32 --cpu-ratio 2,4,8 --arms lru-cpu,clairvoyant    # CPU tier at 2x, 4x and 8x the pool
lh-sim mrc DATA/weka-x32 --shared-models --csv mrc.csv                    # miss-ratio curves
lh-sim replay-log DATA/weka-x32 runs/w0/aiperf/profile_export.jsonl      # a measured run through the engine model
```

The replay options default to `configs/agentic_replay.yaml` with the W0 settings: 16 lanes, 600 s plus a 60 s grace, a 120 s think-time cap, seed 11, start ratios 0 to 1, burst starts and a 10 s idle cap. Capacities default to the frozen box (`--pool-tokens 36400`, a 20 GiB CPU tier; `--cpu-ratio` or `--cpu-gib` change it), costs to the E0 fits (`--costs` takes the `summary.json` of `lh-bench tiers`; `--itl-ms` sets the decode rate). `run` writes a JSON summary (`--json`), one CSV row per arm and capacity (`--csv`) and one CSV row per request (`--requests`). `--token-scale 32` replays a 32× corpus at its original token counts.

## Validation against W0

W0 replayed the full 32× corpus against the frozen box: AIPerf's sequential sampler gave its 16 lanes the first 16 traces in name order, and its 10 recycled plays the next 10. The simulator runs the same configuration with seed 11 and shared models (one served model). `replay-log` feeds the engine model W0's own requests at their measured start and end times, which separates the cache model from the replay timing.

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
- **The timeline.** The full simulation sends 685 of W0's 688 requests (matched by conversation and turn; W0's other 3 were sent in the last 15 s), but its service times are the idle box's: a mean modeled time to first token of 237 ms and service time of 783 ms, where W0 measured 541 ms and 1,269 ms. Median think times are short (1.6 s in this window), so service time sets much of the gap between a session's turns; with shorter gaps, fewer blocks pass through the pool between them and more prompts still hit the GPU. Slowing decode in the model moves the split toward the measurement: an inter-token latency of 20.9 ms (W0's token-weighted mean) gives 61.1 / 32.4 / 6.5%, 30 ms gives 60.1 / 33.0 / 6.9%, and 45 ms gives 56.4 / 37.4 / 6.2%. A contention-aware service model is the fix.
- **Latency.** Without queueing behind long prefills, modeled time to first token is lower throughout, most at the tail.
- **Prompt length.** The engine counted chat-template and cache-bust tokens the corpus does not hold (4.7% more).
- **Spread.** The seed sets where each lane starts in its trace, so it changes the workload itself. Seeds 11–15 give 639–955 requests and 60.9–72.2% / 21.8–32.5% / 4.4–7.4% (GPU / CPU / computed): one 10-minute, 16-lane run is a small sample.
- **The 16-trace subset** in `testdata/` is not W0's trace pool. Replayed the same way it is a lighter workload (559 requests, 0.74 M prompt tokens, KV pressure 0.9 against 3.3; 72.4 / 13.3 / 14.3% with seed 11), and it serves as the end-to-end test.

Of the 6.11% computed in the simulation, 0.91 points are cold first turns, 3.15 new context and 2.05 recomputed after eviction from both tiers; the clairvoyant reference computes 4.06%, the first two alone. Its resume time to first token is 77 / 224 / 4,235 ms (p50 / p90 / p99) against `lru-cpu`'s 79 / 317 / 4,287 ms, at the same capacities and timeline.

## Miss-ratio curves

`lh-sim mrc` computes curves for the GPU tier alone on the reference arm's admission sequence: LRU at 41 capacities from 1/16 to 64 times the pool in one pass of Mattson stack distances, and MIN at chosen capacities. Each request looks up its blocks, then its blocks become the most recent in the order vLLM frees them. Hits are token weighted, per block and per prefix (a block counts only when its whole prefix hits). MIN's per-block hit ratio is a bound: no policy hits more blocks of this sequence at that capacity. Its prefix hit ratio is a reference.

Three sequences: the 32× replay; the same admission order with every prompt cut at its original token counts (64-token hash blocks, outputs ×32, capacities ×32); and the replay run again at the original token counts. Miss ratios on the W0 configuration:

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

Release builds, one core each, on the full corpus (393 traces, 98,827 requests, 1.7 GB of JSON). The full-corpus runs use `--duration 100000000 --grace 0 --max-sessions 393 --start-max 0`, so every trace plays once from its start.

| Run | Apple M5 Max | cyan: Xeon Silver 4310 at 2.1 GHz, Linux (`taskset -c 20`) |
|---|---|---|
| The validation run, all four arms | 0.36 s | 0.53 s, 108 MB |
| `mrc` | 1.35 s | — |
| All 393 traces live at once, `lru-cpu` | 5.9 s, 650 MB (about 2.5 s of it reading and planning the traces) | 15.2 s, 293 MB |
| Through 16 recycling lanes, all four arms | 10.2 s, 1.6 GB (the clairvoyant pass holds the full timeline) | 29.7 s, 1.36 GB |

The two machines produce identical results for every arm of all three runs: request counts and the token counts by source.

## Tests

`cargo test -p lh-sim`: block identity and namespacing; LRU tail-first freeing, pinning, recomputed copies, CPU write-through and restore; S3-FIFO; both clairvoyant tiers and MIN against exhaustive search on small sequences; Mattson stack distances against a direct LRU simulation; the replay scheduler on small synthetic traces (the think-time and idle caps, concurrent subagents and joins, recycling); AIPerf's seed derivation against numpy; and, on the 16-trace subset, the plan against AIPerf's and every arm end to end.

`testdata/weka-x32-sub16` holds 16 traces of [semianalysisai/cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) (Apache-2.0), block-scaled 32× by `lh-bench scale-traces`. To regenerate the pinned AIPerf plan, see `scripts/aiperf_reference.py`.
