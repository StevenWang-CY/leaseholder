# leaseholder-bench

Load generation and latency probes for Leaseholder. It replays the SemiAnalysis Claude Code traces
(`semianalysisai/cc-traces-weka-062126`: 393 sessions, 98,827 requests, subagents, think times)
against a 16-32k context, with Leaseholder's per-session metadata on every request. It also measures
the tier cost model (E0) and server-side voice turns.

| Piece | Module | Purpose |
|---|---|---|
| `lh-bench scale-traces` | `traces.py` | Shrinks every request of the corpus by 32x or 64x while keeping its prefix structure |
| AIPerf loader `weka_trace_scaled` | `aiperf_plugin.py` | AIPerf's Weka loader, fixed to read each trace's own block size |
| AIPerf endpoint `leaseholder_chat` | `aiperf_plugin.py` | Chat completions plus `X-Session-ID`, `X-LH-Class`, `prompt_cache_key` and `priority` |
| `configs/agentic_replay.yaml`, `configs/aiperf.env` | | AIPerf `agentic_replay` run: steady-state session trees with recycling |
| `lh-bench mock-server` | `mock_server.py` | Recording OpenAI-compatible mock on 127.0.0.1 for CPU-only dry runs |
| `lh-bench tiers` | `tiers.py` | E0: time to first token from the GPU pool, the CPU tier and recompute, verified per sample by vLLM's prefix-cache counters; optional background load |
| `lh-bench voice-turn` | `voice_turn.py` | One server-side voice turn timed per stage (STT, LLM first sentence, TTS first audio); optional background load |
| `lh-bench kv-metrics` | `kv_metrics.py` | KV block lifecycle histograms compared across a run |
| `lh-bench e2` | `e2/` (`arm.py`, `run.py`, `record.py`, `report.py`, `figures.py`) | E2 points: one measured from its arm file on cyan (`run`), the frontier across points (`report`), and their compact records (`export`) |
| background load | `load.py` | `prefill:<tokens>x<n>` (cold prefills) or `decode:<tokens>x<n>` (long decodes) beside a probe |

Python 3.12, pinned to `aiperf==0.13.0`. Installing the package registers both AIPerf plugins
through the `aiperf.plugins` entry point; `aiperf plugins endpoint` lists `leaseholder_chat`.
On cyan, `deploy/cyan/bootstrap.sh` builds `~/leaseholder/venvs/bench` from `uv.lock`; for
development:

```bash
uv sync --python 3.12 --extra test      # in python/leaseholder_bench: its own .venv, from uv.lock
```

On macOS, use Python 3.13 (`--python 3.13`): `lh-bench e2 run` needs `os.waitid`, which macOS has from
3.13 on. On 3.12 it refuses to start, and its tests skip with that reason.

## Trace scaling

Each request in the corpus is a list of 64-token prefix hash blocks; the median main-agent
prompt is 254k tokens. AIPerf's own caps do not help: `--synthesis-max-isl 32768` keeps 2% of
main-agent requests and does not filter subagents, and `--max-context-length 32768` keeps no
trace. Instead, `scale-traces` shrinks the tokens per block and leaves the hash ids alone:

    block_size -> block_size / factor      in -> len(hash_ids) * new block_size      out -> ceil(out / factor)

Prefix identity, subagent forks and the cache hit rate are preserved exactly, and every request
shrinks by the same factor, outputs included, so a cache miss keeps its cost relative to decode
work. Because `ceil(ceil(out / f) / (64 / f)) == ceil(out / 64)`, AIPerf's split of each new turn
into assistant and user blocks is unchanged. Sessions, subagent entries and timing are copied
as recorded; there is no length cap.

| Factor | Context | Largest `in + out` | Use |
|---|---|---|---|
| 32 (`block_size` 2) | 32k | 31,144 | Default. Main-agent context p50 7.9k, p90 20.3k |
| 64 (`block_size` 1) | 16k | 15,572 | Smaller KV pools |

```bash
lh-bench scale-traces traces.jsonl weka-x32 --factor 32       # the HF JSONL or a directory of *.json
lh-bench scale-traces traces.jsonl weka-sub --ids ids.txt     # a subset; or --limit N
```

The full corpus takes about 22 s and 1.8 GB of RAM per factor on cyan; each output is 1.8 GB
(the hash ids are kept).

## Wire metadata

The AIPerf endpoint renders each request's metadata through
`leaseholder_bench.protocol.SessionTagger`. Names and values mirror the Rust crate `lh-proto`.

| Field | Value |
|---|---|
| Header `X-Session-ID` | The session-tree root: a parent agent and all of its subagents share it. AIPerf mints one per session play (`root_correlation_id`) |
| Header `X-LH-Class` | `realtime`, `interactive` or `background` |
| Body `prompt_cache_key` | The session id |
| Body `priority` | vLLM priority from `ENGINE_PRIORITY`: realtime 0, interactive 1, background 2. vLLM schedules lower values first |

Agent replay defaults to `background`. The plugin reads its configuration from the environment,
since AIPerf builds the endpoint itself, in every worker:

| Variable | Default | Meaning |
|---|---|---|
| `LH_BENCH_CLASS` | `background` | Class of every session |
| `LH_BENCH_SEND_PRIORITY` | `true` | `false` sends no `priority`, as a plain client would. vLLM 0.32 acts on it only under `--scheduling-policy priority`; its FCFS scheduler ignores it |

AIPerf also sends its own `X-Correlation-ID` and `X-Session-Affinity`, which identify each agent
rather than the tree. Keep `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` off: when on, AIPerf
overwrites the plugin's `X-Session-ID` with the agent's own id. vLLM 0.32 reads `X-Session-ID`
as the request's session id and ignores `prompt_cache_key`; the control plane strips client
`priority` and sets its own.

## The `weka_trace_scaled` loader

AIPerf 0.13.0 registers `weka_trace` with `default_block_size: 64` and passes that value to the
loader as an override, so the stock loader ignores a trace file's own `block_size`. On a scaled
trace it decodes 64 tokens per block from the first 1/32 of each request's hash ids. On the 32x
corpus it reproduced about 77% of the shared prefix, with prompts 22% short at the median and
no assistant turns. `weka_trace_scaled` is the same loader class with that override dropped;
on the wire it reproduced the shared prefix to within 20 tokens. `tests/test_loader.py` runs
both loaders through AIPerf's own reconstruction. With the fix, every shared prefix and every
prompt length matches the trace exactly; without it they do not.

## AIPerf `agentic_replay`

`agentic_replay` keeps N session trees live. Each lane starts at a random point of a trace,
replays it with recorded think times and subagents, then recycles a trace with a fresh
cache-bust marker. AIPerf 0.13.0 selects it only from YAML. `configs/agentic_replay.yaml` is the
template. AIPerf substitutes `${VAR:default}` when it loads the file:

| Variable | Default |
|---|---|
| `LH_BENCH_URL` | `http://127.0.0.1:8000/v1/chat/completions` |
| `LH_BENCH_API_KEY` | empty: no `Authorization` header. Through `lh-gateway`, a tenant's key. Only a trusted tenant's `X-Session-ID` and `X-LH-Class` are honored, so a replay that keeps its sessions and classes uses the `agents` key |
| `LH_BENCH_DATA` | `data/weka-x32` (a `scale-traces` output directory) |
| `LH_BENCH_MODEL`, `LH_BENCH_TOKENIZER` | `Qwen/Qwen3-4B-Instruct-2507-FP8` |
| `LH_BENCH_SESSIONS` | 16 live session trees |
| `LH_BENCH_DURATION` | 1800 s |
| `LH_BENCH_GRACE` | 120 s |
| `LH_BENCH_THINK_CAP` | 600 s cap on each think time |
| `LH_BENCH_BURST_STARTS` | `true` |
| `LH_BENCH_ARTIFACTS` | `runs/agentic-replay` |
| `LH_BENCH_RAW` | `false`; `true` also exports every request with its headers and payload |
| `LH_BENCH_SEED` | 11 |

Source `configs/aiperf.env` first. It disables AIPerf's dataset cache, which otherwise writes a
second, memory-mapped copy of each dataset under `~/.cache/aiperf` (it filled green's disk once),
keeps the session-id override off, and raises the dataset timeouts that the full corpus (about
300 s to load and 17-22 GB of RAM) exceeds. AIPerf takes these settings only from the
environment. The cache would not shorten a run anyway: AIPerf 0.13.0 caches only its own trace
types, and `weka_trace_scaled` is not one, so every run builds its dataset (about 4 minutes on
cyan).

Things to know:

- **Burst starts.** With spread starts, each lane first waits out its recorded gap after its
  random start point, and `interTurnDelayCapSeconds` does not cap that wait. A 60 s check
  reached 2 of 8 session trees and 31 requests; with burst starts it reached 8 of 8 and 236.
  Set `LH_BENCH_BURST_STARTS=false` for a faithful arrival replay.
- **Grace period.** AIPerf waits the whole grace period whenever session trees are still open,
  which in `agentic_replay` is nearly always.
- **vLLM flags.** Cached-token counts need `--enable-prompt-tokens-details`; the `priority` field
  takes effect only with `--scheduling-policy priority`.
- **Assistant history.** AIPerf replays assistant content synthesized from hash blocks, never the
  model's reply, so the KV the model writes while decoding turn k is never matched by turn k+1, as
  it is when a real agent resends its own output. AIPerf 0.13.0 threads live replies into history
  only in the `deltas_without_responses` context mode, and the Weka loader hardcodes
  `deltas_with_responses`.

## Latency probes (`lh-bench tiers`, `lh-bench voice-turn`)

`tiers` measures what resuming a session costs from each tier (E0). For each prompt length, a unique prompt of random token ids goes out three times, each asking for one token:
1. cold, so every token is recomputed;
2. again at once, a GPU prefix-cache hit;
3. after other prompts have pushed it out of the GPU pool, restored from the CPU tier.

vLLM's `prefix_cache_hits` and `external_prefix_cache_hits` counters, read around each request, verify where its tokens came from. A sample whose eviction or restore went wrong is flagged invalid instead of being mislabeled. The run fits GPU hit `a + b·L`, restore `a + b·L` and recompute `a + b·L + c·L²`; `lh-core` reads those fits.

`voice-turn` times one turn of the cascade a voice agent runs:
- STT on the utterance;
- the LLM's first sentence, counted once it reaches 20 characters as in LiveKit's tokenizer;
- TTS first audio for that sentence, sent while the reply is still streaming.

Both take `--background prefill:8192x1` or `decode:1024x8` to measure under agent load. They write `run.json`, the samples and `summary.json` to `--out`; each summary cell has the count, the mean and p10, p50, p90, p95 (where the SLOs are set) and p99. `tiers` exits 3 if any sample was invalid.

```bash
lh-bench tiers --url http://127.0.0.1:18000 --reps 5 --out runs/e0-idle
lh-bench tiers --url http://127.0.0.1:18000 --lengths 1024,4096,16384 --background decode:1024x8 --out runs/e0-decode
lh-bench voice-turn --stt-url http://127.0.0.1:18100/v1 --llm-url http://127.0.0.1:18000/v1 \
  --tts-url http://127.0.0.1:18880/v1 --audio turn.wav --reps 30 --background prefill:8192x1 --out runs/voice-x1
```

## KV block lifecycle (`lh-bench kv-metrics`)

Snapshot an engine's `/metrics` around a run, then compare vLLM's sampled KV block histograms over it. The engine needs `--kv-cache-metrics`. The histograms are `vllm:kv_block_reuse_gap_seconds`, `vllm:kv_block_idle_before_evict_seconds` and `vllm:kv_block_lifetime_seconds`.

```bash
lh-bench kv-metrics snapshot http://127.0.0.1:18000/metrics before.prom
# ... run the load ...
lh-bench kv-metrics snapshot http://127.0.0.1:18000/metrics after.prom
lh-bench kv-metrics compare before.prom after.prom
```

On the frozen box, 10 minutes of the replay above with 16 sessions (2026-10-22):

```text
vllm:kv_block_reuse_gap_seconds: n=1763 p10=0.495s p50=2.54s p90=6.33s
vllm:kv_block_idle_before_evict_seconds: n=4930 p10=1.32s p50=6.22s p90=27.3s
vllm:kv_block_lifetime_seconds: n=4930 p10=1.52s p50=7.91s p90=46.3s
evicted blocks idle no longer than the reuse-gap p50 (2.54s): 23.8%
evicted blocks idle no longer than the reuse-gap p90 (6.33s): 50.4%
```

When evicted blocks were idle for less time than blocks typically wait between reuses, LRU is evicting KV that would have been reused. Reuse gaps are recorded only for blocks that survived until reuse, so the comparison is a signal, not a bound.

## E2 points

`lh-bench e2 run ARM.toml --out DIR` measures one point of E2 on cyan: the measurement calls beside the AIPerf agentic replay, through the gateway, on the engine the GPU node runs, or on several GPU nodes. [`benchmarks/e2`](../../benchmarks/e2/README.md) holds the arm files and `campaign.sh`, which measures a list of them from a workstation and switches the GPU node's engine between them.

```bash
lh-bench e2 run ~/leaseholder/src/benchmarks/e2/arms/a1-voice.toml --out ~/leaseholder/runs/e2/w2/a1-voice
lh-bench e2 run ~/leaseholder/src/benchmarks/e2/arms/disc-a9-s24.toml --out ~/leaseholder/runs/e2/demo/disc-a9-s24 \
  --save-audio                           # the callers save each call's audio, for the demo video
lh-bench e2 engine a2-stock-s16.toml     # the engine an arm needs, as variant.sh use takes it
lh-bench e2 report ~/leaseholder/runs/e2/w2/{a1-voice,a2-stock-s16} --out ~/leaseholder/runs/e2/first
lh-bench e2 report ~/leaseholder/runs/e2/{w2c,a9,r1,r2}/disc-* --out ~/leaseholder/runs/e2/disc \
  --waterfall disc-frozen-s24 --waterfall disc-a9-s24   # repeats across rounds; two configurations' waterfall
lh-bench e2 export ~/leaseholder/runs/e2/w2b/* --out benchmarks/results/2026-10-23-e2   # compact records
lh-bench e2 reset                        # cyan's units as gateway.toml and the worker's defaults have them
```

A point runs in this order:
1. **The engine.** Its `/server_info` (on vLLM it needs `VLLM_SERVER_DEV_MODE=1`, which `lh-engine.service` sets) gives the server, its version, the KV pool and its settings. They must be those of the engine's command line as `variant.sh use` makes it, or the point fails before anything starts and names the `variant.sh use` command that fixes it: on vLLM the frozen box's policy, step cap and long-prefill threshold, then the flags of the command the variant's `lh-engine.conf` starts, then the arm's `[engine]` flags, the last of a repeated flag winning. On vLLM the scheduler class (`scheduler_cls`) must be the one `--scheduler-cls` names, vLLM's own without it, and `additional_config` exactly what the `--additional-config.KEY` flags build (make-room's switches), so that a frozen, make-room or pr54530 engine, or make-room with another switch, cannot pass for another. A flag whose setting the engine does not report is recorded as unchecked.
2. **cyan's units.** SMG runs only for `router = "smg"`. With a `[gateway]` table the gateway runs on `~/.config/leaseholder/gateway-e2.toml` (`gateway.toml` with the table merged in, mode 600) through the `e2-gateway` drop-in, and without one on `gateway.toml`. The voice worker's preemptive generation is switched with the `no-preemptive-generation` drop-in, and its lifecycle events with `no-events`. Only a unit whose configuration changes restarts, and the worker must register again within 90 s. The worker must end turns with fixed endpointing (`vad-turns`), or nothing runs.
3. **A cold start.** The engine's prefix cache, and vLLM's CPU tier, are emptied.
4. **The replay.** With agent sessions, AIPerf starts on cores 16–23 and the calls wait for its profiling phase plus the warm-up.
5. **The window.** The gateway's and the engine's `/metrics` are saved, `lh-voice bots` runs in the voice-plane namespace on cores 24–31, and they are saved again. The calls are the window every metric covers; AIPerf must outlast them. With `--save-audio`, a run option rather than the arm's, the callers also save each call's audio for the demo video (`lh-voice bots --save-audio`), which measures nothing differently.
6. **The end.** AIPerf gets SIGINT, cancels its requests and writes its exports. The gateway's request lines for the window go to `gateway.jsonl`, and the voice worker's journal from the start of the calls to `voice-worker.log`, once every call's session has logged its end (at most 30 s). Then `summary.json` is written.

**A point on several GPU nodes.** An arm whose `[gateway]` names GPU nodes (`[gateway.nodes.<name>]`, each with the cyan end of its tunnel as `upstream`, and `[gateway.placement]`) runs on all of them, each on the arm's `[engine]`. First the run starts each node's tunnel, `lh-tunnel-<name>`, and waits up to 60 s for the node's engine to answer `/health` through it; a tunnel that runs is left alone, and none is stopped after the point. Each engine is then checked (step 1), emptied (step 3) and snapshotted (step 5). The gateway serves the nodes in place of `gateway.toml`'s `upstream`, and the voice worker gets them as `LH_NODES` from a drop-in the run writes, `variant-nodes.conf`, so that each call's STT and TTS follow its node; the next point on one node, or `lh-bench e2 reset`, removes it. A node is named after its host's first label (`phastform`, `gray`), which also names its tunnel unit, its sockets on cyan and its files in the run directory.

**The arm file.** Unknown keys and wrong types are errors. The point is named after the file.

| Key | Default | Meaning |
|---|---|---|
| `arm` | required | E2's arm, which names the point's frontier: 1 to 10, then a letter for a sub-arm (`"7a"`) or a suffix for a factor (`"4-no-preemptive"`) |
| `label` | required | What the point is, for the report |
| `router` | none | `"smg"` starts `lh-smg` (arm 8b); the `[gateway]` table points the gateway at it |
| `[voice] callers`, `turns` | 2, 25 | Concurrent calls, and turns per call |
| `[voice] barge_in` | 0.25 | Share of turns on which the caller interrupts the reply |
| `[voice] seed` | 0 | The callers' seed |
| `[voice] warmup_turns` | 0 | Turns that open each call before its `turns`, played but not measured (`lh-voice bots --warmup-turns`): marked `"warmup": true` in `bots/turns.jsonl`, they are in no V2V sample, and the worker's lines for them, each session's first, are in no stage |
| `[voice] preemptive_generation` | `true` | LiveKit's preemptive generation in the voice worker (`LH_PREEMPTIVE_GENERATION`) |
| `[voice] events` | `true` | The voice worker's lifecycle events to the gateway (`LH_EVENTS`); `false` measures a point without them |
| `[agents] sessions` | 0 | Session trees AIPerf keeps live, through the gateway as the `agents` tenant with `LH_GATEWAY_AGENTS_KEY` from `~/.config/leaseholder/gateway-keys.env`; 0 is voice alone |
| `[agents] think_cap` | 120 | Cap on each recorded think time, in seconds |
| `[agents] warmup` | 60 | Seconds of profiling before the calls start |
| `[agents] data` | `"weka-x32"` | The corpus, a `scale-traces` directory under `~/leaseholder/data` |
| `[agents] seed` | 11 | AIPerf's seed, which a campaign's `agents-seed` line replaces for its points (`lh-bench e2 run --agents-seed`) |
| `[gateway]` | none | Merged into `gateway.toml` as it is, tables key by key: `upstream`, `upstream_priority`, `[gateway.admission]` and any other key of [`lh-gateway`'s configuration](../../crates/lh-gateway/README.md) |
| `[gateway.nodes.<name>]`, `[gateway.placement]` | none | A point on several GPU nodes: two or more, each with an `http://` `upstream`, and how the gateway places sessions on them ([placement](../../crates/lh-gateway/README.md#placement)). It takes no `router` |
| `[engine] variant` | `"frozen"` | A directory of [`deploy/phastform/variants`](../../deploy/phastform/variants/README.md), or `frozen` |
| `[engine] flags` | `[]` | Engine flags, each a plain word, added to its command line |

**The run directory.**

| File | Contents |
|---|---|
| `arm.toml` | The arm file as run |
| `run.json` | `status` (`running`, then `complete`, or `failed` with the `error`); the host; start and end; the arm as read; `save_audio`, the run option; `agents_seed`, the seed AIPerf replayed with (the campaign's `--agents-seed`, else the arm file's; null for voice alone); versions (cyan's deployed commit from `~/leaseholder/src/DEPLOYED`, each GPU node's beside it as `gpu_source` (by node, from `--gpu-source NODE=REVISION`, which `campaign.sh` reads from the node's own `DEPLOYED` as the point starts; null when nothing named it), `leaseholder-bench`, AIPerf, `lh-gateway`); the engine (server, version, KV pool, flags checked and unchecked; by node on several GPU nodes); what cyan ran (which units restarted, the gateway's configuration without key digests, the worker's nodes); the window; AIPerf's exit status |
| `run.log` | Each step, timestamped |
| `engine.json` | The engine's whole configuration from `/server_info`, without credentials; `engine-NODE.json` for each node on several |
| `aiperf.log`, `aiperf/` | AIPerf's output and artifacts: `profile_export.jsonl` holds one record per request |
| `bots.log`, `bots/` | The callers' output and run directory (`turns.jsonl`, `summary.json`, and with `--save-audio` each call's `call-<i>.wav` and `call-<i>.json`) |
| `metrics/` | `gateway-start.prom`, `gateway-end.prom`, `engine-start.prom`, `engine-end.prom`; `engine-NODE-start.prom` and `engine-NODE-end.prom` for each node on several |
| `gateway.jsonl` | The gateway's request lines for the window (`journalctl --user -u lh-gateway --since --until`, the JSON lines whose message is `request`): class, status, outcome, `first_byte_ms`, `total_ms`, `admission_wait_ms`, tokens and, on several GPU nodes, the `node` of each request. The first bytes come from them: a gateway that logs text (`log_format` other than `"json"`) logs none, and the point fails when it is summarized |
| `voice-worker.log` | The voice worker's journal from the start of the calls: a line per turn with its stage times, a line per session at its end with its lifecycle-event counts ([`leaseholder-voice`](../leaseholder_voice/README.md#lifecycle-events-and-turn-lines)) and, on several GPU nodes, a line per call placed, with its room and node |
| `gpu.csv` | Written by `campaign.sh`, not by `e2 run`: the GPU node's `nvidia-smi` samples, once a second while the point ran (timestamps in UTC, SM and memory clocks, temperature, power, clock event reasons); `gpu-NODE.csv` for each node on several |
| `summary.json` | The point's E2 metrics |

A component that fails fails the point: `run.json` keeps `failed` and the error, and a tool's error gives its exit status and the last lines of its log. AIPerf and the callers run in process groups of their own and are stopped through them, with SIGINT and then SIGKILL for whatever is left, also when the point fails or `lh-bench e2 run` gets SIGINT or SIGTERM. Closing the SSH session that started it does not stop a point. A complete point is never overwritten; an incomplete one is moved aside to `DIR.incomplete-<UTC time>`, and a directory that holds anything else is refused. One `e2 run` or `e2 reset` runs at a time; the lock file, `~/leaseholder/runs/.lh-bench-e2.lock`, holds its pid.

**`summary.json`.** Every metric covers the calls. A metric the gateway or the engine does not export is null, never zero, and a counter that went down (its process restarted during the calls) fails the point.

| Key | Contents |
|---|---|
| `window_s` | Length of the calls |
| `voice` | From the callers: turns, errors, `v2v_ms` and `barge_in_to_silence_ms` (n, p50, p95, p99) and barge-ins |
| `worker` | Null without the worker's lines. From `voice-worker.log`: `sessions` that logged their end, `turns` (without each session's first `[voice] warmup_turns` turn lines, its replies to the warm-up turns), the `waterfall` (each stage of the turn lines: `transcript_ms`, `end_of_turn_ms`, `llm_ttft_ms`, `llm_first_sentence_ms`, `turn_to_first_sentence_ms`, `tts_first_audio_ms`, `e2e_ms`, with n, mean, p10, p50, p90, p95, p99) and the lifecycle `events` its sessions sent, dropped and failed to deliver (null when it sent none: `events = false`, or the LLM bypassed the gateway) |
| `background` | Null for voice alone. From AIPerf's records of the agent requests that ended during the calls: requests, errors (failed and cancelled requests count as lost), requests/s, prompt, completion and total tokens/s from the server's usage counts, and `ttft_ms`; and the same goodput over the calls' first 240 s (`GOODPUT_WINDOW_S`), the same span for every point, where the calls' own length differs by configuration (`fixed_window`: its `seconds`, fewer for calls that lasted less, `requests`, `requests_per_s` and `tokens_per_s`) |
| `gateway` | `realtime_first_byte_ms` and `background_first_byte_ms` (n, mean, p50, p95, p99): exact, from `gateway.jsonl` (`"source": "log"`); what its admission point shed, refused and resubmitted, and the prompt tokens of the attempts it dropped; `background_tokens_per_s`, its own count from response usage; and every counter that moved |
| `engine` | vLLM's preemptions; its prompt tokens by source at their first scheduling (local compute, GPU-pool and CPU-tier hits; vLLM 0.32 counts none at a preempted request's resume); queue-limit rejections; and the CPU tier's own loads into the GPU pool and stores from it, every one of them, in bytes (`kv_offload_load_bytes`, `kv_offload_store_bytes`, from `vllm:kv_offload_load_bytes_total` and `vllm:kv_offload_store_bytes_total`); on several GPU nodes, added up over them |
| `nodes` | On several GPU nodes only, each node's share. `voice`: the calls the worker placed on the node, from their callers' `bots/turns.jsonl` (calls, turns, errors, `v2v_ms`, `barge_in_to_silence_ms`, barge-ins; null without the worker's placement lines; a call whose placement failed counts in the totals only). `background`: the agent requests the gateway sent the node, from `gateway.jsonl` (requests served, errors, requests/s, tokens/s, `first_byte_ms`). `realtime_first_byte_ms`: its realtime requests' first byte, from the log. `engine`: its engine's counts. In a compact record, `gpu` as well: its GPU's, from `gpu-NODE.csv` |
| `gpu` | Only in a compact record (`lh-bench e2 export`): from `gpu.csv`, the GPU's median SM clock (`sm_clock_p50_mhz`) and highest temperature (`temperature_max_c`) during the calls; on several GPU nodes, over every node's `gpu-NODE.csv` |

The background goodput comes from AIPerf's records rather than from the gateway's request outcomes, whose `completed` counts a response relayed in full whatever its status: it cannot tell a request the engine refused from one it served. The gateway's token rate stays beside it as a check. A node's share, which AIPerf cannot tell apart, counts the gateway's request lines that completed with status 200.

**The report.** `lh-bench e2 report RUN_DIR... --out DIR` writes `frontier.csv`, `frontier.md` and, where matplotlib is installed (AIPerf brings it), a frontier figure per workload and `waterfall.svg`. A point's workload is the agents' corpus (`[agents] data`) and the number of callers (`frontier.csv`'s `workload`, such as `weka-x32, 2 callers`), and points of different workloads are never compared: a point is on its arm's frontier when no other point of the arm and workload harvests at least as much background goodput at a V2V p95 at least as low, and one of the two strictly better. `frontier.md` has a section per workload, which opens with its voice SLO and the name of its figure, with a table per arm in it, ordered by background goodput, where • marks the arm's frontier when the arm has more than one configuration (a lone configuration is its own). Run directories without a complete point are named and left out.
- **The frontier figures.** `frontier.svg` draws the arm file's own workload (`weka-x32`, 2 callers); another workload's figure is named after its corpus and its callers where they differ (`frontier-weka-x32-long.svg`, `frontier-4-callers.svg`). An arm's frontier is drawn in its family's color with the arm's marker and joined by a line where it spans load levels (the session trees an arm is swept over); configurations at one load, as in the discriminating scenario, are markers. Open markers are an arm's dominated configurations, and the arm's name sits by the last point of its frontier, at the end of a leader line where points crowd (arms whose points coincide share one, `6, 8b`; the names of an arm's variants that find no clear place are left out, and the figure's notes say how many). The color is the arm's family: stock vLLM (arms 1–5), the baseline, in dark gray; operator answers (6), engines (7), routers (8), Leaseholder (9, blue) and MPS (10) in five hues that stay apart from one another in every pair, as a scatter needs, for color-blind readers too. The marker tells the arms of a family apart. A dashed line marks the voice SLO: 250 ms above voice alone (arm 1) with as many callers, from whichever corpus, since voice alone replays none; a workload without that point says so. A configuration past its workload's fence for far outliers (three interquartile ranges above the upper quartile of the V2V p95s) is left off the scale and named with its numbers, so that one collapse does not flatten the rest.
- **Configurations of their own.** Arm 9's ablations and the vLLM PRs' configurations say `arm = 9` in their arm files, as run, but each is a configuration of its own: the report gives each its own arm (`9-no-headroom`, `9-pr-ab48`; `CONFIGURATION_ARMS` in `e2/report.py`), so that none is placed on arm 9's frontier or joined to it. The records stay as run.
- **Repeats.** Points measured from the same arm file in several campaigns, under the same point name, are repeats of one configuration. `frontier.md` keeps a row per run, named by its campaign (`frontier.csv`'s `campaign`: the directory that holds the run directory), then gives the configuration's median and range (min–max) of V2V p50/p95/p99, background tokens/s (over the whole calls and over their first 240 s), the agents' TTFT p50 and p99, the realtime first byte p95 and failed turns, with its run count. The frontier places the configuration at its medians, and its figure draws it there with whiskers over its runs' range on both axes: one marker per configuration keeps each frontier a line through what `frontier.md` marks, and a run that collapsed still shows as a long whisker. A configuration measured once is shown as before.
- **The goodput over a fixed window.** Beside each point's background goodput over its whole calls, whose length differs by configuration, `frontier.md` gives its goodput over their first 240 s (`frontier.csv`'s `fixed_window_tokens_per_s`, with its `fixed_window_s`), and for a repeated configuration their median and range; † marks one over a point's whole calls, which lasted less. The frontier stays placed by the goodput over the whole calls.
- **Failed turns.** V2V comes from the callers' measured turns in `bots/turns.jsonl` (not the warm-up turns that open each call under `[voice] warmup_turns`), where a turn the agent did not answer within the callers' reply timeout counts as a sample censored at 30,000 ms, in every percentile: `frontier.md` marks a percentile such a sample sets with `≥`. A configuration with failed turns is not placed on the frontier and keeps no other point off it; its figure draws it as a cross and its notes name it with its failed turns, and the stage waterfall leaves its runs out.
- **The realtime first byte** is exact from the run directory's `gateway.jsonl`, even one fetched after the run (from the journal, for the window in `run.json`), and otherwise the summary's, which took it from the same log; `realtime_first_byte_source` says so (`log`, in every record).
- **The stage waterfall** (`waterfall.svg`, and its table in `frontier.md`) splits each configuration's voice turn into stages that follow one another, as means over the replies of its runs: endpointing (`end_of_turn_ms`), the reply to its first sentence (`turn_to_first_sentence_ms`) and TTS to first audio (`tts_first_audio_ms`), from the worker's turn lines, then the rest of the callers' V2V, transport and playout (the mean V2V less the three). So a bar is as long as the mean V2V; a line marks the median V2V and a triangle the p95, over the callers' samples in `bots/turns.jsonl`. A thin bar under it is the LLM's own time to first token (`llm_ttft_ms`), from its request: under preemptive generation the request starts at the final transcript, before the end of the turn, so it lies across the first two stages. `--waterfall POINT` (repeatable) draws only those configurations, in that order; by default every one with turn lines.
- **Columns only some points have** appear when one of the points has them: the worker's stage times (p50 and p95 of each, in a table of their own in `frontier.md`) and event counts; from `gpu.csv` (or a compact record's summary), the GPU's median SM clock and highest temperature during the calls; and under the gateway's policy `leaseholder`, from its counters, what the policy did: the background requests it deferred by what held them back when they arrived (`deferred_headroom`, `deferred_critical`) and the realtime replies it capped (`degraded_replies` of `realtime_replies`, `degraded_share`), in a table of their own in `frontier.md`. A mechanism that deferred or capped nothing in a point did not act there. On vLLM, the engine's prompt tokens by source at their first scheduling (`first_schedule_computed_tokens`, `first_schedule_gpu_hit_tokens`, `first_schedule_cpu_hit_tokens`) and the CPU tier's loads and stores of every scheduling, resumes included (`cpu_tier_loaded_tokens`, `cpu_tier_stored_tokens`), in tokens at the bytes of KV a token takes in the engine's pool, its KV cache's bytes over its tokens from the point's `engine.json` (`kv_bytes_per_token`, 147,492 on the frozen box against the model's exact 147,456), in a table of their own in `frontier.md` with what the CPU tier loaded beyond first scheduling: the loads of requests resumed after a preemption, give or take those of requests whose load and first scheduling are not both within the calls.
- **A point on several GPU nodes** is placed by its totals, and says so: its label in `frontier.md` gives the number of nodes, the `nodes` column of `frontier.csv` names them, a table of its own in `frontier.md` gives each node's share (a line per run and node, named by its campaign when the point is repeated), and its GPU columns cover every node's GPU.

**The compact record.** `lh-bench e2 export RUN_DIR... --out DIR` writes each complete point's record into `DIR/CAMPAIGN/POINT`, where its run directory sits in its campaign: `run.json` (with `versions.gpu_source` null where the run did not record it), `arm.toml`, the callers' `summary.json` and `turns.jsonl`, a `summary.json` that keeps what the report takes from the files the record leaves out (the first bytes of both classes exactly from `gateway.jsonl`, `"source": "log"`, from `gpu.csv` the GPU's clock and temperature, `gpu`, from AIPerf's records the goodput over the calls' first 240 s, and from the engine's metric snapshots the CPU tier's loads and stores, both of which a point measured before its summary kept them gains this way), and an `engine.json` reduced to what says which engine the point ran on. For vLLM that is the model by its Hugging Face id and revision, `max_model_len`, dtype and quantization; from `cache_config` the block size, prefix caching, the KV dtype and the pool (`kv_cache_memory_bytes`, `num_gpu_blocks`); from `scheduler_config` the policy, `max_num_batched_tokens`, `long_prefill_token_threshold`, `max_num_seqs`, async scheduling, `watermark`, `max_num_queued_tokens` and the scheduler class (`scheduler_cls`, null for vLLM's own), the settings `e2 run` checks the arms' flags against; the CPU tier's connector with its `kv_connector_extra_config`; and `additional_config`. For SGLang it is the same of its server arguments (`SGLANG_ENGINE` in `e2/record.py`). A point on several GPU nodes keeps an `engine-NODE.json` for each node, and its summary the clock and temperature of every node's GPU together (`gpu`) and of each in the node's share. Any path under a home directory is written from `~`. A record is about 15 kB and holds no credential: the callers' `run.json`, which names LiveKit's API key, stays out. The report gives the same tables and figures from the records as from the run directories, byte for byte (its figures carry no date), so the records are what goes into the repository, under [`benchmarks/results/2026-10-23-e2`](../../benchmarks/results/2026-10-23-e2/README.md), and `make reproduce-e2` regenerates the report from them with this venv. The raw run directories (the gateway's request log, AIPerf's exports, the metric snapshots, the worker's journal, the GPU's samples, the tools' logs) are an asset of the [v0.3 release](https://github.com/StevenWang-CY/leaseholder/releases/tag/v0.3), `leaseholder-e2-runs-2026-10-24.tar.gz`.

## Mock server

`lh-bench mock-server --port P --log FILE` binds 127.0.0.1 only and streams chat completions.
Each request appends one JSON line to the log: arrival and end times, every header, every body
field except `messages`, a role/size digest of the messages, and `cached_tokens` (the longest
prefix shared with one of the session's recent prompts, also returned in `usage`).

## Quickstart (cyan)

```bash
# Environment: the venv that deploy/cyan/bootstrap.sh builds
export PATH=$HOME/leaseholder/venvs/bench/bin:$PATH

# Data
cd ~/leaseholder/data
curl -sSL --fail -o cc-traces-weka-062126/traces.jsonl --create-dirs \
  https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl
lh-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32 --factor 32
lh-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x64 --factor 64
# A quick-check subset; sub16.ids lists 16 short traces, one id per line (8 with subagents)
lh-bench scale-traces cc-traces-weka-062126/traces.jsonl weka-x32-sub16 --factor 32 --ids sub16.ids

# Mock on loopback (keep clear of cyan's deployed ports 18000, 18080, 18100, 18880 and 17880-17882)
mkdir -p ~/leaseholder/runs/w0-e2e && cd ~/leaseholder/runs/w0-e2e
setsid nohup lh-bench mock-server --port 19180 --log $PWD/mock-aiperf.jsonl < /dev/null > mock.out 2>&1 &
echo $! > mock.pid
ss -lntp | grep 19180          # 127.0.0.1:19180 only

# AIPerf agentic_replay, 60 s
cd ~/leaseholder/src/python/leaseholder_bench
. configs/aiperf.env
LH_BENCH_URL=http://127.0.0.1:19180/v1/chat/completions LH_BENCH_DATA=$HOME/leaseholder/data/weka-x32-sub16 \
LH_BENCH_SESSIONS=8 LH_BENCH_DURATION=60 LH_BENCH_GRACE=15 LH_BENCH_THINK_CAP=3 LH_BENCH_RAW=true \
LH_BENCH_ARTIFACTS=$HOME/leaseholder/runs/w0-e2e/aiperf \
  aiperf profile --config configs/agentic_replay.yaml

# What reached the server
python -c 'import json,sys; r=json.loads(open(sys.argv[1]).readlines()[-1]); print(r["headers"], r["body"])' \
  ~/leaseholder/runs/w0-e2e/mock-aiperf.jsonl
kill $(cat ~/leaseholder/runs/w0-e2e/mock.pid)
```

Results on 2026-10-22, against the mock:

| Run | Requests | Every request carries the four fields | `X-Session-ID` is the tree root |
|---|---|---|---|
| AIPerf, 8 session trees, 60 s | 7 warmup + 236, 0 errors | 243/243 | 243/243 (103 subagent requests) |

## Tests

The tests are offline and take a few seconds. CI runs them with `make test-python`:

```bash
uv run --extra test pytest      # in python/leaseholder_bench
```

They cover the scaling math and schema, the loader's prefix reproduction (both loaders, through
AIPerf's own reconstruction with a tokenizer-free prompt generator), the plugin's headers and
body as AIPerf's transport sends them, the wire names and class-to-priority mapping against
`lh-proto`, the YAML template and the probes against in-process fake servers. `tests/test_e2.py`
runs whole E2 points against a fake gateway and engine on
loopback, a recording `systemctl` and stand-ins for AIPerf and the callers that run as real
processes (`tests/fakes/`): the sequence, the units each arm restarts, the engine check, failures
at each step with every process stopped, the point directory's rules, the committed arm files, and
the logs a point keeps: the gateway's request lines and the worker's journal, waiting for its
sessions to end only when they will. `tests/test_e2_results.py` covers the summary over the window,
the exact first bytes from a request log, the Pareto marking within each arm and workload, each
workload's section, voice SLO and figure (its families, the arms' names and what it leaves off its
scale), repeated points placed at their medians with every run kept, the stage waterfall's means
and V2V marks, the report's columns that only some points have, compact records that give the
report of their run directories byte for byte, the committed records (complete, without
credentials, exported again to themselves, and reported as `make reproduce-e2` reports them), and the names of the lines the voice
worker writes. `tests/test_campaign.py` runs `benchmarks/e2/campaign.sh` against stand-ins for the hosts (an ssh that runs each command in a directory per host, rsync, cyan's `lh-bench` and the GPU node's `variant.sh`): what each point is measured with. `tests/test_demo.py` runs the demo
video's renderer on a synthetic pair of calls ([`benchmarks/e2`](../../benchmarks/e2/README.md#the-demo-video));
its encoding test is marked `integration` and left out of CI.
