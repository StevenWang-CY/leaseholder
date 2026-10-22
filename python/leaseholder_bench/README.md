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
| `lh-bench replay` | `replay.py` | Live-reply driver: feeds the model's actual reply into the next turn |
| `lh-bench mock-server` | `mock_server.py` | Recording OpenAI-compatible mock on 127.0.0.1 for CPU-only dry runs |
| `lh-bench tiers` | `tiers.py` | E0: time to first token from the GPU pool, the CPU tier and recompute, verified per sample by vLLM's prefix-cache counters; optional background load |
| `lh-bench voice-turn` | `voice_turn.py` | One server-side voice turn timed per stage (STT, LLM first sentence, TTS first audio); optional background load |
| `lh-bench kv-metrics` | `kv_metrics.py` | KV block lifecycle histograms compared across a run |
| `lh-bench e2` | `e2.py`, `frontier.py` | E2 points: one measured from its arm file on cyan (`run`), and the frontier across points (`report`) |
| background load | `load.py` | `prefill:<tokens>x<n>` (cold prefills) or `decode:<tokens>x<n>` (long decodes) beside a probe |

Python 3.12, pinned to `aiperf==0.13.0`. Installing the package registers both AIPerf plugins
through the `aiperf.plugins` entry point; `aiperf plugins endpoint` lists `leaseholder_chat`.
On cyan, `deploy/cyan/bootstrap.sh` builds `~/leaseholder/venvs/bench` from `uv.lock`; for
development:

```bash
uv sync --python 3.12 --extra test      # in python/leaseholder_bench: its own .venv, from uv.lock
```

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

Both the AIPerf endpoint and the replay driver render requests through
`leaseholder_bench.protocol.SessionTagger`, so they send identical metadata. Names and values
mirror the Rust crate `lh-proto`.

| Field | Value |
|---|---|
| Header `X-Session-ID` | The session-tree root: a parent agent and all of its subagents share it. AIPerf mints one per session play (`root_correlation_id`); the driver mints a UUID per play |
| Header `X-LH-Class` | `realtime`, `interactive` or `background` |
| Body `prompt_cache_key` | The session id |
| Body `priority` | vLLM priority from `ENGINE_PRIORITY`: realtime 0, interactive 1, background 2. vLLM schedules lower values first |

Agent replay defaults to `background`. Configuration, read from the environment by the AIPerf
plugin (AIPerf builds the endpoint itself, in every worker) and by `lh-bench replay`, whose
`--class`, `--class-map` and `--no-priority` flags override it:

| Variable | Default | Meaning |
|---|---|---|
| `LH_BENCH_CLASS` | `background` | Class of every session |
| `LH_BENCH_CLASS_MAP` | unset | JSON file `{trace id: class}`; overrides the default per trace |
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
- AIPerf replays a synthesized assistant history, never the model's reply. See below.

## Live-reply driver

AIPerf's history replays assistant content synthesized from hash blocks, so the KV the model
writes while decoding turn k is never matched by turn k+1, as it is when a real agent resends its
own output. E1 and E3 need that reuse. Options considered:

- **SGLang's multi-turn benchmarks.** `benchmark/hicache/bench_multiturn.py` and
  `python -m sglang.benchmark.serving --dataset-name agentic-trace` both append the generated
  text to the next round. Neither can replay this workload. `bench_multiturn.py` synthesizes
  random or ShareGPT rounds of set lengths at a global Poisson rate. `agentic-trace` needs full
  message text (OpenHands-style JSON) and sends rounds back to back with one output length for
  every turn. Neither has per-session think time, subagents or per-session headers, and both
  import the `sglang` package.
- **An AIPerf option.** There is none for Weka traces. AIPerf threads live replies into history
  only in the `deltas_without_responses` context mode. The Weka loader hardcodes
  `deltas_with_responses`, whose deltas already carry synthesized assistant blocks, and no
  setting changes it. The live-reply mode also assumes linear traversal, which `agentic_replay`'s
  mid-trace starts break. Using it would mean forking AIPerf's Weka reconstruction (about 3,600
  lines).
- **Our own thin driver.** It reads the same scaled trace files and keeps their structure,
  in one module of under 400 lines.

**Decision: our own driver, `lh-bench replay`.** AIPerf stays the tool for steady-state
background load (`agentic_replay`, E2). The driver serves E1 and E3, where decode-written KV
reuse, `offload_prompt_only` and chat-template re-render breaks matter. On cyan, 248 of 248
driver requests with assistant history carried the model's earlier replies; 0 of 197 AIPerf
requests did.

How it works:

- **Prompt text.** Each hash block becomes `block_size` words, seeded by trace and hash id. Equal
  hash ids give equal text, so block-level prefix identity holds on the wire. Every word is a
  single Qwen3 token, so prompts track the trace's `in`. A per-play system line keeps two plays
  of one trace from sharing KV.
- **History.** Each request continues the chain of its agent whose last prompt shares the
  longest hash prefix with it, or starts a new one, as AIPerf does for interleaved side chains.
  If the request extends that prompt, the live reply stands for the first `ceil(out / block_size)`
  new blocks and the rest become one user message; a reply longer than the extension counts
  against the next request's new blocks. If it keeps only part of it (a compaction or an edit),
  the messages inside the shared prefix stay and the rest is resent.
- **Decode length.** Each request asks for the trace's `out` tokens with `ignore_eos`, so
  contexts grow as recorded (`--no-ignore-eos` to let the model stop).
- **Timing.** Think times are waited out, capped by `--think-cap`. Subagents run concurrently
  under the parent's session id. A parent waits for a subagent only where the trace shows it
  did, that is, the subagent ended before the parent's next request.
- **Load.** `--sessions` lanes each replay one trace at a time from turn 0, cycling through the
  directory in seeded random order. No new request starts after `--duration`, and in-flight
  requests are cancelled after `--grace`.

Output: one JSON line per request with `start` (wall-clock send time), `session_id`, `trace_id`,
`agent_id` (null for the main agent), `chain`, `turn`, `class`, `status`, `error`, `trace_in`,
`trace_out`, `messages`, `prompt_tokens`, `cached_tokens`
(`usage.prompt_tokens_details.cached_tokens`), `completion_tokens`, `ttft_ms` and `latency_ms`.
A summary prints at the end, and the command exits 1 if no request succeeded.

Not yet: mid-trace starts (sessions start at turn 0, so a run should span several session
lengths), and parallel replay of side chains within one agent (they run in recorded order).

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

`lh-bench e2 run ARM.toml --out DIR` measures one point of E2 on cyan: the measurement calls beside the AIPerf agentic replay, through the gateway, on the engine the GPU node runs. [`benchmarks/e2`](../../benchmarks/e2/README.md) holds the arm files and `campaign.sh`, which measures a list of them from a workstation and switches the GPU node's engine between them.

```bash
lh-bench e2 run ~/leaseholder/src/benchmarks/e2/arms/a1-voice.toml --out ~/leaseholder/runs/e2/w2/a1-voice
lh-bench e2 engine a2-stock-s16.toml     # the engine an arm needs, as variant.sh use takes it
lh-bench e2 report ~/leaseholder/runs/e2/w2/{a1-voice,a2-stock-s16} --out ~/leaseholder/runs/e2/first
lh-bench e2 reset                        # cyan's units as gateway.toml and the worker's defaults have them
```

A point runs in this order:
1. **The engine.** Its `/server_info` (on vLLM it needs `VLLM_SERVER_DEV_MODE=1`, which `lh-engine.service` sets) gives the server, its version, the KV pool and its settings. The arm's `[engine]` flags, and on vLLM the frozen box's policy, step cap and long-prefill threshold wherever the flags say nothing else, must match them, or the point fails before anything starts and names the `variant.sh use` command that fixes it. A flag whose setting the engine does not report is recorded as unchecked.
2. **cyan's units.** SMG runs only for `router = "smg"`. With a `[gateway]` table the gateway runs on `~/.config/leaseholder/gateway-e2.toml` (`gateway.toml` with the table merged in, mode 600) through the `e2-gateway` drop-in, and without one on `gateway.toml`. The voice worker's preemptive generation is switched with the `no-preemptive-generation` drop-in. Only a unit whose configuration changes restarts, and the worker must register again within 90 s. The worker must end turns with fixed endpointing (`vad-turns`), or nothing runs.
3. **A cold start.** The engine's prefix cache, and vLLM's CPU tier, are emptied.
4. **The replay.** With agent sessions, AIPerf starts on cores 16–23 and the calls wait for its profiling phase plus the warm-up.
5. **The window.** The gateway's and the engine's `/metrics` are saved, `lh-voice bots` runs in the voice-plane namespace on cores 24–31, and they are saved again. The calls are the window every metric covers; AIPerf must outlast them.
6. **The end.** AIPerf gets SIGINT, cancels its requests and writes its exports, and `summary.json` is written.

**The arm file.** Unknown keys and wrong types are errors. The point is named after the file.

| Key | Default | Meaning |
|---|---|---|
| `arm` | required | E2's arm, which names the point's frontier: 1 to 10, then a letter for a sub-arm (`"7a"`) or a suffix for a factor (`"4-no-preemptive"`) |
| `label` | required | What the point is, for the report |
| `router` | none | `"smg"` starts `lh-smg` (arm 8b); the `[gateway]` table points the gateway at it |
| `[voice] callers`, `turns` | 2, 25 | Concurrent calls, and turns per call |
| `[voice] barge_in` | 0.25 | Share of turns on which the caller interrupts the reply |
| `[voice] seed` | 0 | The callers' seed |
| `[voice] preemptive_generation` | `true` | LiveKit's preemptive generation in the voice worker (`LH_PREEMPTIVE_GENERATION`) |
| `[agents] sessions` | 0 | Session trees AIPerf keeps live; 0 is voice alone |
| `[agents] target` | `"gateway"` | `"gateway"`, as the `agents` tenant with `LH_GATEWAY_AGENTS_KEY` from `~/.config/leaseholder/gateway-keys.env`, or `"engine"`, straight through the tunnel |
| `[agents] think_cap` | 120 | Cap on each recorded think time, in seconds |
| `[agents] warmup` | 60 | Seconds of profiling before the calls start |
| `[agents] data` | `"weka-x32"` | The corpus, a `scale-traces` directory under `~/leaseholder/data` |
| `[agents] seed` | 11 | AIPerf's seed |
| `[agents] send_priority` | `true` | The plugin's body `priority`. The gateway strips it, so it reaches only an engine targeted directly |
| `[gateway]` | none | Merged into `gateway.toml` as it is, tables key by key: `upstream`, `upstream_priority`, `[gateway.admission]` and any other key of [`lh-gateway`'s configuration](../../crates/lh-gateway/README.md) |
| `[engine] variant` | `"frozen"` | A directory of [`deploy/phastform/variants`](../../deploy/phastform/variants/README.md), or `frozen` |
| `[engine] flags` | `[]` | Engine flags, each a plain word, added to its command line |

**The run directory.**

| File | Contents |
|---|---|
| `arm.toml` | The arm file as run |
| `run.json` | `status` (`running`, then `complete`, or `failed` with the `error`); the host; start and end; the arm as read; versions (the deployed commit from `~/leaseholder/src/DEPLOYED`, `leaseholder-bench`, AIPerf, `lh-gateway`); the engine (server, version, KV pool, flags checked and unchecked); what cyan ran (which units restarted, the gateway's configuration without key digests); the window; AIPerf's exit status |
| `run.log` | Each step, timestamped |
| `engine.json` | The engine's whole configuration from `/server_info`, without credentials |
| `aiperf.log`, `aiperf/` | AIPerf's output and artifacts: `profile_export.jsonl` holds one record per request |
| `bots.log`, `bots/` | The callers' output and run directory (`turns.jsonl`, `summary.json`) |
| `metrics/` | `gateway-start.prom`, `gateway-end.prom`, `engine-start.prom`, `engine-end.prom` |
| `summary.json` | The point's E2 metrics |

A component that fails fails the point: `run.json` keeps `failed` and the error, and a tool's error gives its exit status and the last lines of its log. AIPerf and the callers run in process groups of their own and are stopped through them, with SIGINT and then SIGKILL for whatever is left, also when the point fails or `lh-bench e2 run` gets SIGINT or SIGTERM. Closing the SSH session that started it does not stop a point. A complete point is never overwritten; an incomplete one is moved aside to `DIR.incomplete-<UTC time>`, and a directory that holds anything else is refused. One `e2 run` or `e2 reset` runs at a time; the lock file, `~/leaseholder/runs/.lh-bench-e2.lock`, holds its pid.

**`summary.json`.** Every metric covers the calls. A metric the gateway or the engine does not export is null, never zero, and a counter that went down (its process restarted during the calls) fails the point.

| Key | Contents |
|---|---|
| `window_s` | Length of the calls |
| `voice` | From the callers: turns, errors, `v2v_ms` and `barge_in_to_silence_ms` (n, p50, p95, p99) and barge-ins |
| `background` | Null for voice alone. From AIPerf's records of the agent requests that ended during the calls: requests, errors (failed and cancelled requests count as lost), requests/s, prompt, completion and total tokens/s from the server's usage counts, and `ttft_ms` |
| `gateway` | `realtime_first_byte_ms` from its histogram; what its admission point shed, refused and resubmitted, and the prompt tokens of the attempts it dropped; `background_tokens_per_s`, its own count from response usage; and every counter that moved |
| `engine` | vLLM's preemptions, prompt tokens by source (local compute, GPU-pool and CPU-tier hits) and queue-limit rejections |

The background goodput comes from AIPerf's records rather than from the gateway's request outcomes, whose `completed` counts a response relayed in full whatever its status: it cannot tell a request the engine refused from one it served. The gateway's token rate stays beside it as a check.

**The report.** `lh-bench e2 report RUN_DIR... --out DIR` writes `frontier.csv`, `frontier.md` (a table per arm, ordered by background goodput) and, where matplotlib is installed (AIPerf brings it), `frontier.svg`. A point is on its arm's frontier when no other point of the arm harvests at least as much background goodput at a V2V p95 at least as low, and one of the two strictly better. Run directories without a complete point are named and left out.

## Mock server

`lh-bench mock-server --port P --log FILE` binds 127.0.0.1 only and streams chat completions.
Each request appends one JSON line to the log: arrival and end times, every header, every body
field except `messages`, a role/size digest of the messages, `cached_tokens` (the longest prefix
shared with one of the session's recent prompts, also returned in `usage`), and `echoed_replies`
(assistant messages that repeat a reply it streamed to the same session). The replay tests use
the same recorder in process.

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

# Mock on loopback (keep clear of cyan's deployed ports 18000, 18080, 18100, 18880, 15558 and 17880-17882)
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

# Live-reply driver, 60 s (restart the mock with --log .../mock-replay.jsonl to keep the logs apart)
lh-bench replay ~/leaseholder/data/weka-x32-sub16 --url http://127.0.0.1:19180/v1/chat/completions \
  --sessions 8 --duration 60 --grace 15 --think-cap 3 --out ~/leaseholder/runs/w0-e2e/replay/requests.jsonl

# What reached the server
python -c 'import json,sys; r=json.loads(open(sys.argv[1]).readlines()[-1]); print(r["headers"], r["body"])' \
  ~/leaseholder/runs/w0-e2e/mock-aiperf.jsonl
kill $(cat ~/leaseholder/runs/w0-e2e/mock.pid)
```

Results on 2026-10-22, against the mock:

| Run | Requests | Every request carries the four fields | `X-Session-ID` is the tree root | Assistant history that is the live reply |
|---|---|---|---|---|
| AIPerf, 8 session trees, 60 s | 7 warmup + 236, 0 errors | 243/243 | 243/243 (103 subagent requests) | 0/197 |
| `lh-bench replay`, 8 sessions, 60 s | 345, 0 errors | 345/345 | yes (158 subagent requests) | 248/248 |

## Tests

The tests are offline and take a few seconds. CI runs them with `make test-python`:

```bash
uv run --extra test pytest      # in python/leaseholder_bench
```

They cover the scaling math and schema, the loader's prefix reproduction (both loaders, through
AIPerf's own reconstruction with a tokenizer-free prompt generator), the plugin's headers and
body as AIPerf's transport sends them, the wire names and class-to-priority mapping against
`lh-proto`, the YAML template, the probes against in-process fake servers, and the driver against
an in-process server: live replies, compaction cuts, side chains, subagent joins, think-time caps
and failures. `tests/test_e2.py` runs whole E2 points against a fake gateway and engine on
loopback, a recording `systemctl` and stand-ins for AIPerf and the callers that run as real
processes (`tests/fakes/`): the sequence, the units each arm restarts, the engine check, failures
at each step with every process stopped, the point directory's rules and the committed arm files.
`tests/test_frontier.py` covers the summary over the window and the Pareto marking.
