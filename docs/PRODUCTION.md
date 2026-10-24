# Running a voice agent beside background work on one GPU

This guide is for an operator who serves a realtime voice agent (speech to text, an LLM, text to speech) from their own GPU with vLLM, and wants the same GPU to run background LLM work: coding agents, post-call summaries, batch jobs. It gives the configuration E2 measured, what each part buys, what to watch and how it fails. The numbers come from one RTX 5060 Ti 16 GB serving Qwen3-4B-Instruct-2507-FP8 beside Whisper and Kokoro ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)); the mechanisms do not depend on the card, and E1's regime map places other hardware by a model ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)).

## What it buys

Two callers, 25 turns each, beside 24 coding-agent sessions whose contexts fill the GPU's KV pool (E2's discriminating scenario). Voice-to-voice latency (V2V) runs from the end of the caller's speech to the agent's first audio, read from waveforms; the SLO is voice alone plus 250 ms at p95, 2,017 ms. Medians of the runs, with the p95's range:

| Configuration | Runs | V2V p95 | Background tokens/s | Agent TTFT p99 |
|---|---:|---:|---:|---:|
| Voice alone | 5 | 1,767 ms (1,744–1,805) | — | — |
| Shared, stock vLLM 0.32 (priority, step cap 512/256) | 5 | 10,138 ms (9,257–10,811) | 3,710 | 99 s |
| Shared, abort-and-resubmit in a router, stock engine | 5 | 2,591 ms (2,110–2,663) | 3,630 | 160 s |
| **Shared, this configuration** | 3 | **2,193 ms (2,168–2,286)** | 3,493 | 97 s |
| Shared, this configuration with replies capped at 16 tokens (Leaseholder's arm 9) | 5 | 2,057 ms (1,980–2,134) | 3,411 | 103 s |

Sharing on stock vLLM leaves the voice agent unusable: a waiting voice turn cannot take KV from running agents ([vLLM #40004](https://github.com/vllm-project/vllm/issues/40004)), and once it runs, agent prefill chunks share every step of its reply. With this configuration the tail comes within 180 ms of the SLO, the agents keep 94% of their throughput, and their tail is unchanged. The alternative is a GPU per workload.

## Is your deployment in this regime?

The engine change matters when background KV fills the pool: agent contexts of 10k tokens or more against a pool that holds a few of them. On stock vLLM the signs are KV cache usage near 1 (`vllm:kv_cache_usage_perc`) while requests wait (`vllm:num_requests_waiting`), and a realtime time to first token of seconds at p95. Where the pool has room, the change has nothing to do: at the standard scenario's realistic load (16 agent sessions of the whole corpus) it and stock vLLM are within noise of each other (2,196 and 2,323 ms over three runs each), both over the SLO by about 200 ms, which is what agent prefill beside the reply costs there.

## The configuration

### The engine

- **Priority scheduling with a step cap**: `--scheduling-policy priority --max-num-batched-tokens 512 --long-prefill-token-threshold 256`. The step cap bounds how long a running decode waits behind a prefill chunk.
- **Prefix caching and a CPU KV tier written through**: vLLM's `OffloadingConnector` with `offload_prompt_only: false`, about four times the GPU pool ([`engine.yaml`](../deploy/phastform/engine.yaml) has the frozen box's). A preempted agent then restores its KV from host memory instead of recomputing it: 509 ms against 3.26 s for a 16k-token context ([E0](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)).
- **A fixed KV pool** (`--kv-cache-memory`) when the audio models share the GPU, so the pool depends neither on the profiler nor on the audio servers' memory at start.
- **The scheduler change**, proposed to vLLM as [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351):
  - a waiting request of a higher priority preempts running requests of a lower priority when it lacks KV, at most 2 a step: `--max-priority-preemptions-per-step 2`;
  - prefills of a lower priority sit out the steps of its first tokens: `--priority-prefill-budget-cap --priority-prefill-budget-cap-tokens 48` (below for the 48).

  On vLLM 0.32.0, the [`make-room` variant](../deploy/phastform/variants/README.md) generates the same scheduler from the installed vLLM, without modifying it (`python deploy/phastform/variants/make-room/build.py OUT_DIR`), and the engine loads it with `PYTHONPATH=OUT_DIR` and `--scheduler-cls lh_make_room_scheduler.AsyncScheduler`. Its switches are keys of `--additional-config`: `make_room.max_preemptions_per_step 2`, `make_room.budget_cap true`, `make_room.budget_cap_tokens 48` and `make_room.reserve_tokens 0`. The repository's own unit for it (`variants/make-room/lh-engine.conf`) runs arm 9's configuration, the reserve and a 16-token window; these flags go over it. It refuses any other vLLM version, since its edits match 0.32.0's source, and a bad `make_room` key or value fails the engine start, naming it.
- **No reserve.** The variant's third switch keeps KV free for priority 0; beside the KV preemption it changed nothing measurable: with a 48-token window, 2,205 ms with it and Leaseholder's gateway policy against 2,193 ms for the PRs alone, three runs each. The PRs leave it out.

### The priorities

Voice requests go at priority 0 and background work at 2; in vLLM a lower value is served first. A client can send `priority` in the request body or the `X-Vllm-Priority` header. A gateway in front should set it from the caller's identity instead, so that a background client cannot raise its own; `lh-gateway` maps each tenant to a class and caps it with `max_class`. Nothing else in front is needed: the measured configuration ran without an admission policy. `lh-gateway` is the harness that measured it, not a production gateway: what one would add is listed in [its README](../crates/lh-gateway/README.md#what-it-does-not-do), and the engine must be reachable only through the gateway.

### The reply's first tokens

The budget cap protects a reply for its first N tokens. A reply whose first sentence is longer decodes the rest of it beside agent prefill again: its tokens come 62–73 ms apart instead of 16–20 ms ([the prefill case](BENCHMARKS.md#the-kv-case-make-room)). Choose N to cover your agent's first sentence at its longest. This voice agent's replies run 20 tokens at the median and 39 at most:
- at N = 16 with full replies the p95 is 3,229 ms (3,220–4,772);
- at N = 48 it is 2,193 ms;
- at N = 16 with every reply capped at 16 tokens (`max_tokens`) it is 2,057 ms, and two of five runs meet the SLO, at the price of truncated replies.

A longer window holds background prefill for longer each turn: the agents' throughput is 2% lower at 48 than at 16 (3,493 against 3,568 tokens/s).

### The audio models

- Whisper and Kokoro share the GPU by time-slicing; MPS does not help ([Voice on the shared GPU](BENCHMARKS.md#voice-on-the-shared-gpu)): the interference that matters is inside the LLM engine. Sharing still costs them a little: endpointing takes about 825 ms instead of 757, and TTS's first audio 222–249 ms instead of 199.
- Pin each server to its own CPU cores on the GPU's NUMA node ([`deploy/phastform/systemd`](../deploy/phastform/systemd)).

## What not to deploy

Measured in the same scenario:
- **Lifecycle-timed protection in a gateway** (KV headroom held from speech onset, a prefill throttle while a voice turn is answered, the voice agent's events): with agents' KV filling the pool these could not act, since a gateway can only defer new requests while running ones hold the KV. Without the engine change, a gateway doing all of it leaves the p95 at 10,993 ms, as stock.
- **vLLM's development mode** (`VLLM_SERVER_DEV_MODE=1`), which the frozen box runs for the harness's `/server_info` and cache resets: it serves routes such as `/collective_rpc` and `/sleep` that a deployment must not expose.
- **Engine knobs**: `--watermark` (no change), `--max-num-queued-tokens` (it refuses voice turns too: 48 of 50 failed), concurrency caps (two agent contexts fill the pool), [#54530](https://github.com/vllm-project/vllm/pull/54530) (it preempts at the request-count cap, which does not bind here).
- **Another engine**: SGLang 0.5.22's priority preemption, with or without its host tier (11,948 ms with HiCache, five runs).
- **A static KV reservation** halves the tail (5,671 ms) but stays seconds over the SLO.
- **#63350 alone**: 5,773 ms. The KV preemption without the prefill cap gets the turn into the batch, and its reply then decodes beside agent prefill.

Without any engine change, aborting background requests that have no first byte yet and resubmitting them (`lh-gateway`'s `preempt-resubmit` policy) holds 2,591 ms at 98% of stock's throughput, at the cost of the agents' tail (TTFT p99 160 s against 99): the best option on stock vLLM. It cannot reach a request that is already decoding, which the engine change can.

## What to watch

- **Voice**: V2V p95 per call from the voice stack, and the realtime time to first token at p95 (the gateway's request log, or vLLM's per-request metrics). The voice agent's per-turn stage times split a slow turn into endpointing, the LLM's first sentence and TTS ([the stages](BENCHMARKS.md#voice-under-agent-load-e2)).
- **The engine**: KV usage, running and waiting requests, and preemptions; `vllm:num_preemptions_total` counts a victim when its next output reaches the frontend, so it lags. The CPU tier's hits against computed prompt tokens tell restored preemptions from recomputed ones.
- **Background**: completed tokens a second and the agents' TTFT p99, where a preemption policy moves the cost.

## How it fails

- **A voice turn that two preemptions cannot make room for** waits as on stock: the change preempts at most `N` requests a step, and none if `N` would not do.
- **Only realtime work** leaves nothing to preempt: the engine schedules it by priority and arrival, as stock.
- **Many calls at once.** The budget cap holds lower-priority prefill while any realtime request's first tokens are due, so replies that overlap hold it continuously. In a model on vLLM's real scheduler, about one realtime reply a second in aggregate (some six calls turning every 6 s, with the 48-token window) stops the background tier; E2 measured two calls. The engine logs, at most every 10 s, the steps it held, the prefill tokens and admissions it held back, and its preemptions to make room: watch them.
- **A KV load that never completes** holds lower-priority prefill for its first `budget_cap_tokens` steps, then nothing.
- **What E2 measured** ran the v0.3 variant, with three gaps W5 fixed ([ARCHITECTURE.md §2](ARCHITECTURE.md#2-the-engine-change)); W5 re-measures the fixed variant beside it.
- **Host memory pressure.** The CPU tier lives in shared memory charged to the engine's cgroup, so a memory-pressure killer such as `systemd-oomd` picks the engine first under a host-wide stall ([the runbook](../ops/RUNBOOK.md#failures)). Keep the host's swap and memory pressure low; where `systemd-oomd` acts for the engine's own manager, `ManagedOOMPreference=avoid` on the engine's unit makes it a last resort.
- **Another process on the GPU** slows every stage: keep the GPU exclusive, as `gpu-guard` does for the box's units.
- **A reserve too large to start** (only with `make_room.reserve_tokens` above 0): the engine refuses to start if the reserve and one `max_model_len` request do not fit in the pool together.

## Checking it on your hardware

- `deploy/phastform/variants/make-room/probe.py URL kv` sends a voice-sized request at priority 0 while two long priority-2 requests hold the pool, and `probe.py URL prefill` sends it behind three long prefills; each takes about a minute and prints the turn's first token and its 16th, with the engine's state before and after.
- `tail.py URL SECONDS [AGENTS] [fresh|grow]` runs closed-loop agents beside two calls and prints the calls' TTFT distribution.
- The whole measurement: an E2 arm file with your traces and `lh-bench e2 run` ([benchmarks/e2](../benchmarks/e2/README.md)).
