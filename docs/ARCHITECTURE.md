# Architecture

Leaseholder runs a realtime voice agent beside background LLM work on one GPU. The part that makes this work is a change to vLLM's scheduler: a waiting realtime request can take KV from running background requests, and background prefill sits out the realtime reply's first tokens. It is proposed upstream as [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) and runs on vLLM 0.32 as the engine variant `make-room` (§2). Around it are the deployment it was measured on (§3), the harness that measured it against everything an operator can deploy today (§4) and a KV-cache simulator (§5). Two mechanisms were built, measured and not adopted (§6), and the rest of the original design was dropped (§7).

This document describes what is built, as of v0.3 (2026-10-24). [PRODUCTION.md](PRODUCTION.md) gives the configuration to deploy, [BENCHMARKS.md](BENCHMARKS.md) the numbers, and [PRIOR-ART.md](PRIOR-ART.md) what is ours and what is not.

## 1. The problem in the engine

A voice turn on a GPU shared with coding agents loses its time in the LLM's first sentence. On stock vLLM 0.32, tuned (priority scheduling, a 512-token step cap and a write-through CPU KV tier), with 24 agent sessions of long contexts beside two calls, the reply's first sentence takes 3.8 s of a 5.2 s mean turn, against 0.3 s with voice alone ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)). Two mechanisms in the scheduler cause it:
- **A waiting request cannot take KV.** vLLM preempts among running requests, but never for a waiting one ([#40004](https://github.com/vllm-project/vllm/issues/40004)). When running agents hold the KV pool, a voice turn waits for one of them to finish: 4.9 s to the first token of a 746-token turn behind two 18k-token agent requests ([the KV case](BENCHMARKS.md#the-kv-case-make-room)).
- **Prefill shares the reply's steps.** Once the turn runs, every step of its reply also carries the agents' prefill chunks, up to about 270 ms each after a 30k-token context, so the first sentence decodes slowly.

Neither depends on the card: the first appears wherever agents' contexts are long against the pool, the second wherever background prefill runs beside a realtime decode ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking) places other GPUs by a model).

## 2. The engine change

The variant `make-room` ([`deploy/phastform/variants`](../deploy/phastform/variants/README.md)) is generated at each engine start from the installed vLLM 0.32.0's scheduler by `build.py`, with asserted anchors, without modifying vLLM; the engine loads it with `--scheduler-cls`. It adds three switches to priority scheduling, each a key of `--additional-config` and off by default:

| Switch | Variant key | Upstream |
|---|---|---|
| KV make-room | `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#63350](https://github.com/vllm-project/vllm/pull/63350)) |
| The budget cap | `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#63351](https://github.com/vllm-project/vllm/pull/63351)) |
| The reserve | `make_room.reserve_tokens` | none: carried in the variant only |

- **KV make-room.** When the highest-priority waiting request lacks KV for its whole sequence, it preempts running requests of a strictly lower priority through vLLM's own preemption path, before the running loop: lowest priority first, then the most blocks its sequence takes, then the latest arrival. It preempts as few as needed and at most N a step, and none if N would not suffice. A victim, its blocks still cached, waits until the prefills in flight have the blocks they still need. With the `OffloadingConnector`, a victim's pending stores are flushed, and its KV stays in the CPU tier for its resume.
- **The budget cap.** Lower-priority prefill sits out the steps of a request's first tokens: from the step in which the highest-priority waiting request fits (deferred frees counted), while it loads its KV from the CPU tier or prefills, until it has its first `budget_cap_tokens` output tokens (16 by default). Running lower-priority prefill chunks are held, and waiting lower-priority requests are not admitted; decodes are never cut. A turn's prefix is usually evicted from the GPU between turns under agent load and loads back from the CPU tier, which is why the hold starts before the turn's prefill.
- **The reserve.** Requests of a positive priority value are admitted only while their whole sequences leave the reserve free: a class-aware watermark, where `--watermark` is class-blind.

What E2 found ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)): the budget cap is necessary, and either KV make-room or the reserve does the rest. The window has to cover the reply's first sentence: this voice agent's replies run 20 tokens at the median and 39 at most, and at 16 tokens full replies decode beside agent prefill again (3,229 ms at p95), where 48 tokens covers them (2,193 ms). The configuration to deploy is the two PRs' switches with a 48-token window and no reserve.

**Known gaps**, in the variant and in both PRs, to be fixed after v0.3 (E2's numbers include their effect):
- the budget cap does not hold a lower-priority request that already holds its KV, such as one whose CPU-tier load has finished (vLLM's `kv_holding_waiting` queue, which it drains before `waiting`);
- the KV shortfall counts the blocks of a request's cached prefix as needed, so the reserve and the in-flight gate can refuse a request that stock vLLM admits;
- victims are sized before the next step's decode growth, so a turn can need a second preemption a step later.

**Tests.** CPU tests run the variant on vLLM's real scheduler with CPU torch; CI builds the generated module from v0.32.0's source. GPU probes measure the KV case, the prefill case and closed-loop agents beside calls ([`probe.py`, `tail.py`](../deploy/phastform/variants/make-room)).

## 3. The deployment

### 3.1 The frozen box

Every measurement runs on one pinned configuration with the audio models on the same GPU ([`deploy/phastform`](../deploy/phastform); measured values in [BENCHMARKS.md](BENCHMARKS.md#the-frozen-box)):

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), prefix caching, priority scheduling, step cap 512/256, KV cache metrics and events, the `OffloadingConnector` (20 GiB in `/dev/shm`, write-through, generated tokens included) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions` | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1-rc1 at `/v1/audio/speech` | ≈ 1.0 GiB |

- **Units.** Each server is a `systemd --user` unit, restarted on failure. `leaseholder.target` starts STT and TTS first, then the LLM, so vLLM's free-memory check sees the audio servers resident; a unit counts as started once its health endpoint answers. `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU.
- **NUMA.** vLLM runs on the GPU's node (node 1, cores 8–15), the audio servers on node 0. The CPU tier's shared memory prefers node 1.
- **Pinned environment.** vLLM 0.32.0 with torch 2.13.0+cu130 serves the LLM and Whisper, Kokoro runs on torch 2.8.0+cu128; lock files are in `deploy/phastform/requirements/`.
- **Time-slicing, not MPS.** As MPS clients, STT and TTS gain a little under agent prefill, but the LLM stage and its tail get worse: the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).
- **Engine variants** are drop-ins switched by `deploy/phastform/variant.sh`: `make-room`, vLLM #54530 ported to 0.32.0, SGLang 0.5.22, MPS, and engine flags from an arm file.

### 3.2 Hosts

```text
┌──────────────────────────────── cyan: control host, 48 threads, CPU only ───────────────────────────────┐
│  host network (loopback only)                      │  voice plane: network namespace, no route out      │
│  lh-gateway (Rust)                                 │  LiveKit server · LiveKit Agents voice worker      │
│  AIPerf agent replay (pinned cores)                │  measurement callers (livekit-rtc)                 │
│  SSH tunnels → 127.0.0.1:18000/18100/18880         │  reach out only through Unix sockets               │
└──────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                       │ SSH local forwards (systemd --user, restricted key)
                                                       ▼
┌─────────────── Phastform: RTX 5060 Ti 16 GB, 188 GiB RAM, the frozen box (Gray: the same) ──────────────┐
│  vLLM 0.32 + make-room · Qwen3-4B-FP8 · KV pool 36,400 tokens · CPU tier 20 GiB          :8000          │
│  vLLM Whisper large-v3-turbo (STT) :8100 · Kokoro-82M (TTS) :8880                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Host | Role |
|---|---|
| **Phastform** | The GPU node every measurement runs on, paused when idle |
| **Gray** | A second, identical GPU node, used in booked windows for the two-node points; cyan reaches it through `lh-tunnel-gray` |
| **cyan** | The control host: `lh-gateway`, LiveKit and the voice worker, the measurement callers and AIPerf, each on its own cores |
| **peacock** | Linux release builds of `lh-gateway` and `lh-sim`, and simulator runs |

### 3.3 Network and security

See [ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md).
- **Loopback and tunnels.** Every service binds loopback. Hosts reach each other only through SSH local forwards run as `systemd --user` units, with a key that may forward only to the engine, STT and TTS ports and may open no listener. The engine publishes no KV events: they carry every cached block's token ids, and vLLM sends them to whoever binds its TCP endpoint.
- **The voice plane** runs in a network namespace with no route out. libwebrtc binds ICE sockets on every non-loopback interface and gathers no candidate on loopback alone, so the namespace holds loopback plus one dummy interface (10.255.0.1/32).
- **Sanitized inputs.** The gateway strips the request fields and headers that vLLM, SGLang and SMG trust for scheduling, caching and routing, and sets its own: the class's priority, `X-Session-ID` and the tenant's `cache_salt`, an HMAC of the tenant's name under a secret only the gateway reads, so tenants never share prefix-cache entries and a client that reaches the engine directly cannot name another tenant's salt ([PROTOCOL.md](PROTOCOL.md#headers)). Sessions are scoped to the tenant: a session id from an untrusted tenant is hashed with the tenant's name.
- **KV never crosses the network.** A session stays on the node that holds its KV; when a node fails, its sessions are placed again and re-prefill there.
- **Known gap.** The hosts' other users share their loopback: on the GPU nodes they reach the engine (in vLLM's development mode, for the harness), STT and TTS directly, and on cyan the tunnels' TCP ports, SMG and the gateway's `/metrics`. Closing it takes Unix sockets end to end and a production unit without development mode, W6's production hardening ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md#co-tenants-2026-10-24)).

## 4. The measurement harness

### 4.1 The gateway

`lh-gateway` ([`crates/lh-gateway`](../crates/lh-gateway/README.md)) is a small Rust proxy in front of the engines, built to run E2's arms on one GPU without Kubernetes. A deployment of the engine change needs only what it does first: the class's priority set from the caller's identity.
1. **Authenticate** the API key and resolve the tenant and its class, capped by the tenant's `max_class`.
2. **Sanitize** and **resolve the session** ([PROTOCOL.md](PROTOCOL.md#sessions)).
3. **Place**, with several GPU nodes (`[nodes]`): a session keeps its node until it has gone `idle_s` without a request. The `[placement]` policy chooses a new session's node: `dedicated` by the classes a node serves, `least-loaded` by requests in flight, `leaseholder` an agent session by the background KV each node holds and a call by the fewest calls. A node that fails three readiness probes takes no new session, and its sessions are placed again on their next request.
4. **Admit** at the node's admission point, by the `[admission]` policy: E2's baselines (a background cap, a token reservation, llm-d-style flow control with eviction, preempt-resubmit before the first byte) or `leaseholder`, arm 9's gateway half (§6).
5. **Forward** with the priority where the upstream reads it (`X-Vllm-Priority`, SGLang's body field or SMG's header), `X-Session-ID` and `cache_salt`.
6. **Stream back** through a bounded buffer: backpressure on a slow client, an upstream abort when the client leaves.

It serves its metrics at `/metrics` (requests, first bytes, the admission point's decisions by policy and reason, the realtime sessions' turn states) and logs one JSON line per request with its first byte to the millisecond, which E2 reads.

The Rust workspace has four crates: `lh-proto` (wire types and their JSON Schemas), `lh-core` (pure models of the frozen box, the E0 tier cost model, KV geometry and a step-time model of vLLM 0.32, which `lh-sim` runs, and the admission and placement policies, which only the gateway runs and whose throttle prices by the step model, compiled in: no I/O and no clock), `lh-gateway` and `lh-sim` (§5).

### 4.2 The voice pipeline

- **Components.** Stock LiveKit server 1.13 and LiveKit Agents 1.8.8 in the voice-plane namespace on cyan (`lh-voice-worker`, [`python/leaseholder_voice`](../python/leaseholder_voice)), with Silero VAD on CPU; STT and TTS reach the GPU node through Unix-socket forwards, and the LLM goes through the gateway's socket with `X-LH-Class: realtime` and the room's session id.
- **Turn handling.** LiveKit's defaults, with preemptive generation on. Experiments end each turn a fixed delay after the end of speech, because LiveKit's end-of-turn model waits its 3 s maximum on some synthesized lines ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). Barge-in pauses the reply after 0.5 s of speech and interrupts it on the final transcript.
- **Worker settings.** Bound to loopback; a small pool of idle processes; a 30 s LLM timeout, since a cold prefill can exceed LiveKit's 10 s default, and one retry instead of LiveKit's three, for a request that fails fast; no AEC warm-up, since the callers play files.
- **Lifecycle events and stage times.** The worker posts speech onset, end of turn, playback and barge-in to the gateway's `/v1/events`, never blocking the session (one sender task per session, a 1 s timeout, no retry), and logs each reply's stage times (end of speech to the transcript and the end-of-turn decision, the LLM's first token and first sentence, TTS's first audio), from which E2 draws its waterfall.
- **A call's node.** With several GPU nodes the worker asks the gateway for a new call's node (`POST /v1/sessions`) and builds the call's STT and TTS on that node's sockets.
- **Measurement callers** (`lh-voice bots`) are `livekit-rtc` callers in the same namespace: they play utterances with known end times, record the agent's track and read V2V and barge-in to silence from the waveforms on one clock.

### 4.3 E2's harness

`lh-bench e2` ([`benchmarks/e2`](../benchmarks/e2/README.md)) runs an arm file as a point: it switches the GPU node to the arm's engine, checks the engine and each flag it reports against its `/server_info` (the rest are listed as unchecked), starts AIPerf's agent replay through the gateway, runs the calls, and saves the point's records: the callers' turns, the gateway's request log, AIPerf's exports, the worker's journal, the GPU's clocks once a second and the metrics around the calls. A campaign measures a list of arm files in order and resumes where it stopped. `lh-bench e2 export` writes each point's compact record, committed in [`benchmarks/results`](../benchmarks/results/2026-10-23-e2/README.md); `make reproduce-e2` rebuilds the report (tables, frontiers and the stage waterfall) from them, and `make e2-figures` the docs' figures.

## 5. The simulator (E1)

`lh-sim` ([`crates/lh-sim`](../crates/lh-sim/README.md)) replays the SemiAnalysis Claude Code traces, block-scaled 32× (32k context), as AIPerf's `agentic_replay` sends them, through a model of vLLM 0.32's two-tier prefix cache and its step loop (the step cap, chunked prefill, first come first served, as at one priority), with a step time fitted to the box. It reproduces resume TTFT p50 and p90 within 8% at 8 and 16 session trees and is 18–30% optimistic at 24. Its arms are LRU with and without the CPU tier, TTL holds, prefetch timed by session history or by an oracle, simplified Tokencake, MORI and InferCept, and a clairvoyant reference (Belady's MIN in each tier). `lh-sim hardware` places an L4 and an H100 SXM on the same regime map, with the box's costs scaled by their published peaks, labeled as modeled.

## 6. Tested and not adopted

- **Protection timed by the conversation, in the control plane** (E2). Under the `leaseholder` admission policy the gateway follows each call's turn from the worker's lifecycle events: it holds KV headroom from speech onset by deferring new background requests, throttles background prefill in the critical window from the end of the turn to the reply's first audio (priced by the step model), caps the reply at 16 tokens when the headroom was not there (degrade), and admits deferred work after 10 s (aging). In E2's discriminating scenario these mechanisms could not act: running agent requests held the pool, which the headroom can protect only by deferring new ones, so it never had its room (degrade capped 60 of 60 replies in every run), and the throttle, checked after the headroom, deferred nothing while the headroom ran. Room for the turn came from the engine alone: the same policy in front of stock vLLM left the tail where stock does (10,993 ms). So a control plane in front cannot make room while running work holds the KV; whether lifecycle timing helps where its mechanisms engage was not isolated. The policy stays as an experiment arm, with its ablations ([BENCHMARKS.md](BENCHMARKS.md#voice-under-agent-load-e2)).
- **Retention and prefetch of agents' KV by their lifecycle** (E1). In the simulator, even an oracle that knows when each agent returns from its tool closes under 9% of resume TTFT at p50 over LRU with the CPU tier, since the median resume is already a GPU hit or a short restore; at p90 it closes 28% at low pressure, but session history, what a deployment knows without the tools' return times, closes nothing ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)). The CPU-tier policy built for it, `LeasePolicy`, and the connector that carried its hint, `LeaseConnector`, were checked live on the GPU, never used, and removed after v0.3 ([the v0.3 tree](https://github.com/StevenWang-CY/leaseholder/tree/v0.3/python/leaseholder_vllm) keeps them).

## 7. Not built

The original design put leases in the control plane: each session's claim on KV residency, with prefetch timed from lifecycle events, a sidecar per GPU node (`lh-node`) with engine epochs, cache contracts from Anthropic's `cache_control`, and a ledger attributing every recomputed token to a cause. After E1 and E2 these were dropped: §6's two results remove their premise, and sticky placement already does the routing leases were also for. They would be reopened by a source of tool-return times ahead of the return, which the oracle's tail gain needs, or by an engine whose running KV a control plane can reclaim. The types and plugins built for them (`lh-proto`'s lease hint and ledger rows with their schemas, `X-LH-Synthetic`, the `leaseholder_vllm` plugins) were removed after v0.3.

Still planned: the two-node points (dedicate against pool, in a window on Gray), the background tier's cost of make-room (restored against recomputed prompt tokens, from the counters every E2 point keeps), and dashboards and alerts as code.

## 8. Failure handling

| Failure | Behavior | Evidence |
|---|---|---|
| Engine crash | The unit restarts; requests in flight fail, and their sessions re-prefill on the next turn | Once in E2, an out-of-memory kill; no drill |
| GPU node loss | Readiness probes (`/health` every `probe_interval_s`, down after three failures): the node takes no new session, and its sessions are placed again and re-prefill on another node | Two mock engines in the gateway's tests |
| Slow client | Backpressure through the bounded buffer, then an upstream abort | A stalled reader |
| Client disconnect | Upstream abort; vLLM aborts the request | Mid-stream disconnects |
| Host memory pressure | `systemd-oomd` may pick the engine; a campaign run again resumes at its failed point | [RUNBOOK.md](../ops/RUNBOOK.md#failures) |
| Another user's GPU work | `gpu-guard` refuses to start the box; a point measured beside foreign GPU work is measured again | [RUNBOOK.md](../ops/RUNBOOK.md#checks-before-heavy-work-on-a-shared-host) |

## 9. Testing

| Layer | Tests |
|---|---|
| `lh-proto` | Wire formats pinned by examples; schemas kept current |
| `lh-core` | The cost model against E0's cells, the KV geometry, the step-time model; each admission policy's decisions on hand-made sequences of requests and turns; each placement policy's choice |
| `lh-sim` | The AIPerf port against AIPerf's own plan; the step loop on hand-made sequences; the clairvoyant tiers against exhaustive search; every arm end to end on a 16-trace subset; `make reproduce-e1` regenerates E1's tables (not in CI: the corpus stays outside the repository) |
| `lh-gateway` | Against an in-process mock engine: disconnects, backpressure, sanitization and tenant-scoped sessions, event idempotency, every admission policy, placement on two mock engines, the placement table under concurrent requests; the proxy's overhead in process |
| Engine | The generated `make-room` module built from v0.32.0's source; its behavior on vLLM's real scheduler with CPU torch; the GPU probes |
| Voice | A hermetic LiveKit test in the voice-plane namespace (barge-in, headers, lifecycle events in order); unit tests of the event delivery and of a call's node |
| E2 | The report, its figures and the demo renderer on synthetic points; `make reproduce-e2` and `make e2-figures` rebuild them from the committed records (not in CI) |
| CI | GitHub Actions on CPU: rustfmt, clippy and ruff, every Rust test, the engine variants' build tests against vLLM's source, the Python unit tests |
