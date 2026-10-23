# Architecture

Leaseholder is SLO-tiered serving on a shared GPU: a **never-shed realtime tier beside a harvested best-effort tier on one vLLM replica**, with the protection timed by conversation lifecycle events.
- **Position.** A thin control plane between API clients and vLLM. Its policy lives in a library, `lh-core`, so it can also be proposed into routers that already run in production.
- **Inputs.** Lifecycle events from both workloads: tool waits from agents; speech onset, end of turn and barge-in from voice.
- **Leases.** From those events it manages KV-cache leases: where each session's KV lives (GPU or host RAM), when the session will be back, and how much GPU headroom to hold for the next realtime turn.
- **Ledger.** It records why every recomputed prompt token was recomputed.

*One GPU, two clocks.* Agents run on tool time, with seconds to minutes between turns. Voice runs on speech time: the user expects an answer about a second after they stop talking. Stock engines schedule both with one queue; Leaseholder schedules each by its own clock.

**What is ours, and what is not.** Priority classes, budgets, eviction of best-effort work and parking with prefetch already ship (llm-d, SMG, Dynamo) or are published (ConServe, Tokencake, MORI); they are baselines here. The contributions are headroom timed by lifecycle events and freed by parking, an engine fix that lets a waiting high-priority request preempt running work under KV pressure, and a validated recompute ledger. [PRIOR-ART.md](PRIOR-ART.md) gives the sources.

**What exists and what is planned.** This document is the target design. After W3, the third week of a 12-week plan:
- **deployed:** the frozen box on Phastform (§5) and, for booked windows, on Gray; on cyan the SSH tunnels, the control plane (`lh-gateway`) and the voice plane (§7);
- **in code:** the wire types; the gateway with its admission point, E2's baseline policies and Leaseholder's own admission policy (turn-imminent headroom, the critical-window throttle, degrade and aging: §4.3); the voice worker's lifecycle events (§7); the engine's "make room" scheduler change, as an engine variant (§4.6); the cost and step-time models and KV geometry; the simulator with its contention model; the engine plugins and the E2 harness (§4.1);
- **planned:** placement across two GPU nodes (W4); leases, prefetch and the ledger with `lh-node` (from W5). Sections that depend on them are marked *planned*.

## 1. Tenets

1. **Mechanism in the engine, policy in the control plane.** The engine decides *what to keep* from the contract carried on each request. The control plane decides *when to bring it back*, from predictions. There is no second lease table inside the engine ([ADR-0004](adr/0004-native-cpu-offload-tier.md)).
2. **One lifecycle-event schema for every workload.** A tool return and a speech onset are two triggers of the same prefetch-and-headroom mechanism ([ADR-0002](adr/0002-one-lifecycle-event-schema.md)).
3. **Realtime is never shed.** Admission for the realtime class means *make room*, then *degrade*; never *reject* ([ADR-0003](adr/0003-realtime-is-never-shed.md)).
4. **KV never crosses the network.** Sessions stick to the node that holds their KV; failover re-prefills from the transcript ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)).
5. **Stock knobs first.** Leaseholder sets headers, budgets and admission, and does not reimplement scheduling. Engine code exists only behind vLLM's own plugin points, apart from the "make room" scheduler fix, which is proposed upstream as a vLLM PR rather than carried as a fork ([ADR-0001](adr/0001-control-plane-in-front-of-stock-vllm.md)).
6. **Every number can be rerun.** Each plot comes from a scripted run that writes its configuration, host and raw samples next to its results ([EXPERIMENTS.md](EXPERIMENTS.md#how-runs-are-made)).

**Non-goals:**
- new kernels or a new engine;
- owning API semantics, which agentic-api does;
- multi-tenant billing;
- Kubernetes on shared hosts;
- STT and TTS model optimization;
- cross-node KV transfer.

## 2. Topology

```text
┌──────────────────────────────── cyan: control host, 48 threads, CPU only ───────────────────────────────┐
│  host network (loopback only)                      │  voice plane: network namespace, no route out      │
│  Leaseholder control plane (Rust)                  │  LiveKit server · LiveKit Agents voice worker      │
│  agentic-api (Claude Code / Codex via base_url)    │  measurement callers (livekit-rtc)                 │
│  AIPerf agent replay (pinned cores)                │  reach out only through Unix sockets               │
│  SSH tunnel → 127.0.0.1:18000/18100/18880/15558    │  in $XDG_RUNTIME_DIR/lh/                           │
└──────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                       │ SSH local forwards (systemd --user, restricted key)
                                                       ▼
┌─────────────────────── Phastform: RTX 5060 Ti 16 GB, 188 GiB RAM, the frozen box ───────────────────────┐
│  vLLM 0.32 · Qwen3-4B-Instruct-2507-FP8 · KV pool 36,400 tokens (5 GiB) · 32k context        :8000      │
│  native CPU offload tier, 20 GiB in /dev/shm · KV events (ZMQ) · priority scheduling                    │
│  vLLM Whisper large-v3-turbo (STT) :8100 · Kokoro-82M on Kokoro-FastAPI (TTS) :8880                     │
│  lh-node sidecar (W5): KV-event relay, engine epoch, health                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Gray: second RTX 5060 Ti, booked windows; same image; dedicate-vs-pool experiment, node-loss drill
mustard: Prometheus · Grafana · datasets · run archive      peacock: Linux builds · simulator runs
```

In the diagram, `lh-node`, agentic-api in front of the control plane, Prometheus and Grafana on mustard, and Gray's experiments are planned; the rest, Gray's frozen box included, is deployed ([RUNBOOK.md](../ops/RUNBOOK.md)).

| Host | Role |
|---|---|
| **Phastform** | The GPU node every experiment runs on, holding the frozen box (§5) and paused when idle. vLLM is pinned to NUMA node 1, the audio servers to node 0 |
| **Gray** | Second GPU node in booked windows, with the identical image: the equal-hardware dedicate-vs-pool experiment and the node-loss drill. Nothing on the critical path depends on it |
| **cyan** | Control host: the control plane, agentic-api, LiveKit and its worker, the measurement callers and AIPerf, each on its own cores ([RUNBOOK.md](../ops/RUNBOOK.md)) |
| **mustard** | Observability and datasets: Prometheus, Grafana, trace datasets, the run archive |
| **peacock** | Linux release builds (`lh-gateway`, `lh-sim`, agentic-api) and simulator runs |

**Network and security** ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)):
- **Loopback and tunnels.** Every service binds loopback. Hosts reach each other only through SSH local forwards, run as `systemd --user` units, and the tunnel key may forward only the listed loopback ports.
- **The voice plane** runs in a network namespace with no route out. libwebrtc clients bind ICE sockets on every non-loopback interface, whatever the server's `rtc` block says, and gather no candidate on loopback alone, so the namespace holds loopback plus one dummy interface (10.255.0.1/32) for them to bind.
- **Sanitized inputs and cache isolation.** The control plane strips the client inputs that vLLM trusts and sets its own. Each tenant's `cache_salt` is derived from the tenant's name, so tenants never share prefix-cache entries and rotating a key keeps the cache ([PROTOCOL.md](PROTOCOL.md#headers)).

## 3. Core model

The wire types are in [`crates/lh-proto`](../crates/lh-proto) and are described in [PROTOCOL.md](PROTOCOL.md).

### 3.1 Sessions

A session is one conversation, scoped to a tenant: the routing key, the accounting key and the OTel `gen_ai.conversation.id`. [PROTOCOL.md](PROTOCOL.md#sessions) gives the order in which the control plane resolves it from a request; the voice worker sends `voice-<room SID>`, one per LiveKit room.

### 3.2 Lifecycle events

One event type serves every workload ([ADR-0002](adr/0002-one-lifecycle-event-schema.md)). For agents, a tool wait shows in a response that ends in `tool_calls`, or exactly in agentic-api's `tool_wait_started` and `tool_wait_ended` for server-side tools. Voice signals speech onset, end of turn, playback and barge-in (`interrupted`, with how much of the reply was played). Events reach `POST /v1/events`, which `lh-gateway` accepts and deduplicates. The voice worker sends the voice events (§7), and under the `leaseholder` admission policy the gateway turns them into each realtime session's turn state (§4.3). agentic-api's tool-wait events, and turn events inferred from proxied traffic, are *planned*. [PROTOCOL.md](PROTOCOL.md#lifecycle-events) lists every type with its fields and its effect on the lease.

### 3.3 Leases (*planned*)

A lease is a session's time-bounded claim on KV residency. It records when the session will be back and how fast it must resume:

```text
Lease {
  session_id
  holder            = (node, engine_epoch)   // fenced: an epoch change voids the lease
  class             = realtime | interactive | background
  floor             = CPU | NONE              // held until expires_at where capacity allows (refusals counted); GPU residency is best-effort
  expires_at                                  // contract term (e.g. Anthropic cache_control 5m / 1h); renewed on hit
  next_arrival                                // ETA or distribution, from lifecycle events, else session history
  resume_target_ms                            // from the class SLO
  state             = ACTIVE | TOOL_WAIT | HUMAN_WAIT | USER_SPEAKING | PLAYBACK | CLOSED
}
```

**Operations.** *Grant* at the end of a turn; *renew* on a cache hit; *revise* on a lifecycle event; *revoke* on an epoch change or under capacity pressure (counted as a violation); *expire*.

**Invariants.** Each one is a test:
1. never route to a holder whose engine epoch has changed;
2. never grant a CPU floor the tier cannot hold, because admission applies to lease grants as well as requests;
3. every resume either meets `resume_target_ms` or produces a ledger row with a cause.

| Concern | How the lease drives it |
|---|---|
| Retention | `floor`, carried to the engine as the lease hint |
| Prefetch | starts at `next_arrival − restore_time` |
| Admission | imminent hot leases reserve KV and prefill headroom |
| Routing | `holder` plus epoch |
| Accounting | lease-seconds per tier, and violations |

### 3.4 SLO classes

`realtime` (voice turns), `interactive` (agent loops with a person waiting) and `background` (batch agents, replays). Targets and admission rules are in [SLO.md](SLO.md).

### 3.5 The recompute ledger (*planned*)

There is one row per upstream request: prompt, cached (device and host), expected-reusable and recomputed tokens, with one cause for each recomputed span ([PROTOCOL.md](PROTOCOL.md#ledger-rows); the row type is in `lh-proto` v0). Rows are written as Parquet and queried with DuckDB, with per-cause counters in Prometheus. Prefetch warm-ups are tagged and excluded from every indicator. An injected-cause suite validates the attribution ([EXPERIMENTS.md](EXPERIMENTS.md#e3-ledger)).

## 4. Control plane

### 4.1 Crates

The control plane is a Rust workspace. The policy is a library without I/O or a clock, so another router can embed it. The state column says what exists after W3.

| Crate | Responsibility | State |
|---|---|---|
| `lh-proto` | Wire types: lifecycle events, headers, engine lease hints, ledger rows. JSON Schemas for the Python components | v0 |
| `lh-core` | The deterministic policy core: admission, placement, the lease table, the prefetch planner, ledger attribution. Pure code with no I/O and no clock: callers pass counts and events, and get durations and decisions back | The E0 tier cost model, KV geometry, a step-time model of vLLM 0.32's step loop (W2), and `admission`, a state machine driven by arrivals, first bytes and departures: the E2 baselines (static cap, token reservation, flow control, preempt-resubmit) and, since W3, Leaseholder's `leaseholder` policy, which also reads the realtime sessions' turns and prices background prefill with the step model (`PrefillCost`). Placement, leases, prefetch and attribution are to follow |
| `lh-gateway` | The service (tokio, hyper/axum, tower): ingress and auth, SSE passthrough, cancellation, backpressure, upstream client, event ingest, prefetch executor, ledger sink, metrics. It listens on loopback TCP for the host network and on a Unix socket for the voice plane | Deployed on cyan: everything but the prefetch executor and the ledger sink, which come with leases. Its admission point runs `lh-core`'s policies (the E2 baselines since W2, `leaseholder` since W3), it keeps each realtime session's turn state from lifecycle events, and it sends each class's priority where the upstream reads it (`upstream_priority`) |
| `lh-node` | A small sidecar on each GPU node. It binds the KV-event subscriber that the engine connects to, detects sequence gaps and resyncs through the engine's replay endpoint, relays events to the control plane, and reports the engine epoch and health | W5 |
| `lh-sim` | A trace-driven two-tier KV simulator over `lh-core`'s cost and step-time models, with a clairvoyant reference (Belady's MIN per tier) | The stock and E1 arms, the clairvoyant reference, miss-ratio curves, the contention model (the default service since W2, calibrated at two of three operating points) and the E1 regime sweep |

### 4.2 Request path

Steps marked *planned* arrive with placement and leases; `lh-gateway` runs the rest ([`lh-gateway` README](../crates/lh-gateway/README.md)).

1. **Authenticate** the API key and resolve the tenant.
2. **Sanitize** the engine-control headers and body fields.
3. **Resolve** the session and class.
4. **Admit** by the configured `[admission]` policy: one of E2's baselines (a background cap, a token reservation, llm-d-style flow control with eviction, preempt-resubmit before the first byte) or Leaseholder's `leaseholder` (§4.3). Under `leaseholder` a realtime request goes upstream at once, never queued or refused, with its reply capped if the headroom was not there (*degrade*); a background request is *deferred* while it would cut into the armed realtime sessions' headroom or overrun the critical window's prefill budget, and admitted regardless after `max_defer_s` (*aging*). A waiting request is woken on a channel, and one whose client leaves gives its place up. *Planned* for the interactive class: admission by predicted time to first token, which is queue time plus uncached tokens over the measured prefill rate, plus restore cost from `lh-core`'s E0 model ([BENCHMARKS.md](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)).
5. **Place** (*planned*) on the holder if its epoch is valid, else on the least-loaded healthy node. v0 has one upstream.
6. **Forward** with `X-Session-ID`, the class's priority where the upstream reads it (`upstream_priority`: `X-Vllm-Priority` by default, the body's `priority` for SGLang, `X-SMG-Priority` for SMG) and the tenant's `cache_salt`, and (*planned*) the lease hint in `kv_transfer_params`.
7. **Stream back** through a bounded per-stream buffer.
   - On client disconnect, abort upstream (vLLM aborts on disconnect).
   - On completion (*planned*), update the lease and write the ledger row.

### 4.3 Turn-imminent headroom

vLLM preempts among running requests but never on behalf of a waiting one, so a KV pool full of running background agents can hold a voice turn for seconds ([#40004](https://github.com/vllm-project/vllm/issues/40004)): 4.9 s to the first token of a 746-token voice turn behind two 18k-token agent requests, on the frozen box's image ([BENCHMARKS.md](BENCHMARKS.md#the-kv-case-make-room)). Long background prefills also slow every stage of the turn ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).

Under the `leaseholder` admission policy (E2 arm 9; the [`lh-gateway` README](../crates/lh-gateway/README.md#policy-leaseholder) gives its keys and exact semantics), the control plane follows each realtime session through its turn, from lifecycle events (§7) in the order it receives them: `user_speaking` from speech onset; `critical` from the end of the turn, or from the realtime request, which LiveKit's preemptive generation sends earlier, until the reply's first audio; then `playback`. Every state but idle times out, so a lost event holds a session at most `state_timeout_s`, and a session without events is critical while its request is in flight. From those states the control plane:
- **keeps KV headroom** for each armed session, from speech onset (`trigger = "onset"`) or within the critical window alone (`"request"`): a background request goes upstream only if the estimated KV of the background requests in flight and its own leave `headroom_tokens` free per armed session. Deferring new background turns lets the running ones finish, and their KV parks in the write-through CPU tier;
- **throttles background prefill in the critical window** to a budget of predicted prefill time, not of tokens: `lh-core`'s step model prices a request's new tokens after its context, and a 512-token chunk after a 30k-token context costs about seven times one after none;
- **degrades rather than sheds**: a realtime request that arrives while the headroom is not there goes upstream at once, with its reply capped at `degrade_max_tokens`;
- **ages** deferred work: a background request deferred `max_defer_s` is admitted regardless, so agents never starve, and one larger than the headroom leaves is counted as `oversize`.

Each mechanism can be switched off, and E2 ablates each alone. Headroom armed by speech onset that no realtime request follows is a *false alarm*, charged in token-seconds of background work held back. The gateway works from an estimate of the KV in flight; the engine's "make room" change (§4.6) is its exact counterpart inside the engine, where a waiting realtime request that lacks KV preempts lower-priority running work, whose KV stays in the CPU tier, and a class-aware reserve keeps blocks free. *Planned*, with prefetch (§4.4): restoring the speaking session's KV when it is not GPU-resident. A 2k-token restore adds about 55 ms over a GPU hit ([BENCHMARKS.md](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0)), well inside 1–3 s of speech.

All of this runs on top of the engine's own step cap (`--max-num-batched-tokens 512 --long-prefill-token-threshold 256`), which E2 sweeps.

### 4.4 Prefetch (*planned*)

Prefetch triggers on `next_arrival − restore_time` (ETA from session history, or exact from a tool lifecycle) and on `user_speech_started`. A prefetch is a warm-up request with `max_tokens = 1` that:
- reproduces the exact token prefix, because re-rendering a chat template can drift;
- is rate-limited against a PCIe budget set by E0's restore rate, about 6.2 GB/s of KV or roughly 2.6 restores of 16k tokens per second ([BENCHMARKS.md](BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0));
- is never cancelled mid-load, because an abort discards the restored KV;
- is tagged `X-LH-Synthetic` and excluded from SLIs, usage and the ledger.

GPU residency after a prefetch is not guaranteed: restored blocks return to the LRU, so a prefetch is timed close to the arrival. vLLM draft PR #58198 adds a request-free prefetch primitive, which would replace the warm-up request.

### 4.5 Cache contracts (*planned*)

Anthropic `cache_control` breakpoints and `ttl` (`5m`, `1h`) become lease floors and `expires_at`. Usage is reported in Anthropic's fields: `cache_read_input_tokens`, `cache_creation_input_tokens`, and the 5-minute and 1-hour ephemeral counts.

| Term | Meaning |
|---|---|
| Intent | Inside the TTL (renewed on each use), the prefix up to the breakpoint stays at least in host RAM; GPU residency is best-effort. Providers promise billing eligibility, not residency, so this is an objective, not a guarantee |
| Exceptions | Node failure; model or config change; prefix mismatch |
| Honor rate | The share of hits inside the TTL, outside the exceptions: an SLI, reported with every miss counted. A cache read is reported only for a real hit |

OpenAI `prompt_cache_options` is deferred.

### 4.6 Engine-side policy

The CPU tier is vLLM's native `OffloadingConnector` with `offload_prompt_only: false`, so generated tokens are written through as well. Leaseholder's plugins in [`python/leaseholder_vllm`](../python/leaseholder_vllm) are optional, and both exist in v0:
- **`LeaseConnector`** adds per-request hit receipts (GPU tokens and CPU tokens) and mirrors `X-Session-ID` into the lease hint.
- **`LeasePolicy`** makes CPU-tier eviction lease-aware. It ships only where the simulator shows that CPU-tier eviction binds at the chosen RAM ratio. It is soft: it refuses a store only where LRU would refuse it too.

A live check on the frozen box confirmed the contract end to end: the hint reaches both plugins, a prompt written through to the CPU tier is restored from it after a GPU cache reset, and `X-Session-ID` appears in GPU-tier KV events but not in CPU-tier ones ([BENCHMARKS.md](BENCHMARKS.md#engine-plugins)).

**"Make room"**, the scheduler change for #40004's KV case that is proposed upstream, runs as the engine variant `make-room` ([variants README](../deploy/phastform/variants/README.md)): `build.py` generates it from vLLM 0.32.0's scheduler at each start, with asserted anchors, and the engine loads it with `--scheduler-cls`. It adds three mechanisms to priority scheduling, each a key of `--additional-config` and off by default:
- **KV make-room** (`make_room.max_preemptions_per_step`). When the highest-priority waiting request lacks KV for its whole sequence, it preempts running requests of a strictly lower priority through vLLM's own preemption path, before the running loop: lowest priority first, then the most blocks its sequence takes (the most KV, unless shared), then the latest arrival. It preempts as few as needed and at most N a step, and none if N would not suffice. A victim, its blocks still cached, waits until the prefills in flight have the blocks they still need; resuming it earlier takes them back between the chunks of the request it made room for. Deferred block frees count as free. With the OffloadingConnector, a victim's pending stores are flushed, and its KV stays in the CPU tier for its resume.
- **The budget cap** (`make_room.budget_cap`). Running prefills of a lower priority leave the waiting request's first chunk in the step's token budget; decodes are never cut.
- **The reserve** (`make_room.reserve_tokens`). Requests of a positive priority value (interactive and background) are admitted only while their whole sequences leave the reserve free: a class-aware watermark, where `--watermark` is class-blind.

CPU tests check it on vLLM's real scheduler, and a GPU probe measures the KV case with and without it ([BENCHMARKS.md](BENCHMARKS.md#the-kv-case-make-room)).

## 5. Data plane: the frozen box

Every experiment runs on one frozen configuration with audio co-located: [`deploy/phastform`](../deploy/phastform), with the measured values in [BENCHMARKS.md](BENCHMARKS.md#the-frozen-box).

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), prefix caching, priority scheduling, step cap 512/256, prompt token details, KV cache metrics, KV events, native CPU offload (20 GiB, write-through) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions` | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1-rc1 at `/v1/audio/speech` | ≈ 1.0 GiB |

**Placement and environment:**
- **Units.** Each server is a `systemd --user` unit. `leaseholder.target` starts STT and TTS first, then the LLM, so vLLM's free-memory check sees the audio servers resident. Each unit counts as started only once its health endpoint answers.
- **NUMA.** vLLM runs on node 1, local to the GPU (cores 8–15, memory preferred on node 1); the audio servers run on node 0. The CPU tier's shared-memory region prefers node 1 and spills to node 0, where pinned bandwidth is about the same.
- **Pinned environment.** vLLM 0.32.0 with torch 2.13.0+cu130 serves both the LLM and Whisper. It uses the CUDA 13.0 toolkit for DeepGEMM's JIT, curand header links for FlashInfer's JIT, and `nvidia-cudnn-cu13==9.27.0.42`. Kokoro runs on torch 2.8.0+cu128. Lock files are in `deploy/phastform/requirements/`.
- **Time-slicing, not MPS.** With the three servers as MPS clients, STT and TTS gain a little under agent prefills, but the voice turn's LLM stage and its tail get worse and background prefill throughput drops, because the interference that matters is inside the engine ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). The `mps` variant (`deploy/phastform/variant.sh on mps`) stays available as an E2 arm.
- **Fallbacks.** STT can fall back to faster-whisper INT8 (1.2 GiB). TTS can switch to Qwen3-TTS via vLLM-Omni (4.3 GiB, which shrinks the KV pool to about 20k tokens).

## 6. API layer: agentic-api

This layer is *planned*; agentic-api's own cost in front of the engine is measured ([BENCHMARKS.md](BENCHMARKS.md#the-api-layer-agentic-api-in-front-of-the-engine)).
- **Upstream.** agentic-api's single upstream, `llm_api_base`, points at the control plane.
- **Clients.** Claude Code connects through Anthropic Messages, Codex through Responses (whose default `prompt_cache_key` is its session id).
- **Direct paths.** Voice and AIPerf traffic speak Chat Completions straight to the control plane, which keeps the realtime path to one hop.
- **Upstream contributions** that help: a header allowlist on the executor path (#461), a typed lifecycle-event sink (#364), cache-usage metrics (#315), and performance-validation data (#367).

## 7. Realtime voice pipeline

- **Components.** All stock: LiveKit server 1.13 and LiveKit Agents 1.8.8 in the voice-plane namespace on cyan (`lh-voice-worker`, [`python/leaseholder_voice`](../python/leaseholder_voice)), with Silero VAD and LiveKit's audio end-of-turn model (`turn-detector-v1-mini`, run locally) on CPU. The OpenAI STT and TTS plugins reach the frozen box through Unix-socket forwards; the LLM plugin reaches the control plane's socket.
- **Turn handling.** LiveKit's defaults: endpointing between 0.5 and 3 s, and preemptive generation on, which starts the LLM request on the final transcript before the end of turn is confirmed and discards it if the user continues. E2 runs preemptive generation on and off and reports the discarded tokens. Experiments end each turn a fixed delay after the end of speech (`LH_TURN_DETECTION=vad`, the `vad-turns` variant), because the end-of-turn model waits its 3 s maximum on some synthesized turns ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)).
- **Headers.** The LLM plugin sends `X-LH-Class: realtime` and the room-scoped session id (`voice-<room SID>`) through its `extra_headers` option. The worker creates one LLM instance per session.
- **Interruption handling.** The stock LiveKit setting pauses the reply audio after 0.5 s of barge-in speech and interrupts the reply only on the barge-in's final transcript, so the LLM stream stays open through the barge-in. With `resume_false_interruption=False` the reply is interrupted after 0.5 s of barge-in speech, at the cost of cutting it on a cough ([BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). The setting is fixed across E2 arms, and the wasted tokens are reported.
- **Worker settings.**
  - `AgentServer(host="127.0.0.1")`, because the default binds all interfaces.
  - An explicit small `num_idle_processes`, because production mode defaults to the CPU count.
  - An LLM request timeout of 30 s, because `llm_conn_options` defaults to 10 s and a cold prefill can exceed it.
  - Each call's room is deleted when the caller hangs up, so finished calls do not hold a job.
  - Interruptions by VAD, which production mode picks anyway: adaptive detection needs LiveKit Inference, and the namespace has no route out.
  - No AEC warm-up, because the callers play files: the default 3 s warm-up would only discard a barge-in on each call's first reply.
- **Lifecycle events** ([`leaseholder-voice` README](../python/leaseholder_voice/README.md#lifecycle-events-and-turn-lines)).
  - The worker posts `user_speech_started`, `user_turn_ended`, `playback_started`, `playback_ended`, `interrupted` (with `played_ms`, what the caller heard) and `session_closed`, each from one LiveKit hook, to the control plane's `/v1/events`, over the socket its LLM requests use. Posting never blocks the session: one background task per session sends what is queued, with a 1 s timeout and no retry, because a late timing signal would mislead.
  - Speech onset arms the headroom, and the end of the turn opens the critical window (§4.3). On barge-in, LiveKit closes the LLM stream, the control plane aborts upstream, and the reply stays in the context only as far as it was heard.
  - `LH_EVENTS=off` (the `no-events` worker variant) stops the events, for E2's ablation; they stop as well when the LLM bypasses the gateway.
  - Each reply logs one line with its stage times (end of speech to the final transcript and to the end-of-turn decision, the LLM's first token and first sentence, TTS first audio, end to end), which E2 reports as the stage waterfall.
- **Measurement callers** (`lh-voice bots`). These are `livekit-rtc` Python callers in the same namespace. They play utterances with known end timestamps and record the agent's track, and compute V2V and barge-in to silence from waveforms on one clock; audio gaps, the continuity indicator, are not reported yet. VAD and turn-detection settings are fixed across arms, and E2 reports end-of-utterance delay as its own stage.

## 8. Simulator

`lh-sim` decides, before any engine code is written, how much a retention policy can win over LRU plus CPU offload, and where. Its [README](../crates/lh-sim/README.md) gives the model, the simplifications and the validation.
- **Replay.** The SemiAnalysis Claude Code traces, block-scaled 32× (32k context) or 64× (16k), sent as AIPerf's `agentic_replay` sends them, subagents and capped think times included. The port of AIPerf's loader and lane plan matches AIPerf's own field for field on the full corpus.
- **Engine model.** vLLM 0.32's prefix cache over 16-token blocks: a pinned GPU pool freed tail first into an LRU queue, and the OffloadingConnector's write-through CPU tier. Requests are served through a model of vLLM's step loop (the 512/256 step cap, chunked prefill, priority-then-arrival order) with a step time fitted to the E0 cells and three operating points; `--service idle` keeps v0's idle-box costs without queueing.
- **Arms.**
  - in v0: GPU-only LRU; LRU + CPU; S3-FIFO + CPU;
  - the clairvoyant reference, in v0: Belady's MIN in each tier, replayed on the LRU + CPU timeline. It is a reference, not a bound: MIN is optimal only for one tier's hit count over unit-size blocks, and here hits are prefixes served from two tiers;
  - since W2, the E1 arms: TTL holds, prefetch timed by session history or by an oracle, and simplified Tokencake, MORI and InferCept, with pauses from the traces or from TraceLab ([EXPERIMENTS.md](EXPERIMENTS.md#e1-parking)).
- **Outputs.** Per arm: prompt-token shares by source, with recompute split by cause; time to first token overall and for resumes; PCIe bytes, evictions and stalls; the regime coordinates (KV pressure, π, ρ). Miss-ratio curves of the GPU tier, at 32× and at the original token counts, show what block scaling keeps (reuse structure) and what it changes (timing).
- **Calibration.** Driven by W0's measured request times, the engine model matches the measured GPU and CPU shares closely. With the contention model, its own replay reproduces resume TTFT p50/p90 within 8% at 8 and 16 session trees and is 18–30% optimistic at 24, where restores take longer than modeled ([`lh-sim` README](../crates/lh-sim/README.md#calibration-against-the-gpu-operating-points)). The E1 result built on it, G-sim's negative branch, is in [EXPERIMENTS.md](EXPERIMENTS.md#e1-parking).
- **Code reuse.** The cost model, the step-time model and KV geometry are `lh-core`'s. The E1 arms are the simulator's own. The admission policies, Leaseholder's headroom among them, live in `lh-core` and run in the gateway, which prices background prefill with the same step model; the simulator does not run them. Leases are to live there too, so that the simulator and the gateway run the same code.

## 9. Observability

- **Indicators, targets and alerts.** See [SLO.md](SLO.md).
- **What exists.** `lh-gateway` serves its metrics at `/metrics`, among them the admission point's (`lh_gateway_admission_*`: admitted, waiting, wait time, refused by reason, resubmits, discarded tokens; under `leaseholder`, deferrals by reason, aged and oversize admissions, degraded replies and false alarms with their token-seconds) and the realtime sessions by state, with state timeouts. It logs one JSON line per request, with its admission wait and what deferred it, and per lifecycle event ([`lh-gateway` README](../crates/lh-gateway/README.md#metrics-and-logs)). The voice worker logs one line per reply with its stage times (§7). Each E2 point keeps the gateway's request lines (the exact first-byte percentiles), the worker's journal (the stage waterfall and the event counts), the GPU's clocks and temperature once a second, and snapshots of the gateway's and the engine's metrics around the calls (`lh-bench e2`).
- **Dashboards** (*planned*, as code in `ops/`; the voice turn's stage waterfall exists as E2's `waterfall.svg`, from the worker's turn lines):
  1. the voice turn waterfall and SLOs;
  2. KV and leases: pool occupancy, tier residency, prefetch outcomes, PCIe budget;
  3. admission and the ledger: decisions, deferrals, causes, rejections.
- **Runs.** See [EXPERIMENTS.md](EXPERIMENTS.md#how-runs-are-made).

## 10. Failure handling

| Failure | Detection | Behavior | Drill |
|---|---|---|---|
| Engine crash or restart | Epoch change reported by `lh-node` | Revoke that node's leases; re-prefill from the transcript; count violations | Kill vLLM mid-stream |
| Node loss | Health check or tunnel down | Drain; move sessions; re-prefill | Gray window |
| Control-plane restart | — | Rebuild the lease table from KV-event replay plus the session store; epochs fence stale state | Restart under load |
| CPU tier full | Grant admission | New grants get `floor = NONE` and the shortfall is reported; existing floors are kept | Undersized tier |
| KV-event gap | Sequence numbers | Resync through the replay endpoint; affected ledger rows are marked `unknown` | Injected drop |
| PCIe saturation | Prefetch budget | Prefetches are shed before on-demand loads | Burst of resumes |
| Audio server crash | Health check | The worker falls back (faster-whisper INT8 on CPU, or a text reply); the session survives | Kill the TTS server |
| Slow client | Bounded buffer | Backpressure, then abort upstream | Stalled reader |

In v0, `lh-gateway` implements the slow-client row, tested with a stalled reader. The other rows need `lh-node` and the `lh-core` policies; the drills are in [EXPERIMENTS.md](EXPERIMENTS.md#reliability-drills-and-the-two-node-window).

## 11. Testing

| Layer | Tests |
|---|---|
| `lh-proto` | Wire format pinned by examples; schemas kept current; ledger accounting identities |
| `lh-core` | The cost model against the measured E0 cells, the KV geometry, the step-time model, and each admission policy's decisions on hand-made sequences of requests and turns (for `leaseholder`: each mechanism alone and together, aging and oversize, degrade, realtime never held). With leases: unit and property tests for the lease invariants, and deterministic simulation with seeded event sequences |
| `lh-sim` | The AIPerf port against AIPerf's own plan; engine semantics and the step loop on hand-made sequences; the clairvoyant tiers and MIN against exhaustive search; Mattson stack distances against a direct LRU simulation; every arm end to end on a 16-trace subset; the idle service reproducing v0's outputs exactly |
| `lh-gateway` | Integration against an in-process mock engine: mid-stream disconnect, backpressure, header sanitization, event idempotency, the admission baselines (caps, queue limits and timeouts, eviction, transparent resubmission, a streaming response never preempted), and `leaseholder` driven by events through `/v1/events` (deferral while a session is armed, the critical budget, aging, the degrade rewrite of `max_tokens`, state timeouts and lost events). The proxy tax in process and in place ([BENCHMARKS.md](BENCHMARKS.md#the-control-plane-lh-gateway-in-front-of-the-engine)) |
| Ledger | *Planned*: the injected-cause suite, at least 95% correct attribution, with the `unknown` share reported |
| Engine plugins and variants | CPU tests against vLLM's real offload manager, and builds of the generated scheduler variants from v0.32.0's source (in CI); `make-room`'s behavior on vLLM's real scheduler, on CPU with torch (run on the GPU node's environment); a live check of the plugins and the KV probe on the GPU node |
| Voice | Hermetic LiveKit test in the voice-plane namespace: barge-in, headers, and the lifecycle events in order; unit tests of the event mapping and delivery; *planned*: a nightly caller smoke test on the frozen box |
| Performance | *Planned*: a nightly fixed scenario on the GPU node against stored baselines; a weekly compatibility run against the newest vLLM |
| CI | GitHub Actions on CPU: rustfmt, clippy and ruff; the Rust unit, simulator and integration tests; the engine plugin tests against vLLM's offload manager; the Python unit tests |
