# Architecture

Leaseholder is a session-aware serving control plane. It lets **one commodity GPU serve live voice conversations and a swarm of tool-using agents at the same time, within explicit SLOs**.
- **Position.** It sits between API clients and vLLM.
- **Inputs.** It consumes lifecycle events from both workloads: tool waits from agents; speech onset, end of turn and barge-in from voice.
- **Leases.** From those events it manages KV-cache leases: where each session's KV lives (GPU or host RAM), when the session will be back, and how much GPU headroom to hold for the next realtime turn.
- **Ledger.** It records why every recomputed prompt token was recomputed.

*One GPU, two clocks.* Agents run on tool time, with seconds to minutes between turns. Voice runs on speech time: the user expects an answer about a second after they stop talking. Stock engines schedule both with one queue; Leaseholder schedules each by its own clock.

## 1. Tenets

1. **Mechanism in the engine, policy in the control plane.** The engine decides *what to keep* from the contract carried on each request. The control plane decides *when to bring it back*, from predictions. There is no second lease table inside the engine ([ADR-0004](adr/0004-native-cpu-offload-tier.md)).
2. **One lifecycle-event schema for every workload.** A tool return and a speech onset are two triggers of the same prefetch-and-headroom mechanism ([ADR-0002](adr/0002-one-lifecycle-event-schema.md)).
3. **Realtime is never shed.** Admission for the realtime class means *make room*, then *degrade*; never *reject* ([ADR-0003](adr/0003-realtime-is-never-shed.md)).
4. **KV never crosses the network.** Sessions stick to the node that holds their KV; failover re-prefills from the transcript ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)).
5. **Stock knobs first.** Leaseholder sets headers, budgets and admission. It does not reimplement scheduling, and engine code exists only behind vLLM's own plugin points ([ADR-0001](adr/0001-control-plane-in-front-of-stock-vllm.md)).
6. **Every number can be rerun.** Each plot comes from a scripted run that writes its configuration, host and raw samples next to its results.

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
│  host network (loopback only)                      │  voice plane: network namespace with only lo     │
│  Leaseholder control plane (Rust)                  │  LiveKit server · LiveKit Agents voice worker    │
│  agentic-api (Claude Code / Codex via base_url)    │  measurement bots (livekit-rtc)                  │
│  AIPerf agent replay (pinned cores)                │  reach out only through Unix sockets             │
│  SSH tunnel → 127.0.0.1:18000/18100/18880/15558    │  in $XDG_RUNTIME_DIR/lh/                         │
└──────────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                       │ SSH local forwards (systemd --user, restricted key)
                                                       ▼
┌────────────────────────── Phastform: RTX 5060 Ti 16 GB, 188 GiB RAM, always on ─────────────────────────┐
│  vLLM 0.32 · Qwen3-4B-Instruct-2507-FP8 · KV pool 36,400 tokens (5 GiB) · 32k context        :8000     │
│  native CPU offload tier, 20 GiB in /dev/shm · KV events (ZMQ) · priority scheduling                    │
│  vLLM Whisper large-v3-turbo (STT) :8100 · Kokoro-82M on Kokoro-FastAPI (TTS) :8880                     │
│  lh-node sidecar: KV-event relay, engine epoch, health                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Gray: second RTX 5060 Ti, booked windows; same image; reproduction, failover drill, partitioned-GPU upper bound
mustard: Prometheus · Grafana · datasets · run archive      yellow, red: simulator sweeps · Linux builds
```

| Host | Role |
|---|---|
| **Phastform** | The always-on GPU node, running the frozen box (§5). vLLM is pinned to NUMA node 1, the audio servers to node 0 |
| **Gray** | Second GPU node in booked windows, with the identical image. Reproduction on a second card, the node-loss drill, the partitioned-GPU upper bound. Nothing on the critical path depends on it |
| **cyan** | Control host: the control plane, agentic-api, LiveKit and its worker, voice bots and AIPerf, each on its own cores. Every run checks that the load generator itself is not saturated |
| **mustard** | Observability and datasets: Prometheus, Grafana, trace datasets, the run archive |
| **yellow, red** | Simulator sweeps and Linux release builds |

**Network and security** ([ADR-0005](adr/0005-loopback-services-and-node-sticky-kv.md)):
- **Loopback and tunnels.**
  - Every service binds loopback.
  - Hosts reach each other only through SSH local forwards, run as `systemd --user` units.
  - The tunnel key may forward only the listed loopback ports.
- **The voice plane.** It runs in a network namespace whose only interface is loopback. libwebrtc clients bind ICE sockets on every other interface, and a server-side `rtc` block cannot stop that.
- **Sanitized inputs.** The control plane strips the client inputs that vLLM trusts (`X-Vllm-Priority`, `X-Session-ID`, `X-data-parallel-rank`, and the matching body fields) and sets its own.
- **Cache isolation.** Each tenant's `cache_salt` is derived from its API key.

## 3. Core model

The wire types are in [`crates/lh-proto`](../crates/lh-proto) and are described in [PROTOCOL.md](PROTOCOL.md).

### 3.1 Sessions

A session is one conversation, scoped to a tenant. The control plane resolves it from, in order: a trusted caller's `X-Session-ID`; `X-Claude-Code-Session-ID`; the OpenAI `prompt_cache_key`; the LiveKit room and participant; otherwise a minted id. The resolved id is the routing key, the accounting key and the OTel `gen_ai.conversation.id`.

### 3.2 Lifecycle events

There is one event type for every workload, delivered to `POST /v1/events` (idempotent by `event_id`) or inferred from proxied traffic.

| Event | Emitted by | What it changes |
|---|---|---|
| `turn_finished` with `stop_reason = tool_calls` | inferred from the response | Lease → `TOOL_WAIT`; next arrival estimated from session history |
| `tool_wait_started` / `tool_wait_ended` | agentic-api, for server-side tools | An exact ETA, so prefetch can be timed exactly |
| `user_speech_started` | voice worker (VAD onset) | Lease → `USER_SPEAKING`: speech-onset prefetch plus turn-imminent headroom |
| `user_turn_ended` | voice worker (end-of-turn decision) | The realtime turn request follows within milliseconds |
| `interrupted` with `played_ms` | voice worker (barge-in) | Cancel the in-flight stream; truncate the context to what the user heard |
| `session_closed` | any source | Lease → `CLOSED`; the KV becomes ordinary cache |

### 3.3 Leases

A lease is a session's time-bounded claim on KV residency. It records when the session will be back and how fast it must resume:

```text
Lease {
  session_id
  holder            = (node, engine_epoch)   // fenced: an epoch change voids the lease
  class             = realtime | interactive | background
  floor             = CPU | NONE              // guaranteed until expires_at; GPU residency is always best-effort
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

### 3.5 The recompute ledger

There is one row per upstream request, written as Parquet and queried with DuckDB, with per-cause counters in Prometheus.
- **Fields.** Each row records prompt, cached (device and host), expected-reusable and recomputed tokens. Each recomputed span carries one cause: `cold_first_turn`, `prefix_break`, `evicted_gpu_and_cpu`, `lease_expired`, `holder_miss`, `preempted`, `epoch_change` or `unknown`.
- **Validation.** An injected-cause test suite triggers each cause deliberately and requires at least 95% correct attribution.
- **Synthetic requests.** Prefetch warm-ups are tagged and excluded from every indicator.

## 4. Control plane

### 4.1 Crates

The control plane is a Rust workspace of about 3–4k lines, excluding tests. The policy is a library behind a narrow trait, so another router can embed it.

| Crate | Responsibility |
|---|---|
| `lh-proto` | Wire types: lifecycle events, headers, engine lease hints, ledger rows. JSON Schemas for the Python components |
| `lh-core` | The deterministic policy core: lease table, admission, placement, prefetch planner, ledger attribution. Pure functions over an injected clock, with no I/O. `trait Policy { admit, place, on_event, on_complete }` |
| `lh-gateway` | The service (tokio, hyper/axum, tower): ingress and auth, SSE passthrough, cancellation, backpressure, upstream client, event ingest, prefetch executor, ledger sink, metrics. It listens on loopback TCP for the host network and on a Unix socket for the voice plane |
| `lh-node` | A small sidecar on each GPU node. It binds the KV-event subscriber that the engine connects to, detects sequence gaps and resyncs through the engine's replay endpoint, relays events to the control plane, and reports the engine epoch and health |
| `lh-sim` | A trace-driven KV simulator that runs the same `lh-core` code against the tier cost model, plus a Belady-OPT bound |

### 4.2 Request path

1. **Authenticate** the API key and resolve the tenant.
2. **Sanitize** the engine-control headers and body fields.
3. **Resolve** the session and class.
4. **Admit.** Choose among *admit now*, *defer* (queue with a deadline), *make room* (realtime only: tighten the background budget, demote) and *degrade*. Predicted TTFT is queue time plus uncached tokens over the measured prefill rate, plus restore cost (GPU hit ≈ 60 ms + 7.2 µs/token; a CPU restore adds ≈ 8 ms + 23.6 µs/token).
5. **Place** on the holder if its epoch is valid, else on the least-loaded healthy node.
6. **Forward** with `X-Session-ID`, `X-Vllm-Priority` and the lease hint in `kv_transfer_params`.
7. **Stream back** through a bounded per-stream buffer.
   - On client disconnect, abort upstream (vLLM aborts on disconnect).
   - On completion, update the lease and write the ledger row.

### 4.3 Turn-imminent headroom

vLLM's priority orders only its waiting queue and never preempts running requests to admit a waiting one, so a KV pool full of running background agents can hold a voice turn for seconds. While any realtime session is `USER_SPEAKING` or waiting for its first sentence, the control plane:
- caps background prefill admission at a small token budget;
- keeps free KV at least equal to the realtime working set, by deferring new background turns and letting finished ones park in the CPU tier;
- prefetches the speaking session's KV if it is not GPU-resident. A 2k-token restore adds about 60 ms over a GPU hit, well inside 1–3 s of speech.

Aging prevents background starvation. All of this runs on top of the engine's own step cap (`--max-num-batched-tokens 512 --long-prefill-token-threshold 256`), which E2 sweeps.

### 4.4 Prefetch

Prefetch triggers on `next_arrival − restore_time` (ETA from session history, or exact from a tool lifecycle) and on `user_speech_started`. A prefetch is a warm-up request with `max_tokens = 1` that:
- reproduces the exact token prefix, because re-rendering a chat template can drift;
- is rate-limited against a PCIe budget of about 6.25 GB/s effective, roughly 2.6 restores of 16k tokens per second;
- is never cancelled mid-load, because an abort discards the restored KV;
- is tagged `X-LH-Synthetic` and excluded from SLIs, usage and the ledger.

GPU residency after a prefetch is not guaranteed: restored blocks return to the LRU, so a prefetch is timed close to the arrival. vLLM draft PR #58198 adds a request-free prefetch primitive, which would replace the warm-up request.

### 4.5 Cache contracts

Anthropic `cache_control` breakpoints and `ttl` (`5m`, `1h`) become lease floors and `expires_at`. Usage is reported in Anthropic's fields: `cache_read_input_tokens`, `cache_creation_input_tokens`, and the 5-minute and 1-hour ephemeral counts.

| Term | Meaning |
|---|---|
| Guarantee | Inside the TTL, the prefix up to the breakpoint stays at least in host RAM. GPU residency is best-effort |
| Exceptions | Node failure; model or config change; prefix mismatch |
| Violations | Every other miss inside the TTL. The count is an SLI with target 0 |

OpenAI `prompt_cache_options` is deferred.

### 4.6 Engine-side policy

The CPU tier is vLLM's native `OffloadingConnector` with `offload_prompt_only: false`, so generated tokens are written through as well. Leaseholder's plugins in [`python/leaseholder_vllm`](../python/leaseholder_vllm) are optional:
- **`LeaseConnector`** adds per-request hit receipts (GPU tokens and CPU tokens) and mirrors `X-Session-ID` into the lease hint.
- **`LeasePolicy`** makes CPU-tier eviction lease-aware. It ships only where the simulator shows that CPU-tier eviction binds at the chosen RAM ratio. It is soft: it refuses a store only where LRU would refuse it too.

The live check on the frozen box confirmed three things:
- the hint reaches both plugins;
- `X-Session-ID` appears in GPU-tier KV events (CPU-tier events carry no session id);
- a 7,921-token prompt is restored from the CPU tier after a GPU cache reset.

## 5. Data plane: the frozen box

Every experiment runs on one frozen configuration with audio co-located: [`deploy/phastform/`](../deploy/phastform/), with the measured values in [BENCHMARKS.md](BENCHMARKS.md).

| Component | Configuration | GPU memory |
|---|---|---|
| LLM | vLLM 0.32.0, `Qwen/Qwen3-4B-Instruct-2507-FP8`, 32k context, `--kv-cache-memory` 5 GiB (36,400 tokens), prefix caching, priority scheduling, step cap 512/256, prompt token details, KV cache metrics, KV events, native CPU offload (20 GiB, write-through) | ≈ 10.6 GiB |
| STT | vLLM 0.32.0 serving `openai/whisper-large-v3-turbo` at `/v1/audio/transcriptions` | ≈ 2.4 GiB |
| TTS | Kokoro-82M on Kokoro-FastAPI 0.9.1 at `/v1/audio/speech` | ≈ 1.0 GiB |

**Placement and environment:**
- **Units.** Each server is a `systemd --user` unit. `leaseholder.target` starts STT and TTS first, then the LLM, so vLLM's free-memory check sees the audio servers resident. Each unit counts as started only once its health endpoint answers.
- **NUMA.** vLLM runs on node 1, local to the GPU (cores 8–15, memory preferred on node 1); the audio servers run on node 0. The CPU tier's shared-memory region prefers node 1 and spills to node 0; the measured cost of a remote tier is under 1% of pinned bandwidth.
- **Pinned environment.** vLLM 0.32.0 with torch 2.13.0+cu130 serves both the LLM and Whisper. It uses the CUDA 13.0 toolkit for DeepGEMM's JIT, curand header links for FlashInfer's JIT, and `nvidia-cudnn-cu13==9.27.0.42`. Kokoro runs on torch 2.8.0+cu128. Lock files are in `deploy/phastform/requirements/`.
- **MPS vs time-slicing.** Under time-slicing the audio servers slow by 43–44% during long prefills. One measurement in W1 decides between the two.
- **Fallbacks.** STT can fall back to faster-whisper INT8 (1.2 GiB). TTS can switch to Qwen3-TTS via vLLM-Omni (4.3 GiB, which shrinks the KV pool to about 20k tokens).

## 6. API layer: agentic-api

- **Upstream.** agentic-api's single upstream, `llm_api_base`, points at the control plane.
- **Clients.** Claude Code connects through Anthropic Messages, Codex through Responses (whose default `prompt_cache_key` is its session id).
- **Direct paths.** Voice and AIPerf traffic speak Chat Completions straight to the control plane, which keeps the realtime path to one hop.
- **Upstream contributions** that help: a header allowlist on the executor path (#461), a typed lifecycle-event sink (#364), cache-usage metrics (#315), and performance-validation data (#367).

## 7. Realtime voice pipeline

- **Components.** All stock: LiveKit server 1.13 and LiveKit Agents 1.8.8 in the voice-plane namespace on cyan, with Silero VAD and the turn detector on CPU. The OpenAI STT, LLM and TTS plugins reach the frozen box through Unix-socket forwards.
- **Headers.** The LLM plugin sends `X-LH-Class: realtime` and the room-scoped session id.
- **Lifecycle.**
  - The worker posts `user_speech_started`, `user_turn_ended`, `playback_started`, `playback_ended` and `interrupted` to the control plane.
  - Speech onset triggers prefetch and headroom.
  - On barge-in, the LLM stream closes, the control plane aborts upstream, and the transcript is truncated to `played_ms`.
- **Measurement bots.** These are `livekit-rtc` Python bots in the same namespace. They play utterances with known end timestamps and record the agent's track; V2V, barge-in-to-silence and gaps are computed from waveforms on one clock. VAD and turn-detection settings are fixed across arms, and end-of-utterance delay is reported as its own stage.

## 8. Simulator

`lh-sim` decides, before any engine code is written, how much a retention policy can win over LRU plus CPU offload, and where.
- **Traces.** It replays the SemiAnalysis Claude Code traces, block-scaled 32× (32k context) or 64× (16k), with outputs scaled the same way; miss-ratio curves show fidelity before and after scaling.
- **Arms.**
  - GPU-only LRU; LRU + CPU; ARC + CPU;
  - a fixed-TTL hold;
  - prefetch by session history; prefetch by oracle;
  - lease-aware CPU eviction; active-session admission;
  - Belady-OPT.
- **Calibration.** The tier cost model is checked against 2–3 GPU operating points, with a target error of 10–15%.
- **Code reuse.** The arms are `lh-core` policies, so simulator and production share code.

## 9. Observability

- **Indicators and targets.** See [SLO.md](SLO.md).
- **Dashboards, as code in `ops/`:**
  1. the voice turn waterfall and SLOs;
  2. KV and leases: pool occupancy, tier residency, prefetch outcomes, PCIe budget;
  3. admission and the ledger: decisions, deferrals, causes, rejections.
- **Alerts.** Multi-window SLO burn rate, KV-event sequence gaps, engine epoch changes, tunnel loss.
- **Runs.** Each run writes its configuration, host, start and end times and raw samples next to its results, and results are archived on mustard.

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

## 11. Testing

| Layer | Tests |
|---|---|
| `lh-proto` | Wire format pinned by examples; schemas kept current; ledger accounting identities |
| `lh-core` | Unit and property tests for the lease invariants; deterministic simulation with seeded event sequences and an injected clock |
| `lh-gateway` | Integration against a mock upstream (llm-d-inference-sim or a recorded server): mid-stream disconnect, backpressure, header sanitization, admission decisions, event idempotency |
| Ledger | The injected-cause suite: at least 95% correct attribution, with the `unknown` share reported |
| Engine plugins | CPU tests against vLLM's real offload manager (in CI); a live check on the GPU node |
| Voice | Hermetic LiveKit barge-in and header test in the voice-plane namespace; a nightly bot smoke test on the frozen box |
| Performance | A nightly fixed scenario on the GPU node against stored baselines; a weekly compatibility run against the newest vLLM |
| CI | GitHub Actions on CPU: rustfmt, clippy, ruff, unit, property, simulator and integration tests |

## 12. Prior art

See [PRIOR-ART.md](PRIOR-ART.md). In short:
- leases and TTL holds: llm-d-router #2273, AIBrix #2814, Continuum, SGLang;
- active-session admission: ThunderAgent, vllm-project/router Progress-TTL, llm-d;
- CPU-tier prefetch: Tokencake, MORI, vLLM #58198;
- turn priority and session affinity: llm-d.

Leaseholder's contribution is a combination on one commodity GPU:
- one lifecycle-event schema shared by agents and voice;
- a never-shed realtime class with turn-imminent headroom, measured against tuned stock vLLM;
- a cache contract honored with a stated guarantee;
- a validated recompute ledger;
- simulator bounds against Belady-OPT.
