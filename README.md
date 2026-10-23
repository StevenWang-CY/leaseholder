# Leaseholder

[![CI](https://github.com/StevenWang-CY/leaseholder/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenWang-CY/leaseholder/actions/workflows/ci.yml)

**SLO-tiered serving on a shared GPU.** Leaseholder runs a never-shed realtime tier beside a harvested best-effort tier on one vLLM replica, and times the protection from conversation lifecycle events. Voice calls are the hardest realtime client; coding agents are the heavy-KV background. *One GPU, two clocks.*

API providers sell the same two tiers (OpenAI Fast and Flex, Google Priority and Flex, Anthropic Fast mode and Batch) and harvest the cheap one by shedding it and having clients retry. On one replica that is not enough: long agent prefills slow every stage of a voice turn, and a waiting voice turn cannot preempt running agent work when the KV cache is full. Leaseholder takes the lifecycle events of both workloads (speech onset, end of turn and barge-in from voice; tool waits from agents) and:
- reserves GPU headroom for the next spoken turn before its request exists, and frees it by parking background sessions in host RAM rather than killing them;
- throttles background prefill while a voice turn is in flight;
- brings parked agent sessions back before their tools return;
- attributes every recomputed prompt token to a cause.

These mechanisms are built from week 3 (W3) of a 12-week plan; [Status](#status) says what runs today. Classes, budgets, eviction and parking with prefetch already ship or are published; here they are baselines. [docs/PRIOR-ART.md](docs/PRIOR-ART.md) says what is new and what is not.

## The problem, measured on one RTX 5060 Ti 16 GB

| | vLLM 0.32, Qwen3-4B-FP8, the frozen box (W1, 2026-10-22) |
|---|---|
| Resuming a 16k-token agent turn | **97 ms** from the GPU cache · **509 ms** restored from host RAM · **3.26 s** recomputed |
| GPU KV pool with speech-to-text, LLM and text-to-speech co-located | **36,400 tokens**: two or three agent contexts |
| One voice turn on the server (STT → first LLM sentence → first TTS audio), p50 | **589 ms** idle · **1,261 ms** beside one stream of cold 8k-token agent prefills · **2,279 ms** beside two |
| Voice-to-voice (V2V) through LiveKit with fixed endpointing, read from waveforms, p95 | **1,760 ms** alone · **2,312 ms** beside a 16-session agent replay, against an SLO of at most +250 ms |
| What stock priority scheduling does about it | vLLM preempts among running requests, but a waiting voice turn cannot preempt running work when the KV cache is full ([#40004](https://github.com/vllm-project/vllm/issues/40004)) |
| What a control plane in front costs | `lh-gateway` adds **0.38 ms** to a voice request's first token: the paired median against the engine's own socket |

The frozen box is the one pinned configuration every experiment runs on: the LLM, STT and TTS servers sharing the GPU. More numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## How it works

```text
 voice: LiveKit worker ──┐   Claude Code / Codex → agentic-api ──┐   agent replay (AIPerf) ──┐
 speech · turn · barge-in      cache_control TTLs · tool lifecycle       background class    │
                         ▼                                       ▼                           ▼
          ┌─────────────────────────────── Leaseholder (Rust) ───────────────────────────────┐
          │ lifecycle events → lease table (epoch-fenced) → admission · headroom · prefetch  │
          │ placement (holder-first) · SSE passthrough with cancellation · recompute ledger  │
          └───────────────────────────────────────────┬──────────────────────────────────────┘
                                                      ▼
            vLLM 0.32: Qwen3-4B-FP8 · native CPU offload (20 GiB) · KV events · priority · step cap
            + Whisper (STT) and Kokoro (TTS) on the same 16 GB GPU
```

- **Mechanism in the engine, policy in the control plane.**
  - The engine decides what to keep, from a lease hint carried on each request.
  - The control plane decides when to bring KV back, from predictions.
  - Engine-side code is limited to vLLM's own plugin points, plus the scheduler fix proposed upstream.
- **One event schema for agents and voice.** A tool return and a speech onset trigger the same prefetch and headroom mechanism.
- **Realtime is never shed.** Under pressure the control plane makes room by parking and deferring background work, then degrades the reply. It never rejects a voice turn.
- **KV never crosses the network.** Sessions stick to the node that holds them. Every service binds loopback; hosts talk over SSH tunnels; the voice plane runs in a network namespace with no route out.

The full design, with what exists and what is planned, is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

State after week 2 (W2) of a 12-week plan:

| Component | State |
|---|---|
| Frozen box on the GPU node: vLLM 0.32 LLM + Whisper STT + Kokoro TTS, pinned environment, NUMA-pinned `systemd --user` units, smoke test | **Deployed** on Phastform, and on Gray as the second GPU node for booked windows; paused when idle ([`deploy/phastform`](deploy/phastform), [`deploy/gray`](deploy/gray)) |
| Engine variants for E2: vLLM #54530 ported to 0.32.0, SGLang 0.5.22, CUDA MPS, engine flags from an arm file | **Deployed** ([`deploy/phastform/variants`](deploy/phastform/variants)) |
| Control host: restricted SSH tunnels, `lh-gateway`, and LiveKit with the voice agent in a network namespace with no route out; SMG for one E2 arm | **Deployed** ([`deploy/cyan`](deploy/cyan)) |
| `lh-proto`: lifecycle events, headers, engine lease hint, ledger rows, with JSON Schemas | **v0** ([`crates/lh-proto`](crates/lh-proto)) |
| `lh-core`: the tier cost model, KV geometry, the step-time model fitted on the frozen box, and the admission state machine with E2's baseline policies (static cap, token reservation, llm-d flow control emulated, preempt-resubmit) | **W2** ([`crates/lh-core`](crates/lh-core)); Leaseholder's own policy (headroom, critical-window throttle, degrade) arrives in W3 |
| `lh-gateway`: auth, sanitization, sessions and classes, SSE relay with cancellation and backpressure, lifecycle-event ingest, metrics, and an admission point running `lh-core`'s policies, with the class sent upstream as vLLM's priority header, SGLang's body field or SMG's header | **W2, deployed on cyan** ([`crates/lh-gateway`](crates/lh-gateway)) |
| `lh-sim`: the agent replay through a model of vLLM's two-tier KV cache and its step loop (calibrated within 8% of resume TTFT at 8 and 16 sessions), with stock arms, E1's information-level arms, a clairvoyant reference and the regime map | **W2** ([`crates/lh-sim`](crates/lh-sim)); E1's gate (G-sim) came out negative |
| `lh-node` | W5, with leases and engine epochs |
| `leaseholder_vllm`: lease-aware CPU-tier `CachePolicy`, connector with per-request hit receipts | **v0** ([`python/leaseholder_vllm`](python/leaseholder_vllm)); CPU-tested against vLLM's real offload manager, live-checked on the GPU; off in E2, as G-sim decided |
| `leaseholder_bench`: block-scaled Claude Code traces, AIPerf loader and endpoint plugin, live-reply replay, KV lifecycle comparison, tier and voice-turn probes, and `lh-bench e2`, the E2 harness (arm files, campaigns, per-point summaries, the frontier report) | **W2** ([`python/leaseholder_bench`](python/leaseholder_bench), [`benchmarks/e2`](benchmarks/e2)) |
| `leaseholder_voice`: the voice agent (`lh-voice-worker`), measurement callers that read latency from waveforms (`lh-voice bots`), LiveKit plugin factory, barge-in test | **v0** ([`python/leaseholder_voice`](python/leaseholder_voice)); measured alone and beside the agent replay in every E2 point |

The experiments are planned in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The headline is E2: the voice SLO frontier against tuned stock vLLM while background agents harvest the GPU.

## Quickstart

Development needs Rust 1.98, uv and Python 3.12. No GPU is required.

```sh
make check                                   # lint and every test CI runs (below)
make schema                                  # regenerate crates/lh-proto/schema after changing a wire type
VLLM_SRC=~/src/vllm-v0.32.0 make test-vllm   # the engine plugin tests against an existing vLLM checkout
```

`make check` runs rustfmt, clippy and ruff, the Rust tests, the engine plugin tests against vLLM's real offload manager (in a sparse vLLM v0.32.0 checkout it clones once into `~/.cache/leaseholder`), and the Python unit tests. Deploying to the hosts is in [ops/RUNBOOK.md](ops/RUNBOOK.md#deploy).

## Repository layout

```text
crates/lh-proto/          wire types and their JSON Schemas
crates/lh-core/           the cost and step-time models and KV geometry, shared with the simulator; the admission policies
crates/lh-gateway/        the control-plane service in front of vLLM, with its overhead benchmark
crates/lh-sim/            the KV cache simulator: AIPerf's replay, vLLM's two tiers, policy arms
python/leaseholder_vllm/  engine-side plugins for vLLM 0.32, with CPU tests and a live GPU check
python/leaseholder_bench/ trace conversion, AIPerf plugins and configs, live-reply replay driver, latency probes
python/leaseholder_voice/ the voice agent, measurement callers, LiveKit integration and its hermetic barge-in test
benchmarks/               runs against upstream projects, with their scripts (agentic-api's HTTP baseline)
deploy/                   per-host bootstrap, systemd units, tunnels, smoke test
ops/                      the runbook; dashboards and alert rules are planned
docs/                     architecture, protocol, SLOs, experiments, benchmarks, prior art, ADRs
```

## Why not llm-d, Dynamo or vllm-project/router?

- **Most mechanisms already ship.**
  - llm-d v1.0 has flow control with priority bands, holdback ceilings, SLO-deadline ordering and in-flight eviction of sheddable requests.
  - SMG, a Rust gateway, reserves slots per priority class and preempts lower classes before their first byte.
  - Dynamo queues by priority but does not reserve capacity.
  - vllm-project/router admits agent Programs with a logical KV TTL.
  - In the engines, vLLM and SGLang preempt by priority, but neither lets a waiting request preempt running work when the KV cache is full.

  Leaseholder claims none of these. llm-d flow control (emulated), SMG and both engines' priority preemption are baselines it is measured against.
- **What it adds:**
  - KV headroom for the next voice turn, held from speech onset or end of turn and freed by parking background sessions in host RAM instead of killing them;
  - an engine fix that lets a waiting request preempt running work when the KV cache is full, keeping the victim's KV (open in vLLM as #40004; a draft in SGLang, #45686);
  - a recompute ledger validated by injected causes;
  - measurements of when lifecycle timing beats static reservation and engine priority, and when it does not.
- **The gateway is a harness.** `lh-gateway` is a small Rust proxy for experiments on one GPU without Kubernetes. It runs `lh-core`'s admission policies: E2's baselines since W2, Leaseholder's own from W3. It does not compete with fleet routers. The policy is to be proposed where production runs:
  - the engine fix, as a vLLM scheduler PR under #40004;
  - the agent-side lease and prefetch policy, in vllm-project/router on RFCs #427 (engine-confirmed retention) and #428 (prefetch before a tool returns);
  - the lifecycle-event sink and cache-control pass-through, in agentic-api (#364, #330).

  None of these is merged yet.

See [docs/PRIOR-ART.md](docs/PRIOR-ART.md) for sources and status.

## License

Apache-2.0
