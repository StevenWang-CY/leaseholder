# Leaseholder

[![CI](https://github.com/StevenWang-CY/leaseholder/actions/workflows/ci.yml/badge.svg)](https://github.com/StevenWang-CY/leaseholder/actions/workflows/ci.yml)

**SLO-tiered serving on a shared GPU.** Leaseholder runs a never-shed realtime tier beside a harvested best-effort tier on one vLLM replica, and times the protection from conversation lifecycle events. Voice calls are the hardest realtime client; coding agents are the heavy-KV background. *One GPU, two clocks.*

API providers sell the same two classes (OpenAI Fast and Flex, Google Priority and Flex, Anthropic Priority and Batch) and harvest the cheap one by killing and retrying it. On one replica that is not enough: long agent prefills slow every stage of a voice turn, and a waiting voice turn cannot preempt running agent work when the KV cache is full. Leaseholder takes the lifecycle events of both workloads (speech onset, end of turn and barge-in from voice; tool waits from agents) and:
- reserves GPU headroom for the next spoken turn before its request exists, and frees it by parking background sessions in host RAM rather than killing them;
- throttles background prefill while a voice turn is in flight;
- brings parked agent sessions back before their tools return;
- attributes every recomputed prompt token to a cause.

Classes, budgets, eviction, and parking with prefetch already ship or are published; here they are baselines. [docs/PRIOR-ART.md](docs/PRIOR-ART.md) says what is new and what is not.

## The problem, measured on one RTX 5060 Ti 16 GB

| | vLLM 0.32, Qwen3-4B-FP8, the frozen box (W1, 2026-10-22) |
|---|---|
| Resuming a 16k-token agent turn | **97 ms** from the GPU cache · **509 ms** restored from host RAM · **3.26 s** recomputed |
| GPU KV pool with speech-to-text, LLM and text-to-speech co-located | **36,400 tokens**: two or three agent contexts |
| One voice turn on the server (STT → first LLM sentence → first TTS audio), p50 | **589 ms** idle · **1,261 ms** beside one stream of cold 8k-token agent prefills · **2,279 ms** beside two |
| What stock priority scheduling does about it | vLLM preempts among running requests, but a waiting voice turn cannot preempt running work when the KV cache is full ([#40004](https://github.com/vllm-project/vllm/issues/40004)) |

More numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## How it works

```text
 voice: LiveKit worker ──┐   Claude Code / Codex → agentic-api ──┐   agent replay (AIPerf) ──┐
 speech · turn · barge-in      cache_control TTLs · tool lifecycle       background class      │
                         ▼                                       ▼                              ▼
          ┌─────────────────────────────── Leaseholder (Rust) ───────────────────────────────┐
          │ lifecycle events → lease table (epoch-fenced) → admission · headroom · prefetch   │
          │ placement (holder-first) · SSE passthrough with cancellation · recompute ledger   │
          └───────────────────────────────────────────┬──────────────────────────────────────┘
                                                      ▼
            vLLM 0.32: Qwen3-4B-FP8 · native CPU offload (20 GiB) · KV events · priority · step cap
            + Whisper (STT) and Kokoro (TTS) on the same 16 GB GPU
```

- **Mechanism in the engine, policy in the control plane.**
  - The engine decides what to keep, from a lease hint carried on each request.
  - The control plane decides when to bring KV back, from predictions.
  - Engine-side code is limited to vLLM's own plugin points.
- **One event schema for agents and voice.** A tool return and a speech onset trigger the same prefetch and headroom mechanism.
- **Realtime is never shed.** Under pressure the control plane makes room by parking and deferring background work, then degrades the reply. It never rejects a voice turn.
- **KV never crosses the network.** Sessions stick to the node that holds them. Every service binds loopback; hosts talk over SSH tunnels; the voice plane runs in a network namespace with no route out.

The full design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Week 1 of a 12-week plan.

| Component | State |
|---|---|
| Frozen box on the GPU node: vLLM 0.32 LLM + Whisper STT + Kokoro TTS, pinned environment, NUMA-pinned `systemd --user` units, smoke test | **Deployed**; paused when idle ([deploy/phastform](deploy/phastform)) |
| Control host: restricted SSH tunnels, `lh-gateway`, and LiveKit with the voice agent in a network namespace with no route out | **Deployed** ([deploy/cyan](deploy/cyan)) |
| `lh-proto`: lifecycle events, headers, engine lease hint, ledger rows, with JSON Schemas | **v0** ([crates/lh-proto](crates/lh-proto)) |
| `leaseholder_vllm`: lease-aware CPU-tier `CachePolicy`, connector with per-request hit receipts | **v0**. CPU-tested against vLLM's real offload manager; live-checked on the GPU |
| `leaseholder_bench`: block-scaled Claude Code traces, AIPerf loader and endpoint plugin, live-reply replay, KV lifecycle comparison, tier and voice-turn probes | **v0**. Measured E0 and the server-side voice turn on the frozen box; replays the 32× corpus against it (688 requests, 0 errors) |
| `leaseholder_voice`: the voice agent (`lh-voice-worker`), measurement callers that read latency from waveforms (`lh-voice bots`), LiveKit plugin factory, barge-in test | **v0**. Voice alone: 1,614 ms voice-to-voice p50, 1,760 ms p95. Beside a 16-session agent replay, p95 rises to 2,312 ms, the problem E2 takes on |
| `lh-gateway`: auth, sanitization, sessions and classes, SSE relay with cancellation and backpressure, lifecycle-event ingest, metrics | **v0, deployed on cyan** ([crates/lh-gateway](crates/lh-gateway)). Carries the voice agent and the agent replay without errors; adds 0.4 ms to a voice request's first token, paired against the engine's own socket |
| `lh-core`: the tier cost model and KV geometry | **v0** ([crates/lh-core](crates/lh-core)) |
| `lh-sim`: the agent replay through a model of vLLM's two-tier KV cache, with stock arms, a clairvoyant reference and miss-ratio curves | **v0** ([crates/lh-sim](crates/lh-sim)). Driven by W0's measured request times, it reproduces the engine's GPU and CPU hit shares within half a point; the full corpus simulates in 6 s on one core |
| `lh-node` | W3 |

The experiments are planned in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The headline is E2: the voice SLO frontier against tuned stock vLLM while background agents harvest the GPU.

## Quickstart

Development needs Rust 1.98, uv and Python 3.12. No GPU is required.

```sh
make check                        # rustfmt, clippy, cargo test, ruff, and the Python unit tests
make schema                       # regenerate crates/lh-proto/schema after changing a wire type
VLLM_SRC=~/src/vllm-v0.32.0 make test-vllm   # engine plugin tests against vLLM's offload manager (source checkout)
```

To deploy to the hosts, see [ops/RUNBOOK.md](ops/RUNBOOK.md):

```sh
export LH_SSH="ssh -F $HOME/.ssh/penn_config"
deploy/deploy.sh phastform.cis.upenn.edu   # then deploy/phastform/bootstrap.sh once on the node
deploy/deploy.sh cyan.cis.upenn.edu
```

## Repository layout

```text
crates/lh-proto/          wire types and their JSON Schemas (lh-node joins in W3)
crates/lh-core/           the tier cost model and KV geometry, shared by the simulator and the gateway
crates/lh-gateway/        the gateway in front of vLLM, with its overhead benchmark
crates/lh-sim/            the KV cache simulator: AIPerf's replay, vLLM's two tiers, policy arms
python/leaseholder_vllm/  engine-side plugins for vLLM 0.32, with CPU tests and a live GPU check
python/leaseholder_bench/ trace conversion, AIPerf plugins and configs, live-reply replay driver, latency probes
python/leaseholder_voice/ the voice agent, measurement callers, LiveKit integration and its hermetic barge-in test
benchmarks/               runs against upstream projects, with their scripts (agentic-api's HTTP baseline)
deploy/                   per-host bootstrap, systemd units, tunnels, smoke test
ops/                      runbook (dashboards and alert rules arrive with the control plane)
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
- **The gateway is a harness.** `lh-gateway` is a small Rust proxy that carries the `lh-core` policy for experiments on one GPU without Kubernetes; it does not compete with fleet routers. The policy is to be proposed where production runs:
  - the engine fix, as a vLLM scheduler PR under #40004;
  - the agent-side lease and prefetch policy, in vllm-project/router on RFCs #427 (engine-confirmed retention) and #428 (prefetch before a tool returns);
  - the lifecycle-event sink and cache-control pass-through, in agentic-api (#364, #330).

  None of these is merged yet.

See [docs/PRIOR-ART.md](docs/PRIOR-ART.md) for sources and status.

## License

Apache-2.0
