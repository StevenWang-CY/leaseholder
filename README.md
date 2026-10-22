# Leaseholder

**One GPU, two clocks.** Leaseholder is a session-aware serving control plane for vLLM. It lets one 16 GB GPU serve realtime voice calls and a swarm of coding agents at the same time, within explicit SLOs.

Agents run on tool time: seconds to minutes between turns, with long contexts they come back to. Voice runs on speech time: the caller expects an answer about a second after they stop talking. Stock engines schedule both from one queue and evict idle sessions by LRU. Leaseholder turns the lifecycle events of both workloads into **KV-cache leases**. The inputs are tool waits for agents, and speech onset, end of turn and barge-in for voice. With the leases it:
- holds GPU headroom for the next spoken turn;
- parks idle agent sessions in host RAM and brings them back before their tools return;
- honors API cache TTLs;
- attributes every recomputed prompt token to a cause.

## The problem, measured on one RTX 5060 Ti 16 GB

| | vLLM 0.32, measured 2026-10-21/22 |
|---|---|
| Resuming a 16k-token agent turn | **0.18 s** from the GPU cache · **0.57 s** restored from host RAM · **4.75 s** recomputed |
| GPU KV pool with speech-to-text, LLM and text-to-speech co-located | **36,400 tokens**: two or three agent contexts |
| One voice turn on the server (STT → first LLM sentence → first TTS audio) | **415 ms** idle, **876 ms** median while background agents run long prefills |
| What stock priority scheduling does about it | Orders only the waiting queue; it never preempts running work to admit a waiting voice turn |

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
- **Realtime is never shed.** Under pressure the control plane makes room by deferring and demoting background work, then degrades the reply. It never rejects a voice turn.
- **KV never crosses the network.** Sessions stick to the node that holds them. Every service binds loopback; hosts talk over SSH tunnels; the voice plane runs in a network namespace with no route out.

The full design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Week 0 of a 12-week plan.

| Component | State |
|---|---|
| Frozen box on the GPU node: vLLM 0.32 LLM + Whisper STT + Kokoro TTS, pinned environment, NUMA-pinned `systemd --user` units, smoke test | **Running** ([deploy/phastform](deploy/phastform)) |
| Control host: restricted SSH tunnels, LiveKit in a network namespace with no route out | **Running** ([deploy/cyan](deploy/cyan)) |
| `lh-proto`: lifecycle events, headers, engine lease hint, ledger rows, with JSON Schemas | **v0** ([crates/lh-proto](crates/lh-proto)) |
| `leaseholder_vllm`: lease-aware CPU-tier `CachePolicy`, connector with per-request hit receipts | **v0**. CPU-tested against vLLM's real offload manager; live-checked on the GPU |
| `leaseholder_bench`: block-scaled Claude Code traces, AIPerf loader and endpoint plugin, live-reply replay | v0 |
| `leaseholder_voice`: LiveKit plugin factory, barge-in and header integration test | v0 |
| `lh-core`, `lh-gateway`, `lh-node`, `lh-sim` | W1 |

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
crates/lh-proto/          wire types and their JSON Schemas (lh-core, lh-gateway, lh-node, lh-sim join in W1)
python/leaseholder_vllm/  engine-side plugins for vLLM 0.32, with CPU tests and a live GPU check
python/leaseholder_bench/ trace conversion, AIPerf plugins and configs, live-reply replay driver
python/leaseholder_voice/ LiveKit Agents integration and its hermetic barge-in test
deploy/                   per-host bootstrap, systemd units, tunnels, smoke test
ops/                      runbook (dashboards and alert rules arrive with the control plane)
docs/                     architecture, protocol, SLOs, experiments, benchmarks, prior art, ADRs
```

## Why not llm-d, Dynamo or vllm-project/router?

- **Scope.** Those are fleet routers built around Kubernetes and endpoint scoring. Leaseholder targets single-node timing policies on shared hosts without Kubernetes, and makes no attempt to beat them at routing.
- **Overlap.** Its parts overlap known work: leases (llm-d-router #2273), active-session admission (ThunderAgent, Progress-TTL), CPU-tier prefetch (vLLM #58198) and turn priority (llm-d), which is a baseline here.
- **What it adds** is a combination, measured on one commodity GPU:
  - one lifecycle-event schema shared by agents and voice;
  - a never-shed realtime class with turn-imminent headroom;
  - Anthropic cache TTLs honored with a stated guarantee;
  - a recompute ledger validated by injected causes;
  - simulator bounds against Belady-OPT.

See [docs/PRIOR-ART.md](docs/PRIOR-ART.md) for sources and status.

## License

Apache-2.0
