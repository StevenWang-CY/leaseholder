# ADR-0001: A session-aware control plane in front of stock vLLM

- **Status:** accepted, 2026-10-22
- **Deciders:** Chuyue Wang

## Context

One 16 GB GPU has to serve realtime voice turns and a swarm of tool-using coding agents. Stock vLLM 0.32 schedules everything from one queue:
- priority scheduling orders the waiting queue and picks which running request to preempt when one cannot grow, but never preempts running work to admit a waiting request;
- prefix-cache eviction is LRU, with no notion of when a session will be back;
- a 16k-token agent turn costs 0.18 s from the GPU cache, 0.57 s from host RAM and 4.75 s recomputed (measured on this card on 2026-10-21 with BF16 weights; [BENCHMARKS.md](../BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0) has the frozen box's W1 values).

There are three ways to add session awareness:
1. patch or fork the vLLM scheduler;
2. adopt an external router (llm-d, Dynamo, vllm-project/router);
3. put a thin control plane in front of the engine. It sets headers, budgets and admission, and uses the engine's supported extension points for anything that must run inside it.

Option 1 tracks a codebase that releases every two weeks, and its patches rarely merge. Option 2 brings Kubernetes and endpoint scoring. The problems here are timing policies on a single node, plus accounting.

## Decision

Build option 3: a Rust service (tokio, hyper/axum, tower).
- The policy is a pure, deterministic library (`lh-core`) behind a narrow trait: `admit`, `place`, `on_event`, `on_complete`. The simulator (`lh-sim`) runs the same code, and another router could embed it.
- The engine stays stock. Engine-side code exists only behind vLLM's own plugin points (`CachePolicy` and connector modules, see ADR-0004).
- Wire types are defined once, in `lh-proto` (ADR-0002).

## Consequences

- Upgrading vLLM is a pin change plus a compatibility run. There is no fork to rebase.
- Some policies cannot be expressed outside the engine, the main one being preemption of running work. The control plane compensates with admission and turn-imminent headroom (ADR-0003). The engine's step cap (`--max-num-batched-tokens`, `--long-prefill-token-threshold`) bounds the rest.
- The control plane adds one hop. Its budget is ≤ 2 ms per request at p99, and it is measured on every run.
- Rust matches agentic-api, the API layer in front of it, and keeps the data path cheap on slow hosts.

## Update (2026-10-22, goal-fit review)

The decision stands: the control plane stays in front of stock vLLM, and its policy lives in `lh-core` so routers that already run in production can adopt it. The one engine change Leaseholder needs, letting a waiting high-priority request preempt running work under KV pressure while keeping the victim's KV ([vLLM #40004](https://github.com/vllm-project/vllm/issues/40004)), is proposed upstream as a vLLM PR rather than carried as a fork.

