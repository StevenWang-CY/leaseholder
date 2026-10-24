# ADR-0006: The engine change carries the realtime tier; the control plane is a harness

- **Status:** accepted, 2026-10-24
- **Deciders:** Chuyue Wang
- **Supersedes in part:** [ADR-0001](0001-control-plane-in-front-of-stock-vllm.md) (the engine stays stock, and the control plane compensates with admission and headroom), [ADR-0003](0003-realtime-is-never-shed.md) (room made by the control plane), and the lease and prefetch consequences of [ADR-0002](0002-one-lifecycle-event-schema.md) and [ADR-0004](0004-native-cpu-offload-tier.md)

## Context

E2 measured two calls beside 24 coding-agent sessions whose contexts fill the GPU's KV pool, on one RTX 5060 Ti ([BENCHMARKS.md](../BENCHMARKS.md#voice-under-agent-load-e2)):
- **Nothing deployable today fixes it.** Tuned stock vLLM leaves the voice tail at 10.1 s at p95. Every gateway baseline and both engines' priority preemption leave it at 8.8–12.1 s. Aborting background work before its first byte and resubmitting it reaches 2.6 s, at the cost of the agents' tail.
- **The engine change does.** With the two vLLM PRs and a 48-token window the tail is 2.19 s, against an SLO of 2.02 s, at 94% of stock's background throughput. Leaseholder's arm 9 reaches 2.06 s with every reply capped at 16 tokens.
- **The control plane could not make room.** Leaseholder's admission policy holds KV headroom from speech onset and throttles background prefill while a turn is answered. In front of stock vLLM it left the tail where stock does (10,993 ms). Beside the engine change its mechanisms could not act. Running agent requests held the pool, and a control plane can only defer new ones, so the headroom was never there: degrade capped 60 of 60 replies in every run. The throttle, checked after the headroom, deferred nothing.
- **E1 measured the agent side the same way.** In the simulator, even an oracle that knows when each agent returns closes under 7% of resume TTFT at p50 ([EXPERIMENTS.md](../EXPERIMENTS.md#e1-parking)).

## Decision

- **The engine carries the realtime tier.** It runs priority scheduling with two changes:
  - [#63350](https://github.com/vllm-project/vllm/pull/63350): a waiting request of a higher priority preempts running lower-priority work for KV, at most two requests a step;
  - [#63351](https://github.com/vllm-project/vllm/pull/63351): lower-priority prefill sits out the request's first tokens, with a window long enough for the reply's first sentence (48 tokens for this voice agent).

  There is no reserve. Until the PRs merge, the `make-room` variant runs the same change on vLLM 0.32.0.
- **The priority comes from the caller's identity.** A gateway maps each tenant to a class and caps it, so a client cannot raise its own priority.
- **No admission policy in front.** `lh-gateway` stays the experiment harness: its baselines and the `leaseholder` policy remain E2's arms, and the measured configuration ran without either.
- **Not pursued:**
  - leases, prefetch, the per-node sidecar (`lh-node`), cache contracts and the recompute ledger;
  - they would be reopened for a regime where knowing an agent's return closes more than 10% of resume TTFT, or for an engine whose running KV a control plane can reclaim.

## Consequences

- **What deploys is a set of vLLM flags** ([PRODUCTION.md](../PRODUCTION.md)). The change is carried upstream, not as a fork; until it merges, the variant pins vLLM 0.32.0, and its build refuses any other version.
- **The other ADRs stand with these changes:**
  - ADR-0001's design describes the experiment harness, not the deployment;
  - ADR-0003's decision stands, with the room made in the engine;
  - a window that covers the reply's first sentence makes degrade unnecessary.
- **The v0 lease and ledger code is removed after v0.3:** `lh-proto`'s lease hint and ledger rows, `X-LH-Synthetic`, and `leaseholder_vllm`'s `LeasePolicy` and `LeaseConnector`.
- **The change's known gaps** ([ARCHITECTURE.md §2](../ARCHITECTURE.md#2-the-engine-change)) are fixed in the variant and in both PRs. E2 measured the change with them.
