# ADR-0004: The host tier is vLLM's native CPU offload; lease-aware eviction only where it binds

- **Status:** accepted, 2026-10-22
- **Deciders:** Chuyue Wang

## Context

Idle sessions must park somewhere cheaper than the 36,400-token GPU pool. Restoring 16k tokens from host RAM takes 0.57 s against 4.75 s recomputed, and restore beats recompute from about 64–128 tokens (the 2026-10-21 BF16 probe; [BENCHMARKS.md](../BENCHMARKS.md#tier-cost-model-resuming-an-agent-turn-e0) has the frozen box's W1 values). The candidates were LMCache, a custom connector, and vLLM 0.32's native `OffloadingConnector`. The native connector already has:
- a pluggable `CachePolicy` (`cache_policy_module_path`);
- write-through of generated tokens (`offload_prompt_only: false`);
- per-request `kv_transfer_params`;
- KV events for the CPU tier.

## Decision

- **Host tier.** The native `OffloadingConnector` with `offload_prompt_only: false`, sized at 20 GiB (four times the GPU pool) in the frozen box. In vLLM 0.32 it is a shared-memory region in `/dev/shm`, placed by the engine's NUMA policy.
- **Split of work.** The engine decides what to keep from the contract carried on each request; the control plane decides when to bring KV back, by prediction. Prefetch is a `max_tokens=1` warm-up timed close to the predicted arrival. It is never cancelled mid-load, because an abort discards the restored KV.
- **Default eviction.** CPU-tier eviction stays LRU. `leaseholder_vllm.policy.LeasePolicy` is switched on only where the simulator shows that CPU-tier eviction binds at the configured RAM ratio (gate G-sim, [EXPERIMENTS.md](../EXPERIMENTS.md#e1-parking)).
- **Policy behavior.** The policy reads only `kv_transfer_params.leaseholder` (`EngineLeaseHint`), plus recency. It is soft: it refuses a store only where plain LRU would refuse it too, because a refused eviction makes vLLM drop the store silently.
- **Hit receipts.** `leaseholder_vllm.connector.LeaseConnector` wraps the native connector to return per-request hit receipts (GPU tokens and CPU tokens). The recompute ledger needs that split; vLLM's metrics give it only in aggregate.

## Consequences

- No KV transfer code of our own. The policy is tested on CPU against vLLM's real offload manager, loaded from source in CI.
- GPU residency after a prefetch is best-effort: restored blocks go back into the GPU LRU, so prefetch timing matters.
- The CPU tier competes with other tenants for host RAM and `/dev/shm`. Lease grants are admission-controlled, so a floor is never promised beyond what the tier can hold.
