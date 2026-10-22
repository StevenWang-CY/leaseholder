# leaseholder-vllm

Leaseholder's engine-side plugins for vLLM 0.32. They are loaded by vLLM's own extension points, so the engine stays stock.

| Module | Plugs into | What it does |
|---|---|---|
| `leaseholder_vllm.policy.LeasePolicy` | the native CPU offload tier (`cache_policy_module_path`) | Soft leases over LRU. Chunks without a live lease are evicted first; leased chunks go only when nothing else is evictable, lowest class first. A store is refused only where plain LRU would refuse it too |
| `leaseholder_vllm.connector.LeaseConnector` | `kv_connector_module_path` | vLLM's `OffloadingConnector`, plus session-id mirroring into the lease hint and a per-request hit receipt (`gpu_hit_tokens`, `cpu_hit_tokens`) returned in `kv_transfer_params` |

The control plane talks to both through one field: `kv_transfer_params.leaseholder`, the `EngineLeaseHint` type defined in [`crates/lh-proto`](../../crates/lh-proto) (JSON Schema in `crates/lh-proto/schema/engine_lease_hint.schema.json`):

```json
{"kv_transfer_params": {"leaseholder": {"session_id": "s-1", "class": "realtime", "ttl_s": 300, "expect_return_s": 2.5}}}
```

## Engine configuration

```yaml
kv-transfer-config:
  kv_connector: LeaseConnector
  kv_connector_module_path: leaseholder_vllm.connector
  kv_role: kv_both
  kv_connector_extra_config:
    cpu_bytes_to_use: 21474836480
    offload_prompt_only: false
    eviction_policy: LeasePolicy
    cache_policy_module_path: leaseholder_vllm.policy
```

The frozen box ships the stock `OffloadingConnector` with LRU. The policy is switched on only where the simulator shows that CPU-tier eviction binds (the G-sim gate of [E1](../../docs/EXPERIMENTS.md#e1-parking)).

## Tests

The offload manager and its policies are plain Python, so the tests run without a GPU, torch or an installed vLLM. They load vLLM's real `CPUOffloadingManager` from a v0.32.0 source checkout. CI runs them with `make test-vllm`, which makes a sparse checkout in `~/.cache/leaseholder/vllm-v0.32.0` on first use; `VLLM_SRC` points the tests at another checkout:

```sh
VLLM_SRC=~/src/vllm-v0.32.0 uv run --extra test pytest      # in python/leaseholder_vllm
```

Without `VLLM_SRC` the tests use the installed vLLM (the engine environment on a GPU node), and they skip when neither is available. `tests/test_pr54530.py` builds the `pr54530` engine variant ([`deploy/phastform/variants`](../../deploy/phastform/variants)) from the same checkout's scheduler and checks that it changes nothing but the PR's lines.

The live check runs on the GPU node against the running engine. `tests/live/run.sh` links the `engine-check.conf` drop-in into `lh-engine` (the frozen configuration with `LeaseConnector` and a recording `LeasePolicy`), restarts the engine, runs `tests/live/test_live_engine.py`, and restores the frozen configuration, also on failure. It checks that the hint reaches both plugins, that `X-Session-ID` reaches the GPU-tier KV events and the hint, and that a CPU restore shows up in the hit receipt.
