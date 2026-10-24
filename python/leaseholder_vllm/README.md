# leaseholder-vllm

The tests of Leaseholder's engine variants for vLLM 0.32 ([`deploy/phastform/variants`](../../deploy/phastform/variants)). The package holds no runtime code: each variant builds its scheduler module from the engine's own vLLM at engine start.

The engine-side plugins it held were built for the lease design and removed after v0.3 ([ADR-0006](../../docs/adr/0006-the-engine-change-carries-the-realtime-tier.md)): `LeasePolicy`, a lease-aware CPU-tier cache policy, and `LeaseConnector`, the native `OffloadingConnector` with per-request hit receipts. The v0.3 tree keeps them, with their tests and live check.

## Tests

`tests/test_pr54530.py` and `tests/test_make_room_build.py` build the `pr54530` and `make-room` variants from a vLLM v0.32.0 source checkout's scheduler and check that each changes nothing but its own lines. They need no GPU, torch or installed vLLM. CI runs them with `make test-vllm`, which makes a sparse checkout in `~/.cache/leaseholder/vllm-v0.32.0` on first use; `VLLM_SRC` points them at another checkout, and they skip without one:

```sh
VLLM_SRC=~/src/vllm-v0.32.0 uv run --extra test pytest      # in python/leaseholder_vllm
```

`tests/test_make_room.py` checks the `make-room` variant's behavior on vLLM's real v1 scheduler, on CPU, with hand-made requests and no model, as vLLM's own `tests/v1/core` do: preemption for a waiting higher-priority request under KV pressure (the request chosen, the victims and their order, shared prefixes, the per-step cap, deferred block frees, LoRA adapters, a request the waiting loop skips), the budget cap (lower-priority prefill sitting out a request's first tokens from the step it fits or its KV load is done, and the first steps of a load only, and not for a request that does not fit), the reserve, the switches' validation, and stock behavior with the switches off. A stand-in KV connector loads a request's KV asynchronously or delays it. `tests/test_tail.py` checks `tail.py`'s accounting of failed call turns and stopped agents; it needs aiohttp, which the engine environment has. The scheduler needs torch and vLLM's dependencies, too much for CI, so these tests skip without them and run on the GPU node, against the engine environment with pytest beside it. `VLLM_TARGET_DEVICE=cpu` keeps them off the GPU:

```sh
tools=~/leaseholder/cache/testtools
[[ -d $tools/pytest ]] || ~/.local/bin/uv pip install -q --target "$tools" pytest
cd ~/leaseholder/src/python/leaseholder_vllm
VLLM_TARGET_DEVICE=cpu PYTHONPATH=$tools ~/leaseholder/venvs/engine/bin/python -m pytest -q -p no:cacheprovider tests/test_make_room.py
```

Elsewhere, an environment with CPU torch 2.13 and vLLM 0.32.0's Python dependencies runs them on the source checkout instead: `VLLM_TARGET_DEVICE=cpu PYTHONPATH=~/.cache/leaseholder/vllm-v0.32.0 python -m pytest tests/test_make_room.py`.
