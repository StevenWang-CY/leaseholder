# Engine variants of the frozen box

Every experiment runs on the frozen box ([`engine.yaml`](../engine.yaml) and the units in [`systemd/`](../systemd)). An E2 point changes it in two ways, both named by its arm file's `[engine]` table ([`benchmarks/e2`](../../../benchmarks/e2)):

- **a variant**, a directory here of systemd drop-ins, one `UNIT.conf` per unit it changes, linked as `UNIT.service.d/variant-NAME.conf`;
- **engine flags**, added to the engine's command line by a generated drop-in, `lh-engine.service.d/variant-flags.conf`, which sets `LH_ENGINE_FLAGS`. vLLM takes the command line over `engine.yaml`, and the last of a repeated flag wins, so they beat a variant's own flags too.

`variant.sh use NAME|frozen [FLAG...]` makes the box run exactly that: it removes every other variant, links NAME's drop-ins, writes the flags and restarts what changed. When only the engine's drop-ins change and the audio servers run, only the engine restarts (about 2 minutes with warm caches); otherwise the whole box does. It does nothing when the box already runs NAME with those flags. `benchmarks/e2/campaign.sh` calls it before each point with what `lh-bench e2 engine ARM` prints, and `lh-bench e2 run` checks the result against the engine's own `/server_info`. `systemctl --user cat lh-engine` shows what the box runs.

`variant.sh on|off NAME` adds or removes one variant and restarts the box, for work outside E2.

## Variants

| Variant | Units | What it changes | E2 arm |
|---|---|---|---|
| `frozen` | none | Nothing: the frozen box | 1, and every arm that changes flags only |
| [`mps`](mps) | engine, STT, TTS | The three servers run as clients of one CUDA MPS daemon (`lh-mps.service`) instead of time-slicing | the MPS factor |
| [`pr54530`](pr54530) | engine | vLLM PR [#54530](https://github.com/vllm-project/vllm/pull/54530) ported to 0.32.0: under priority scheduling, a waiting request preempts the lowest-priority running one once the running queue is full. `build.py` writes the scheduler module at each start, the engine loads it with `--scheduler-cls lh_pr54530_scheduler.AsyncScheduler`, and `--max-num-seqs 12` lets the queue fill, since the PR acts on the request count, never on KV pressure. `probe.py` is the check | 7a (with the PR) |
| [`sglang`](sglang) | engine | SGLang 0.5.22 on the engine's port, with the same model, step cap (`--chunked-prefill-size 512`) and KV pool (36,400 tokens), and priority scheduling that preempts on any priority gap. Its environment is built once with `install.sh` | 7b |
| [`make-room`](make-room) | engine | Leaseholder's scheduler change for the KV case of vLLM [#40004](https://github.com/vllm-project/vllm/issues/40004), proposed upstream. Under priority scheduling, the waiting request of the highest priority preempts running requests of a lower priority when it lacks KV for its whole sequence (the largest first, the latest arrival among equals, at most 2 a step, none if 2 would not do), and a victim waits until the prefills in flight have the blocks they still need; running prefills of a lower priority leave its first chunk in the step's budget; and requests of a positive priority value leave a reserve of 2,048 tokens free. `build.py` writes the scheduler module at each start and the engine loads it with `--scheduler-cls lh_make_room_scheduler.AsyncScheduler`; `--max-num-seqs` stays the frozen box's. `probe.py` is the KV check | 9, engine side |

The frozen box schedules asynchronously (vLLM's default here), so `pr54530` and `make-room` load the stock `AsyncScheduler` over their patched `Scheduler`: with and without each change differ by the change alone. `python/leaseholder_vllm/tests/test_pr54530.py` and `test_make_room_build.py` build the modules from vLLM v0.32.0's source and check that; `test_make_room.py` checks `make-room`'s behavior on the real scheduler.

`make-room`'s three switches are keys of vLLM's `--additional-config`, read once at engine start: `make_room.max_preemptions_per_step` (0 is off), `make_room.budget_cap` and `make_room.reserve_tokens` (0 is off). Its `lh-engine.conf` sets arm 9's values, and an arm file ablates one with engine flags, which come after the variant's and win (vLLM warns of the duplicate key): `flags = ["--additional-config.make_room.reserve_tokens", "0"]`. The engine reports them under `additional_config` in `/server_info`, so each point's `engine.json` records them; `lh-bench e2 run` lists such flags as unchecked, as it checks only the cache and scheduler settings.

## Engine flags of the E2 arms

Each flag was checked against vLLM v0.32.0 (`vllm/engine/arg_utils.py`, `vllm/config/scheduler.py`). The frozen box runs `--scheduling-policy priority --max-num-batched-tokens 512 --long-prefill-token-threshold 256`.

| Arm | `[engine]` | What it sets |
|---|---|---|
| 2, stock | `--scheduling-policy fcfs --max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | vLLM's defaults on this GPU: first come, first served; a step cap of 2,048 tokens (the API server's default below 70 GiB of GPU memory); no per-request prefill cap. Nothing signals priority (the gateway's `upstream_priority = []`) |
| 3, priority | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 0` | Priority scheduling with vLLM's default chunking |
| 4, step cap | `--max-num-batched-tokens B --long-prefill-token-threshold T` | B in {512, 1024}, T in {256, 512, 1024}. A threshold at or above B never binds (each request's share of a step is capped by B anyway), so the grid has five distinct cells: 512/256 (the frozen box), 512/0, 1024/256, 1024/512 and 1024/0 |
| 5, watermark | best arm 4 + `--watermark W` | W in {0.05, 0.1, 0.2}: the share of KV blocks kept free when the scheduler admits a waiting or preempted request |
| 5, queued tokens | best arm 4 + `--max-num-queued-tokens N` | New requests get 503 while the prompt tokens of the requests in prefill exceed N. It applies to every class, voice included, and counts as `vllm:admission_rejections_total` |
| 7a, without the PR | `--max-num-seqs 12` | The `pr54530` engine without the PR |
| 7a, with the PR | variant `pr54530` | |
| 7b | variant `sglang` | |
| 9, engine side | variant `make-room` | KV make-room at 2 preemptions a step, the budget cap and a reserve of 2,048 tokens, beside the gateway's policies |
| 9, an ablation | variant `make-room` + `--additional-config.make_room.max_preemptions_per_step 0`, `--additional-config.make_room.budget_cap false` or `--additional-config.make_room.reserve_tokens 0` | The engine side without one mechanism; all three off is the frozen box's scheduling, through the generated module |

vLLM 0.32's FCFS scheduler ignores request priorities rather than rejecting them: nothing in 0.32.0 enforces the "will raise an error" of the OpenAI protocol's `priority` description. Arm 2 sends none so that it is the engine out of the box.
