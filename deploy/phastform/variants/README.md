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
| [`sglang`](sglang) | engine | SGLang 0.5.22 on the engine's port, with the same model and step cap (`--chunked-prefill-size 512`, without vLLM's 256-token threshold per request), a KV pool asked for at 36,400 tokens, which SGLang's static memory fraction brings to 33,683–34,679, and priority scheduling that preempts on any priority gap. Its environment is built once with `install.sh` | 7b |
| [`make-room`](make-room) | engine | Leaseholder's scheduler change for the KV case of vLLM [#40004](https://github.com/vllm-project/vllm/issues/40004), proposed upstream as [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) (without the reserve). Under priority scheduling, the waiting request of the highest priority that can be scheduled in the step (not one loading its KV, compiling its grammar, paused for streaming input or awaiting output from before its preemption, nor one whose LoRA adapter could not join the running ones') preempts running requests of a lower priority when it lacks KV for its whole sequence, in vLLM's own preemption order (the lowest priority first, the latest arrival among equals), at most 2 a step, none if 2 would not do; the room counts a block as released once all its holders go, not the request's own prefix-cache hits, and the running decodes' growth until the victims' frees land; it makes no room again for a request the waiting loop has not tried since; and a victim waits until the prefills in flight have the blocks they still need. Prefills of a lower priority, running or waiting (a waiting one that holds its KV is skipped), sit out the steps from the one in which it fits, or for one loading its KV from the one in which its load is done, until its first 16 output tokens (the budget cap); a load in flight holds them back for its first 16 steps only, so a stuck load costs them no more. Requests of a positive priority value leave a reserve of 2,048 tokens free. The engine logs, at most every 10 s, the steps in which the budget cap held prefill back, the prefill tokens and admissions it held back in them (summed over the steps) and the requests make-room preempted. `build.py` writes the scheduler module at each start and the engine loads it with `--scheduler-cls lh_make_room_scheduler.AsyncScheduler`; `--max-num-seqs` stays the frozen box's. `probe.py` checks a voice turn in a full KV pool (`kv`) and behind agent prefills (`prefill`), and `tail.py` the calls' first-token tail beside closed-loop agents | 9, engine side |
| [`make-room-v03`](make-room-v03) | engine | W5's control: `make-room` as the v0.3 tag built it, its `build.py` and unit verbatim but for the paths, and its module renamed `lh_make_room_v03_scheduler` so that the engine's scheduler class tells it from `make-room`'s. W5 re-measures the fixed variant beside it |

The frozen box schedules asynchronously (vLLM's default here), so `pr54530` and `make-room` load the stock `AsyncScheduler` over their patched `Scheduler`: with and without each change differ by the change alone. `python/leaseholder_vllm/tests/test_pr54530.py` and `test_make_room_build.py` build the modules from vLLM v0.32.0's source and check that; `test_make_room.py` checks `make-room`'s behavior on the real scheduler.

`make-room`'s three switches are keys of vLLM's `--additional-config`, read once at engine start: `make_room.max_preemptions_per_step` (0 is off), `make_room.budget_cap` (with `make_room.budget_cap_tokens`, 16) and `make_room.reserve_tokens` (0 is off). Its `lh-engine.conf` sets arm 9's values, and an arm file ablates one with engine flags, which come after the variant's and win (vLLM warns of the duplicate key): `flags = ["--additional-config.make_room.reserve_tokens", "0"]`. The engine reports them under `additional_config` in `/server_info`, so each point's `engine.json` records them; `lh-bench e2 run` checks them: the engine's `additional_config` must be exactly what the variant's and the arm's flags build, and its scheduler class the one the variant loads.

The engine start fails on an unknown `make_room` key or a bad value (`budget_cap` takes true or false in any case, or 1 or 0; the others an integer, `budget_cap_tokens` at least 1), and on a switch under a policy other than priority; for an encoder-decoder model the room counts a request's cross-attention blocks. `build.py` refuses an installed vLLM other than 0.32.0. The PRs name the same switches as engine flags:

| Variant key | vLLM PR flag |
|---|---|
| `make_room.max_preemptions_per_step` | `--max-priority-preemptions-per-step` ([#63350](https://github.com/vllm-project/vllm/pull/63350)) |
| `make_room.budget_cap`, `make_room.budget_cap_tokens` | `--priority-prefill-budget-cap`, `--priority-prefill-budget-cap-tokens` ([#63351](https://github.com/vllm-project/vllm/pull/63351)) |
| `make_room.reserve_tokens` | none: the reserve is the variant's only |

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
