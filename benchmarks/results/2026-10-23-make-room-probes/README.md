# make-room probe records (2026-10-23)

The outputs behind [BENCHMARKS](../../../docs/BENCHMARKS.md#the-kv-case-make-room)' three probe tables (the KV case, the prefill case and the calls' first-token tail), which vLLM [#63350](https://github.com/vllm-project/vllm/pull/63350) and [#63351](https://github.com/vllm-project/vllm/pull/63351) cite. Every run was on Gray, the second RTX 5060 Ti, with the frozen box's units, pins and pool (vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8, 2,274 blocks of 16 tokens), on 2026-10-23, with [`probe.py`](../../../deploy/phastform/variants/make-room/probe.py) and [`tail.py`](../../../deploy/phastform/variants/make-room/tail.py).

`python3 check.py` recomputes every row of the three tables from these files: each run of a row, or its median and range from four runs on, and for a tail, tail.py's summary line, recomputed from the run's per-turn lines (within their 1 ms rounding; the check fails otherwise). None of the tails had a failed call turn.

The drivers' logs under `logs/` give, for each run, the revision Gray had deployed (its `DEPLOYED` line) and the engine's `additional_config` as `/server_info` read it back. The switches are `make-room`'s: the KV preemption (at most 2 victims a step; #63350), the budget cap (#63351) and the 2,048-token reserve (arm 9 only).

| File | Engine | Revision | Rows |
|---|---|---|---|
| `logs/gray-makeroom.log`, `logs/gray-makeroom-probe.log` | The frozen box, then `make-room` without the reserve. probe.py's first form sent the turn once both priority-2 requests were decoding, with 1,148 blocks free | make-room's branch before it merged | The KV case on an idle pool (4 runs) |
| `logs/gray-makeroom-probe3.log` | The frozen box. The `make-room` run after it (268 ms) is in no table: it ran the unmerged branch's module | make-room's branch before it merged | The KV case, stock (4,923 ms) |
| `pr-probes/stock-kv.out` | The frozen box | d307801 | The KV case, stock (5,069 and 5,077 ms) |
| `pr-probes/a-kv.out` | The KV preemption alone: no budget cap, no reserve | 6967456 | The KV case, #63350 |
| `final-probes/makeroom-kv.out` | The KV preemption and the budget cap, no reserve | e19fba5 | The KV case, #63350 and #63351 |
| `logs/gray-kvtimeline.log` | The frozen box under the KV case's load, `/metrics` every half second, and the engine's log | make-room's branch before it merged | Why stock waits |
| `logs/gray-firsttokens.log` | The frozen box (290 and 302 ms); `make-room` with `budget_cap_tokens 0`, which turns the first-tokens rule off, and the reserve on (324 and 339 ms). Its arm-9 runs are in no table: they ran the same working tree | The working tree that became 77ea7dd | The prefill case: stock (2 of 6 runs), and without the first-tokens rule |
| `pr-probes/stock-prefill.out` | The frozen box | d307801 | The prefill case, stock (4 of 6 runs) |
| `pr-probes/a-prefill.out` | The KV preemption alone | Runs 1–2 6967456, 3–4 d307801 | The prefill case, #63350 |
| `pr-probes/ab-prefill.out` | The KV preemption and the budget cap | Runs 1–2 6967456, 3–4 d307801 | The prefill case, #63350 and #63351 |
| `final-probes/makeroom-prefill.out` | Arm 9's configuration: both switches and the reserve | e19fba5 | The prefill case, arm 9 |
| `final-probes/frozen-tail-fresh.out`, `-grow.out` | The frozen box | e19fba5 | The tail, stock |
| `pr-probes/a-tail-fresh.out`, `-grow.out` | The KV preemption alone | d307801 | The tail, #63350 |
| `tail/v2b-fresh.out`, `tail/tailprobe.out` | Arm 9's configuration, on a module that logged the scheduler's state each step a priority-0 request had waited over 0.3 s (`tail/*-engine.log`: every such request was in `WAITING_FOR_REMOTE_KVS`). tail.py before it was committed, as `tailprobe.py`, the same code; fresh and growing prompts | 77ea7dd and the logging | The tail, before the KV-load rule |
| `pr-probes/ab-tail-fresh.out`, `-grow.out` | The KV preemption and the budget cap | 6967456 | The tail, #63350 and #63351 |
| `final-probes/makeroom-tail-fresh.out`, `-grow.out` | Arm 9's configuration | e19fba5 | The tail, arm 9 |

The revisions after the KV-load rule schedule these probes alike. e19fba5's module and 6967456's differ by 1460bfc only, which holds nothing back when every running slot is taken; that is 256 requests, and no probe comes near it. d307801's module is 6967456's.
