# E2 campaigns

An E2 point is one arm file in [`arms/`](arms): the calls, the agent replay beside them, the gateway's configuration and the engine ([the arm-file reference](../../python/leaseholder_bench/README.md#e2-points)). `lh-bench e2 run` measures one point on cyan. A campaign file lists arm files, and `campaign.sh` measures them in order from a workstation with SSH to both hosts, the way `deploy/deploy.sh` deploys.

```sh
export LH_SSH="ssh -F $HOME/.ssh/penn_config"                  # your SSH command for the hosts, as for deploy.sh
benchmarks/e2/campaign.sh --dry-run benchmarks/e2/w2.campaign   # check every arm file; list the points and engines
benchmarks/e2/campaign.sh --pause benchmarks/e2/w2.campaign     # measure, then pause the GPU node
```

For each point whose run directory on cyan is not complete, `campaign.sh`:

1. switches the GPU node to the arm's engine with [`variant.sh use`](../../deploy/phastform/variants/README.md), which restarts only what changes and returns once the engine answers `/health`;
2. runs `lh-bench e2 run` on cyan, into `~/leaseholder/runs/e2/CAMPAIGN/POINT`;
3. fetches that directory into `$LH_E2_RESULTS/CAMPAIGN/POINT` (default `~/leaseholder/runs/e2`), without AIPerf's `inputs.json`.

A complete point is skipped, so running a stopped campaign again resumes it. The first failure stops the campaign, with the partial run directory fetched and its `run.json` saying what failed. At the end, `lh-bench e2 report` writes the frontier of the campaign's points (fetched into `CAMPAIGN/report`: `frontier.md`, `frontier.csv`, `frontier.svg`) and `lh-bench e2 reset` puts cyan's units back. `--pause` then stops the GPU node's units, which nothing should hold while idle. `campaign.log` beside the points keeps every step.

Points that share an engine go together in a campaign file: a switch restarts the engine, about 2 minutes with warm caches. A point takes about 12 minutes: AIPerf builds its dataset for about 4 (its cache does not apply to this dataset type, see `configs/aiperf.env`), then a minute of warm-up and the calls.

**Stopping.** Ctrl-C stops `campaign.sh`, but a point already running on cyan goes on to its end: closing its SSH session does not stop it. To stop it as well, send SIGINT to its pid, which `~/leaseholder/runs/.lh-bench-e2.lock` holds on cyan; `lh-bench e2 run` then stops AIPerf and the callers and marks the point failed. While a point runs, `campaign.sh` refuses to switch the engine under it.

## Before the first campaign

- Deploy both hosts (`deploy/deploy.sh`), then on cyan rerun `deploy/cyan/bootstrap.sh`: it builds `venvs/bench` and `venvs/smg`.
- cyan: the `vad-turns` worker variant is linked (`lh-bench e2 run` refuses to measure without it); `~/.config/leaseholder/gateway-keys.env` has `LH_GATEWAY_AGENTS_KEY`, the key of the `agents` tenant; `~/leaseholder/data/weka-x32` (`lh-bench scale-traces`) and `~/leaseholder/data/callers` (`lh-voice render-callers`) exist; `lh-gateway` is a build that knows every key the arms' `[gateway]` tables set (`upstream_priority`, `[admission]`).
- Phastform, for arm 7b: `bash ~/leaseholder/src/deploy/phastform/variants/sglang/install.sh` builds SGLang's environment once.

## Points

[`w2.campaign`](w2.campaign) holds the points that depend on no other arm's result:

| Arm | Points | Engine | Gateway |
|---|---|---|---|
| 1 | `a1-voice`: the calls alone | frozen | as configured |
| 2 | `a2-stock-s{8,16,24}` | FCFS, step cap 2,048, no long-prefill threshold | `upstream_priority = []` |
| 3 | `a3-priority-s{8,16,24}` | priority, step cap 2,048, no long-prefill threshold | as configured |
| 4 | `a4-b512-t256-s16` (the frozen box), `a4-b512-t0-s16`, `a4-b1024-t256-s16`, `a4-b1024-t512-s16`, `a4-b1024-t0-s16` | priority with each step cap B and long-prefill threshold T; T at or above B does not bind, so B × T has five distinct cells | as configured |
| 7a | `a7a-seqs12-s16`, `a7a-pr54530-s16` | priority, `--max-num-seqs 12`, without and with vLLM #54530 | as configured |
| 7b | `a7b-sglang-s{8,16}` | SGLang 0.5.22 | `upstream_priority = ["body"]` |

The gateway "as configured" is `~/.config/leaseholder/gateway.toml`, which sends each class's priority as `X-Vllm-Priority`. Every point runs 2 callers × 25 turns with barge-in on a quarter of them, beside AIPerf's agentic replay of the 32× corpus (think times capped at 120 s) through the gateway as the `agents` tenant.

The other arms take an earlier arm's winner, so their files are written from its results. Each starts from the winning arm file, with its own `arm` and `label`:

| Arm | Add to the winner's arm file |
|---|---|
| 4, best two at 8 and 24 sessions | `[agents] sessions = 8` (and 24) |
| 5, best arm 4 + headroom | `"--watermark", "0.05"` (0.1, 0.2) in `[engine] flags`; or `"--max-num-queued-tokens", "N"`, which answers new requests 503 while the prompt tokens in prefill exceed N, voice requests included. vLLM sizes N as the TTFT target times the prefill throughput |
| 6, best arm 5 + an operator's cap | `[gateway.admission]` with `policy = "static-cap"` and `background_max_in_flight = 2` (4, 8); or `policy = "token-reservation"` and `reserved_tokens` set to one voice turn's KV: the realtime class's prompt and completion tokens per completed request, from arm 1's `summary.json` (`gateway.counters`) |
| 8a, best stock + llm-d's flow control | `[gateway.admission]` with `policy = "flow-control"` and `ceilings = { realtime = …, interactive = …, background = … }` |
| 8b, best stock + SMG | `router = "smg"`, and `[gateway]` with `upstream = "http://127.0.0.1:18090"` and `upstream_priority = ["smg-header"]` |
| 8c, best stock + preempt-resubmit | `[gateway.admission]` with `policy = "preempt-resubmit"` |
| factor, frozen and best stock | `[voice] preemptive_generation = false`, `arm = "<arm>-no-preemptive"` |
| calls, best stock | `[voice] callers = 4` (6), at 0 and 16 sessions, `arm = "<arm>-calls"` |
| discriminating scenario | `[agents] sessions = 24` and `data` naming a corpus with longer contexts (`lh-bench scale-traces` with a smaller factor) |

The `[admission]` keys are those of `lh-gateway`'s configuration ([its README](../../crates/lh-gateway/README.md)); `lh-bench e2 run` passes the `[gateway]` table through as it is.

**SMG as shipped (arm 8b).** `lh-smg.service` runs SMG 1.13.0's priority scheduler with its defaults. A vLLM worker behind its `openai` backend reports no capacity, so SMG sizes its pool at 64 slots, and its built-in reservations (system 32, interactive at least 128, default 10%) do not fit: it scales them down to system 19, interactive 39 and default 4, which leaves 2 slots for the bulk class, where the gateway sends background requests. Measured against `lh-bench mock-server`, 40 concurrent bulk requests went through 2 at a time, and the others wait in SMG's bulk queue (1,024 deep, 503 after 300 s). As shipped, arm 8b therefore caps the agents at 2 requests in flight; reservations that fit the pool take a `--priority-scheduler-config` file.

## Results

`lh-bench e2 report RUN_DIR... --out DIR` builds a frontier from any set of run directories, across campaigns too. Results that go into the repository are curated into `benchmarks/results/<date>-<name>/`, as the earlier ones are.
