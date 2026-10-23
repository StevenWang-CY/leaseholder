# E2 campaigns

An E2 point is one arm file in [`arms/`](arms): the calls, the agent replay beside them, the gateway's configuration and the engine ([the arm-file reference](../../python/leaseholder_bench/README.md#e2-points)). `lh-bench e2 run` measures one point on cyan. A campaign file lists arm files, and `campaign.sh` measures them in order from a workstation with SSH to both hosts, the way `deploy/deploy.sh` deploys.

```sh
export LH_SSH="ssh -F $HOME/.ssh/penn_config"                  # your SSH command for the hosts, as for deploy.sh
benchmarks/e2/campaign.sh --dry-run benchmarks/e2/w2.campaign   # check every arm file; list the points and engines
benchmarks/e2/campaign.sh --pause benchmarks/e2/w2.campaign     # measure, then pause the GPU node
```

For each point whose run directory on cyan is not complete, `campaign.sh`:

1. switches the GPU node to the arm's engine with [`variant.sh use`](../../deploy/phastform/variants/README.md), which restarts only what changes and returns once the engine answers `/health`;
2. runs `lh-bench e2 run` on cyan, into `~/leaseholder/runs/e2/CAMPAIGN/POINT`. Meanwhile it samples the GPU node's `nvidia-smi` once a second (SM and memory clocks, temperature, power and clock event reasons, timestamps in UTC), because a consumer card's boost clock follows its temperature, and puts the samples into that directory as `gpu.csv` when the point ends, failed or not;
3. fetches that directory into `$LH_E2_RESULTS/CAMPAIGN/POINT` (default `~/leaseholder/runs/e2`), without AIPerf's `inputs.json`.

A complete point is skipped, so running a stopped campaign again resumes it. The first failure stops the campaign, with the partial run directory fetched and its `run.json` saying what failed. At the end, `lh-bench e2 report` writes the frontier of the campaign's points (fetched into `CAMPAIGN/report`: `frontier.md`, `frontier.csv` and a figure per workload, `frontier.svg` for the arm files' own) and `lh-bench e2 reset` puts cyan's units back. `--pause` then stops the GPU node's units, which nothing should hold while idle. `campaign.log` beside the points keeps every step.

Points that share an engine go together in a campaign file: a switch restarts the engine, about 2 minutes with warm caches. A point takes about 12 minutes: AIPerf builds its dataset for about 4 (its cache does not apply to this dataset type, see `configs/aiperf.env`), then a minute of warm-up and the calls.

**Stopping.** Ctrl-C stops `campaign.sh` and its sampling of the GPU, but a point already running on cyan goes on to its end: closing its SSH session does not stop it. To stop it as well, send SIGINT to its pid, which `~/leaseholder/runs/.lh-bench-e2.lock` holds on cyan; `lh-bench e2 run` then stops AIPerf and the callers and marks the point failed. While a point runs, `campaign.sh` refuses to switch the engine under it.

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

[`w2b.campaign`](w2b.campaign) holds W2's second batch, written from the first batch's results. At 16 sessions the best arm 4 cells, step cap 1024 with no long-prefill threshold and the frozen box (512/256), were within noise of each other (V2V p95 2,167 and 2,215 ms over 50 turns, background goodput within 4%), so both are measured again at 24 sessions, and the frozen box, the configuration every earlier result was measured on, is the engine of the other points: arm 5 (`--watermark 0.1`; `--max-num-queued-tokens 8192`, about 1.6 s of prefill at the step cap), arm 6 (caps of 2 and 8; a reservation of 2,048 tokens, two concurrent voice turns at their longest), arms 8a–8c (flow-control ceilings 6/6/6, so that background fills at most 6 and a realtime request sheds), preemptive generation off, and 4 and 6 callers. The admission arms wait at most 240 s, below the replay's 300 s request timeout.

**The discriminating scenario** (PLAN §3.1) replays only the traces with the largest contexts, so that agents fill the KV pool while voice turns keep arriving. `weka-x32-long` on cyan holds the 48 traces of `weka-x32` whose main agent has the largest median input (13,326 tokens a request, against 4,784 for the whole corpus; 23,848 main-agent requests), linked by [`long_traces.py`](long_traces.py): `python benchmarks/e2/long_traces.py ~/leaseholder/data/weka-x32 ~/leaseholder/data/weka-x32-long 48`. Its points are named `disc-*`.

[`w2c.campaign`](w2c.campaign) holds W2's third batch: arms 4 to 8 in the discriminating scenario, one run each (the arm files `disc-<arm>-s24.toml` are the 16- or 24-session arms with `sessions = 24` and `data = "weka-x32-long"`), and SGLang with its host tier (`--enable-hierarchical-cache --hicache-ratio 4 --hicache-write-policy write_through`) at 8 and 16 sessions, for parity with vLLM's write-through CPU tier. W4 repeats the best of each arm beside arm 9. [`a9.campaign`](a9.campaign) holds arm 9, Leaseholder, in the same scenario: the gateway's policy `leaseholder` (`[gateway.admission]`: 1,024 tokens of KV headroom per armed session, a 150 ms critical-window budget of predicted background prefill, a 16-token reply cap when the headroom was not kept), the engine's `make-room` variant and the voice worker's lifecycle events; its knob, the critical-window budget at 0 and 500 ms; and one run of each single-mechanism ablation (`disc-a9-no-*`, `trigger-request`, `gateway-only`, `engine-only`). A switched-off mechanism's keys are left out: the gateway refuses them.

[`r1.campaign`](r1.campaign) to [`r4.campaign`](r4.campaign) repeat the discriminating scenario in rounds, so that each headline configuration has five runs (w2c and a9 are round 0) and each of arm 9's ablations three (rounds 1 and 2). The headline configurations are the best of each arm in round 0: the frozen box and the best step-cap cell (`disc-a4-b1024-t0-s24`), the static reservation, preempt-resubmit, SGLang with its host tier, and arm 9. A round groups its points by engine, and the groups' order rotates from round to round, so that no configuration always runs at the same hour of a campaign that takes hours.

[`a9-standard.campaign`](a9-standard.campaign) puts arm 9 in the standard scenario at 8, 16 and 24 session trees, as W2 measured arms 2 to 8, so that its frontier stands beside theirs. [`s1.campaign`](s1.campaign) and [`s2.campaign`](s2.campaign) repeat the realistic operating point, 16 session trees, for the frozen box and arm 9, so that each has three runs there.

The `[admission]` keys are those of `lh-gateway`'s configuration ([its README](../../crates/lh-gateway/README.md)); `lh-bench e2 run` passes the `[gateway]` table through as it is.

**SMG as shipped (arm 8b).** `lh-smg.service` runs SMG 1.13.0's priority scheduler with its defaults. A vLLM worker behind its `openai` backend reports no capacity, so SMG sizes its pool at 64 slots, and its built-in reservations (system 32, interactive at least 128, default 10%) do not fit: it scales them down to system 19, interactive 39 and default 4, which leaves 2 slots for the bulk class, where the gateway sends background requests. Measured against `lh-bench mock-server`, 40 concurrent bulk requests went through 2 at a time, and the others wait in SMG's bulk queue (1,024 deep, 503 after 300 s). As shipped, arm 8b therefore caps the agents at 2 requests in flight; reservations that fit the pool take a `--priority-scheduler-config` file.

## Results

`lh-bench e2 report RUN_DIR... --out DIR` builds a frontier from any set of run directories, across campaigns too. It compares points only within a workload, the agents' corpus and the number of callers: the discriminating scenario (`weka-x32-long`) and the calls-per-GPU points (4 and 6 callers) each get their own section of `frontier.md`, with the voice SLO they are read against (250 ms above voice alone with as many callers), and their own figure (`frontier-weka-x32-long.svg`, `frontier-4-callers.svg`), beside `frontier.svg` for the standard one ([the figures](../../python/leaseholder_bench/README.md#e2-points)). A point measured in several campaigns (W4's rounds of the discriminating scenario reuse w2c's and a9's arm files) is one configuration: the report keeps each run, gives the configuration's median and range, and places it on the frontier at its medians. `waterfall.svg` splits each configuration's voice turn into its stages (endpointing, the reply to its first sentence, TTS first audio, transport and playout), stacked to its mean V2V, with the median and p95 marked; `--waterfall POINT` picks the configurations it draws.

The repository keeps a compact record of each point, not its run directory: `lh-bench e2 export RUN_DIR... --out DIR` writes the few small files the report reads (about 15 kB a point, no credential and no home path) into `DIR/CAMPAIGN/POINT`, and the report gives the same tables and figures from them as from the run directories ([the compact record](../../python/leaseholder_bench/README.md#e2-points)). E2's records are in [`benchmarks/results/2026-10-23-e2`](../results/2026-10-23-e2/README.md), and `make reproduce-e2` regenerates the report from them into `build/reproduce-e2` (`E2_REPORT=DIR` to choose), with the bench venv:

```sh
python/leaseholder_bench/.venv/bin/lh-bench e2 export $LH_E2_RESULTS/w2c/* --out benchmarks/results/2026-10-23-e2
make reproduce-e2
```

The run directories themselves, with the raw records the report does not need (the gateway's request log, AIPerf's exports, the metric snapshots, the worker's journal, the GPU's samples, the tools' logs), are an asset of the GitHub release that publishes the results.

## The demo video

Two calls side by side, one stuttering and one smooth, while background agents run: [`demo.py`](demo.py) renders them from two points measured with `lh-bench e2 run --save-audio`, which has the callers save each call's audio and the times of its turns ([`lh-voice bots --save-audio`](../../python/leaseholder_voice/README.md#measurement-callers)). `campaign.sh` does not pass the option, so the two points are measured by hand, each after switching the GPU node to its engine as `campaign.sh` does (`variant.sh use` with what `lh-bench e2 engine ARM` prints), for example the frozen box and arm 9 in the discriminating scenario:

```sh
lh-bench e2 run ~/leaseholder/src/benchmarks/e2/arms/disc-frozen-s24.toml --out ~/leaseholder/runs/e2/demo/disc-frozen-s24 --save-audio
lh-bench e2 run ~/leaseholder/src/benchmarks/e2/arms/disc-a9-s24.toml --out ~/leaseholder/runs/e2/demo/disc-a9-s24 --save-audio
```

Then, on a workstation with the two run directories fetched (`uv` installs the script's pinned numpy, matplotlib and imageio-ffmpeg, whose ffmpeg encodes):

```sh
uv run benchmarks/e2/demo.py $LH_E2_RESULTS/demo/disc-frozen-s24 $LH_E2_RESULTS/demo/disc-a9-s24 \
  --labels "Tuned vLLM, the frozen box" "Leaseholder, arm 9" --out demo.mp4 --gif demo.gif
```

The video is 1280×720 at 30 frames a second. Each side shows its point's first call (`--caller`): its scrolling waveform, the caller above and the agent below; the current turn's wait, from the end of the caller's speech to the agent's first audio, counting live; each reply's V2V as the callers measured it, in a bar that grows while the caller waits; and the background goodput, the point's over its calls and the last 10 s of AIPerf's completed requests on the wall clock. Both calls show the same turns from `--first-turn` (default 0): `--turns` of them, by default as many as fit in `--seconds` (default 90). Turn k of both starts at the same moment, so the same line is answered side by side, and the call that is ahead waits in silence before its next line: only a caller's pause between turns is stretched, never a measured wait. The soundtrack is one call at a time, the left one for the first half and the right one for the second, and the side playing is highlighted. `--gif` writes the first 10 s (`--gif-seconds`) as a 640×360 GIF at 10 frames a second, for a README. A 90-second video takes two to three minutes to render.

`python/leaseholder_bench/tests/test_demo.py` runs the renderer on a synthetic pair of calls, tones in the files `--save-audio` writes: in CI the excerpt, the aligned turns, the soundtrack and the frames, with the bench venv's numpy and matplotlib; and, marked integration since it fetches the script's dependencies, the script itself through `uv run` to an MP4 and a GIF (`uv run --extra test pytest -m integration tests/test_demo.py` in `python/leaseholder_bench`).
