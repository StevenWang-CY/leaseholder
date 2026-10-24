# E2 records (2026-10-23)

The compact records of E2's points ([E2](../../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)), from which `make reproduce-e2` regenerates the report into `build/reproduce-e2` (`E2_REPORT=DIR` to choose): `frontier.md`, `frontier.csv`, a frontier figure per workload (`frontier.svg` for the standard scenario, `frontier-weka-x32-long.svg` for the discriminating one, `frontier-4-callers.svg` and `frontier-6-callers.svg` for the calls-per-GPU points) and `waterfall.svg`, each configuration's voice turn by stage. [The report](../../../python/leaseholder_bench/README.md#e2-points) gives the same tables and figures, byte for byte, from these records as from the points' run directories.

Each point is `CAMPAIGN/POINT/`, as `lh-bench e2 export` writes it from the run directory that [`campaign.sh`](../../e2/README.md) fetched:

| File | Contents |
|---|---|
| `run.json` | The point's record: the arm as read, versions, the engine and its checked flags, what cyan ran, the window of the calls |
| `arm.toml` | The arm file as run |
| `engine.json` | The engine the point ran on, from its `/server_info`: the model and its revision, the context, dtype and quantization, the KV cache and pool, the scheduler settings the arms' flags set, the CPU tier's connector and `additional_config` (for SGLang the same of its server arguments), with home paths written from `~` |
| `summary.json` | The point's E2 metrics, with the first bytes taken exactly from the gateway's request log and, where the campaign sampled the GPU, its median SM clock and highest temperature during the calls (`gpu`) |
| `bots/summary.json`, `bots/turns.jsonl` | The callers' summary and their line per turn |

The raw records, which the report does not need (the gateway's request log, AIPerf's exports, the metric snapshots, the voice worker's journal, the GPU's samples and the tools' logs), are an asset of the GitHub release v0.3.

129 points, in the campaigns of [`benchmarks/e2`](../../e2/README.md), all on vLLM 0.32.0 and Qwen3-4B-Instruct-2507-FP8 on Phastform's RTX 5060 Ti:

| Campaign | Points | What |
|---|---:|---|
| [`w2`](w2) | 16 | W2's first batch: arms 1–8 in the standard scenario, and voice alone |
| [`w2b`](w2b) | 14 | W2's second batch: the arms that take an earlier arm's winner, at 16 and 24 sessions |
| [`w2c`](w2c) | 15 | W2's third batch: arms 4–8 in the discriminating scenario, round 0 of its baselines |
| [`a9`](a9) | 13 | Arm 9 and its ablations in the discriminating scenario, round 0 |
| [`r1`](r1), [`r2`](r2) | 18 each | Repeat rounds 1 and 2: the headline configurations and arm 9's ablations |
| [`r3`](r3), [`r4`](r4) | 7 each | Repeat rounds 3 and 4: the headline configurations and voice alone |
| [`a9-standard`](a9-standard) | 3 | Arm 9 in the standard scenario at 8, 16 and 24 sessions |
| [`s1`](s1), [`s2`](s2) | 3 each | The standard scenario's realistic load repeated: the frozen box and arm 9 at 16 sessions, and voice alone |
| [`a9-window`](a9-window) | 1 | Arm 9 without degrade, with a 48-token first-tokens window |
| [`upstream`](upstream), [`upstream-r1`](upstream-r1), [`upstream-r2`](upstream-r2) | 3, 4, 4 | The engine as vLLM #63350 and #63351 propose it, and arm 9's 48-token window, three rounds |

To add a campaign's points: `lh-bench e2 export $LH_E2_RESULTS/CAMPAIGN/* --out benchmarks/results/2026-10-23-e2`.
