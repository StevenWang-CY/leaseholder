# E2 records (2026-10-23)

The compact records of E2's points ([E2](../../../docs/EXPERIMENTS.md#e2-voice-under-agent-load)), from which `make reproduce-e2` regenerates the report: `frontier.md`, `frontier.csv`, `frontier.svg` and `waterfall.svg`, written into `build/reproduce-e2` (`E2_REPORT=DIR` to choose). [The report](../../../python/leaseholder_bench/README.md#e2-points) gives the same tables and figures, byte for byte, from these records as from the points' run directories.

Each point is `CAMPAIGN/POINT/`, as `lh-bench e2 export` writes it from the run directory that [`campaign.sh`](../../e2/README.md) fetched:

| File | Contents |
|---|---|
| `run.json` | The point's record: the arm as read, versions, the engine and its checked flags, what cyan ran, the window of the calls |
| `arm.toml` | The arm file as run |
| `engine.json` | The engine's configuration from its `/server_info`, without credentials |
| `summary.json` | The point's E2 metrics, with the first bytes taken exactly from the gateway's request log and, where the campaign sampled the GPU, its median SM clock and highest temperature during the calls (`gpu`) |
| `bots/summary.json`, `bots/turns.jsonl` | The callers' summary and their line per turn |

The raw records, which the report does not need (the gateway's request log, AIPerf's exports, the metric snapshots, the voice worker's journal, the GPU's samples and the tools' logs), are an asset of the GitHub release that publishes these results.

| Campaign | Points |
|---|---|
| [`w2b`](w2b) | W2's second batch ([`w2b.campaign`](../../e2/w2b.campaign)): the arms that take an earlier arm's winner, at 16 and 24 sessions |

To add a campaign's points: `lh-bench e2 export $LH_E2_RESULTS/CAMPAIGN/* --out benchmarks/results/2026-10-23-e2`.
