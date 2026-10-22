# agentic-api HTTP baseline (W1, 2026-10-22): input for vllm-project/agentic-api#367

agentic-api's own `response_provider` benchmark, run against the frozen box. Arms: agentic-api over WebSocket, HTTP/SSE and JSON, and direct vLLM over HTTP/SSE and JSON. All five arms start at one barrier and run at once, so each row of a table saw the same engine load.

| | |
|---|---|
| agentic-api | a7dc7b5 (0.9.0), built on peacock with a static OpenSSL 3.0.15; HTTP adapter; SQLite store; `--gateway-host 127.0.0.1`; pinned to cores 8–15 of cyan (2× Xeon Silver 4310, 48 threads) |
| Engine | vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8, the frozen box on Phastform (RTX 5060 Ti 16 GB) with `--enable-auto-tool-choice --tool-call-parser hermes`. Both arms reach it through the same SSH local forward (about +0.45 ms per small request) |
| Benchmark | `response_provider --provider all --seed 20261022` on cores 16–23 of cyan; `run.sh` is the exact script |

**Transport workload.** 4 tool rounds per turn, 10 turns per session. Latency is per turn, p50 / p95 in ms:

| Sessions per arm | agentic-api WS | agentic-api HTTP/SSE | agentic-api JSON | vLLM HTTP/SSE | vLLM JSON | Paired median, agentic-api minus vLLM |
|---|---|---|---|---|---|---|
| 1 | 3,173 / 4,194 | 3,173 / 4,191 | 3,175 / 4,194 | 3,176 / 4,183 | 3,187 / 4,272 | 0 ms |
| 5 | 4,488 / 7,925 | 4,489 / 7,686 | 4,490 / 7,837 | 4,482 / 7,890 | 4,500 / 7,772 | +3 ms |
| 10 | 6,099 / 10,949 | 6,102 / 10,953 | 6,094 / 10,977 | 6,072 / 10,945 | 6,097 / 10,975 | +1 ms |

- **Task correctness.** 100%, 98% and 99% on every arm, except vLLM JSON at 90% with one session.
- **Request size per turn.** 3.6–3.8 KB through the gateway, which holds the state, against 8.4 KB direct.
- **Gateway resources.** About 5.5 ms of CPU per Responses round: 8.6 s for about 1,500 rounds in 71 s at 10 sessions per arm, 12% of one core. Peak RSS was 44–54 MiB (`gateway_usage.py` over `server-usage.txt`).

**History rehydration** (depths 1, 5, 10, 25). Every arm fails the recall at turn 2 with this 4B model: success is 100%, 40%, 20% and 8%, the same on every arm. It confirms equal behavior but needs a larger model to say anything about depth.

This directory keeps each run's `summary.md` (and the history `depth_summary.md`), the campaign log, the script and the usage analysis. The per-turn `turns.csv` and `run.json` files (8 MB) are kept with the run archive.
