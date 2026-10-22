# Service levels

Leaseholder serves three classes on one GPU. Each class has its own clock, admission rule and primary indicator.

| Class | Typical traffic | Resume target | Primary indicator | Admission |
|---|---|---|---|---|
| `realtime` | Voice turns | ≤ 250 ms to GPU-resident KV, restored while the user is still speaking | Voice-to-voice latency; LLM time to first sentence | Never rejected: make room, then degrade the reply ([ADR-0003](adr/0003-realtime-is-never-shed.md)) |
| `interactive` | Agent loops with a person waiting | Warm-turn TTFT p95 ≤ 250 ms at ≤ 16k tokens | Warm and resumed TTFT, by source tier | Predicted-TTFT admission; queued with a deadline |
| `background` | Batch agents, trace replays, memory consolidation | Best effort | Goodput (tokens/s), with a minimum share and aging | Deferred or demoted whenever realtime needs room |

## Indicators and v0 targets

| Indicator | Definition | v0 target |
|---|---|---|
| Voice V2V | End of the user's speech to the bot's first audio, measured from waveforms at the bot | p50 ≤ 900 ms, p95 ≤ 1,200 ms, p99 ≤ 1,800 ms. Recalibrated against voice alone in W2 |
| Barge-in to silence | The user starts speaking during bot speech, until bot audio falls below threshold | p95 ≤ 250 ms beyond the configured minimum interruption duration |
| Audio continuity | Share of bot utterances with any gap of 100 ms or more | ≤ 1% |
| LLM time to first sentence | Realtime request to the end of the first sentence | Reported; this stage dominates V2V under load |
| Resume TTFT | Interactive, by source tier (GPU, CPU, recompute) | Warm p95 ≤ 250 ms at ≤ 16k tokens |
| Harvested background goodput | Background agent tokens/s while the voice SLO holds | Reported; the x-axis of the E2 frontier |
| Contract violations | Prefix misses inside a cache TTL, outside the listed exceptions (node failure, model or config change, prefix mismatch) | 0 |
| Control-plane overhead | Latency added per request and per streamed chunk | p99 ≤ 2 ms per request |

Synthetic requests (prefetch warm-ups, marked `X-LH-Synthetic`) are excluded from every indicator.

## Alerts

- Multi-window burn rate on voice V2V p95 and on contract violations.
- KV-event sequence gaps, engine epoch changes and tunnel loss. Each one invalidates leases or ledger attribution until it is resolved.
