# Service levels

Leaseholder serves three classes on one GPU, the tiers API providers sell as priority, standard and flex. Each class has its own clock, admission rule and primary indicator.

| Class | Typical traffic | Resume target | Primary indicator | Admission |
|---|---|---|---|---|
| `realtime` | Voice turns | ≤ 250 ms to GPU-resident KV, restored while the user is still speaking | Voice-to-voice latency; LLM time to first sentence | Never queued or rejected: background work is deferred to make room, and the reply is capped when the room is not there ([ADR-0003](adr/0003-realtime-is-never-shed.md)) |
| `interactive` | Agent loops with a person waiting | Warm-turn time to first token (TTFT) p95 ≤ 250 ms at ≤ 16k tokens | Warm and resumed TTFT, by source tier | Predicted-TTFT admission, queued with a deadline (planned; not held today) |
| `background` | Batch agents, trace replays, memory consolidation | Best effort | Goodput (tokens/s) | Deferred while realtime turns need the headroom or the critical window's prefill budget; admitted regardless after `max_defer_s` (aging) |

## Indicators and targets

The voice targets are calibrated against voice alone on the frozen box (W1, 2026-10-22, [BENCHMARKS.md](BENCHMARKS.md#voice-on-the-shared-gpu)). Turn-taking settings are fixed across every arm. Experiments end a turn a fixed delay after the end of speech (VAD plus LiveKit's 0.5 s minimum), which contributes about 1 s of the end-to-end figure. So the end-to-end target is stated as the degradation that agent load may add.

| Indicator | Definition | Target |
|---|---|---|
| Server-side voice turn | The utterance sent to STT, until the first byte of TTS audio (`lh-bench voice-turn`): the part serving controls | p95 ≤ 800 ms (voice alone: 589 ms p50, 665 ms p90) |
| Voice V2V | End of the caller's speech to the agent's first audio, from waveforms at the caller (`lh-voice bots`), with fixed endpointing | p95 at most 250 ms above voice alone (voice alone: 1,614 ms p50, 1,760 ms p95, 1,889 ms p99) |
| Barge-in to silence | The caller starts speaking during the agent's speech, until the agent's audio falls below threshold | p95 ≤ 250 ms beyond the configured minimum interruption duration |
| Audio continuity | Share of the agent's utterances with any gap of 100 ms or more | ≤ 1% |
| LLM time to first sentence | Realtime request to the end of the first sentence | Reported; this stage dominates V2V under load |
| Resume TTFT | Interactive, by source tier (GPU, CPU, recompute) | Warm p95 ≤ 250 ms at ≤ 16k tokens |
| Harvested background goodput | Background agent tokens/s while the voice SLO holds | Reported; the x-axis of the E2 frontier |
| Headroom false alarms | Armings of the headroom that no realtime request of the session followed, and the background token-seconds held back meanwhile (`lh_gateway_admission_false_alarm*`) | Reported; the price of arming at speech onset |
| Degraded replies | Realtime replies capped at `degrade_max_tokens` because the headroom was not there (`lh_gateway_admission_degraded_total`) | Reported; zero while the headroom holds |
| Cache-contract honor rate | Hits inside a `cache_control` TTL (renewed on each use) among requests that should hit, outside the listed exceptions (node failure, model or config change, prefix mismatch) | Reported, with every miss counted; providers promise billing eligibility, not residency |
| Control-plane overhead | Latency `lh-gateway` adds per request: from the end of the request body to the upstream write (`lh_gateway_overhead_seconds`), including an admission policy's decision but not the wait it imposes. The client's upload is excluded, because a direct call waits for it too. Also the latency added per streamed chunk | p99 ≤ 2 ms per request (W1 in place on cyan, with the deployed keep-alives: p99 0.44 ms for voice alone, 0.58 ms beside the agent replay, [BENCHMARKS.md](BENCHMARKS.md#the-control-plane-lh-gateway-in-front-of-the-engine)). An admission policy adds 6–24 µs of own time on average, `leaseholder` 9 µs ([`lh-gateway` README](../crates/lh-gateway/README.md#overhead)) |

Synthetic requests (prefetch warm-ups, marked `X-LH-Synthetic`) are excluded from every indicator.

## Alerts

Planned, as rules in `ops/` beside the dashboards:
- Multi-window burn rate on voice V2V p95 and on the cache-contract honor rate.
- KV-event sequence gaps, engine epoch changes and tunnel loss. Each one invalidates leases or ledger attribution until it is resolved.
