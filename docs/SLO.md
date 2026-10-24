# Service levels

Three classes share the GPU, the tiers API providers sell as priority, standard and flex. A gateway maps each tenant to a class, capped by the tenant's `max_class`, and sends the class's priority to the engine, where a lower value is served first.

| Class | Traffic | Engine priority | Indicators |
|---|---|---|---|
| `realtime` | Voice turns | 0 | Voice V2V, the voice turn's stages, the realtime first byte |
| `interactive` | Agent loops with a person waiting | 1 | None measured: no E2 arm runs this class |
| `background` | Batch agents, trace replays | 2 | Background goodput and the agents' time to first token |

The deployed configuration sends the priority and nothing else ([ADR-0006](adr/0006-the-engine-change-carries-the-realtime-tier.md)). Under Leaseholder's own admission policy, an experiment arm, a realtime request is never queued or refused ([ADR-0003](adr/0003-realtime-is-never-shed.md)).

## Indicators

Every E2 point computes these (`lh-bench e2`; [EXPERIMENTS.md](EXPERIMENTS.md#e2-voice-under-agent-load) gives the method). The voice targets are stated against voice alone on the same box: experiments end a turn a fixed delay after the end of speech, which contributes about 1 s of every turn, so the target is the degradation that agent load may add.

| Indicator | Definition | Target, or what is reported |
|---|---|---|
| Voice V2V | The end of the caller's speech to the agent's first audio, from waveforms at the caller (`lh-voice bots`), with fixed endpointing; a turn not answered within 30 s counts at 30 s | **The SLO: p95 at most 250 ms above voice alone with as many callers.** For two callers, voice alone is the median of five runs, 1,767 ms (the four on the W3 build alone: 1,768.5 ms), so the SLO is 2,017 ms. W2's one-run tables used 1,744 ms and 1,994 |
| The voice turn's stages | Endpointing, the reply's first sentence, TTS's first audio, transport and playout, from the voice worker's line per reply | Reported per point; the reply's first sentence dominates under load |
| Realtime first byte | The gateway's request log, exact to the millisecond | Reported per point |
| Barge-in to silence | The caller starts speaking during the agent's speech, until the agent's audio falls below threshold | Reported per point |
| Server-side voice turn | The utterance sent to STT, until the first byte of TTS audio (`lh-bench voice-turn`): the part serving controls | p95 ≤ 800 ms (voice alone: 589 ms p50 and 665 ms p90, W1) |
| Background goodput | The prompt and completion tokens per second of the agent requests that complete during the calls | Reported: the frontier's x-axis. Its window is the calls' duration, which differs by configuration |
| Agents' time to first token | AIPerf's per-request records | p50 and p99, reported |
| The gateway's decisions | Under an admission policy: deferrals by reason, degraded replies, false alarms of the headroom, resubmissions (`lh_gateway_admission_*`) | Reported per point |
| Control-plane overhead | `lh_gateway_overhead_seconds`, on every request: from the end of the request body to the upstream write, with an admission policy's decision but not the wait it imposes | p99 ≤ 2 ms per request. On cyan with the W1 build: p99 0.44 ms for voice alone and 0.58 ms beside the agent replay, and +0.38 ms to a voice request's first token paired against the engine's own socket ([BENCHMARKS.md](BENCHMARKS.md#the-control-plane-lh-gateway-in-front-of-the-engine)). On an Apple M5 Max, an admission policy adds 6–24 µs of its own time on average, `leaseholder` 9 µs ([`lh-gateway` README](../crates/lh-gateway/README.md#overhead)) |

Not computed: audio gaps (the continuity indicator), the interactive class's resume target, and the cache-contract honor rate, which went with leases.

Dashboards and alerts on these indicators are planned as code in `ops/`.
