# lh-gateway

`lh-gateway` is the process in front of one vLLM engine. It carries Leaseholder's request contract: who a request belongs to, which session and SLO class it has, and which engine inputs a client may not set. It relays responses byte for byte, cancels the engine request when the client leaves, and ingests lifecycle events. Its admission point runs `lh-core`'s admission policies: today the E2 baselines it emulates, and from W3 Leaseholder's own policy (turn-imminent headroom, leases, the recompute ledger).

It is a harness, not a general-purpose router: one upstream, loopback listeners, and nothing a stock proxy would not need besides the contract above. Its cost is measured below.

## What it does

- **Routes.** `POST /v1/chat/completions`, `/v1/completions` and `/v1/responses` are proxied with session, class and sanitization. `GET /v1/models` is proxied with sanitized headers. `POST /v1/events` takes lifecycle events. `GET /metrics`, `/health` and `/ready` are the gateway's own and need no key. Any other path is 404, and a wrong method 405, both in OpenAI error format.
- **Listeners.** Any number of loopback TCP ports and Unix sockets. A Unix socket is created with mode 0600 and removed at exit; a socket file that nobody accepts on (left by a crash) is replaced at start, while a live socket or a file that is not a socket is an error. A Unix listener may name a tenant for requests without a known key. TCP listeners may not, because every user of the host can reach loopback.
- **Tenants and keys.** `Authorization: Bearer <key>` is hashed with SHA-256 and compared with every configured digest in constant time; the loop never exits early, so timing reveals neither which key matched nor how much of a digest did. A known key selects its tenant, on any listener. Otherwise the request belongs to the listener's tenant, or gets 401. The voice worker's OpenAI client sends a placeholder key, so on the voice socket an unknown key also means the socket's tenant.
- **Sanitization.** From every request: the engine-control headers (`X-Vllm-Priority`, `X-SMG-Priority`, `X-data-parallel-rank`, and `X-Session-ID` unless the tenant is trusted), `X-Request-Id`, every `X-LH-` header, `X-Claude-Code-Session-ID`, `Authorization`, and hop-by-hop headers, including those named in `Connection`. From every JSON body: `session_id`, `priority`, `kv_transfer_params`, `cache_salt`, `request_id` and `vllm_xargs` (`lh_proto::headers`). Only the top level of the body is parsed; values are copied through as raw JSON, and keys are compared after unescaping. Each removal of an input the client may not control is counted in `lh_gateway_sanitized_total`.
- **What it sets.** `X-Session-ID` (the resolved session), `X-Request-Id` (a fresh id, also returned to the client), the body `cache_salt` (the first 32 hex digits of SHA-256("leaseholder-cache-salt", NUL, tenant)), and the class's priority in the signals `upstream_priority` lists: `X-Vllm-Priority` by default (realtime 0, interactive 1, background 2, as vLLM's priority scheduling reads it), the same number as the body's `priority` (SGLang's), or `X-SMG-Priority` (`system`, `interactive` and `bulk`, for SMG's priority scheduler). Every proxied inference response carries `X-LH-Session-ID`.
- **Sessions.** The first present source in [PROTOCOL.md](../../docs/PROTOCOL.md#sessions) order: a trusted tenant's `X-Session-ID`, `X-Claude-Code-Session-ID`, the body's `prompt_cache_key`. A valid session id is used as it is; any other value becomes `h-<first 32 hex of SHA-256(tenant, NUL, value)>`. With no source the gateway mints `m-<32 random hex>`.
- **Classes.** A trusted tenant may ask for one with `X-LH-Class` (an unknown value is 400); otherwise the tenant's `default_class` applies. A class stricter than the tenant's `max_class` is clamped to it and counted.
- **Admission.** Under an `[admission]` policy, a request whose session and class are resolved waits there until the policy admits it, and is refused when its queue is full or its wait reaches `queue_timeout_s` ([Admission](#admission)). Nothing polls: a waiting request learns of its admission on a channel. An admitted request holds its place until its response ends or its client leaves; a client that leaves while it waits gives its place up at once.
- **Relay.** A pump task reads the upstream body into a bounded channel (`stream_buffer_frames`, 64 by default) that the client's response body drains. A full channel stops the pump from reading the upstream, so a slow client slows the engine's socket, not the gateway's memory. If the channel stays full for `stall_timeout_s`, the pump aborts the upstream request and the client's response ends with an error. When the client leaves, its response body is dropped, the pump sees the channel close and drops the upstream body at once, which closes the upstream connection; vLLM aborts a request whose client disconnects. A client that leaves before the response headers is handled the same way: hyper drops the handler, and with it the upstream request. A response cut short, by a stall, by the upstream or by a shed, never ends cleanly, so the client cannot mistake it for a complete one.
- **Usage.** Token usage is read from non-streaming JSON bodies and from SSE `data:` lines, for all three APIs (`prompt_tokens`/`input_tokens`, `completion_tokens`/`output_tokens`, `cached_tokens`). Only lines that carry a non-null `usage` or `finish_reason` are parsed. The stream itself is never rewritten.
- **Lifecycle events.** `POST /v1/events` accepts one `lh_proto::LifecycleEvent` or a JSON array of them, from trusted tenants only (403 otherwise). A batch is validated in full before any of it is recorded; an unknown `type` is 400 with the event's index. Events are deduplicated by `(source, event_id)` within a window of `event_dedup_capacity` ids and `event_dedup_ttl_s`. New events go to an `EventSink`; v0's `LogSink` writes one log line per event. The answer is 202 `{"accepted": n, "duplicates": m}`.
- **Lifecycle.** SIGTERM or SIGINT stops the listeners, lets in-flight requests finish for up to `drain_timeout_s`, removes the socket files and exits 0. A configuration error exits 2 with the file and the cause; a listener that cannot bind exits 1.

## What it does not do

- Leaseholder's own admission (headroom timed by lifecycle events, deferral, a realtime class that is never shed), leases, prefetch or the ledger: these are `lh-core`'s, from W3. The admission policies today are baselines.
- More than one upstream, retries, or load balancing. A session lives on the node that holds its KV.
- TLS. Every listener binds loopback or a private socket, and the upstream is the loopback end of an SSH tunnel.
- The Anthropic Messages API: agentic-api translates it and calls the gateway.
- Inferring `turn_started` and `turn_finished` from traffic. The finish reason is logged per request, which is what that inference will use.
- The Responses API's stop reasons (its usage is read).

## Running

```sh
lh-gateway --config ~/.config/leaseholder/gateway.toml
```

[`deploy/cyan/gateway.example.toml`](../../deploy/cyan/gateway.example.toml) is the configuration for cyan, and [`deploy/cyan/systemd/lh-gateway.service`](../../deploy/cyan/systemd/lh-gateway.service) its unit. The [runbook](../../ops/RUNBOOK.md#the-gateway) covers installing and checking it.

## Configuration

Unknown keys are errors, so a misspelled setting fails at start instead of silently taking its default.

| Key | Default | Meaning |
|---|---|---|
| `upstream` | required | The engine, `http://host:port` |
| `upstream_idle_timeout_s` | 4 | How long an idle upstream connection is kept for reuse. Keep it below the engine's keep-alive (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, 5 s by default), or a request can go out on a connection the engine is closing |
| `upstream_priority` | `["vllm-header"]` | Where the class's priority goes upstream: any of `"vllm-header"`, `"body"` and `"smg-header"`. The empty list sends none, as a plain client does: E2's stock arm. vLLM 0.32's FCFS scheduler ignores priorities rather than rejecting them |
| `max_body_bytes` | 33554432 (32 MiB) | Larger request bodies get 413 |
| `stream_buffer_frames` | 64 | Frames buffered between the upstream and a client |
| `stall_timeout_s` | 30 | How long that buffer may stay full before the stream is aborted |
| `first_byte_timeout_s` | 600 | How long to wait for the upstream's response headers before 504; for a request preempt-resubmit may preempt, for its first body byte |
| `drain_timeout_s` | 30 | How long a shutdown waits for in-flight requests |
| `event_dedup_capacity` | 100000 | Lifecycle-event ids remembered for deduplication |
| `event_dedup_ttl_s` | 600 | How long an event id is remembered |
| `log_format` | `"text"` | `"text"` or `"json"` (one object per line) |
| `log_level` | `"info"` | `error`, `warn`, `info`, `debug` or `trace` |

Each `[[listen]]` entry sets exactly one of `tcp = "127.0.0.1:18080"` or `unix = "/absolute/path.sock"`, and a Unix entry may set `tenant`. Each `[tenants.<name>]` table (names from `[A-Za-z0-9._-]`) sets:

| Key | Default | Meaning |
|---|---|---|
| `keys_sha256` | `[]` | Hex SHA-256 digests of the tenant's API keys; the keys never appear in the file. A tenant with no keys must be named by a listener |
| `trusted` | `false` | May set `X-Session-ID` and `X-LH-Class`, and post lifecycle events |
| `default_class` | `"interactive"` | The class of requests that do not choose one |
| `max_class` | `default_class` | The strictest class the tenant may use |

The `[admission]` table sets the admission policy. A key that belongs to other policies than the chosen one is an error, like an unknown key.

| Key | Default | Policies | Meaning |
|---|---|---|---|
| `policy` | `"none"` | | `"none"` (every request goes upstream at once), `"static-cap"`, `"token-reservation"`, `"flow-control"` or `"preempt-resubmit"` |
| `background_max_in_flight` | required | `static-cap` | Background requests in flight at most |
| `pool_tokens` | 36400 | `token-reservation` | The KV pool in tokens; the default is the frozen box's |
| `reserved_tokens` | required | `token-reservation` | Pool tokens background requests leave free; less than `pool_tokens` |
| `ceilings` | required | `flow-control` | `{ realtime = 16, interactive = 12, background = 8 }`: each class is dispatched while fewer requests than its ceiling are in flight. The ceilings must not increase from realtime to background |
| `queue_limit` | 1024 | all but `none` | Requests each class's queue holds; one more gets 429 |
| `queue_timeout_s` | 60 | all but `none` | How long a request may wait before it gets 503 |

## Admission

The admission point is between resolving a request (tenant, sanitization, session, class) and the upstream request. `lh_core::admission` decides, from the order of events alone: a request's arrival, its response's first byte, its departure. The gateway does the waiting, its timeouts and the cancellation. A policy holds some classes; requests of the others go upstream at once. Within a class, waiting requests are admitted in arrival order, and none goes ahead of a waiting request of a stricter class.

The policies are E2's baselines, the simple answers an operator would try and what routers ship, run here so that every arm has the same engine, clients and measurement. The emulations of routers say what they reproduce and what they do not.

- **`static-cap`** (E2 arm 6): at most `background_max_in_flight` background requests in flight; the rest wait. Realtime and interactive requests are not held. It is a background concurrency limit, as an operator would put one in front of the engine.
- **`token-reservation`** (E2 arm 6, the equal-size static reservation): a background request is admitted while the estimated tokens of the background requests in flight, its own and `reserved_tokens` fit in `pool_tokens`, so that the reservation stays free for realtime and interactive requests, which are not held. A request too large to fit beside the reservation is admitted once no other background request is in flight, so it cannot hold up the queue for good.
  - *The estimate.* There is no tokenizer. A request's prompt tokens are its session's last reported `prompt_tokens` plus the growth of the body since that request (an agent's next turn resends the conversation and adds to it), at 4 bytes per token. A body smaller than the session's last is another conversation (a subagent shares its parent's session) or a compacted one, and is estimated as for a session without a report: its whole body at 4 bytes per token. The body is the one sent upstream, after sanitization. The last reports of 4,096 sessions are kept, the earliest forgotten first.
  - *Its error on the replay.* On the W2 operating points, the agent replay at 8, 16 and 24 session trees (415 to 1,521 requests), the estimate is off by 0.4 to 0.5% of all prompt tokens, against 1.9 to 2.2% for bytes / 4 alone. Of the requests of 1,000 tokens or more, 90% are within 1.3 to 1.6% (bytes / 4: 3.4 to 4.5%), 99% within 4.1 to 5.8% (4.5 to 6.5%) and all within 9.4% (8.2%). AIPerf's synthetic prompts run at 3.9 to 4.0 bytes per token, close to the estimate's 4; the session's report leaves only a turn's growth to that rate, so text further from it moves the estimate less than it moves bytes / 4. `cargo run --release -p lh-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...` replays an export as the gateway would see it; its byte counts are AIPerf's, headers included, where the gateway counts the body alone.
  - *What it leaves out.* The estimate counts a request's prompt, not the KV it holds as it decodes, and requests that share a prefix share its blocks in the engine. The sum is a proxy for the pool's use, not a measurement of it.
- **`flow-control`** (E2 arm 8, llm-d's flow control, emulated): each class is a priority band, and the bands are dispatched strictly, a stricter band's waiting requests always first. `ceilings` holds the bands back: a request goes upstream while the requests in flight, of every class, are fewer than its class's ceiling, and since the ceilings fall from realtime to background, background is held back first as the engine fills. Within a band, requests go in arrival order. A realtime or interactive request blocked by its ceiling sheds the newest background requests in flight, as many as it takes, if that is enough to let it in. Their upstream requests are cancelled at once, and their clients get 429 `shed`, or an error that ends a stream already begun. When shedding is not enough, because realtime and interactive requests fill the ceiling, a realtime request waits like any other and can be refused at the queue limit or timeout, which Leaseholder's own admission never does to realtime ([ADR-0003](../../docs/adr/0003-realtime-is-never-shed.md)).
  - *What it reproduces.* llm-d v1.0's flow control with its priority-holdback usage limit and in-flight eviction (`flowControl.enableEviction`): strict priority bands, a ceiling per band that holds the lower priorities back as utilization rises, and the eviction of sheddable requests in flight when stricter demand is blocked, which their clients see as HTTP 429.
  - *What it does not.* Utilization is the number of requests in flight, where llm-d reads a saturation detector over the engine's metrics, so the ceilings are counts rather than fractions of saturation. llm-d orders a band by an SLO deadline from the `x-llm-d-slo-ttft-ms` header; here the requests of a band share one class, and with one target per class, earliest deadline first is arrival order. Eviction takes effect at once and picks the newest background request, the one with the least work done, where llm-d paces evictions behind a confirmation gate (the evicted streams end, then a grace period; at most 2 per decision). Fairness between the flows of a band, and llm-d's limits on queued bytes, are left out.
- **`preempt-resubmit`** (E2 arm 8, a router's abort-and-resubmit of background work on a voice turn, with SMG's rule that only a request without a first byte may be preempted): a realtime arrival preempts the background requests in flight that have no first body byte yet. Their upstream requests are dropped, which aborts them in the engine, and they wait at the front of the queue; background requests that arrive meanwhile wait behind them. Once every realtime request has its first byte, or has ended, they are all sent again. Since a background request can be preempted until its first byte, the gateway holds its response back until then: the client sees nothing of a dropped attempt and gets one response, whose first byte comes later. For a streaming response this delays only the headers, and the first-byte timeout covers each attempt to its first byte. A response already streaming is never preempted, and interactive requests are not held.
  - *What it counts.* Resubmissions, and the prompt tokens of the dropped attempts, estimated as for token-reservation: an upper bound on the prefill they discarded, since an attempt may be dropped while it still waits in the engine, and the blocks it finished stay in the prefix cache for its resubmission.
  - *What it does not reproduce.* SMG preempts to free a concurrency slot for a higher class, and answers the preempted request 503 with `Retry-After`, leaving the retry to the client; E2 runs SMG itself for that (arm 8b). This emulation preempts to clear the voice turn's prefill whatever the slots, and resubmits on the client's behalf.

## Errors

Errors the gateway generates use OpenAI's format, `{"error": {"message", "type", "code"}}`, and carry `X-Request-Id`. Responses from the engine pass through unchanged.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_class`, `invalid_json`, `invalid_event`, `unreadable_body` | An unknown `X-LH-Class`; a body that is not a JSON object, or that cannot be read; an invalid lifecycle event |
| 401 | `invalid_api_key` | No known key, on a listener without a tenant |
| 403 | `untrusted_tenant` | Lifecycle events from an untrusted tenant |
| 404, 405 | `not_found`, `method_not_allowed` | Any other route or method |
| 413 | `body_too_large` | A body over `max_body_bytes` |
| 429 | `queue_full`, `shed` | The request's class queue at the admission point is full (`queue_limit`); a background request shed by flow control before its response began. The `type` is `rate_limit_error` |
| 502 | `upstream_unavailable` | The engine could not be reached, or closed the connection before answering |
| 503 | `queue_timeout` | Not admitted within `queue_timeout_s` |
| 504 | `upstream_timeout` | No response headers within `first_byte_timeout_s` |

## Metrics and logs

`GET /metrics` serves the OpenMetrics text format from a registry the gateway owns. Label values come from fixed sets or from the configuration, so the number of series is bounded. Every series except those of `lh_gateway_requests_total` and `lh_gateway_events_total` exists from the start, at zero.

| Metric | Labels | Meaning |
|---|---|---|
| `lh_gateway_requests_total` | `tenant`, `route`, `class`, `status` | API requests. `tenant` is empty when unauthenticated, `class` for routes without one; status 499 means the client left before any response |
| `lh_gateway_request_outcomes_total` | `class`, `outcome` | Proxied requests by how they ended: `completed` (relayed in full, whatever the status), `cancelled` (the client left first), `stalled`, `error` (the upstream failed), `shed` (the admission policy gave its place to a stricter class) |
| `lh_gateway_overhead_seconds` | `route` | The end of a request's body to the moment hyper writes the request to an upstream connection: parse, sanitization, resolution, admission and connection checkout, without the time a request waits for admission. The number behind the 2 ms SLO; buckets from 50 µs, with 2 ms as a bound |
| `lh_gateway_request_body_seconds` | `route` | Receipt of a request's headers to the end of its body: the client's upload, which a direct call to the engine waits for too |
| `lh_gateway_first_byte_seconds` | `class` | Receipt to the first response body byte from the upstream |
| `lh_gateway_duration_seconds` | `class` | Receipt to the end of the upstream response |
| `lh_gateway_streams_active` | `class` | Responses being relayed |
| `lh_gateway_sanitized_total` | `field` | Inputs removed because clients may not set them |
| `lh_gateway_class_clamped_total` | | Requests whose class was stricter than their tenant's `max_class` |
| `lh_gateway_upstream_errors_total` | `kind` | `connect`, `request`, `timeout`, `body` |
| `lh_gateway_events_total` | `type`, `source`, `duplicate` | Lifecycle events received |
| `lh_gateway_usage_tokens_total` | `class`, `kind` | `prompt`, `completion` and `cached` tokens from response usage |
| `lh_gateway_admission_admitted_total` | `class` | Requests the admission point let go upstream, whether they waited or not, and those of classes the policy does not hold; zero under policy `none` |
| `lh_gateway_admission_waiting` | `class` | Requests waiting at the admission point |
| `lh_gateway_admission_wait_seconds` | `class` | Arrival at the admission point to admission |
| `lh_gateway_admission_refused_total` | `class`, `reason` | Requests the admission point refused: `queue_full` (429) or `timeout` (503) |
| `lh_gateway_admission_resubmits_total` | `class` | Requests sent upstream again after a preemption dropped their attempt |
| `lh_gateway_admission_discarded_tokens_total` | `class` | Estimated prompt tokens of the attempts preemptions dropped: at most the prefill they discarded |

Each request writes one log line at its end (target `lh_gateway::request`, level `warn` for 5xx, `error` and `stalled`): `request_id`, `tenant`, `session`, `class`, `route`, `status`, `outcome`, `admission_wait_ms` (under an admission policy), `resubmits` (when preempted), `first_byte_ms`, `total_ms`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `finish_reason` and, for the gateway's own errors, `error`. Lifecycle events log under `lh_gateway::events`.

`GET /health` answers 200 while the process runs. `GET /ready` answers 200 if the engine's `/health` answered within the last second, asking it again otherwise, and 503 if it does not answer.

## Design notes

- **Upstream client: hyper-util's pooled client rather than reqwest.** The request is already an `http::Request` with a sanitized `HeaderMap`, so it goes upstream without conversion, and the response body is hyper's `Incoming`, read frame by frame. reqwest wraps the same client and would add a URL parse, header conversions and redirect handling per request, none of which a fixed loopback upstream needs. Idle upstream connections close after `upstream_idle_timeout_s`, before the engine's keep-alive closes them under a request. With vLLM's default of 5 s, a voice call, whose turns are often more than 4 s apart, opens a new upstream connection for most turns; on cyan the engine keeps connections for 60 s and the gateway for 55 s.
- **Request ids** are a random 64-bit prefix chosen at start and a counter. They need to be unique, not unpredictable (a client cannot set them), and they cost no system call. Minted session ids use 128 random bits.
- **The overhead metric starts when the request body is complete and ends at the first poll of the upstream request body,** which hyper does on a checked-out connection as it writes the request, so connection checkout is included. The body's arrival is timed separately (`lh_gateway_request_body_seconds`): a client that sends its body some time after its headers would otherwise have its own delay counted as the gateway's.

## Overhead

`cargo run --release -p lh-gateway --example overhead` measures the latency the gateway adds and its CPU time. A mock engine that answers at once, the client and two proxies run in one process, each on its own tokio runtime with 4 worker threads, over loopback TCP with pooled keep-alive connections. Each scenario runs direct to the engine, through a bare reverse proxy on the same stack (axum server, hyper-util client, bodies passed through untouched: the HTTP hop alone), and through `lh-gateway` with a trusted tenant, a key, `X-Session-ID`, `X-LH-Class` and a `prompt_cache_key` body. The gateway formats its per-request log line, as in production. "Added" is the gateway's quantile minus the direct one. CPU is the busy time of the proxy runtime's workers, system calls included, per request (per 8-chunk stream for the streaming rows), which is also milliseconds per 1,000 requests.

cyan (2× Xeon Silver 4310 at 2.1 GHz, Linux 5.15), the process pinned to 16 logical CPUs (`taskset -c 0-15`), release build, 2026-10-22. Microseconds, p50 / p99:

| Scenario | Concurrency | Samples | Direct | Bare proxy | lh-gateway | Added by lh-gateway | CPU per request, bare / lh-gateway |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-streaming, 2 KiB body | 1 | 5000 | 44 / 54 | 156 / 171 | 270 / 306 | 226 / 251 | 80 / 191 |
| Non-streaming, 2 KiB body | 8 | 40000 | 43 / 123 | 107 / 212 | 125 / 395 | 81 / 271 | 27 / 57 |
| Non-streaming, 2 KiB body | 64 | 40000 | 243 / 481 | 690 / 1405 | 927 / 2400 | 684 / 1919 | 40 / 55 |
| Non-streaming, 64 KiB body | 1 | 1000 | 36 / 44 | 71 / 80 | 381 / 433 | 345 / 389 | 32 / 254 |
| Non-streaming, 64 KiB body | 8 | 8000 | 106 / 178 | 153 / 321 | 206 / 527 | 100 / 349 | 41 / 109 |
| Streaming, first byte | 1 | 500 | 95 / 129 | 199 / 258 | 270 / 357 | 175 / 228 | 300 / 508 |
| Streaming, per chunk | 1 | 3500 | 31 / 40 | 61 / 71 | 65 / 85 | 34 / 45 | 300 / 508 |
| Streaming, first byte | 8 | 4000 | 97 / 156 | 230 / 342 | 303 / 421 | 206 / 265 | 302 / 452 |
| Streaming, per chunk | 8 | 28000 | 44 / 81 | 92 / 158 | 94 / 162 | 51 / 82 | 302 / 452 |
| Streaming, first byte | 64 | 4000 | 143 / 404 | 422 / 1865 | 440 / 1874 | 296 / 1469 | 247 / 292 |
| Streaming, per chunk | 64 | 28000 | 72 / 223 | 210 / 775 | 234 / 893 | 163 / 670 | 247 / 292 |

The gateway's own histogram over the same run: mean 32 µs from the end of the body to the upstream write; 94.3% of requests within 100 µs, 99.6% within 500 µs, 99.99% within 2 ms.

What the numbers say:
- **The SLO holds, without much room at the extreme.** The largest added p99 is 1.92 ms, for 64 concurrent non-streaming requests in a process that also runs the client and the engine. The gateway's own time stays within 2 ms for 99.99% of requests.
- **At one client, the hop and the gateway's work cost about the same.** The bare proxy adds 112 µs at p50 and the gateway 114 µs more. Per request, the gateway's CPU falls from 191 µs at one client to 55 µs at 64; the bare proxy's falls from 80 to 40 µs. So under load the gateway's own work costs about 15 µs of CPU per request beyond the hop.
- **Large bodies cost a parse and a copy.** A 64 KiB body adds 345 µs at one client: one validating pass over the JSON and one copy. Agent prompts are that size, so they pay a few hundred microseconds against times to first token of 100 ms and more.
- **Streamed chunks cost about the same as the bare hop:** 65 against 61 µs at p50 for one stream.
- **The same benchmark on an Apple M5 Max** (18 cores, macOS 26.6) adds 39 µs at one client, and 0.82 ms at most at p99. cyan's 2.1 GHz Xeon costs several times more per request.

**In place on cyan**, with the voice agent and the agent replay as clients ([BENCHMARKS](../../docs/BENCHMARKS.md#the-control-plane-lh-gateway-in-front-of-the-engine)):
- the gateway's own time is 0.13 ms on average for voice requests;
- paired against the engine's own socket, the gateway adds 0.38 ms to a voice request's first token.

**Admission policies.** The benchmark's second table prices each policy: background requests, which every policy holds, through one gateway per policy, with limits the runs never reach so that a policy adds its bookkeeping and never a wait, and through a gateway without a policy, which runs the same requests just before. Each gateway's own histogram follows the table. It has not run on cyan yet. On the M5 Max, the medians of eight runs on 2026-10-22, with the machine busy with other work, for the 2 KiB non-streaming rows:

| Policy | Own time, mean | CPU per request over `none`, 1 / 8 / 64 clients | p50 over `none`, 1 / 8 / 64 clients |
|---|---:|---:|---:|
| `none` | 33 µs | | |
| `static-cap` | 35 µs | 1 / 2 / 2 µs | 1 / 2 / 28 µs |
| `token-reservation` | 36 µs | 1 / 2 / 2 µs | 1 / 2 / 30 µs |
| `flow-control` | 36 µs | 1 / 3 / 3 µs | 1 / 2 / 34 µs |
| `preempt-resubmit` | 50 µs | −2 / −1 / −2 µs | −5 / −7 / −31 µs |

What the numbers say:
- **Policy `none` is unchanged.** In six runs alternating with the base commit's benchmark, the gateway's own time had a mean of 29 to 32 µs, against 29 to 31 µs at the base (0.5 µs more on average), and the main table's rows moved both ways within the run-to-run spread.
- **The bookkeeping costs a few microseconds.** `static-cap`, `token-reservation` and `flow-control` take the admission lock as a request arrives and as it leaves: 3 to 4 µs of own time and up to 3 µs of CPU per request. `flow-control` also watches every streamed frame for a shed, about 1 µs of CPU per frame. At 64 clients the gateway's 4 workers are saturated, 16 requests to a worker, so a request's latency is about 16 times its CPU, and 2 to 3 µs of CPU show as about 30 µs at p50.
- **`preempt-resubmit` costs no CPU, and its own time is higher.** It takes the lock a third time, at the first byte, yet its CPU and latency are at or below `none`'s: it holds a background response until the first frame, and answers with the headers and that frame together. The hold is what raises its own time, by 10 to 21 µs: forced onto the other policies' requests in an experiment, it raised theirs by 13 to 15 µs.
- **Every policy keeps 99.96% of requests within 2 ms of own time.** The p99 differences between policies are the machine's: single runs show spikes of 0.1 to 1.7 ms at p99, each time in another policy, `none`'s among them.

## Tests

`cargo test -p lh-gateway` runs in under a second and needs only loopback. The integration tests (`tests/gateway/`) put the gateway between a raw HTTP/1.1 client and an in-process mock engine that records each request, streams SSE with set delays, holds a request's answer until the test releases it, and reports when its response body (or a pending handler) is dropped. They cover header and body sanitization, each upstream priority signal, session order, hashing and minting, classes and clamping, per-tenant cache salts, 401/400/413/404/405, byte-exact SSE relay with usage, a client leaving mid-stream (the engine's body is dropped within 100 ms) and before the headers, a stalled reader (aborted after the stall timeout, ending in an error), 502 and 504, events, the metrics exposition, the Unix socket (stale file, mode, live-socket refusal, cleanup), a graceful drain, and the admission point: a static cap that holds while realtime passes, its first-in, first-out queue, 429 at the queue limit and 503 at the queue timeout, a waiting client that leaves and gives its place up, token reservation priced from a session's last usage, flow control's strict band order with arrival order within a band, the shedding of the newest background request for a blocked realtime one, before its response (429, the engine's request dropped) and in the middle of a stream (an error), and preempt-resubmit: a background request preempted before its first byte, waiting for its response headers or in prefill behind them, is dropped at the engine and sent again after the realtime request's first byte, background arriving meanwhile waits too, and the client gets one whole response; a response already streaming is never preempted. The unit tests pin the configuration rules (each admission policy's own keys and defaults among them), the bounded session table of the token estimate, the cache salts and hashed session ids, the constant-time key match at every position, the deduplication window and the SSE scanner under every split of a stream.
