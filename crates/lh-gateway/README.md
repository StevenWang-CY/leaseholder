# lh-gateway

`lh-gateway` is the process in front of one vLLM engine. It carries Leaseholder's request contract: who a request belongs to, which session and SLO class it has, and which engine inputs a client may not set. It relays responses byte for byte, cancels the engine request when the client leaves, and ingests lifecycle events. Leaseholder's policy (`lh-core`: admission, turn-imminent headroom, leases, the recompute ledger) will run inside it from W3.

It is a harness, not a general-purpose router: one upstream, loopback listeners, and nothing a stock proxy would not need besides the contract above. Its cost is measured below.

## What it does

- **Routes.** `POST /v1/chat/completions`, `/v1/completions` and `/v1/responses` are proxied with session, class and sanitization. `GET /v1/models` is proxied with sanitized headers. `POST /v1/events` takes lifecycle events. `GET /metrics`, `/health` and `/ready` are the gateway's own and need no key. Any other path is 404, and a wrong method 405, both in OpenAI error format.
- **Listeners.** Any number of loopback TCP ports and Unix sockets. A Unix socket is created with mode 0600 and removed at exit; a socket file that nobody accepts on (left by a crash) is replaced at start, while a live socket or a file that is not a socket is an error. A Unix listener may name a tenant for requests without a known key. TCP listeners may not, because every user of the host can reach loopback.
- **Tenants and keys.** `Authorization: Bearer <key>` is hashed with SHA-256 and compared with every configured digest in constant time; the loop never exits early, so timing reveals neither which key matched nor how much of a digest did. A known key selects its tenant, on any listener. Otherwise the request belongs to the listener's tenant, or gets 401. The voice worker's OpenAI client sends a placeholder key, so on the voice socket an unknown key also means the socket's tenant.
- **Sanitization.** From every request: the engine-control headers (`X-Vllm-Priority`, `X-data-parallel-rank`, and `X-Session-ID` unless the tenant is trusted), `X-Request-Id`, every `X-LH-` header, `X-Claude-Code-Session-ID`, `Authorization`, and hop-by-hop headers, including those named in `Connection`. From every JSON body: `session_id`, `priority`, `kv_transfer_params`, `cache_salt`, `request_id` and `vllm_xargs` (`lh_proto::headers`). Only the top level of the body is parsed; values are copied through as raw JSON, and keys are compared after unescaping. Each removal of an input the client may not control is counted in `lh_gateway_sanitized_total`.
- **What it sets.** `X-Session-ID` (the resolved session), `X-Vllm-Priority` (from the class: realtime 0, interactive 1, background 2), `X-Request-Id` (a fresh id, also returned to the client), and the body `cache_salt`: the first 32 hex digits of SHA-256("leaseholder-cache-salt", NUL, tenant). Every proxied inference response carries `X-LH-Session-ID`.
- **Sessions.** The first present source in [PROTOCOL.md](../../docs/PROTOCOL.md#sessions) order: a trusted tenant's `X-Session-ID`, `X-Claude-Code-Session-ID`, the body's `prompt_cache_key`. A valid session id is used as it is; any other value becomes `h-<first 32 hex of SHA-256(tenant, NUL, value)>`. With no source the gateway mints `m-<32 random hex>`.
- **Classes.** A trusted tenant may ask for one with `X-LH-Class` (an unknown value is 400); otherwise the tenant's `default_class` applies. A class stricter than the tenant's `max_class` is clamped to it and counted.
- **Relay.** A pump task reads the upstream body into a bounded channel (`stream_buffer_frames`, 64 by default) that the client's response body drains. A full channel stops the pump from reading the upstream, so a slow client slows the engine's socket, not the gateway's memory. If the channel stays full for `stall_timeout_s`, the pump aborts the upstream request and the client's response ends with an error. When the client leaves, its response body is dropped, the pump sees the channel close and drops the upstream body at once, which closes the upstream connection; vLLM aborts a request whose client disconnects. A client that leaves before the response headers is handled the same way: hyper drops the handler, and with it the upstream request. A response cut short, by a stall or by the upstream, never ends cleanly, so the client cannot mistake it for a complete one.
- **Usage.** Token usage is read from non-streaming JSON bodies and from SSE `data:` lines, for all three APIs (`prompt_tokens`/`input_tokens`, `completion_tokens`/`output_tokens`, `cached_tokens`). Only lines that carry a non-null `usage` or `finish_reason` are parsed. The stream itself is never rewritten.
- **Lifecycle events.** `POST /v1/events` accepts one `lh_proto::LifecycleEvent` or a JSON array of them, from trusted tenants only (403 otherwise). A batch is validated in full before any of it is recorded; an unknown `type` is 400 with the event's index. Events are deduplicated by `(source, event_id)` within a window of `event_dedup_capacity` ids and `event_dedup_ttl_s`. New events go to an `EventSink`; v0's `LogSink` writes one log line per event. The answer is 202 `{"accepted": n, "duplicates": m}`.
- **Lifecycle.** SIGTERM or SIGINT stops the listeners, lets in-flight requests finish for up to `drain_timeout_s`, removes the socket files and exits 0. A configuration error exits 2 with the file and the cause; a listener that cannot bind exits 1.

## What it does not do

- Admission, deferral, headroom, leases, prefetch or the ledger: these are `lh-core`'s, from W3.
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
| `max_body_bytes` | 33554432 (32 MiB) | Larger request bodies get 413 |
| `stream_buffer_frames` | 64 | Frames buffered between the upstream and a client |
| `stall_timeout_s` | 30 | How long that buffer may stay full before the stream is aborted |
| `first_byte_timeout_s` | 600 | How long to wait for the upstream's response headers before 504 |
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

## Errors

Errors the gateway generates use OpenAI's format, `{"error": {"message", "type", "code"}}`, and carry `X-Request-Id`. Responses from the engine pass through unchanged.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_class`, `invalid_json`, `invalid_event`, `unreadable_body` | An unknown `X-LH-Class`; a body that is not a JSON object, or that cannot be read; an invalid lifecycle event |
| 401 | `invalid_api_key` | No known key, on a listener without a tenant |
| 403 | `untrusted_tenant` | Lifecycle events from an untrusted tenant |
| 404, 405 | `not_found`, `method_not_allowed` | Any other route or method |
| 413 | `body_too_large` | A body over `max_body_bytes` |
| 502 | `upstream_unavailable` | The engine could not be reached, or closed the connection before answering |
| 504 | `upstream_timeout` | No response headers within `first_byte_timeout_s` |

## Metrics and logs

`GET /metrics` serves the OpenMetrics text format from a registry the gateway owns. Label values come from fixed sets or from the configuration, so the number of series is bounded. Series with fixed labels exist from the start, at zero.

| Metric | Labels | Meaning |
|---|---|---|
| `lh_gateway_requests_total` | `tenant`, `route`, `class`, `status` | API requests. `tenant` is empty when unauthenticated, `class` for routes without one; status 499 means the client left before any response |
| `lh_gateway_request_outcomes_total` | `class`, `outcome` | Proxied requests by how they ended: `completed` (relayed in full, whatever the status), `cancelled` (the client left first), `stalled`, `error` (the upstream failed) |
| `lh_gateway_overhead_seconds` | `route` | Receipt of a request to the moment hyper writes it to an upstream connection: body read, parse, sanitization, resolution and connection checkout. The number behind the 2 ms SLO; buckets from 50 µs, with 2 ms as a bound |
| `lh_gateway_first_byte_seconds` | `class` | Receipt to the first response body byte from the upstream |
| `lh_gateway_duration_seconds` | `class` | Receipt to the end of the upstream response |
| `lh_gateway_streams_active` | `class` | Responses being relayed |
| `lh_gateway_sanitized_total` | `field` | Inputs removed because clients may not set them |
| `lh_gateway_class_clamped_total` | | Requests whose class was stricter than their tenant's `max_class` |
| `lh_gateway_upstream_errors_total` | `kind` | `connect`, `request`, `timeout`, `body` |
| `lh_gateway_events_total` | `type`, `source`, `duplicate` | Lifecycle events received |
| `lh_gateway_usage_tokens_total` | `class`, `kind` | `prompt`, `completion` and `cached` tokens from response usage |

Each request writes one log line at its end (target `lh_gateway::request`, level `warn` for 5xx, `error` and `stalled`): `request_id`, `tenant`, `session`, `class`, `route`, `status`, `outcome`, `first_byte_ms`, `total_ms`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `finish_reason` and, for the gateway's own errors, `error`. Lifecycle events log under `lh_gateway::events`.

`GET /health` answers 200 while the process runs. `GET /ready` answers 200 if the engine's `/health` answered within the last second, asking it again otherwise, and 503 if it does not answer.

## Design notes

- **Upstream client: hyper-util's pooled client rather than reqwest.** The request is already an `http::Request` with a sanitized `HeaderMap`, so it goes upstream without conversion, and the response body is hyper's `Incoming`, read frame by frame. reqwest wraps the same client and would add a URL parse, header conversions and redirect handling per request, none of which a fixed loopback upstream needs. Idle upstream connections close after 4 s, before vLLM's 5 s keep-alive timeout closes them under a request.
- **Request ids** are a random 64-bit prefix chosen at start and a counter. They need to be unique, not unpredictable (a client cannot set them), and they cost no system call. Minted session ids use 128 random bits.
- **The overhead metric ends at the first poll of the upstream request body,** which hyper does on a checked-out connection as it writes the request, so connection checkout is included.

## Overhead

`cargo run --release -p lh-gateway --example overhead` measures the latency the gateway adds and its CPU time. A mock engine that answers at once, the client and two proxies run in one process, each on its own tokio runtime with 4 worker threads, over loopback TCP with pooled keep-alive connections. Each scenario runs direct to the engine, through a bare reverse proxy on the same stack (axum server, hyper-util client, bodies passed through untouched: the HTTP hop alone), and through `lh-gateway` with a trusted tenant, a key, `X-Session-ID`, `X-LH-Class` and a `prompt_cache_key` body. The gateway formats its per-request log line, as in production. "Added" is the gateway's quantile minus the direct one. CPU is the busy time of the proxy runtime's workers, system calls included, per request (per 8-chunk stream for the streaming rows), which is also milliseconds per 1,000 requests.

Apple M5 Max (18 cores), macOS 26.6, release build, 2026-10-22. Microseconds, p50 / p99:

| Scenario | Concurrency | Samples | Direct | Bare proxy | lh-gateway | Added by lh-gateway | CPU per request, bare / lh-gateway |
|---|---:|---:|---:|---:|---:|---:|---:|
| Non-streaming, 2 KiB body | 1 | 5000 | 21 / 49 | 45 / 67 | 60 / 85 | 39 / 37 | 13 / 29 |
| Non-streaming, 2 KiB body | 8 | 40000 | 59 / 112 | 122 / 193 | 147 / 225 | 88 / 113 | 27 / 48 |
| Non-streaming, 2 KiB body | 64 | 40000 | 380 / 749 | 746 / 1102 | 905 / 1496 | 525 / 747 | 41 / 54 |
| Non-streaming, 64 KiB body | 1 | 1000 | 36 / 59 | 65 / 156 | 82 / 116 | 47 / 58 | 21 / 40 |
| Non-streaming, 64 KiB body | 8 | 8000 | 73 / 133 | 156 / 255 | 189 / 296 | 116 / 163 | 37 / 68 |
| Streaming, first byte | 1 | 500 | 56 / 112 | 122 / 218 | 132 / 234 | 76 / 122 | 101 / 135 |
| Streaming, per chunk | 1 | 3500 | 34 / 68 | 67 / 121 | 68 / 114 | 33 / 47 | 101 / 135 |
| Streaming, first byte | 8 | 4000 | 71 / 131 | 146 / 250 | 142 / 267 | 71 / 136 | 79 / 96 |
| Streaming, per chunk | 8 | 28000 | 36 / 77 | 83 / 147 | 81 / 133 | 44 / 56 | 79 / 96 |
| Streaming, first byte | 64 | 4000 | 233 / 430 | 559 / 986 | 564 / 1248 | 331 / 818 | 88 / 113 |
| Streaming, per chunk | 64 | 28000 | 79 / 190 | 304 / 580 | 304 / 597 | 225 / 407 | 88 / 113 |

The gateway's own histogram over the same run: mean 30 µs from receipt to the upstream write, 91.3% of requests within 100 µs, 99.8% within 500 µs, all within 2 ms. In a second run the concurrency-1 rows moved by at most 4 µs at p50 and 55 µs at p99; the largest changes were in the first-byte row at concurrency 64, 60 µs at p50 and 213 µs at p99 (605 µs added there instead of 818).

What the numbers say:
- **The SLO holds with room.** The largest added p99 is 0.82 ms (first byte, 64 concurrent streams), against a 2 ms target.
- **Most of the tax is the hop.** At concurrency 1 the bare proxy adds 24 µs at p50 and the gateway 39 µs; the gateway's own work (body read and rewrite, sanitization, the pump task, metrics, the log line) is about 16 µs of CPU per request, and the gateway uses 29 ms of CPU per 1,000 requests in all.
- **Streamed chunks cost the same as the bare hop:** the pump and its channel add no measurable per-chunk latency (67 vs 68 µs at p50).
- **At 64 concurrent requests the cost is queueing,** on 4 worker threads shared with nothing else in production but here sharing a laptop with the client and the engine; the bare proxy adds 366 µs at p50 there too.

On cyan the gateway runs on Linux with 8 cores (`CPUAffinity=0-7`), where system calls, about half of the gateway's CPU on macOS, cost less. These numbers are to be re-measured there.

## Tests

`cargo test -p lh-gateway` runs in under a second and needs only loopback. The integration tests (`tests/gateway/`) put the gateway between a raw HTTP/1.1 client and an in-process mock engine that records each request, streams SSE with set delays, and reports when its response body (or a pending handler) is dropped. They cover header and body sanitization, session order, hashing and minting, classes and clamping, per-tenant cache salts, 401/400/413/404/405, byte-exact SSE relay with usage, a client leaving mid-stream (the engine's body is dropped within 100 ms) and before the headers, a stalled reader (aborted after the stall timeout, ending in an error), 502 and 504, events, the metrics exposition, the Unix socket (stale file, mode, live-socket refusal, cleanup) and a graceful drain. The unit tests pin the configuration rules, the cache salts and hashed session ids, the constant-time key match at every position, the deduplication window and the SSE scanner under every split of a stream.
