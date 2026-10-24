# lh-proto

The wire types every Leaseholder component shares: what crosses a process boundary is defined here once.

| Module | What it defines |
|---|---|
| `event` | Lifecycle events (`POST /v1/events`): one schema for agents and voice ([PROTOCOL.md](../../docs/PROTOCOL.md#lifecycle-events), [ADR-0002](../../docs/adr/0002-one-lifecycle-event-schema.md)) |
| `headers` | The headers and body fields the control plane reads, sets and strips ([PROTOCOL.md](../../docs/PROTOCOL.md#headers)) |
| `hint` | The lease hint for the engine's `kv_transfer_params`: unused, removed after v0.3 ([ADR-0006](../../docs/adr/0006-the-engine-change-carries-the-realtime-tier.md)) |
| `ledger` | One recompute-ledger row per upstream request, with its accounting checks: unused, removed after v0.3 |
| `class`, `session` | The SLO class and the session id the others carry |

**JSON Schemas.** `schema/` holds the schemas generated from these types, the reference for the Python components; `make schema` regenerates them, and a test fails when they fall behind the code. Today the engine plugins' contract test reads the lease hint's, and the voice worker's `tests/test_events.py` the lifecycle event's.

**Tests.** `tests/wire.rs` pins the wire format with the examples in `testdata/events.jsonl`; `tests/schema.rs` keeps the schemas current.
