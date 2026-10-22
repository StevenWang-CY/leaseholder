//! Wire types shared by every Leaseholder component.
//!
//! This crate is the single definition of what crosses a process boundary:
//! - [`event`]: lifecycle events (`POST /v1/events`), one schema for agents and voice;
//! - [`headers`]: the HTTP headers and body fields the control plane reads, sets and strips;
//! - [`hint`]: the lease hint carried to the engine in `kv_transfer_params`;
//! - [`ledger`]: one recompute-ledger row per upstream request.
//!
//! Python components consume the JSON Schemas generated from these types (`schema/`, written by the
//! `lh-proto-schema` binary).

pub mod class;
pub mod event;
pub mod headers;
pub mod hint;
pub mod ledger;
mod session;

pub use class::SloClass;
pub use event::{EventKind, EventSource, LifecycleEvent, StopReason};
pub use hint::{EngineLeaseHint, HintOp};
pub use ledger::{LedgerError, LedgerRow, Outcome, RecomputeCause, RecomputeSpan};
pub use session::{SessionId, SessionIdError};

/// The JSON Schema of every top-level wire type, keyed by file stem.
pub fn schemas() -> Vec<(&'static str, schemars::Schema)> {
    vec![
        ("lifecycle_event", schemars::schema_for!(LifecycleEvent)),
        ("engine_lease_hint", schemars::schema_for!(EngineLeaseHint)),
        ("ledger_row", schemars::schema_for!(LedgerRow)),
    ]
}

/// Renders a schema exactly as it is checked in: pretty-printed JSON with a trailing newline.
pub fn schema_json(schema: &schemars::Schema) -> String {
    let mut out = serde_json::to_string_pretty(schema).expect("a schema always serializes");
    out.push('\n');
    out
}
