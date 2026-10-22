//! Wire types shared by every Leaseholder component.
//!
//! This crate is the single definition of what crosses a process boundary:
//! - [`event`]: lifecycle events (`POST /v1/events`), one schema for agents and voice;
//! - [`headers`]: the HTTP headers and body fields the control plane reads, sets and strips;
//! - [`hint`]: the lease hint carried to the engine in `kv_transfer_params`;
//! - [`ledger`]: one recompute-ledger row per upstream request;
//! - [`class`] and [`session`]: the SLO class and the session id the others carry.
//!
//! The JSON Schemas generated from these types (`schema/`, written by the `lh-proto-schema` binary) are
//! the reference for the Python components; today the engine plugins' contract test reads the lease hint's.

pub mod class;
pub mod event;
pub mod headers;
pub mod hint;
pub mod ledger;
pub mod session;

pub use class::{SloClass, UnknownClass};
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
