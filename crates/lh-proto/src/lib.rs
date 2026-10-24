//! Wire types shared by every Leaseholder component.
//!
//! This crate is the single definition of what crosses a process boundary:
//! - [`event`]: lifecycle events (`POST /v1/events`), one schema for agents and voice;
//! - [`headers`]: the HTTP headers and body fields the control plane reads, sets and strips;
//! - [`class`] and [`session`]: the SLO class and the session id the others carry.
//!
//! The JSON Schemas generated from these types (`schema/`, written by the `lh-proto-schema` binary) are
//! the reference for the Python components: the voice worker's tests read the lifecycle event's, and the
//! benchmark's and the voice worker's the SLO class's.

pub mod class;
pub mod event;
pub mod headers;
pub mod session;

pub use class::{SloClass, UnknownClass};
pub use event::{EventKind, EventSource, LifecycleEvent, StopReason};
pub use session::{SessionId, SessionIdError};

/// The JSON Schema of every top-level wire type, keyed by file stem.
pub fn schemas() -> Vec<(&'static str, schemars::Schema)> {
    vec![
        ("lifecycle_event", schemars::schema_for!(LifecycleEvent)),
        ("slo_class", schemars::schema_for!(SloClass)),
    ]
}

/// Renders a schema exactly as it is checked in: pretty-printed JSON with a trailing newline.
pub fn schema_json(schema: &schemars::Schema) -> String {
    let mut out = serde_json::to_string_pretty(schema).expect("a schema always serializes");
    out.push('\n');
    out
}
