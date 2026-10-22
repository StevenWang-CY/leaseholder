//! AIPerf 0.13.0's `agentic_replay` of Weka traces, mirrored so the simulator sends the requests the
//! measured runs sent, at the moments they sent them.
//!
//! - [`plan`]: what AIPerf's Weka loader builds from a trace: one conversation per detected agent
//!   chain (main agent, flattened agent chains, subagent chains), each turn's think-time delay,
//!   SPAWN branches and SPAWN_JOIN gates, and the cross-stream replay predecessors;
//! - [`snapshot`]: the trajectory of a lane: its snapshot instant `t*`, drawn as AIPerf draws it, and
//!   the streams live at that instant;
//! - [`rng`]: the SHA-256 seed and numpy PCG64 draw behind `t*`;
//! - `chains`: the LCP chain detection the loader runs at both layers.
//!
//! The dispatch rules (warmup, burst starts, delays, spawn offsets, joins, barriers, recycling, the
//! system idle cap) are in [`crate::replay`].

mod chains;
pub mod plan;
pub mod reference;
pub mod rng;
pub mod snapshot;

pub use plan::{Branch, ConvKind, Conversation, PlanOptions, ReplayPlan, Turn};
pub use snapshot::{Snapshot, State};

/// A request as the loader sees it: start time normalized to the trace clock, model interned per trace.
#[derive(Clone, Debug)]
pub(crate) struct Req {
    pub t: f64,
    pub api: Option<f64>,
    pub think: Option<f64>,
    pub model: u8,
    pub input: u64,
    pub out: u64,
    pub hash: Vec<u32>,
}

impl Req {
    /// Interval end in seconds; a missing, negative or non-finite duration counts as zero.
    pub fn end(&self) -> f64 {
        request_end(self.t, self.api)
    }
}

pub(crate) fn request_end(t: f64, api: Option<f64>) -> f64 {
    t + api.filter(|a| a.is_finite()).unwrap_or(0.0).max(0.0)
}
