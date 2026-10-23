//! `lh-sim`: a trace-driven simulator of vLLM's two-tier KV cache under agent replays.
//!
//! It replays the SemiAnalysis Claude Code traces as AIPerf's `agentic_replay` sends them
//! ([`aiperf`], [`replay`]; [`pauses`] draws TraceLab's tool waits in place of their think times),
//! through a model of vLLM 0.32's GPU prefix cache and OffloadingConnector CPU tier ([`engine`]),
//! served by vLLM's step loop ([`scheduler`], priced by `lh-core`'s step-time model) or at E0's
//! idle-box cost. [`run`] runs the arms and the clairvoyant reference, [`retention`] holds the E1
//! arms' policies, [`metrics`] summarizes a run, [`regime`] sweeps the E1 regime map, [`mrc`]
//! computes miss-ratio curves, and [`replay_log`] drives the engine model with a measured AIPerf run.

pub mod aiperf;
pub mod corpus;
pub mod engine;
pub mod metrics;
pub mod mrc;
pub mod pauses;
pub mod prefix;
pub mod regime;
pub mod replay;
pub mod replay_log;
pub mod retention;
pub mod run;
pub mod scheduler;
pub mod trace;
