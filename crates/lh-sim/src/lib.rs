//! `lh-sim`: a trace-driven simulator of vLLM's two-tier KV cache under agent replays.
//!
//! It replays the SemiAnalysis Claude Code traces as AIPerf's `agentic_replay` sends them
//! ([`aiperf`], [`replay`]), through a model of vLLM 0.32's GPU prefix cache and OffloadingConnector
//! CPU tier ([`engine`]), priced by the E0 tier cost model from `lh-core`. [`run`] runs the arms and
//! the clairvoyant reference, [`metrics`] summarizes them, [`mrc`] computes miss-ratio curves, and
//! [`replay_log`] drives the engine model with a measured AIPerf run.

pub mod aiperf;
pub mod corpus;
pub mod engine;
pub mod metrics;
pub mod mrc;
pub mod prefix;
pub mod replay;
pub mod replay_log;
pub mod run;
pub mod trace;
