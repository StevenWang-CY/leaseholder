//! The deterministic policy core shared by the simulator and, later, the gateway.
//!
//! Pure code: no I/O and no clock. Callers pass token counts and get durations back.
//! - [`cost`]: what serving a prompt costs from each tier (the E0 model), and decode time;
//! - [`geometry`]: KV bytes per token, the engine block, and tier capacities in blocks.

pub mod cost;
pub mod geometry;

pub use cost::{CostError, Linear, PromptSources, Quadratic, TierCost};
pub use geometry::{FROZEN_BOX_CPU_BYTES, FROZEN_BOX_POOL_TOKENS, KvGeometry, TierCapacity};
