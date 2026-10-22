//! The deterministic policy core shared by the simulator and the gateway.
//!
//! Pure code: no I/O and no clock. Callers pass token counts and events, and get durations and decisions back.
//! - [`admission`]: whether a request goes upstream now, waits or is refused (the E2 baselines the gateway
//!   emulates);
//! - [`cost`]: what serving a prompt costs from each tier (the E0 model), and decode time;
//! - [`step`]: what one engine step costs from what it carries, for a busy engine;
//! - [`geometry`]: KV bytes per token, the engine block, and tier capacities in blocks.

pub mod admission;
pub mod cost;
pub mod geometry;
pub mod step;

pub use cost::{CostError, Linear, PromptSources, Quadratic, TierCost};
pub use geometry::{FROZEN_BOX_CPU_BYTES, FROZEN_BOX_POOL_TOKENS, KvGeometry, TierCapacity};
pub use step::{StepCost, StepLoad};
