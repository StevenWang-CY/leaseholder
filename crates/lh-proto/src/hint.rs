//! The engine lease hint: the whole contract between the control plane and the engine-side cache policy.
//!
//! The control plane attaches it to each engine request as `kv_transfer_params.leaseholder`. The engine decides
//! what to keep from this hint plus recency and size; the control plane decides when to bring KV back. There is
//! no second lease table inside the engine.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{SessionId, SloClass};

/// The key under `kv_transfer_params` that holds the hint.
pub const KV_TRANSFER_PARAMS_KEY: &str = "leaseholder";

/// The lease hint on one engine request, at `kv_transfer_params.leaseholder`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EngineLeaseHint {
    pub session_id: SessionId,
    pub class: SloClass,
    /// Seconds the session's KV should stay at least in host RAM, renewed by every request: the contract term
    /// (300 for a 5-minute Anthropic `cache_control` TTL). 0 means no floor, which is plain LRU.
    #[serde(default)]
    pub ttl_s: f64,
    /// Expected seconds until the session's next request (a tool pause). Caps the floor at that return plus a
    /// short grace period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect_return_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<HintOp>,
}

/// An operation on the session's floor beyond renewing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HintOp {
    /// Drop the session's floor at once (the session closed); its KV falls back to LRU.
    Release,
}

impl EngineLeaseHint {
    /// The `kv_transfer_params` object to send with an engine request.
    pub fn to_kv_transfer_params(&self) -> serde_json::Value {
        serde_json::json!({ KV_TRANSFER_PARAMS_KEY: self })
    }
}
