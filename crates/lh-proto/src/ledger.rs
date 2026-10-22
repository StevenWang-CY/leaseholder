//! The recompute ledger: one row per upstream request, saying how much of the prompt was served from cache and
//! why every recomputed token was recomputed.
//!
//! Rows are written as Parquet and queried with DuckDB; per-cause counters are exported to Prometheus.
//!
//! Token accounting, per request:
//! - `prompt_tokens = cached_device_tokens + cached_host_tokens + uncached`;
//! - `expected_reusable_tokens` is the part of the prompt the session already paid for: the longest common token
//!   prefix with its previous prompt plus output (for a first turn, the prefix shared with other sessions);
//! - `recomputed_tokens` is the part of that expected-reusable prefix that was computed again. The `recompute`
//!   spans attribute it to causes and sum to it exactly.

use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{SessionId, SloClass};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LedgerRow {
    pub request_id: String,
    pub session_id: SessionId,
    pub class: SloClass,
    /// The GPU node that served the request.
    pub node: String,
    /// The engine's identity when it served the request. A change voids every lease held on the old epoch.
    pub engine_epoch: String,
    /// Name of the serving configuration, so rows from different setups never mix.
    pub config: String,
    /// A control-plane prefetch warm-up. Synthetic rows are excluded from every SLI and from usage.
    #[serde(default)]
    pub synthetic: bool,
    pub outcome: Outcome,

    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub expected_reusable_tokens: u32,
    pub cached_device_tokens: u32,
    pub cached_host_tokens: u32,
    pub recomputed_tokens: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recompute: Vec<RecomputeSpan>,

    /// Receipt at the control plane, in Unix epoch milliseconds.
    pub received_at_ms: u64,
    /// Stage durations in milliseconds; absent when the engine did not report the stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restore_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<f64>,
    /// Realtime only: receipt to the end of the first sentence, the stage that dominates voice-to-voice latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_sentence_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    Cancelled,
    Error,
}

/// A run of recomputed tokens and the reason they were recomputed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RecomputeSpan {
    pub tokens: u32,
    #[serde(flatten)]
    pub cause: RecomputeCause,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "cause", rename_all = "snake_case")]
pub enum RecomputeCause {
    /// A first turn whose shared prefix (system prompt, tools) was not cached.
    ColdFirstTurn,
    /// The prompt diverged from what the session sent before, for example a timestamp or a chat template
    /// re-rendering tool-call JSON. `position` is the first differing token.
    PrefixBreak {
        position: u32,
        /// Index of the message that differs, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<u32>,
        /// The field that differs, when known (for example `tool_calls[0].arguments`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        field: Option<String>,
    },
    /// Evicted from both the GPU pool and the CPU tier.
    EvictedGpuAndCpu,
    /// The lease expired before the session came back.
    LeaseExpired,
    /// Routed to a node that does not hold the session's KV.
    HolderMiss,
    /// The engine preempted the request and recomputed it.
    Preempted,
    /// The engine restarted (a new epoch), so its caches were empty.
    EpochChange,
    /// No cause could be determined, for example after a KV-event gap. Reported as its own share.
    Unknown,
}

impl RecomputeCause {
    /// A stable label for metrics.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::ColdFirstTurn => "cold_first_turn",
            Self::PrefixBreak { .. } => "prefix_break",
            Self::EvictedGpuAndCpu => "evicted_gpu_and_cpu",
            Self::LeaseExpired => "lease_expired",
            Self::HolderMiss => "holder_miss",
            Self::Preempted => "preempted",
            Self::EpochChange => "epoch_change",
            Self::Unknown => "unknown",
        }
    }
}

impl LedgerRow {
    /// Prompt tokens computed by this request: new tokens plus recomputed ones.
    pub fn uncached_tokens(&self) -> u32 {
        self.prompt_tokens
            .saturating_sub(self.cached_device_tokens + self.cached_host_tokens)
    }

    /// Checks the accounting identities every writer must uphold.
    pub fn check(&self) -> Result<(), LedgerError> {
        let cached = self.cached_device_tokens + self.cached_host_tokens;
        if cached > self.prompt_tokens {
            return Err(LedgerError::CachedExceedsPrompt);
        }
        if self.expected_reusable_tokens > self.prompt_tokens {
            return Err(LedgerError::ReusableExceedsPrompt);
        }
        if self.recomputed_tokens > self.expected_reusable_tokens {
            return Err(LedgerError::RecomputedExceedsReusable);
        }
        let attributed: u32 = self.recompute.iter().map(|span| span.tokens).sum();
        if attributed != self.recomputed_tokens {
            return Err(LedgerError::UnattributedTokens {
                recomputed: self.recomputed_tokens,
                attributed,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    CachedExceedsPrompt,
    ReusableExceedsPrompt,
    RecomputedExceedsReusable,
    UnattributedTokens { recomputed: u32, attributed: u32 },
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CachedExceedsPrompt => f.write_str("cached tokens exceed prompt tokens"),
            Self::ReusableExceedsPrompt => {
                f.write_str("expected-reusable tokens exceed prompt tokens")
            }
            Self::RecomputedExceedsReusable => {
                f.write_str("recomputed tokens exceed expected-reusable tokens")
            }
            Self::UnattributedTokens {
                recomputed,
                attributed,
            } => write!(
                f,
                "recompute spans attribute {attributed} tokens, but {recomputed} were recomputed"
            ),
        }
    }
}

impl std::error::Error for LedgerError {}
