//! The E1 retention arms: what the engine does with a session's KV while the session pauses between
//! requests (a tool call, or think time), given what it knows of when the session returns.
//!
//! A pause starts when a request completes and its session's next turn is due after a delay; it ends
//! when that turn is sent. Its KV is the session's latest prompt. The information levels:
//! - none: nothing about the return (`lru-cpu`, and `ttl-*`, which holds for a fixed time);
//! - history: the session's own past pauses ([`PauseHistory`]);
//! - oracle: the exact time the next turn is sent.
//!
//! The arms, each a simplification of the published policy to this engine model:
//! - `ttl-*`, Continuum (arXiv 2511.02230): hold the session's cached blocks for a fixed TTL.
//!   Continuum picks the TTL per tool call; a fixed TTL is swept instead.
//! - `history-prefetch` and `oracle-prefetch`: load the session's blocks back from the CPU tier so
//!   the load ends at the predicted (exact) return, and hold them until it comes.
//! - `tokencake`, Tokencake (arXiv 2510.18586): offload at the call start (here the CPU tier already
//!   has a write-through copy, so the session's free blocks are queued to be reused first), then the
//!   history prefetch. Tokencake's space scheduling across agents is not modeled.
//! - `mori`, MORI (arXiv 2606.00866): the pool reuses first the free blocks of the session expected
//!   back last, by its history. MORI's proactive moves between HBM and DRAM are not modeled.
//! - `infercept`, InferCept (Abhyankar et al., ICML 2024): at each pause, preserve the KV (hold it)
//!   or swap it (offload, as Tokencake) by minimum waste in KV-time: preserving wastes the context
//!   for the predicted pause; swapping wastes the running batch for the step-loop stall of its
//!   restore (the restore stall times E0's restore term). Discarding wastes the batch for the
//!   recompute, never less than the restore here (ρ < 1), so it is chosen only without a CPU tier.
//!
//! Holds give way to requests that need their blocks ([`crate::engine`]), and a prefetch takes only
//! free blocks, so no arm makes a request wait for memory that vLLM would give it.

use serde::Serialize;

use crate::prefix::IdMap;

/// What the engine does with a pausing session's KV.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Retention {
    /// Nothing: vLLM's.
    None,
    /// Hold the session's cached blocks for this long, in seconds.
    Ttl(f64),
    /// Prefetch for the predicted return.
    HistoryPrefetch,
    /// Prefetch for the exact return.
    OraclePrefetch,
    Tokencake,
    Mori,
    InferCept,
}

/// What InferCept does with a pausing session's KV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interception {
    Preserve,
    Swap,
    Discard,
}

/// InferCept's minimum-waste choice for a pause of `pause_ms` over `context` tokens of KV, with
/// `batch` tokens of KV running: the restore of the context stalls the batch for `stall_ms`, its
/// recompute for `recompute_ms`; `swap` is whether a CPU tier can take it.
pub fn intercept(
    context: u64,
    pause_ms: f64,
    batch: u64,
    stall_ms: f64,
    recompute_ms: f64,
    swap: bool,
) -> Interception {
    let preserve = context as f64 * pause_ms;
    let (other, choice) = if swap {
        (batch as f64 * stall_ms, Interception::Swap)
    } else {
        (batch as f64 * recompute_ms, Interception::Discard)
    };
    if preserve <= other {
        Interception::Preserve
    } else {
        choice
    }
}

/// The history information level: an exponentially weighted mean (α = 0.5) of each session's
/// pauses, per tool when the pauses name one; a session that has not paused yet (or not after that
/// tool) is predicted by its own mean over every tool, then by the mean over every session.
#[derive(Debug, Default)]
pub struct PauseHistory {
    by_tool: IdMap<f64>,
    by_session: IdMap<f64>,
    all: Option<f64>,
}

const ALPHA: f64 = 0.5;

fn ewma(mean: Option<f64>, ms: f64) -> f64 {
    mean.map_or(ms, |m| m + ALPHA * (ms - m))
}

fn key(session: u32, tool: Option<u16>) -> u64 {
    (u64::from(session) << 17) | tool.map_or(0, |t| 1 + u64::from(t))
}

impl PauseHistory {
    /// Session `session` paused `ms` after calling `tool`.
    pub fn observe(&mut self, session: u32, tool: Option<u16>, ms: f64) {
        let k = key(session, tool);
        self.by_tool
            .insert(k, ewma(self.by_tool.get(&k).copied(), ms));
        let s = u64::from(session);
        self.by_session
            .insert(s, ewma(self.by_session.get(&s).copied(), ms));
        self.all = Some(ewma(self.all, ms));
    }

    /// The predicted pause of `session` after calling `tool`, in ms.
    pub fn predict(&self, session: u32, tool: Option<u16>) -> Option<f64> {
        self.by_tool
            .get(&key(session, tool))
            .or_else(|| self.by_session.get(&u64::from(session)))
            .copied()
            .or(self.all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_predicts_per_tool_then_per_session_then_overall() {
        let mut h = PauseHistory::default();
        assert_eq!(h.predict(1, None), None);
        h.observe(1, Some(3), 1_000.0);
        h.observe(1, Some(3), 3_000.0);
        h.observe(1, Some(4), 10_000.0);
        assert_eq!(h.predict(1, Some(3)), Some(2_000.0));
        assert_eq!(h.predict(1, Some(4)), Some(10_000.0));
        // Session 1's mean over every tool: 1,000, then 2,000, then 6,000.
        assert_eq!(h.predict(1, Some(5)), Some(6_000.0));
        assert_eq!(h.predict(2, Some(3)), Some(6_000.0));
        h.observe(2, None, 500.0);
        assert_eq!(h.predict(2, None), Some(500.0));
    }

    #[test]
    fn infercept_preserves_short_pauses_and_swaps_long_ones() {
        // 8k tokens of context beside 20k running; the restore stalls the batch for 160 ms.
        let choice = |pause_ms| intercept(8_000, pause_ms, 20_000, 160.0, 1_100.0, true);
        assert_eq!(choice(300.0), Interception::Preserve);
        assert_eq!(choice(400.0), Interception::Preserve);
        assert_eq!(choice(401.0), Interception::Swap);
        // Without a CPU tier the alternative is recomputing, which wastes more.
        assert_eq!(
            intercept(8_000, 2_000.0, 20_000, 160.0, 1_100.0, false),
            Interception::Preserve
        );
        assert_eq!(
            intercept(8_000, 3_000.0, 20_000, 160.0, 1_100.0, false),
            Interception::Discard
        );
    }
}
