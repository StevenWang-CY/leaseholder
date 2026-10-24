//! Per-run results: where prompt tokens came from, modeled latencies, traffic, and the dimensionless
//! regime coordinates.

use lh_core::{KvGeometry, TierCapacity, TierCost};
use serde::Serialize;

use crate::aiperf::ConvKind;
use crate::engine::EngineStats;
use crate::replay::{ReplayStats, RequestRecord};

/// Shares of prompt tokens (each in [0, 1]).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Shares {
    /// Served from the GPU prefix cache.
    pub gpu: f64,
    /// Restored from the CPU tier.
    pub cpu: f64,
    /// Computed: `first_turn + new_context + recomputed`.
    pub computed: f64,
    /// Computed on the first turn of a conversation (a cold, cache-busted session start).
    pub first_turn: f64,
    /// Computed on later turns, in blocks this run had not seen: new context.
    pub new_context: f64,
    /// Computed on later turns, in blocks the run had seen: evicted from every tier (plus the last
    /// block of a fully cached prompt, which vLLM always recomputes).
    pub recomputed: f64,
}

/// The count, mean and percentiles of a sample.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Percentiles {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

impl Percentiles {
    /// Linear-interpolation percentiles (numpy's default).
    pub fn of(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(f64::total_cmp);
        let at = |p: f64| {
            let rank = p / 100.0 * (values.len() - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(values.len() - 1);
            values[lo] + (rank - lo as f64) * (values[hi] - values[lo])
        };
        Self {
            n: values.len(),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p50: at(50.0),
            p90: at(90.0),
            p99: at(99.0),
        }
    }
}

/// Where a run sits: KV pressure, pause over restore (π), restore over recompute (ρ).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Regime {
    /// Live working set over the GPU pool: time-weighted mean and peak.
    pub kv_pressure_mean: f64,
    pub kv_pressure_peak: f64,
    /// Median pause over the CPU-restore time of the median prompt.
    pub pi: f64,
    /// CPU-restore time over recompute time of the median prompt.
    pub rho: f64,
    /// The median pause: the delay before a turn, over the turns that have one (turns sent at once
    /// are not pauses).
    pub median_think_ms: f64,
    pub median_prompt_tokens: f64,
    pub restore_ms_at_median: f64,
    pub recompute_ms_at_median: f64,
}

/// Requests by the kind of conversation that sent them.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RequestMix {
    pub main: u64,
    pub subagent: u64,
    pub flattened: u64,
}

/// The results of one arm. Request and token counts, shares, latencies, restored bytes and stalls
/// cover the profiling requests that completed (AIPerf's results); store and eviction counts, and
/// `engine`, cover the whole run, warmup included.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Summary {
    pub arm: String,
    /// The arm whose timeline a clairvoyant reference replayed.
    pub reference: Option<String>,
    /// vLLM's `num_gpu_blocks` (the pool plus the null block).
    pub gpu_blocks: u64,
    /// Blocks in the CPU tier; 0 for none.
    pub cpu_blocks: u64,
    pub requests: u64,
    pub mix: RequestMix,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub gpu_tokens: u64,
    pub cpu_tokens: u64,
    pub computed_tokens: u64,
    pub shares: Shares,
    /// Modeled time to first token, including waits for free blocks.
    pub ttft_ms: Percentiles,
    /// The same for resumed turns (every turn after a conversation's first).
    pub resume_ttft_ms: Percentiles,
    /// KV bytes restored from the CPU tier over PCIe.
    pub pcie_restored_bytes: u64,
    /// KV bytes loaded from the CPU tier ahead of requests (prefetch), warmup included; left out of
    /// the JSON when zero.
    #[serde(skip_serializing_if = "is_zero")]
    pub pcie_prefetched_bytes: u64,
    /// KV bytes written to the CPU tier.
    pub pcie_stored_bytes: u64,
    pub gpu_evictions: u64,
    pub cpu_evictions: u64,
    /// Requests that waited for free GPU blocks, and how long in total.
    pub stalls: u64,
    pub stall_s: f64,
    pub regime: Regime,
    pub replay: ReplayStats,
    pub engine: EngineStats,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// Summarizes the profiling requests of a run.
pub fn summarize(
    arm: &str,
    capacity: TierCapacity,
    records: &[RequestRecord],
    replay: &ReplayStats,
    engine: &EngineStats,
    cost: &TierCost,
    geometry: &KvGeometry,
) -> Summary {
    let mut s = Summary {
        arm: arm.to_owned(),
        gpu_blocks: capacity.gpu_blocks,
        cpu_blocks: capacity.cpu_blocks,
        replay: replay.clone(),
        engine: engine.clone(),
        ..Default::default()
    };
    let pool_tokens = (capacity.gpu_blocks * geometry.block_tokens).max(1);
    let (mut first, mut recomputed) = (0u64, 0u64);
    let mut ttft = Vec::with_capacity(records.len());
    let mut resume = Vec::new();
    let mut prompts = Vec::with_capacity(records.len());
    let mut thinks = Vec::new();
    for r in records {
        s.requests += 1;
        match r.kind {
            ConvKind::Root => s.mix.main += 1,
            ConvKind::Subagent => s.mix.subagent += 1,
            ConvKind::Flat => s.mix.flattened += 1,
        }
        s.prompt_tokens += r.prompt_tokens;
        s.output_tokens += r.output_tokens;
        s.gpu_tokens += r.sources.gpu;
        s.cpu_tokens += r.sources.cpu;
        s.computed_tokens += r.sources.computed;
        if r.turn == 0 {
            first += r.sources.computed;
        } else {
            recomputed += r.recomputed.min(r.sources.computed);
        }
        let ms = r.ttft_ns as f64 / 1e6;
        ttft.push(ms);
        if r.turn > 0 {
            resume.push(ms);
        }
        if r.blocked_ns > 0 {
            s.stalls += 1;
            s.stall_s += r.blocked_ns as f64 / 1e9;
        }
        prompts.push(r.prompt_tokens as f64);
        if let Some(d) = r.delay_ms.filter(|&d| d > 0.0) {
            thinks.push(d);
        }
    }
    let total = s.prompt_tokens.max(1) as f64;
    let new = s.computed_tokens - first - recomputed;
    s.shares = Shares {
        gpu: s.gpu_tokens as f64 / total,
        cpu: s.cpu_tokens as f64 / total,
        computed: s.computed_tokens as f64 / total,
        first_turn: first as f64 / total,
        new_context: new as f64 / total,
        recomputed: recomputed as f64 / total,
    };
    s.ttft_ms = Percentiles::of(ttft);
    s.resume_ttft_ms = Percentiles::of(resume);
    s.pcie_restored_bytes = geometry.bytes_for_tokens(s.cpu_tokens);
    s.pcie_prefetched_bytes = engine.prefetched_blocks * geometry.block_bytes();
    s.pcie_stored_bytes = engine.cpu_stored_blocks * geometry.block_bytes();
    s.gpu_evictions = engine.gpu_evictions;
    s.cpu_evictions = engine.cpu_evictions;
    let median_prompt = Percentiles::of(prompts).p50;
    let median_think = Percentiles::of(thinks).p50;
    let restore = cost.restore_ms(median_prompt.round() as u64);
    let recompute = cost.recompute_ms(median_prompt.round() as u64);
    s.regime = Regime {
        kv_pressure_mean: replay.working_set_mean_tokens / pool_tokens as f64,
        kv_pressure_peak: replay.working_set_peak_tokens as f64 / pool_tokens as f64,
        pi: if restore > 0.0 {
            median_think / restore
        } else {
            0.0
        },
        rho: if recompute > 0.0 {
            restore / recompute
        } else {
            0.0
        },
        median_think_ms: median_think,
        median_prompt_tokens: median_prompt,
        restore_ms_at_median: restore,
        recompute_ms_at_median: recompute,
    };
    s
}

/// CSV header and one row per summary.
pub fn csv(summaries: &[Summary]) -> String {
    let mut out = String::from(
        "arm,reference,gpu_blocks,cpu_blocks,requests,prompt_tokens,gpu_share,cpu_share,computed_share,first_turn_share,new_context_share,\
         recomputed_share,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,resume_ttft_p50_ms,resume_ttft_p90_ms,\
         resume_ttft_p99_ms,pcie_restored_bytes,pcie_stored_bytes,gpu_evictions,cpu_evictions,stalls,\
         kv_pressure_mean,kv_pressure_peak,pi,rho\n",
    );
    for s in summaries {
        let sh = &s.shares;
        out.push_str(&format!(
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4}\n",
            s.arm,
            s.reference.as_deref().unwrap_or(""),
            s.gpu_blocks,
            s.cpu_blocks,
            s.requests,
            s.prompt_tokens,
            sh.gpu,
            sh.cpu,
            sh.computed,
            sh.first_turn,
            sh.new_context,
            sh.recomputed,
            s.ttft_ms.p50,
            s.ttft_ms.p90,
            s.ttft_ms.p99,
            s.resume_ttft_ms.p50,
            s.resume_ttft_ms.p90,
            s.resume_ttft_ms.p99,
            s.pcie_restored_bytes,
            s.pcie_stored_bytes,
            s.gpu_evictions,
            s.cpu_evictions,
            s.stalls,
            s.regime.kv_pressure_mean,
            s.regime.kv_pressure_peak,
            s.regime.pi,
            s.regime.rho,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lh_core::PromptSources;

    #[test]
    fn percentiles_interpolate_like_numpy() {
        let p = Percentiles::of(vec![4.0, 1.0, 3.0, 2.0]);
        assert_eq!((p.n, p.mean, p.p50), (4, 2.5, 2.5));
        assert!((p.p90 - 3.7).abs() < 1e-12);
        assert!((p.p99 - 3.97).abs() < 1e-12);
        assert_eq!(Percentiles::of(vec![]).n, 0);
    }

    #[test]
    fn pi_counts_only_the_turns_that_pause() {
        let record = |turn, delay_ms| RequestRecord {
            trace: 0,
            conv: 0,
            turn,
            kind: ConvKind::Root,
            warmup: false,
            prompt_tokens: 1_000,
            output_tokens: 1,
            sources: PromptSources::default(),
            recomputed: 0,
            delay_ms,
            issued_ns: 0,
            admitted_ns: 0,
            blocked_ns: 0,
            done_ns: 0,
            ttft_ns: 0,
        };
        // A first turn, two turns sent at once and two pauses: the median pause is 3 s, not 1 s.
        let records = [
            record(0, None),
            record(1, Some(0.0)),
            record(2, Some(2_000.0)),
            record(3, Some(0.0)),
            record(4, Some(4_000.0)),
        ];
        let geometry = KvGeometry::QWEN3_4B_BF16;
        let cost = TierCost::frozen_box();
        let s = summarize(
            "lru-cpu",
            TierCapacity::frozen_box(&geometry),
            &records,
            &ReplayStats::default(),
            &EngineStats::default(),
            &cost,
            &geometry,
        );
        assert_eq!(s.regime.median_think_ms, 3_000.0);
        assert_eq!(s.regime.pi, 3_000.0 / cost.restore_ms(1_000));
    }
}
