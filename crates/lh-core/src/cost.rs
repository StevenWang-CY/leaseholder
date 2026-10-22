//! The tier cost model: time to first token by where a prompt's tokens come from, plus decode time.
//!
//! E0 (`lh-bench tiers`) measures three pure cases on the frozen box, each a time to first token as a
//! function of the prompt length `L`:
//!
//! | Case | Fit |
//! |---|---|
//! | every token a GPU prefix-cache hit | `gpu_hit(L) = a_g + b_g·L` |
//! | every token restored from the CPU tier | `gpu_hit(L) + cpu_restore(L)`, with `cpu_restore(c) = a_c + b_c·c` |
//! | every token computed | `recompute(L) = a_r + b_r·L + c_r·L²` |
//!
//! A request with `g` GPU-hit, `c` restored and `n` computed tokens (`L = g + c + n`, `p = g + c`)
//! composes the three:
//!
//! ```text
//! ttft = gpu_hit(L) + [c > 0]·cpu_restore(c) + [n > B]·prefill(n | p)
//! prefill(n | p) = (a_r − a_g) + (b_r − b_g)·n + c_r·((p + n)² − p²)
//! ```
//!
//! Every prompt pays the cached path (scheduling, hashing, one forward step); restored tokens add the
//! PCIe transfer; computed tokens add their prefill, including attention over the cached prefix. With
//! `g = c = 0` the sum is `recompute(L)` exactly, and with `n = 0` it is the measured hit. vLLM always
//! recomputes at least the last prompt token and the hit fits were measured that way, so a computed
//! tail of at most one engine block `B` is priced as part of the hit.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// A latency that grows linearly with a token count: `ms + us_per_token · tokens`, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Linear {
    pub ms: f64,
    pub us_per_token: f64,
}

impl Linear {
    /// The latency at `tokens`, in milliseconds.
    pub fn at(&self, tokens: u64) -> f64 {
        self.ms + self.us_per_token * tokens as f64 / 1e3
    }

    fn is_finite(&self) -> bool {
        self.ms.is_finite() && self.us_per_token.is_finite()
    }
}

/// A latency quadratic in a token count: `ms + us_per_token · n + ns_per_token_squared · n²`, in
/// milliseconds. The quadratic term is attention.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quadratic {
    pub ms: f64,
    pub us_per_token: f64,
    pub ns_per_token_squared: f64,
}

impl Quadratic {
    /// The latency at `tokens`, in milliseconds.
    pub fn at(&self, tokens: u64) -> f64 {
        let n = tokens as f64;
        self.ms + self.us_per_token * n / 1e3 + self.ns_per_token_squared * n * n / 1e6
    }

    fn is_finite(&self) -> bool {
        self.ms.is_finite()
            && self.us_per_token.is_finite()
            && self.ns_per_token_squared.is_finite()
    }
}

/// Where the prompt tokens of one request come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PromptSources {
    /// Served from the GPU prefix cache.
    pub gpu: u64,
    /// Restored from the CPU tier.
    pub cpu: u64,
    /// Computed (prefilled).
    pub computed: u64,
}

impl PromptSources {
    /// The prompt length.
    pub const fn total(&self) -> u64 {
        self.gpu + self.cpu + self.computed
    }
}

/// The measured cost of serving a prompt from each tier, and of decoding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierCost {
    /// Time to first token of a prompt served entirely from the GPU prefix cache.
    pub gpu_hit: Linear,
    /// Extra time to first token for the tokens restored from the CPU tier.
    pub cpu_restore: Linear,
    /// Time to first token of a prompt that is computed entirely.
    pub recompute: Quadratic,
    /// Decode time per output token after the first, in milliseconds.
    pub inter_token_ms: f64,
    /// The engine block in tokens. A computed tail no longer than this is priced as part of a hit.
    pub block_tokens: u64,
}

impl TierCost {
    /// The frozen box, measured by E0 on 2026-10-22: vLLM 0.32.0, Qwen3-4B-Instruct-2507-FP8 with
    /// the audio servers resident, the 36,400-token pool, the 20 GiB CPU tier and the 512/256 step
    /// cap, idle, 5 samples per cell (`lh-bench tiers`). The inter-token latency is the p50 of the W0
    /// agent replay with 16 sessions.
    pub const fn frozen_box() -> Self {
        Self {
            gpu_hit: Linear {
                ms: 35.0,
                us_per_token: 3.769,
            },
            cpu_restore: Linear {
                ms: 10.3,
                us_per_token: 23.64,
            },
            recompute: Quadratic {
                ms: 47.8,
                us_per_token: 63.11,
                ns_per_token_squared: 8.1258,
            },
            inter_token_ms: 15.6,
            block_tokens: 16,
        }
    }

    /// Loads the tier fits from the `summary.json` that `lh-bench tiers` writes (its `fits` object).
    ///
    /// The restore term is `fits.cpu_over_gpu`, or `fits.cpu − fits.gpu` when only the absolute CPU
    /// fit is present. The summary has no decode or block parameters; they keep their
    /// [`frozen_box`](Self::frozen_box) values.
    pub fn from_e0_summary(json: &str) -> Result<Self, CostError> {
        let summary: E0Summary = serde_json::from_str(json).map_err(CostError::Json)?;
        let fits = summary.fits;
        let gpu_hit = fits.gpu.ok_or(CostError::MissingFit("gpu"))?;
        let cpu_restore = match (fits.cpu_over_gpu, fits.cpu) {
            (Some(extra), _) => extra,
            (None, Some(cpu)) => Linear {
                ms: cpu.ms - gpu_hit.ms,
                us_per_token: cpu.us_per_token - gpu_hit.us_per_token,
            },
            (None, None) => return Err(CostError::MissingFit("cpu_over_gpu")),
        };
        let recompute = fits.recompute.ok_or(CostError::MissingFit("recompute"))?;
        let cost = Self {
            gpu_hit,
            cpu_restore,
            recompute,
            ..Self::frozen_box()
        };
        if !cost.gpu_hit.is_finite() {
            return Err(CostError::NotFinite("gpu"));
        }
        if !cost.cpu_restore.is_finite() {
            return Err(CostError::NotFinite("cpu_over_gpu"));
        }
        if !cost.recompute.is_finite() {
            return Err(CostError::NotFinite("recompute"));
        }
        Ok(cost)
    }

    /// The same model with another decode inter-token latency.
    pub fn with_inter_token_ms(mut self, ms: f64) -> Self {
        self.inter_token_ms = ms;
        self
    }

    /// Time to first token, in milliseconds. See the [module docs](self) for the composition.
    pub fn ttft_ms(&self, sources: PromptSources) -> f64 {
        let total = sources.total();
        let mut ms = self.gpu_hit.at(total);
        if sources.cpu > 0 {
            ms += self.cpu_restore.at(sources.cpu);
        }
        if sources.computed > self.block_tokens {
            ms += self.prefill_ms(sources.computed, sources.gpu + sources.cpu);
        }
        ms.max(0.0)
    }

    /// Time to first token.
    pub fn ttft(&self, sources: PromptSources) -> Duration {
        duration_ms(self.ttft_ms(sources))
    }

    /// Decode time of `output_tokens` after the first token: the first arrives with the prefill.
    pub fn decode(&self, output_tokens: u64) -> Duration {
        duration_ms(self.inter_token_ms * output_tokens.saturating_sub(1) as f64)
    }

    /// Time from admission to the last output token.
    pub fn service(&self, sources: PromptSources, output_tokens: u64) -> Duration {
        self.ttft(sources) + self.decode(output_tokens)
    }

    /// Extra time to first token for restoring `tokens` from the CPU tier.
    pub fn restore_ms(&self, tokens: u64) -> f64 {
        if tokens == 0 {
            0.0
        } else {
            self.cpu_restore.at(tokens).max(0.0)
        }
    }

    /// Time to first token of a cold prompt of `tokens`.
    pub fn recompute_ms(&self, tokens: u64) -> f64 {
        self.recompute.at(tokens).max(0.0)
    }

    /// Prefill of `computed` tokens after `cached` ones, beyond the cached path every prompt pays.
    fn prefill_ms(&self, computed: u64, cached: u64) -> f64 {
        let (n, p) = (computed as f64, cached as f64);
        let r = &self.recompute;
        let g = &self.gpu_hit;
        let ms = (r.ms - g.ms)
            + (r.us_per_token - g.us_per_token) * n / 1e3
            + r.ns_per_token_squared * ((p + n) * (p + n) - p * p) / 1e6;
        ms.max(0.0)
    }
}

fn duration_ms(ms: f64) -> Duration {
    if ms.is_finite() && ms > 0.0 {
        Duration::from_secs_f64(ms / 1e3)
    } else {
        Duration::ZERO
    }
}

#[derive(Deserialize)]
struct E0Summary {
    fits: E0Fits,
}

#[derive(Deserialize)]
struct E0Fits {
    gpu: Option<Linear>,
    cpu: Option<Linear>,
    cpu_over_gpu: Option<Linear>,
    recompute: Option<Quadratic>,
}

/// Why an E0 summary cannot be loaded.
#[derive(Debug)]
pub enum CostError {
    /// The summary is not valid JSON or does not have the `lh-bench tiers` shape.
    Json(serde_json::Error),
    /// `fits` lacks a fit the model needs.
    MissingFit(&'static str),
    /// A fit has a non-finite coefficient.
    NotFinite(&'static str),
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "E0 summary: {e}"),
            Self::MissingFit(name) => write!(f, "E0 summary: fits.{name} is missing"),
            Self::NotFinite(name) => write!(f, "E0 summary: fits.{name} is not finite"),
        }
    }
}

impl std::error::Error for CostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTHS: [u64; 7] = [1024, 2048, 4096, 8192, 16384, 24576, 31744];
    // E0 p50 cells (ms) on the frozen box, 2026-10-22.
    const RECOMPUTE_P50: [f64; 7] = [135.0, 198.0, 437.0, 1114.0, 3264.0, 6506.0, 10239.0];
    const GPU_P50: [f64; 7] = [45.0, 40.0, 48.0, 64.0, 97.0, 128.0, 156.0];
    const CPU_P50: [f64; 7] = [64.0, 95.0, 157.0, 281.0, 509.0, 722.0, 903.0];

    fn gpu(tokens: u64) -> PromptSources {
        PromptSources {
            gpu: tokens,
            ..Default::default()
        }
    }

    fn cpu(tokens: u64) -> PromptSources {
        PromptSources {
            cpu: tokens,
            ..Default::default()
        }
    }

    fn computed(tokens: u64) -> PromptSources {
        PromptSources {
            computed: tokens,
            ..Default::default()
        }
    }

    fn close(model: f64, measured: f64) -> bool {
        // The fits are least squares over medians, so the short prompts are the loosest points.
        (model - measured).abs() <= (0.12 * measured).max(10.0)
    }

    #[test]
    fn frozen_box_reproduces_the_measured_cells() {
        let cost = TierCost::frozen_box();
        for (i, &n) in LENGTHS.iter().enumerate() {
            let r = cost.ttft_ms(computed(n));
            let g = cost.ttft_ms(gpu(n));
            let c = cost.ttft_ms(cpu(n));
            assert!(
                close(r, RECOMPUTE_P50[i]),
                "recompute {n}: model {r:.1} vs {}",
                RECOMPUTE_P50[i]
            );
            assert!(
                close(g, GPU_P50[i]),
                "gpu {n}: model {g:.1} vs {}",
                GPU_P50[i]
            );
            assert!(
                close(c, CPU_P50[i]),
                "cpu {n}: model {c:.1} vs {}",
                CPU_P50[i]
            );
        }
    }

    #[test]
    fn pure_cases_are_the_fits() {
        let cost = TierCost::frozen_box();
        let n = 16_384;
        assert!((cost.ttft_ms(computed(n)) - cost.recompute.at(n)).abs() < 1e-9);
        assert!((cost.ttft_ms(gpu(n)) - cost.gpu_hit.at(n)).abs() < 1e-9);
        let restore = cost.gpu_hit.at(n) + cost.cpu_restore.at(n);
        assert!((cost.ttft_ms(cpu(n)) - restore).abs() < 1e-9);
        // 35.0 + 3.769 * 16.384 ms
        assert!((cost.gpu_hit.at(n) - 96.751_296).abs() < 1e-6);
    }

    #[test]
    fn a_tail_within_one_block_is_a_hit() {
        let cost = TierCost::frozen_box();
        let hit = PromptSources {
            gpu: 8_000,
            computed: 16,
            ..Default::default()
        };
        assert_eq!(cost.ttft_ms(hit), cost.gpu_hit.at(8_016));
        let miss = PromptSources {
            gpu: 8_000,
            computed: 17,
            ..Default::default()
        };
        assert!(cost.ttft_ms(miss) > cost.ttft_ms(hit));
    }

    #[test]
    fn a_resume_prices_attention_over_the_cached_prefix() {
        let cost = TierCost::frozen_box();
        let resume = PromptSources {
            gpu: 8_000,
            computed: 200,
            ..Default::default()
        };
        let expected = cost.gpu_hit.at(8_200)
            + (cost.recompute.ms - cost.gpu_hit.ms)
            + (cost.recompute.us_per_token - cost.gpu_hit.us_per_token) * 0.2
            + cost.recompute.ns_per_token_squared * (8_200.0f64.powi(2) - 8_000.0f64.powi(2)) / 1e6;
        assert!((cost.ttft_ms(resume) - expected).abs() < 1e-9);
        // The same 200 new tokens cost more prefill after a long prefix than as a cold prompt.
        let after_prefix = cost.ttft_ms(resume) - cost.ttft_ms(gpu(8_200));
        let cold = cost.ttft_ms(computed(200)) - cost.ttft_ms(gpu(200));
        assert!(after_prefix > cold + 20.0, "{after_prefix} vs {cold}");
    }

    #[test]
    fn worse_placement_costs_more() {
        let cost = TierCost::frozen_box();
        for total in [64, 1_000, 8_000, 30_000] {
            for moved in [17, total / 2, total] {
                let base = PromptSources {
                    gpu: total,
                    ..Default::default()
                };
                let restored = PromptSources {
                    gpu: total - moved,
                    cpu: moved,
                    computed: 0,
                };
                let recomputed = PromptSources {
                    gpu: total - moved,
                    cpu: 0,
                    computed: moved,
                };
                assert!(cost.ttft_ms(restored) > cost.ttft_ms(base));
                assert!(
                    cost.ttft_ms(recomputed) > cost.ttft_ms(restored),
                    "{total} {moved}"
                );
            }
        }
    }

    #[test]
    fn decode_counts_tokens_after_the_first() {
        let cost = TierCost::frozen_box();
        assert_eq!(cost.decode(0), Duration::ZERO);
        assert_eq!(cost.decode(1), Duration::ZERO);
        assert_eq!(cost.decode(11), Duration::from_millis(156));
        let s = gpu(1_000);
        assert_eq!(
            cost.service(s, 11),
            cost.ttft(s) + Duration::from_millis(156)
        );
    }

    #[test]
    fn loads_an_lh_bench_tiers_summary() {
        let json = r#"{
            "p50_ms": {"recompute": {"1024": 135.0}, "gpu": {}, "cpu": {}},
            "cells": {}, "invalid": {},
            "fits": {
                "gpu": {"ms": 35.0, "us_per_token": 3.769},
                "cpu": {"ms": 45.3, "us_per_token": 27.41},
                "cpu_over_gpu": {"ms": 10.3, "us_per_token": 23.64},
                "recompute": {"ms": 47.8, "us_per_token": 63.11, "ns_per_token_squared": 8.1258}
            },
            "cpu_restore_beats_recompute_from": 1024
        }"#;
        let cost = TierCost::from_e0_summary(json).unwrap();
        assert_eq!(cost, TierCost::frozen_box());
    }

    #[test]
    fn derives_the_restore_term_from_the_absolute_cpu_fit() {
        let json = r#"{"fits": {
            "gpu": {"ms": 60.0, "us_per_token": 7.2},
            "cpu": {"ms": 68.0, "us_per_token": 30.8},
            "recompute": {"ms": 0.0, "us_per_token": 120.0, "ns_per_token_squared": 0.0}
        }}"#;
        let cost = TierCost::from_e0_summary(json).unwrap();
        assert!((cost.cpu_restore.ms - 8.0).abs() < 1e-9);
        assert!((cost.cpu_restore.us_per_token - 23.6).abs() < 1e-9);
    }

    #[test]
    fn rejects_incomplete_summaries() {
        let no_recompute = r#"{"fits": {"gpu": {"ms": 1.0, "us_per_token": 1.0},
            "cpu_over_gpu": {"ms": 1.0, "us_per_token": 1.0}}}"#;
        assert!(matches!(
            TierCost::from_e0_summary(no_recompute),
            Err(CostError::MissingFit("recompute"))
        ));
        let no_cpu = r#"{"fits": {"gpu": {"ms": 1.0, "us_per_token": 1.0},
            "recompute": {"ms": 1.0, "us_per_token": 1.0, "ns_per_token_squared": 0.0}}}"#;
        assert!(matches!(
            TierCost::from_e0_summary(no_cpu),
            Err(CostError::MissingFit("cpu_over_gpu"))
        ));
        assert!(matches!(
            TierCost::from_e0_summary("{}"),
            Err(CostError::Json(_))
        ));
    }
}
