//! The step-time model: how long one engine step takes from what it carries, and the host time a
//! request spends before the engine first schedules it. [`TierCost`] is its zero-load limit.
//!
//! vLLM 0.32 serves requests in steps: one forward pass computes a chunk of every scheduled prefill
//! and one token of every scheduled decode. A step that computes `T` tokens takes
//!
//! ```text
//! step = step_ms + us_per_token·T + ns_per_pair·P + ns_per_context·K  (+ mixed_ms)
//! ```
//!
//! - `step_ms`: reading the weights and launching the pass, once per step;
//! - `us_per_token`: the linear layers, per computed token;
//! - `ns_per_pair`: attention compute, per pair of a computed token and a token it attends to (its
//!   cached context and, causally, its own chunk), `P` in all;
//! - `ns_per_context`: attention memory, per token of KV a scheduled request reads (its context after
//!   the step), `K` in all. It is most of what a decode adds to a step;
//! - `mixed_ms`: once more for a step that carries both a prefill chunk and a decode. It is measured,
//!   not derived: such steps take this much longer than the other terms say.
//!
//! Before its first step, a request spends host time: the API server, the tunnel, tokenization and
//! block hashing, linear in its prompt tokens, and `host_per_decode_ms` more for every request
//! decoding meanwhile, whose tokens the API server streams. A CPU restore adds E0's restore term
//! ([`TierCost::restore_ms`]) before the rest of the prompt can be computed, and on this box it is not
//! hidden behind the steps of other requests: the step loop stalls for most of the transfer
//! (`restore_stall` of it).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::cost::{Linear, PromptSources, TierCost};

/// What one step carries, as the model counts it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepLoad {
    /// Tokens computed: every prefill chunk and one per decode.
    pub tokens: u64,
    /// Attention pairs: each computed token times the tokens it attends to.
    pub pairs: u64,
    /// KV tokens read: the context of every scheduled request after the step.
    pub context: u64,
    pub prefills: u32,
    pub decodes: u32,
}

impl StepLoad {
    /// Adds a prefill chunk of `tokens` tokens after `cached` tokens of its context.
    pub fn add_prefill(&mut self, tokens: u64, cached: u64) {
        self.add(tokens, cached);
        self.prefills += 1;
    }

    /// Adds a decode after `context` tokens.
    pub fn add_decode(&mut self, context: u64) {
        self.add(1, context);
        self.decodes += 1;
    }

    fn add(&mut self, tokens: u64, cached: u64) {
        self.tokens += tokens;
        self.pairs += tokens * cached + tokens * (tokens + 1) / 2;
        self.context += cached + tokens;
    }

    /// Whether the step carries both a prefill chunk and a decode.
    pub fn mixed(&self) -> bool {
        self.prefills > 0 && self.decodes > 0
    }
}

/// The fitted cost of an engine step, and of a request's host time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepCost {
    /// Fixed cost of a step, in milliseconds.
    pub step_ms: f64,
    /// Per computed token, in microseconds.
    pub us_per_token: f64,
    /// Per attention pair, in nanoseconds.
    pub ns_per_pair: f64,
    /// Per token of KV read, in nanoseconds.
    pub ns_per_context: f64,
    /// Extra fixed cost of a step that carries both a prefill chunk and a decode, in milliseconds.
    pub mixed_ms: f64,
    /// Host time before the engine first schedules a request, by its prompt tokens.
    pub host: Linear,
    /// Extra host time for every request decoding when a request arrives, in milliseconds.
    pub host_per_decode_ms: f64,
    /// The share of a CPU restore's transfer time for which the step loop stalls.
    pub restore_stall: f64,
}

impl StepCost {
    /// The frozen box ([`TierCost::frozen_box`]'s configuration), fitted on 2026-10-22 by
    /// `crates/lh-sim/scripts/fit_step_cost.py` on the three fresh operating points (8, 16 and 24
    /// sessions, each on a freshly started engine; see the lh-sim README).
    ///
    /// - The step terms and `host` are fitted together, weighted equally on relative error, to
    ///   46,315 decode-only steps (the client's gaps between tokens while every request in flight
    ///   was decoding), 517 requests that ran alone and E0's 14 GPU-hit and recompute cells; the
    ///   median error of each set is within 1.2%. `ns_per_context` is 147,456 bytes of KV per token
    ///   read at 343 GB/s (the card's memory bandwidth is 448 GB/s). `ns_per_pair` is within 13% of
    ///   twice the quadratic term of E0's recompute fit: a cold prompt of `L` tokens makes about
    ///   `L²/2` pairs whatever its chunking. `host` is the chat requests' (the API server renders and
    ///   tokenizes their history); E0's token-id requests spend 5 µs per token less (see the tests).
    /// - `mixed_ms`: the median excess of 1,443 decode steps beside one long prefill.
    /// - `host_per_decode_ms`: 92 GPU hits that arrived while only decodes ran, by how many decoded.
    /// - `restore_stall`: 36 restores of 2,000 tokens or more with only decodes beside them. Over each,
    ///   a request decoding across it fell behind the decode-step model by 0.80 of E0's transfer time
    ///   (median; quartiles 0.74 and 1.01).
    pub const fn frozen_box() -> Self {
        Self {
            step_ms: 11.56,
            us_per_token: 47.99,
            ns_per_pair: 14.17,
            ns_per_context: 429.3,
            mixed_ms: 7.6,
            host: Linear {
                ms: 29.6,
                us_per_token: 7.994,
            },
            host_per_decode_ms: 6.5,
            restore_stall: 0.8,
        }
    }

    /// The duration of a step carrying `load`, in milliseconds.
    pub fn step_ms(&self, load: &StepLoad) -> f64 {
        let mixed = if load.mixed() { self.mixed_ms } else { 0.0 };
        self.step_ms
            + mixed
            + self.us_per_token * load.tokens as f64 / 1e3
            + self.ns_per_pair * load.pairs as f64 / 1e6
            + self.ns_per_context * load.context as f64 / 1e6
    }

    /// The duration of a step carrying `load`.
    pub fn step(&self, load: &StepLoad) -> Duration {
        Duration::from_secs_f64(self.step_ms(load).max(0.0) / 1e3)
    }

    /// Host time of a request with `prompt_tokens` tokens sent while `decoding` requests decode, in
    /// milliseconds.
    pub fn host_ms(&self, prompt_tokens: u64, decoding: u64) -> f64 {
        (self.host.at(prompt_tokens) + self.host_per_decode_ms * decoding as f64).max(0.0)
    }

    /// Time to first token of a request alone on the engine, in milliseconds: its host time, the
    /// restore of its CPU-tier tokens, and the prefill of its computed tokens in steps of at most
    /// `chunk` tokens. vLLM 0.32 gives a lone request the whole step budget.
    pub fn alone_ttft_ms(&self, tier: &TierCost, sources: PromptSources, chunk: u64) -> f64 {
        // vLLM computes at least the last prompt token.
        let computed = sources.computed.max(1);
        let cached = sources.total().saturating_sub(computed);
        self.host_ms(sources.total(), 0)
            + tier.restore_ms(sources.cpu)
            + self.prefill_ms(computed, cached, chunk)
    }

    /// The steps a lone prefill of `tokens` tokens after `cached` takes, in milliseconds, in chunks
    /// of at most `chunk` tokens.
    pub fn prefill_ms(&self, tokens: u64, cached: u64, chunk: u64) -> f64 {
        let (mut left, mut cached, mut ms) = (tokens, cached, 0.0);
        while left > 0 {
            let n = left.min(chunk.max(1));
            let mut load = StepLoad::default();
            load.add_prefill(n, cached);
            ms += self.step_ms(&load);
            cached += n;
            left -= n;
        }
        ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENGTHS: [u64; 7] = [1024, 2048, 4096, 8192, 16384, 24576, 31744];
    // E0 p50 cells (ms) on the frozen box, 2026-10-22 (as in `cost`'s tests).
    const RECOMPUTE_P50: [f64; 7] = [135.0, 198.0, 437.0, 1114.0, 3264.0, 6506.0, 10239.0];
    const GPU_P50: [f64; 7] = [45.0, 40.0, 48.0, 64.0, 97.0, 128.0, 156.0];
    const CPU_P50: [f64; 7] = [64.0, 95.0, 157.0, 281.0, 509.0, 722.0, 903.0];
    /// The frozen box's step budget: a lone request prefills 512 tokens per step.
    const BUDGET: u64 = 512;
    /// E0 measures a GPU hit of a 16-aligned prompt: vLLM computes its last block.
    const BLOCK: u64 = 16;

    fn sources(gpu: u64, cpu: u64, computed: u64) -> PromptSources {
        PromptSources { gpu, cpu, computed }
    }

    /// E0's requests send token ids to `/v1/completions`: their host time is what E0's GPU-hit fit
    /// leaves after the one step a hit takes.
    fn e0_host(cost: &StepCost, tier: &TierCost) -> StepCost {
        let hit_step = |l: u64| {
            let mut load = StepLoad::default();
            load.add_prefill(BLOCK, l - BLOCK);
            cost.step_ms(&load)
        };
        let (a, b) = (1_024, 31_744);
        let slope =
            (tier.gpu_hit.at(b) - hit_step(b) - tier.gpu_hit.at(a) + hit_step(a)) / (b - a) as f64;
        StepCost {
            host: Linear {
                ms: tier.gpu_hit.at(a) - hit_step(a) - slope * a as f64,
                us_per_token: slope * 1e3,
            },
            ..cost.clone()
        }
    }

    fn close(model: f64, measured: f64) -> bool {
        (model - measured).abs() <= (0.12 * measured).max(10.0)
    }

    #[test]
    fn a_step_adds_its_tokens_pairs_and_context() {
        let mut load = StepLoad::default();
        // A 256-token chunk after 1,000 cached tokens, and a decode at context 3,000.
        load.add_prefill(256, 1_000);
        assert!(!load.mixed());
        load.add_decode(3_000);
        assert!(load.mixed());
        assert_eq!(load.tokens, 257);
        assert_eq!(load.pairs, 256 * 1_000 + 256 * 257 / 2 + 3_001);
        assert_eq!(load.context, 1_256 + 3_001);
        let cost = StepCost::frozen_box();
        let expected = cost.step_ms
            + cost.mixed_ms
            + cost.us_per_token * 0.257
            + cost.ns_per_pair * load.pairs as f64 / 1e6
            + cost.ns_per_context * load.context as f64 / 1e6;
        assert!((cost.step_ms(&load) - expected).abs() < 1e-9);
    }

    #[test]
    fn a_decode_step_grows_with_the_context_it_reads() {
        // The decode-only steps the fit used (p50): 13.7 ms for one request at 3.5-4.5k tokens of
        // context, 15.3 ms for two at 7-11k together.
        let cost = StepCost::frozen_box();
        let decodes = |contexts: &[u64]| {
            let mut load = StepLoad::default();
            for &c in contexts {
                load.add_decode(c);
            }
            cost.step_ms(&load)
        };
        let one = decodes(&[3_900]);
        assert!((one - 13.7).abs() < 0.5, "{one}");
        let two = decodes(&[3_300, 5_400]);
        assert!((two - 15.3).abs() < 0.5, "{two}");
    }

    #[test]
    fn alone_it_reproduces_the_e0_cells_with_e0s_host_time() {
        let tier = TierCost::frozen_box();
        let cost = e0_host(&StepCost::frozen_box(), &tier);
        for (i, &n) in LENGTHS.iter().enumerate() {
            let gpu = cost.alone_ttft_ms(&tier, sources(n - BLOCK, 0, BLOCK), BUDGET);
            let cpu = cost.alone_ttft_ms(&tier, sources(0, n - BLOCK, BLOCK), BUDGET);
            assert!(
                close(gpu, GPU_P50[i]),
                "gpu {n}: {gpu:.1} vs {}",
                GPU_P50[i]
            );
            assert!(
                close(cpu, CPU_P50[i]),
                "cpu {n}: {cpu:.1} vs {}",
                CPU_P50[i]
            );
            let recompute = cost.alone_ttft_ms(&tier, sources(0, 0, n), BUDGET);
            if n == 1_024 {
                // 35 ms above any step model with positive costs: 1,024 more tokens (two more
                // 512-token steps, at least 2 × 36 ms) take E0 only 63 ms (198 ms at 2,048).
                // E0's own quadratic fit misses this cell by 10%.
                assert!(
                    (recompute / RECOMPUTE_P50[i] - 0.79).abs() < 0.03,
                    "{recompute}"
                );
            } else {
                assert!(
                    close(recompute, RECOMPUTE_P50[i]),
                    "recompute {n}: {recompute:.1} vs {}",
                    RECOMPUTE_P50[i]
                );
            }
        }
        // E0's token-id requests spend less host time than the replays' chat requests.
        let replay = StepCost::frozen_box();
        assert!(cost.host.us_per_token + 4.5 < replay.host.us_per_token);
    }

    #[test]
    fn its_pair_term_is_twice_e0s_quadratic_term() {
        let cost = StepCost::frozen_box();
        let quadratic = TierCost::frozen_box().recompute.ns_per_token_squared;
        assert!((cost.ns_per_pair / (2.0 * quadratic) - 1.0).abs() < 0.13);
    }

    #[test]
    fn a_chunk_costs_more_the_longer_the_context_it_attends_to() {
        let cost = StepCost::frozen_box();
        // A 512-token chunk: 38 ms on an empty context, 269 ms after 30,000 tokens, where attention
        // over the context is most of the step. A token budget prices both the same.
        let empty = cost.prefill_ms(512, 0, BUDGET);
        let deep = cost.prefill_ms(512, 30_000, BUDGET);
        assert!((empty - 38.2).abs() < 0.5, "{empty}");
        assert!((deep - 268.7).abs() < 0.5, "{deep}");
        assert_eq!(cost.prefill_ms(0, 30_000, BUDGET), 0.0);
        // A longer prefill is its chunks' steps, each after the ones before it.
        let two = cost.prefill_ms(1_024, 30_000, BUDGET);
        assert!((two - deep - cost.prefill_ms(512, 30_512, BUDGET)).abs() < 1e-9);
    }

    #[test]
    fn a_lone_prefill_runs_in_budget_sized_steps() {
        let tier = TierCost::frozen_box();
        let cost = StepCost::frozen_box();
        let s = sources(1_000, 0, 1_200);
        // 512 + 512 + 176 tokens, attending to the growing context.
        let mut expected = cost.host_ms(2_200, 0);
        for (n, cached) in [(512, 1_000), (512, 1_512), (176, 2_024)] {
            let mut load = StepLoad::default();
            load.add_prefill(n, cached);
            expected += cost.step_ms(&load);
        }
        assert!((cost.alone_ttft_ms(&tier, s, BUDGET) - expected).abs() < 1e-9);
        // A restore adds E0's term before the prefill.
        let restored = sources(1_000, 600, 600);
        let mut load = StepLoad::default();
        load.add_prefill(512, 1_600);
        let mut tail = StepLoad::default();
        tail.add_prefill(88, 2_112);
        let expected = cost.host_ms(2_200, 0)
            + tier.restore_ms(600)
            + cost.step_ms(&load)
            + cost.step_ms(&tail);
        assert!((cost.alone_ttft_ms(&tier, restored, BUDGET) - expected).abs() < 1e-9);
    }
}
