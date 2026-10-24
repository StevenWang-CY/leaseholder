//! The arms of a run, at a set of tier capacities.
//!
//! - `gpu-lru`: the GPU pool alone, with vLLM's LRU free queue.
//! - `lru-cpu`: the pool and the write-through CPU tier, both LRU (vLLM's `LRUCachePolicy`): the
//!   frozen box.
//! - `clairvoyant`: the clairvoyant reference. A first pass runs the reference arm (`lru-cpu` unless
//!   configured otherwise) and records the order in which the engine admitted and finished requests.
//!   A second pass replays that fixed order with Belady's MIN in each tier: the pool reuses the free
//!   block whose next use is farthest, and the CPU tier keeps the blocks with the nearest next uses
//!   (it declines to store a block whose next use is farther than every block it holds). The
//!   timeline, including every wait for free blocks, is the reference arm's; time to first token is
//!   re-priced with the new hit pattern. The free-block count evolves exactly as in the first pass
//!   (a request takes the blocks its footprint needs minus those it shares with running requests,
//!   whatever the policy), so every admission of the first pass fits in the second.
//!
//! The E1 arms add a retention policy to `lru-cpu` ([`crate::retention`]): `ttl-5`, `ttl-30` and
//! `ttl-120` (Continuum's hold), `history-prefetch`, `oracle-prefetch`, `tokencake`, `mori` (with
//! its idleness-ranked pool) and `infercept`.
//!
//! The clairvoyant reference is not a bound on what the arms can achieve: MIN per tier ignores that
//! a block's hit counts only when its whole prefix hits, that a CPU copy is worthless while the GPU
//! holds the block, that faster service would change which requests overlap, and PCIe limits. It is
//! optimal only for the hit count of one tier over a fixed sequence of unit-size blocks, the case
//! `lh-sim mrc` reports as a bound.

use std::fmt;
use std::rc::Rc;
use std::str::FromStr;
use std::time::Instant;

use lh_core::{KvGeometry, TierCapacity, TierCost};
use serde::Serialize;

use crate::corpus::Corpus;
use crate::engine::oracle::Oracle;
use crate::engine::{CpuPolicyKind, Engine, EngineConfig, EngineRequest, GpuPolicyKind};
use crate::metrics::{self, Summary};
use crate::prefix::IdSet;
use crate::replay::{
    self, Replay, ReplayConfig, ReplayOutput, RequestRecord, Service, TimelineEvent,
};
use crate::retention::Retention;

/// A cache-policy arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    GpuLru,
    LruCpu,
    Clairvoyant,
    Ttl5,
    Ttl30,
    Ttl120,
    HistoryPrefetch,
    OraclePrefetch,
    Tokencake,
    Mori,
    InferCept,
}

impl Arm {
    /// The stock arms and the reference, in the order a run reports them by default.
    pub const STOCK: [Self; 3] = [Self::GpuLru, Self::LruCpu, Self::Clairvoyant];

    /// The E1 information ladder: `lru-cpu`, the retention arms, and the reference.
    pub const E1: [Self; 10] = [
        Self::LruCpu,
        Self::Ttl5,
        Self::Ttl30,
        Self::Ttl120,
        Self::HistoryPrefetch,
        Self::OraclePrefetch,
        Self::Tokencake,
        Self::Mori,
        Self::InferCept,
        Self::Clairvoyant,
    ];

    pub const ALL: [Self; 11] = [
        Self::GpuLru,
        Self::LruCpu,
        Self::Clairvoyant,
        Self::Ttl5,
        Self::Ttl30,
        Self::Ttl120,
        Self::HistoryPrefetch,
        Self::OraclePrefetch,
        Self::Tokencake,
        Self::Mori,
        Self::InferCept,
    ];

    /// The name on the command line and in the results.
    pub const fn name(self) -> &'static str {
        match self {
            Self::GpuLru => "gpu-lru",
            Self::LruCpu => "lru-cpu",
            Self::Clairvoyant => "clairvoyant",
            Self::Ttl5 => "ttl-5",
            Self::Ttl30 => "ttl-30",
            Self::Ttl120 => "ttl-120",
            Self::HistoryPrefetch => "history-prefetch",
            Self::OraclePrefetch => "oracle-prefetch",
            Self::Tokencake => "tokencake",
            Self::Mori => "mori",
            Self::InferCept => "infercept",
        }
    }

    /// What the arm does with a pausing session's KV.
    pub const fn retention(self) -> Retention {
        match self {
            Self::GpuLru | Self::LruCpu | Self::Clairvoyant => Retention::None,
            Self::Ttl5 => Retention::Ttl(5.0),
            Self::Ttl30 => Retention::Ttl(30.0),
            Self::Ttl120 => Retention::Ttl(120.0),
            Self::HistoryPrefetch => Retention::HistoryPrefetch,
            Self::OraclePrefetch => Retention::OraclePrefetch,
            Self::Tokencake => Retention::Tokencake,
            Self::Mori => Retention::Mori,
            Self::InferCept => Retention::InferCept,
        }
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Arm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|a| a.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|a| a.name()).collect();
                format!("unknown arm {s:?} (the arms are {})", names.join(", "))
            })
    }
}

/// What every arm of a run shares.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunConfig {
    pub replay: ReplayConfig,
    pub cost: TierCost,
    pub service: Service,
    pub geometry: KvGeometry,
    /// The arm whose timeline the clairvoyant reference replays.
    pub reference: Arm,
}

impl RunConfig {
    /// The W0 replay on the frozen box: [`ReplayConfig::default`], the E0 costs, the step loop
    /// under contention, Qwen3-4B KV, and the clairvoyant reference on the `lru-cpu` timeline.
    pub fn frozen_box(replay: ReplayConfig) -> Self {
        Self {
            replay,
            cost: TierCost::frozen_box(),
            service: Service::frozen_box(),
            geometry: KvGeometry::QWEN3_4B_BF16,
            reference: Arm::LruCpu,
        }
    }

    /// The capacities `arm` actually uses: `gpu-lru` has no CPU tier, nor does a clairvoyant
    /// reference on a `gpu-lru` timeline.
    pub fn effective(&self, arm: Arm, capacity: TierCapacity) -> TierCapacity {
        let gpu_only = match arm {
            Arm::GpuLru => true,
            Arm::Clairvoyant => self.reference == Arm::GpuLru,
            _ => false,
        };
        TierCapacity {
            gpu_blocks: capacity.gpu_blocks,
            cpu_blocks: if gpu_only { 0 } else { capacity.cpu_blocks },
        }
    }

    /// The engine of `arm` at `capacity`.
    pub fn engine_config(&self, arm: Arm, capacity: TierCapacity) -> EngineConfig {
        let (gpu_policy, cpu_policy) = match arm {
            Arm::Clairvoyant => (GpuPolicyKind::Clairvoyant, CpuPolicyKind::Clairvoyant),
            Arm::Mori => (GpuPolicyKind::IdleRanked, CpuPolicyKind::Lru),
            _ => (GpuPolicyKind::Lru, CpuPolicyKind::Lru),
        };
        let capacity = self.effective(arm, capacity);
        EngineConfig {
            gpu_blocks: capacity.gpu_blocks as usize,
            cpu_blocks: capacity.cpu_blocks as usize,
            block_tokens: self.replay.block_tokens as u64,
            gpu_policy,
            cpu_policy,
        }
    }

    fn check(&self) -> Result<(), String> {
        if !matches!(self.reference, Arm::GpuLru | Arm::LruCpu) {
            return Err("the clairvoyant reference needs a stock arm as its reference".into());
        }
        if self.geometry.block_tokens != self.replay.block_tokens as u64 {
            return Err(format!(
                "the KV geometry has {}-token blocks but the replay {}-token blocks",
                self.geometry.block_tokens, self.replay.block_tokens
            ));
        }
        Ok(())
    }
}

/// The results of one arm at one capacity.
#[derive(Clone, Debug, Serialize)]
pub struct ArmResult {
    #[serde(flatten)]
    pub summary: Summary,
    /// Wall-clock time the arm took to simulate, in seconds.
    pub wall_s: f64,
    /// The profiling requests the summary covers.
    #[serde(skip)]
    pub records: Vec<RequestRecord>,
}

/// Runs one stock arm. With `timeline`, the output keeps the engine's admission order for a
/// clairvoyant pass.
pub fn run_arm(
    corpus: &mut Corpus,
    cfg: &RunConfig,
    arm: Arm,
    capacity: TierCapacity,
    timeline: bool,
) -> Result<(ArmResult, ReplayOutput), String> {
    if arm == Arm::Clairvoyant {
        return Err(
            "the clairvoyant reference replays a reference run; use run() or clairvoyant()".into(),
        );
    }
    cfg.check()?;
    let start = Instant::now();
    let engine = Engine::new(&cfg.engine_config(arm, capacity), None);
    let out = Replay::new(
        cfg.replay.clone(),
        corpus,
        engine,
        cfg.cost.clone(),
        cfg.service.clone(),
        arm.retention(),
        timeline,
    )
    .run()?;
    let capacity = cfg.effective(arm, capacity);
    let summary = metrics::summarize(
        arm.name(),
        capacity,
        &out.records,
        &out.stats,
        &out.engine,
        &cfg.cost,
        &cfg.geometry,
    );
    let records = out.records.clone();
    Ok((
        ArmResult {
            summary,
            wall_s: start.elapsed().as_secs_f64(),
            records,
        },
        out,
    ))
}

/// The clairvoyant reference on the timeline of `reference`, a run of `cfg.reference` at `capacity`
/// that kept its timeline.
pub fn clairvoyant(
    reference: &ReplayOutput,
    cfg: &RunConfig,
    capacity: TierCapacity,
) -> Result<ArmResult, String> {
    cfg.check()?;
    let start = Instant::now();
    let timeline = reference
        .timeline
        .as_ref()
        .ok_or("the reference run kept no timeline")?;
    let oracle = Rc::new(Oracle::new(
        timeline.requests.iter().map(|r| r.prompt.as_slice()),
    ));
    let mut engine = Engine::new(&cfg.engine_config(Arm::Clairvoyant, capacity), Some(oracle));
    let mut records = reference.records.clone();
    let mut seen = IdSet::default();
    for event in &timeline.events {
        match *event {
            TimelineEvent::Admit(i) => {
                let r = &timeline.requests[i];
                let req = EngineRequest {
                    prompt: r.prompt.clone(),
                    prompt_tokens: r.prompt_tokens,
                    output_tokens: r.output_tokens,
                };
                let adm = engine.submit(i as u64, req).ok_or_else(|| {
                    format!("request {i} of the reference timeline does not fit in the second pass")
                })?;
                let recomputed = replay::recomputed_tokens(
                    &mut seen,
                    &r.prompt,
                    adm,
                    cfg.replay.block_tokens as u64,
                );
                if let Some(rec) = r.record.map(|k| &mut records[k]) {
                    rec.ttft_ns = match &cfg.service {
                        Service::Idle => {
                            rec.blocked_ns + cfg.cost.ttft(adm.sources).as_nanos() as u64
                        }
                        // The reference's queueing stays; the request's own service is re-priced.
                        Service::Contention { budget, steps } => {
                            let alone = |s| {
                                let ms =
                                    steps.alone_ttft_ms(&cfg.cost, s, budget.max_batched_tokens);
                                (ms * 1e6).round() as u64
                            };
                            (rec.ttft_ns + alone(adm.sources)).saturating_sub(alone(rec.sources))
                        }
                    };
                    rec.sources = adm.sources;
                    rec.recomputed = recomputed;
                }
            }
            TimelineEvent::Finish(i) => {
                engine.finish(i as u64, 0);
            }
        }
    }
    // The waits are the reference arm's.
    let mut stats = engine.stats.clone();
    stats.stalls = reference.engine.stalls;
    let capacity = cfg.effective(Arm::Clairvoyant, capacity);
    let mut summary = metrics::summarize(
        Arm::Clairvoyant.name(),
        capacity,
        &records,
        &reference.stats,
        &stats,
        &cfg.cost,
        &cfg.geometry,
    );
    summary.reference = Some(cfg.reference.name().to_owned());
    Ok(ArmResult {
        summary,
        wall_s: start.elapsed().as_secs_f64(),
        records,
    })
}

/// Runs `arms` at every capacity, calling `done` as each result lands. An arm whose effective
/// capacities repeat (`gpu-lru` across CPU sizes) runs once. The clairvoyant reference reuses the
/// reference arm's run when that arm is one of `arms`.
pub fn run(
    corpus: &mut Corpus,
    cfg: &RunConfig,
    arms: &[Arm],
    capacities: &[TierCapacity],
    mut done: impl FnMut(&ArmResult),
) -> Result<Vec<ArmResult>, String> {
    let with_reference = arms.contains(&Arm::Clairvoyant);
    let mut results: Vec<ArmResult> = Vec::new();
    for &capacity in capacities {
        let mut reference = None;
        for &arm in arms.iter().filter(|&&a| a != Arm::Clairvoyant) {
            let effective = cfg.effective(arm, capacity);
            let repeat = results.iter().any(|r| {
                r.summary.arm == arm.name()
                    && (r.summary.gpu_blocks, r.summary.cpu_blocks)
                        == (effective.gpu_blocks, effective.cpu_blocks)
            });
            let keep = with_reference && arm == cfg.reference;
            if repeat && !keep {
                continue;
            }
            let (result, out) = run_arm(corpus, cfg, arm, capacity, keep)?;
            if !repeat {
                done(&result);
                results.push(result);
            }
            if keep {
                reference = Some(out);
            }
        }
        if with_reference {
            let out = match reference {
                Some(out) => out,
                None => run_arm(corpus, cfg, cfg.reference, capacity, true)?.1,
            };
            let result = clairvoyant(&out, cfg, capacity)?;
            done(&result);
            results.push(result);
        }
    }
    Ok(results)
}
