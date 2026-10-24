//! Other GPUs on the E1 regime map, from modeled costs.
//!
//! Every cost `lh-sim` uses is measured on the frozen box, an RTX 5060 Ti 16 GB: E0's tier fits
//! ([`TierCost::frozen_box`]) and the step-time model ([`StepCost::frozen_box`]). A [`Gpu`] holds
//! another card's published peaks (memory, memory bandwidth, dense FP8 and BF16 tensor throughput,
//! the host link), and they are used only as ratios to the box's: each measured term is divided by
//! the ratio of the published peaks of the kind that bounds it, so the box's measured efficiency
//! carries over. Terms the host and the software set stay the box's: the API server's time, the
//! restore's fixed part, the mixed-step extra and the restore stall. The box's own profile gives its
//! measured costs back exactly; every other GPU's are modeled, and labeled so.
//!
//! The pool is the GPU's memory, less what its default ECC keeps and what the box keeps beside its
//! own (the engine, STT and TTS on one card: the edge pattern), in whole blocks, with the CPU tier a
//! multiple of it. [`sweep`] runs the E1 arms in each GPU's cells and records where it sits on the
//! map (KV pressure, π, ρ), how long the agents leave a realtime tier short of KV, and the step a
//! realtime turn waits behind.

use std::fmt::Write as _;
use std::path::Path;
use std::str::FromStr;

use lh_core::{
    FROZEN_BOX_POOL_TOKENS, KvGeometry, Linear, Quadratic, StepCost, StepLoad, TierCapacity,
    TierCost,
};

use crate::aiperf::PlanOptions;
use crate::corpus::Corpus;
use crate::metrics::Summary;
use crate::regime::{self, signed, thousands};
use crate::replay::{Pauses, ReplayConfig, Service};
use crate::run::{Arm, RunConfig};
use crate::scheduler::StepBudget;

/// A PCIe host link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub generation: u8,
    pub lanes: u8,
}

impl Link {
    /// The rate in each direction, in GB/s: 8, 16 or 32 GT/s a lane for PCIe 3.0, 4.0 and 5.0, with
    /// 128b/130b encoding.
    pub fn gb_per_s(self) -> f64 {
        let gt_per_s = match self.generation {
            3 => 8.0,
            4 => 16.0,
            5 => 32.0,
            g => panic!("PCIe {g}.0 is not modeled"),
        };
        gt_per_s * f64::from(self.lanes) * 128.0 / 130.0 / 8.0
    }
}

/// A GPU, by its published peaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gpu {
    /// The name on the command line.
    pub name: &'static str,
    pub title: &'static str,
    /// Memory in MiB, less what the card's default ECC keeps of it.
    pub memory_mib: u64,
    pub bandwidth_gb_s: f64,
    /// Dense tensor throughput, in TFLOPS.
    pub fp8_tflops: f64,
    pub bf16_tflops: f64,
    pub link: Link,
}

/// The frozen box. NVIDIA gives 16 GB of GDDR7 at 448 GB/s and 759 AI TOPS, FP4 with sparsity
/// (<https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/>,
/// <https://www.nvidia.com/en-us/geforce/graphics-cards/compare/>): 379.5 dense FP4 TFLOPS. Every
/// GeForce Blackwell card in the RTX Blackwell whitepaper's tables (appendices A to C,
/// <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>)
/// has half its FP4 rate in FP8 with FP16 accumulate, and a quarter of that in BF16, whose only rate
/// is with FP32 accumulate. FP8 with FP32 accumulate runs at half rate on these cards, but the box's
/// linear layers beat it (47.99 µs for the 7.27 GFLOP of a token, about 150 TFLOPS), so the
/// FP16-accumulate rate is the peak of the same kind. The link is what both hosts negotiate
/// (`nvidia-smi`'s `pcie.link.gen.hostmax` 3, width 8), not the card's PCIe 5.0.
pub const RTX_5060_TI: Gpu = Gpu {
    name: "rtx-5060-ti",
    title: "RTX 5060 Ti 16 GB",
    memory_mib: 16 << 10,
    bandwidth_gb_s: 448.0,
    fp8_tflops: 759.0 / 4.0,
    bf16_tflops: 759.0 / 16.0,
    link: Link {
        generation: 3,
        lanes: 8,
    },
};

/// NVIDIA's L4 datasheet (<https://www.nvidia.com/en-us/data-center/l4/>): 24 GB at 300 GB/s, 485 FP8
/// and 242 BF16 TFLOPS "shown with sparsity", halved, and PCIe Gen4 x16. Its product brief
/// (PB-11316-001, tables 2 and 3,
/// <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf>) gives
/// the 24 GB as GDDR6 with ECC "Enabled (by default)", and with ECC on, GDDR memory keeps its check
/// bits in 6.25% of itself (the CUDA C++ Best Practices Guide, 9.2.1,
/// <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation>).
/// That leaves 22.5 GiB, of which `nvidia-smi` shows 23,034 MiB
/// (<https://github.com/NVIDIA/TensorRT-LLM/issues/1823>), as it shows 16,311 MiB of the box's
/// 16 GiB.
pub const L4: Gpu = Gpu {
    name: "l4",
    title: "L4 24 GB",
    memory_mib: (24 << 10) * 15 / 16,
    bandwidth_gb_s: 300.0,
    fp8_tflops: 485.0 / 2.0,
    bf16_tflops: 242.0 / 2.0,
    link: Link {
        generation: 4,
        lanes: 16,
    },
};

/// NVIDIA's H100 datasheet, SXM (<https://www.nvidia.com/en-us/data-center/h100/>): 80 GB at 3.35 TB/s,
/// 3,958 FP8 and 1,979 BF16 TFLOPS with sparsity, halved, and PCIe Gen5 at 128 GB/s (x16, both ways).
pub const H100_SXM: Gpu = Gpu {
    name: "h100-sxm",
    title: "H100 SXM 80 GB",
    memory_mib: 80 << 10,
    bandwidth_gb_s: 3_350.0,
    fp8_tflops: 3_958.0 / 2.0,
    bf16_tflops: 1_979.0 / 2.0,
    link: Link {
        generation: 5,
        lanes: 16,
    },
};

/// The profiles, the box first.
pub const GPUS: [Gpu; 3] = [RTX_5060_TI, L4, H100_SXM];

/// Qwen3-4B-Instruct-2507's native context
/// (<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8>): vLLM serves no longer request of it.
pub const MODEL_CONTEXT: u64 = 262_144;

/// The KV a realtime tier keeps per voice call: a turn's prompt and reply, about 680 tokens in the
/// W2 runs on the box, rounded up to 64 blocks.
pub const TOKENS_PER_CALL: u64 = 1_024;

/// What the box keeps beside its pool, of its 16 GiB: 9,288 MiB of engine, STT and TTS (14,408 MiB in
/// use with the three servers idle, `docs/BENCHMARKS.md`, less the engine's 5 GiB `kv-cache-memory`:
/// weights, activations, CUDA graphs and contexts) and 1,976 MiB of headroom (`nvidia-smi` shows
/// 16,311 MiB, 1,903 MiB of them free).
const BOX_BESIDE_POOL: u64 = (16 << 30) - (5 << 30);

/// The regime map's largest session count, where the box is at pressure 7 to 7.5.
const BOX_TOP_SESSIONS: f64 = 40.0;

/// A GPU's peaks over the box's.
struct Ratios {
    bandwidth: f64,
    fp8: f64,
    bf16: f64,
    link: f64,
}

impl Gpu {
    /// Whether these are the box's measured costs rather than modeled ones.
    pub fn measured(&self) -> bool {
        self.name == RTX_5060_TI.name
    }

    pub fn label(&self) -> &'static str {
        if self.measured() {
            "measured"
        } else {
            "modeled"
        }
    }

    fn ratios(&self) -> Ratios {
        let b = &RTX_5060_TI;
        Ratios {
            bandwidth: self.bandwidth_gb_s / b.bandwidth_gb_s,
            fp8: self.fp8_tflops / b.fp8_tflops,
            bf16: self.bf16_tflops / b.bf16_tflops,
            link: self.link.gb_per_s() / b.link.gb_per_s(),
        }
    }

    /// The step-time model: the box's fit with its weight and KV reads scaled by memory bandwidth,
    /// its linear layers by FP8 and its attention by BF16 throughput.
    pub fn step_cost(&self) -> StepCost {
        let r = self.ratios();
        let base = StepCost::frozen_box();
        StepCost {
            // Mostly the weight read: Qwen3-4B-FP8 reads 4.41 GB a step (3.63 G FP8 linear weights
            // and the 0.78 GB BF16 embedding, tied to the LM head), 9.8 ms of the 11.56 ms at the
            // box's 448 GB/s. The other 1.7 ms, launches and kernel tails, need not shrink with
            // bandwidth: vLLM 0.30 decoding Qwen3-4B in BF16 at batch 1 on an H100 SXM takes 4.29 ms
            // a step, 2.45 ms of it the weight read at the datasheet's bandwidth and 1.8 ms the rest
            // (decode-decoded, <https://github.com/zhebrak/decode-decoded>, results/cells.csv). So
            // the H100's 1.55 ms here is a floor: with the box's 1.7 ms kept it would be 3.0 ms.
            step_ms: base.step_ms / r.bandwidth,
            // The linear layers.
            us_per_token: base.us_per_token / r.fp8,
            // Attention over the BF16 cache.
            ns_per_pair: base.ns_per_pair / r.bf16,
            // The KV reads.
            ns_per_context: base.ns_per_context / r.bandwidth,
            // The mixed-step extra, the host time and the restore stall are the software's.
            ..base
        }
    }

    /// The tier costs: the restore's transfer scaled by the host link, and E0's cold prefill by the
    /// parts of the step loop that make it up.
    pub fn tier_cost(&self) -> TierCost {
        let r = self.ratios();
        let base = TierCost::frozen_box();
        let steps = StepCost::frozen_box();
        // E0's cold prefill as the step loop runs one alone: E0's own host time stays, the fixed cost
        // of its steps (one per 512 tokens) scales by bandwidth, the rest of the linear term (the
        // linear layers) by FP8 and the quadratic term (attention) by BF16. This keeps within 8% of
        // the scaled step loop's lone prefill from 4k tokens on, for both modeled GPUs. Each term is
        // written so that ratios of 1 give the box's fit back exactly.
        let host = steps.e0_host(&base);
        let per_step = steps.step_ms / StepBudget::FROZEN_BOX.max_batched_tokens as f64 * 1e3;
        let cold = &base.recompute;
        TierCost {
            // The transfer by the link; the fixed part is the connector's. Gen4 x16 gives 5.91 µs a
            // token, 34.5, 58.7 and 107.1 ms for 4k, 8k and 16k tokens, where vLLM 0.29 measured 34,
            // 65 and 104 ms for Qwen3-4B on an L20, also PCIe Gen4
            // (<https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478>).
            cpu_restore: Linear {
                ms: base.cpu_restore.ms,
                us_per_token: base.cpu_restore.us_per_token / r.link,
            },
            recompute: Quadratic {
                ms: cold.ms / r.bandwidth + host.ms * (1.0 - 1.0 / r.bandwidth),
                us_per_token: cold.us_per_token / r.fp8
                    + host.us_per_token * (1.0 - 1.0 / r.fp8)
                    + per_step * (1.0 / r.bandwidth - 1.0 / r.fp8),
                ns_per_token_squared: cold.ns_per_token_squared / r.bf16,
            },
            // The GPU-hit fit and the decode rate are the idle service's, which the step loop
            // replaces.
            ..base
        }
    }

    /// The GPU pool in tokens: whole blocks of what the box would leave to its pool on this card.
    pub fn pool_tokens(&self, geometry: &KvGeometry) -> u64 {
        let bytes = (self.memory_mib << 20).saturating_sub(BOX_BESIDE_POOL);
        geometry.blocks_in_bytes(bytes) * geometry.block_tokens
    }

    /// The pool and a CPU tier `cpu_ratio` times its size.
    pub fn capacity(&self, geometry: &KvGeometry, cpu_ratio: f64) -> TierCapacity {
        TierCapacity::with_cpu_ratio(geometry, self.pool_tokens(geometry), cpu_ratio)
    }

    /// The frozen box's run of `replay`, with this GPU's costs.
    pub fn run_config(&self, replay: ReplayConfig) -> RunConfig {
        RunConfig {
            cost: self.tier_cost(),
            service: Service::Contention {
                budget: StepBudget::FROZEN_BOX,
                steps: self.step_cost(),
            },
            ..RunConfig::frozen_box(replay)
        }
    }
}

impl FromStr for Gpu {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        GPUS.into_iter().find(|g| g.name == s).ok_or_else(|| {
            let names: Vec<_> = GPUS.iter().map(|g| g.name).collect();
            format!("unknown GPU {s:?} (the GPUs are {})", names.join(", "))
        })
    }
}

/// The step of one 512-token background prefill chunk after `context` cached tokens, alone: what a
/// realtime request that arrives during it waits behind, in ms.
pub fn chunk_ms(steps: &StepCost, context: u64) -> f64 {
    let mut load = StepLoad::default();
    load.add_prefill(StepBudget::FROZEN_BOX.max_batched_tokens, context);
    steps.step_ms(&load)
}

/// The largest prompt plus output of a request of `corpus`, in tokens at token scale 1: at token
/// scale `k`, a request needs at most `k` times this, less one, of KV.
pub fn longest_request(corpus: &mut Corpus) -> Result<u64, String> {
    let mut longest = 0;
    for i in 0..corpus.len() {
        let plan = corpus.plan(i)?;
        for t in plan.convs.iter().flat_map(|c| &c.turns) {
            let tokens = u64::from(t.hash_blocks) * u64::from(plan.block_size);
            longest = longest.max(tokens + u64::from(t.max_tokens));
        }
    }
    Ok(longest)
}

/// The token scale closest to the traces' original token counts that `gpu` can serve: the largest
/// power of two up to `max_scale` at which a request of `longest` tokens at token scale 1
/// ([`longest_request`]) fits both its pool and the model's context, as the 32x replay fits the box.
/// `None` when it fits at no scale.
pub fn fitted_token_scale(
    gpu: &Gpu,
    geometry: &KvGeometry,
    longest: u64,
    max_scale: u32,
) -> Option<u32> {
    // vLLM's null block holds no KV.
    let usable = (gpu.pool_tokens(geometry) / geometry.block_tokens).saturating_sub(1);
    let fits = |k: u32| {
        let tokens = u64::from(k) * longest;
        tokens <= MODEL_CONTEXT
            && tokens.saturating_sub(1).div_ceil(geometry.block_tokens) <= usable
    };
    std::iter::successors(Some(1u32), |k| k.checked_mul(2))
        .take_while(|&k| k <= max_scale && fits(k))
        .last()
}

/// The counts of `sessions` that `gpu` runs at `token_scale`: those up to 40·r, where r is its pool
/// over the box's in the replay's tokens. A pool r times larger meets the box's pressures at about r
/// times the sessions, and the box itself runs the regime map's counts.
pub fn sessions_for(
    gpu: &Gpu,
    geometry: &KvGeometry,
    token_scale: u32,
    sessions: &[usize],
) -> Vec<usize> {
    let box_pool = (FROZEN_BOX_POOL_TOKENS * u64::from(token_scale)) as f64;
    let r = gpu.pool_tokens(geometry) as f64 / box_pool;
    sessions
        .iter()
        .copied()
        .filter(|&s| s as f64 <= BOX_TOP_SESSIONS * r.max(1.0))
        .collect()
}

/// One cell of [`sweep`]: a GPU, the token scale and seed of its replay, and the regime cell (whose
/// capacity is the GPU's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub gpu: Gpu,
    pub token_scale: u32,
    pub seed: u64,
    pub regime: regime::Cell,
}

/// The cells of `gpus`, each at its token scale, over the counts of `sessions` it runs and every
/// seed.
pub fn cells(
    gpus: &[(Gpu, u32)],
    sessions: &[usize],
    seeds: &[u64],
    pauses: Pauses,
    cpu_ratio: f64,
) -> Vec<Cell> {
    let geometry = KvGeometry::QWEN3_4B_BF16;
    let mut out = Vec::new();
    for &(gpu, token_scale) in gpus {
        for n in sessions_for(&gpu, &geometry, token_scale, sessions) {
            for &seed in seeds {
                out.push(Cell {
                    gpu,
                    token_scale,
                    seed,
                    regime: regime::Cell {
                        pauses,
                        sessions: n,
                        cpu_ratio,
                        capacity: gpu.capacity(&geometry, cpu_ratio),
                    },
                });
            }
        }
    }
    out
}

/// A cell of [`sweep`] and what the E1 arms did in it, at the cell's seed.
#[derive(Clone, Debug)]
pub struct Placement {
    pub gpu: Gpu,
    pub token_scale: u32,
    pub result: regime::CellResult,
}

/// Runs the E1 arms in every cell on `jobs` threads, each GPU with its own costs. `base` is the rest
/// of the replay; its concurrency, seed, token scale and pauses give way to each cell's.
pub fn sweep(
    traces: &Path,
    plan: PlanOptions,
    base: &ReplayConfig,
    cells: &[Cell],
    jobs: usize,
) -> Result<Vec<Placement>, String> {
    regime::par_map(traces, plan, cells, jobs, |corpus, cell| {
        let replay = ReplayConfig {
            concurrency: cell.regime.sessions,
            seed: cell.seed,
            token_scale: cell.token_scale,
            pauses: cell.regime.pauses,
            ..base.clone()
        };
        let result = regime::run_cell(corpus, &cell.gpu.run_config(replay), cell.regime)
            .map_err(|e| format!("{}: {e}", cell.gpu.name))?;
        Ok(Placement {
            gpu: cell.gpu,
            token_scale: cell.token_scale,
            result,
        })
    })
}

fn opt(x: Option<f64>) -> String {
    x.map_or(String::new(), |v| format!("{v:.4}"))
}

/// One row per cell and arm: the regime sweep's columns between the GPU's and the realtime tier's.
pub fn csv(results: &[Placement]) -> String {
    let mut text = format!(
        "gpu,costs,token_scale,pool_tokens,{},kv_pressure_peak,median_prompt_tokens,\
         realtime_short_share,realtime_blocked_share,chunk_ms\n",
        regime::CSV_COLUMNS
    );
    let block_tokens = KvGeometry::QWEN3_4B_BF16.block_tokens;
    for p in results {
        let steps = p.gpu.step_cost();
        for (s, &arm) in p.result.arms.iter().zip(&Arm::E1) {
            write!(
                text,
                "{},{},{},{},",
                p.gpu.name,
                p.gpu.label(),
                p.token_scale,
                p.result.cell.capacity.gpu_blocks * block_tokens,
            )
            .expect("writing to a String");
            regime::csv_row(&mut text, &p.result, s, arm);
            let median = s.regime.median_prompt_tokens;
            writeln!(
                text,
                ",{:.3},{median:.0},{},{},{:.1}",
                s.regime.kv_pressure_peak,
                opt(s.replay.realtime_short_share),
                opt(s.replay.realtime_blocked_share),
                chunk_ms(&steps, median.round() as u64),
            )
            .expect("writing to a String");
        }
    }
    text
}

/// `x` to a tenth, with thousands separators.
fn tenths(x: f64) -> String {
    let t = (x * 10.0).round() as u64;
    format!("{}.{}", thousands(t / 10), t % 10)
}

/// `mib` in GiB: whole, or to a tenth.
fn gib(mib: u64) -> String {
    if mib.is_multiple_of(1024) {
        thousands(mib / 1024)
    } else {
        tenths(mib as f64 / 1024.0)
    }
}

/// The GPUs: their published peaks and the costs modeled from them.
pub fn profiles(gpus: &[Gpu]) -> String {
    let geometry = KvGeometry::QWEN3_4B_BF16;
    let mut text = String::from(
        "| GPU | Costs | Memory | Bandwidth | Dense FP8 / BF16 | Host link, each way | Pool \
         | Step: fixed | per token | per pair | per KV token read | Restore | Cold prefill |\n\
         |---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|\n",
    );
    for g in gpus {
        let (s, t) = (g.step_cost(), g.tier_cost());
        let c = &t.recompute;
        writeln!(
            text,
            "| {} | {} | {} GiB | {} GB/s | {} / {} TFLOPS | PCIe {}.0 x{}, {:.1} GB/s \
             | {} tokens | {:.2} ms | {:.2} µs | {:.2} ns | {:.1} ns | {:.1} ms + {:.2} µs/token \
             | {:.1} ms + {:.2} µs/token + {:.3} ns/token² |",
            g.title,
            g.label(),
            gib(g.memory_mib),
            thousands(g.bandwidth_gb_s as u64),
            tenths(g.fp8_tflops),
            tenths(g.bf16_tflops),
            g.link.generation,
            g.link.lanes,
            g.link.gb_per_s(),
            thousands(g.pool_tokens(&geometry)),
            s.step_ms,
            s.us_per_token,
            s.ns_per_pair,
            s.ns_per_context,
            t.cpu_restore.ms,
            t.cpu_restore.us_per_token,
            c.ms,
            c.us_per_token,
            c.ns_per_token_squared,
        )
        .expect("writing to a String");
    }
    text
}

/// The placements: per GPU and token scale, one row per session count, each the mean over its seeds.
/// `hash_block` is the corpus's tokens per hash block, for the scaling against the original traces'
/// 64.
pub fn markdown(results: &[Placement], hash_block: u32, realtime_tokens: u64) -> String {
    fn lru(p: &Placement) -> &Summary {
        p.result.arm(Arm::LruCpu)
    }
    let mut groups: Vec<Vec<&Placement>> = Vec::new();
    for p in results {
        let key = |q: &Placement| (q.gpu, q.token_scale, q.result.cell);
        match groups.iter_mut().find(|g| key(g[0]) == key(p)) {
            Some(g) => g.push(p),
            None => groups.push(vec![p]),
        }
    }
    let Some(first) = groups.first() else {
        return String::new();
    };
    let seeds: Vec<_> = first.iter().map(|p| p.result.seed.to_string()).collect();
    let seeds = match &seeds[..] {
        [one] => format!("seed {one}"),
        all => format!("means over seeds {}", all.join(", ")),
    };
    let mut text = String::new();
    writeln!(
        text,
        "**Pauses: {}; the CPU tier {}× the pool; {seeds}.** Pressure: mean / peak. Voice KV short: \
         the share of time the live working set / the running requests' KV leave less than {} \
         tokens of the pool free. Chunk: one 512-token background prefill chunk at the median \
         prompt, in ms. Resume TTFT p50 / p90 in ms; gains against `lru-cpu` at p50 / p90.\n\n\
         | GPU | Sessions | Pressure | π | ρ | Voice KV short | Chunk | `lru-cpu` | `clairvoyant` \
         | `oracle-prefetch` | `history-prefetch` | `tokencake` |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        regime::pauses_name(first[0].result.cell.pauses),
        first[0].result.cell.cpu_ratio,
        thousands(realtime_tokens),
    )
    .expect("writing to a String");
    let mut previous = None;
    for g in &groups {
        let p = g[0];
        let n = g.len() as f64;
        let mean = |f: &dyn Fn(&Placement) -> f64| g.iter().map(|q| f(q)).sum::<f64>() / n;
        let share = |f: &dyn Fn(&Placement) -> Option<f64>| {
            g.iter().map(|q| f(q)).sum::<Option<f64>>().map(|x| x / n)
        };
        let short = share(&|q| lru(q).replay.realtime_short_share);
        let blocked = share(&|q| lru(q).replay.realtime_blocked_share);
        let steps = p.gpu.step_cost();
        let ms = |pct| thousands(mean(&|q| q.result.resume(Arm::LruCpu, pct)).round() as u64);
        let gains = |arm: Arm| {
            format!(
                "{} / {}%",
                signed(100.0 * mean(&|q| q.result.gain(arm, 50)), 0),
                signed(100.0 * mean(&|q| q.result.gain(arm, 90)), 0)
            )
        };
        let gpu = if previous == Some((p.gpu.name, p.token_scale)) {
            String::new()
        } else {
            let scale = 64.0 / f64::from(hash_block * p.token_scale);
            format!("{} ({}), {scale}×", p.gpu.title, p.gpu.label())
        };
        previous = Some((p.gpu.name, p.token_scale));
        writeln!(
            text,
            "| {gpu} | {} | {:.1} / {:.1} | {:.1} | {:.2} | {} | {:.0} | {} / {} | {} | {} | {} | {} |",
            p.result.cell.sessions,
            mean(&|q| lru(q).regime.kv_pressure_mean),
            mean(&|q| lru(q).regime.kv_pressure_peak),
            mean(&|q| lru(q).regime.pi),
            mean(&|q| lru(q).regime.rho),
            match (short, blocked) {
                (Some(s), Some(b)) => format!("{:.0} / {:.0}%", 100.0 * s, 100.0 * b),
                _ => "–".into(),
            },
            mean(&|q| chunk_ms(&steps, lru(q).regime.median_prompt_tokens.round() as u64)),
            ms(50),
            ms(90),
            gains(Arm::Clairvoyant),
            gains(Arm::OraclePrefetch),
            gains(Arm::HistoryPrefetch),
            gains(Arm::Tokencake),
        )
        .expect("writing to a String");
    }
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use lh_core::PromptSources;

    use super::*;
    use crate::metrics::{Percentiles, Regime};
    use crate::replay::ReplayStats;

    const QWEN: KvGeometry = KvGeometry::QWEN3_4B_BF16;

    /// The box with `f` applied to its peaks.
    fn the_box_with(f: impl FnOnce(&mut Gpu)) -> Gpu {
        let mut g = Gpu {
            name: "test",
            ..RTX_5060_TI
        };
        f(&mut g);
        g
    }

    #[test]
    fn the_box_gives_its_measured_costs_back() {
        let b = RTX_5060_TI;
        assert_eq!(b.step_cost(), StepCost::frozen_box());
        assert_eq!(b.tier_cost(), TierCost::frozen_box());
        assert_eq!(b.pool_tokens(&QWEN), FROZEN_BOX_POOL_TOKENS);
        assert_eq!(b.capacity(&QWEN, 4.0).gpu_blocks, 2_275);
        let replay = ReplayConfig::default();
        assert_eq!(b.run_config(replay.clone()), RunConfig::frozen_box(replay));
        assert!(b.measured() && !L4.measured() && !H100_SXM.measured());
        assert_eq!((b.label(), H100_SXM.label()), ("measured", "modeled"));
        // 759 AI TOPS: 189.75 dense FP8 and 47.4 BF16 TFLOPS; PCIe 3.0 x8 is 7.88 GB/s each way.
        assert_eq!((b.fp8_tflops, b.bf16_tflops), (189.75, 47.4375));
        assert!((b.link.gb_per_s() - 7.877).abs() < 1e-3);
        assert!((H100_SXM.link.gb_per_s() / b.link.gb_per_s() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn each_peak_scales_only_its_terms() {
        let (box_steps, box_tier) = (StepCost::frozen_box(), TierCost::frozen_box());
        let software = |s: &StepCost, t: &TierCost| {
            assert_eq!(
                (s.mixed_ms, s.host, s.host_per_decode_ms, s.restore_stall),
                (
                    box_steps.mixed_ms,
                    box_steps.host,
                    box_steps.host_per_decode_ms,
                    box_steps.restore_stall
                )
            );
            assert_eq!(
                (
                    t.gpu_hit,
                    t.cpu_restore.ms,
                    t.inter_token_ms,
                    t.block_tokens
                ),
                (
                    box_tier.gpu_hit,
                    box_tier.cpu_restore.ms,
                    box_tier.inter_token_ms,
                    box_tier.block_tokens
                )
            );
        };
        // Twice the bandwidth: half the fixed step and the KV reads, and a cheaper cold prefill.
        let g = the_box_with(|g| g.bandwidth_gb_s *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.step_ms, box_steps.step_ms / 2.0);
        assert_eq!(s.ns_per_context, box_steps.ns_per_context / 2.0);
        assert_eq!(
            (s.us_per_token, s.ns_per_pair, t.cpu_restore),
            (
                box_steps.us_per_token,
                box_steps.ns_per_pair,
                box_tier.cpu_restore
            )
        );
        assert!(t.recompute.ms < box_tier.recompute.ms);
        assert!(t.recompute.us_per_token < box_tier.recompute.us_per_token);
        assert_eq!(
            t.recompute.ns_per_token_squared,
            box_tier.recompute.ns_per_token_squared
        );
        software(&s, &t);
        // Twice the FP8 throughput: half the linear layers.
        let g = the_box_with(|g| g.fp8_tflops *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.us_per_token, box_steps.us_per_token / 2.0);
        assert_eq!(
            (s.step_ms, s.ns_per_pair, s.ns_per_context),
            (
                box_steps.step_ms,
                box_steps.ns_per_pair,
                box_steps.ns_per_context
            )
        );
        assert!(t.recompute.us_per_token < box_tier.recompute.us_per_token);
        assert_eq!(t.recompute.ms, box_tier.recompute.ms);
        software(&s, &t);
        // Twice the BF16 throughput: half the attention.
        let g = the_box_with(|g| g.bf16_tflops *= 2.0);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(s.ns_per_pair, box_steps.ns_per_pair / 2.0);
        assert_eq!(
            t.recompute.ns_per_token_squared,
            box_tier.recompute.ns_per_token_squared / 2.0
        );
        assert_eq!(
            (t.recompute.ms, t.recompute.us_per_token, s.step_ms),
            (
                box_tier.recompute.ms,
                box_tier.recompute.us_per_token,
                box_steps.step_ms
            )
        );
        software(&s, &t);
        // Twice the link: half the transfer, the same fixed part.
        let g = the_box_with(|g| g.link.generation = 4);
        let (s, t) = (g.step_cost(), g.tier_cost());
        assert_eq!(
            t.cpu_restore.us_per_token,
            box_tier.cpu_restore.us_per_token / 2.0
        );
        assert_eq!((s, t.recompute), (box_steps.clone(), box_tier.recompute));
        // More memory: a larger pool, by what it adds. The L4's ECC leaves it 22.5 of its 24 GiB.
        let g = the_box_with(|g| g.memory_mib = 24 << 10);
        assert_eq!(
            g.pool_tokens(&QWEN),
            QWEN.blocks_in_bytes(13 << 30) * QWEN.block_tokens
        );
        let g = the_box_with(|g| g.memory_mib = L4.memory_mib);
        assert_eq!(g.pool_tokens(&QWEN), L4.pool_tokens(&QWEN));
        assert_eq!(
            g.pool_tokens(&QWEN),
            QWEN.blocks_in_bytes(23 << 29) * QWEN.block_tokens
        );
        // The L4 has less bandwidth than the box and more of everything else; the H100 has more.
        let (l4, h100) = (L4.step_cost(), H100_SXM.step_cost());
        assert!(l4.step_ms > box_steps.step_ms && h100.step_ms < box_steps.step_ms);
        for s in [&l4, &h100] {
            assert!(s.us_per_token < box_steps.us_per_token);
            assert!(s.ns_per_pair < box_steps.ns_per_pair);
        }
        let restore = |g: Gpu| g.tier_cost().cpu_restore.us_per_token;
        assert!(restore(H100_SXM) < restore(L4) && restore(L4) < restore(RTX_5060_TI));
        assert_eq!(
            (L4.pool_tokens(&QWEN), H100_SXM.pool_tokens(&QWEN)),
            (83_728, 502_432)
        );
    }

    #[test]
    fn the_l4_loses_a_sixteenth_of_its_memory_to_ecc() {
        // ECC, on by default, keeps a sixteenth of the L4's 24 GiB of GDDR6. The 22.5 GiB left, less
        // what the box keeps beside its pool, hold 5,233 blocks: 83,728 tokens, where all 24 GiB
        // would hold 94,656.
        let left = (24 << 30) / 16 * 15 - BOX_BESIDE_POOL;
        assert_eq!(QWEN.blocks_in_bytes(left), 5_233);
        assert_eq!(L4.pool_tokens(&QWEN), 83_728);
        let all = QWEN.blocks_in_bytes((24 << 30) - BOX_BESIDE_POOL) * QWEN.block_tokens;
        assert_eq!(all, 94_656);
        // The table gives the memory left; the box's and the H100's keep all theirs.
        let gpus = profiles(&GPUS);
        assert!(
            gpus.contains("| L4 24 GB | modeled | 22.5 GiB | 300 GB/s |"),
            "{gpus}"
        );
        assert!(
            gpus.contains("| H100 SXM 80 GB | modeled | 80 GiB | 3,350 GB/s |"),
            "{gpus}"
        );
    }

    #[test]
    fn the_modeled_cold_prefill_follows_the_scaled_step_loop() {
        // E0's host time, by its own derivation: 22.7 ms + 3.11 µs a token.
        let host = StepCost::frozen_box().e0_host(&TierCost::frozen_box());
        assert!((host.ms - 22.67).abs() < 0.01 && (host.us_per_token - 3.113).abs() < 0.001);
        for gpu in [L4, H100_SXM] {
            let tier = gpu.tier_cost();
            let steps = StepCost {
                host,
                ..gpu.step_cost()
            };
            for tokens in [4_096, 8_192, 16_384, 31_744, 65_536, 131_072, 249_152] {
                let cold = PromptSources {
                    computed: tokens,
                    ..PromptSources::default()
                };
                let alone =
                    steps.alone_ttft_ms(&tier, cold, StepBudget::FROZEN_BOX.max_batched_tokens);
                let fit = tier.recompute_ms(tokens);
                assert!(
                    (fit / alone - 1.0).abs() < 0.08,
                    "{} at {tokens}: fit {fit:.0} ms, step loop {alone:.0} ms",
                    gpu.name
                );
            }
        }
        // The restore on a Gen4 x16 link, against vLLM's on an L20 (34, 65 and 104 ms).
        let l4 = L4.tier_cost();
        for (tokens, measured) in [(4_096, 34.0), (8_192, 65.0), (16_384, 104.0)] {
            let modeled = l4.restore_ms(tokens);
            assert!(
                (modeled / measured - 1.0).abs() < 0.10,
                "{tokens}: {modeled:.1}"
            );
        }
    }

    #[test]
    fn a_corpus_replays_at_the_least_scaling_its_pool_and_the_context_hold() {
        // The 32x corpus's longest request: 31,144 tokens of prompt and output.
        let longest = 31_144;
        let fitted = |g: &Gpu| fitted_token_scale(g, &QWEN, longest, 32);
        // The box holds it as given; the L4 at 16x (twice the tokens); the H100 at 4x, where the
        // model's context binds before its pool does.
        assert_eq!(
            (fitted(&RTX_5060_TI), fitted(&L4), fitted(&H100_SXM)),
            (Some(1), Some(2), Some(8))
        );
        assert!(16 * longest > MODEL_CONTEXT && 16 * longest < H100_SXM.pool_tokens(&QWEN));
        // No scale goes past the original token counts, and the original corpus fits no pool.
        assert_eq!(fitted_token_scale(&H100_SXM, &QWEN, longest, 4), Some(4));
        assert_eq!(fitted_token_scale(&H100_SXM, &QWEN, 996_579, 1), None);
        // The box's 2,274 usable blocks hold 36,384 tokens: a request of 36,385 (36,384 of KV) fits.
        assert_eq!(fitted_token_scale(&RTX_5060_TI, &QWEN, 36_385, 32), Some(1));
        assert_eq!(fitted_token_scale(&RTX_5060_TI, &QWEN, 36_386, 32), None);
        // A card with no room beside the box's footprint has no pool at all.
        let small = the_box_with(|g| g.memory_mib = 8 << 10);
        assert_eq!(small.pool_tokens(&QWEN), 0);
        assert_eq!(fitted_token_scale(&small, &QWEN, 16, 32), None);
    }

    #[test]
    fn a_larger_pool_runs_more_sessions() {
        let all = [5, 8, 12, 16, 24, 32, 40, 64, 96, 128, 192, 256];
        let regime_map = [5, 8, 12, 16, 24, 32, 40];
        assert_eq!(sessions_for(&RTX_5060_TI, &QWEN, 1, &all), regime_map);
        // 2.3 and 13.8 times the box's pool at its scale: up to 92 and 552 sessions.
        assert_eq!(sessions_for(&L4, &QWEN, 1, &all).last(), Some(&64));
        assert_eq!(sessions_for(&H100_SXM, &QWEN, 1, &all).len(), all.len());
        // At 4x, the H100's pool holds 1.7 times the box's at 32x: up to 69 sessions.
        assert_eq!(sessions_for(&H100_SXM, &QWEN, 8, &all).last(), Some(&64));
        let c = cells(
            &[(RTX_5060_TI, 1), (L4, 2)],
            &all,
            &[11, 12],
            Pauses::Trace,
            4.0,
        );
        assert_eq!(c.len(), 2 * (7 + 7));
        assert_eq!((c[0].regime.sessions, c[0].seed, c[1].seed), (5, 11, 12));
        assert_eq!(c[14].regime.capacity, L4.capacity(&QWEN, 4.0));
    }

    #[test]
    fn gpus_parse_by_name() {
        for g in GPUS {
            assert_eq!(g.name.parse::<Gpu>(), Ok(g));
        }
        let err = "a100".parse::<Gpu>().unwrap_err();
        assert!(err.contains("rtx-5060-ti, l4, h100-sxm"), "{err}");
    }

    #[test]
    fn counts_get_thousands_separators() {
        let t = [0, 999, 1_000, 36_400, 1_234_567].map(thousands);
        assert_eq!(t, ["0", "999", "1,000", "36,400", "1,234,567"]);
        let t = [189.75, 47.4375, 121.0, 1_979.0, 999.96].map(tenths);
        assert_eq!(t, ["189.8", "47.4", "121.0", "1,979.0", "1,000.0"]);
        let t = [16 << 10, 23_040, 80 << 10, 1_200 << 10, 1_536].map(gib);
        assert_eq!(t, ["16", "22.5", "80", "1,200", "1.5"]);
    }

    /// A placement whose arms have resume TTFT p50 100 ms and p90 1,000 ms (`oracle` for
    /// `oracle-prefetch`), KV pressure `pressure` and a realtime tier blocked `blocked` of the time.
    fn placement(gpu: Gpu, seed: u64, oracle: f64, pressure: f64, blocked: f64) -> Placement {
        let arms = Arm::E1
            .iter()
            .map(|&arm| Summary {
                arm: arm.name().into(),
                resume_ttft_ms: Percentiles {
                    p50: 100.0,
                    p90: if arm == Arm::OraclePrefetch {
                        oracle
                    } else {
                        1_000.0
                    },
                    ..Percentiles::default()
                },
                regime: Regime {
                    kv_pressure_mean: pressure,
                    kv_pressure_peak: 2.0 * pressure,
                    median_prompt_tokens: 4_000.0,
                    ..Regime::default()
                },
                replay: ReplayStats {
                    realtime_short_share: Some(1.0),
                    realtime_blocked_share: Some(blocked),
                    ..ReplayStats::default()
                },
                ..Summary::default()
            })
            .collect();
        Placement {
            gpu,
            token_scale: 8,
            result: regime::CellResult {
                cell: regime::Cell {
                    pauses: Pauses::Trace,
                    sessions: 16,
                    cpu_ratio: 4.0,
                    capacity: gpu.capacity(&QWEN, 4.0),
                },
                seed,
                arms,
            },
        }
    }

    #[test]
    fn the_tables_have_a_row_per_cell_and_arm_and_a_mean_per_cell() {
        let results = [
            placement(H100_SXM, 11, 700.0, 1.0, 0.1),
            placement(H100_SXM, 12, 900.0, 2.0, 0.3),
        ];
        let table = csv(&results);
        let columns = table.lines().next().unwrap().split(',').count();
        assert_eq!(table.lines().count(), 1 + 2 * Arm::E1.len());
        let row = table
            .lines()
            .find(|l| l.starts_with("h100-sxm,modeled,8,502432,12,trace,16,4,oracle-prefetch,"))
            .expect("the second seed's oracle row");
        assert_eq!(row.split(',').count(), columns);
        let chunk = chunk_ms(&H100_SXM.step_cost(), 4_000);
        assert!(
            row.ends_with(&format!(",4000,1.0000,0.3000,{chunk:.1}")),
            "{row}"
        );
        // One row for both seeds, at the GPU's 4x (2-token hash blocks at token scale 8).
        let map = markdown(&results, 2, 2_048);
        assert!(
            map.contains("seeds 11, 12") && map.contains("less than 2,048 tokens"),
            "{map}"
        );
        let rows: Vec<_> = map.lines().filter(|l| l.starts_with("| H100")).collect();
        assert_eq!(rows.len(), 1, "{map}");
        assert!(
            rows[0].starts_with("| H100 SXM 80 GB (modeled), 4× | 16 | 1.5 / 3.0 |"),
            "{map}"
        );
        // Short of KV 100% of the time and blocked 20% on average; the oracle gains 0 / 20%.
        assert!(
            rows[0].contains("| 100 / 20% |") && rows[0].contains("| +0 / +20% |"),
            "{map}"
        );
        assert!(rows[0].contains("| 100 / 1,000 |"), "{map}");
        let gpus = profiles(&GPUS);
        assert!(
            gpus.contains("| RTX 5060 Ti 16 GB | measured | 16 GiB | 448 GB/s |"),
            "{gpus}"
        );
        assert!(
            gpus.contains("PCIe 4.0 x16, 31.5 GB/s | 83,728 tokens | 17.26 ms"),
            "{gpus}"
        );
        assert!(gpus.contains("| 1,979.0 / 989.5 TFLOPS |"), "{gpus}");
        assert!(gpus.contains("| 502,432 tokens | 1.55 ms |"), "{gpus}");
    }
}
