//! `lh-sim`: replays agent traces through a model of vLLM's two-tier KV cache.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use lh_core::{
    FROZEN_BOX_CPU_BYTES, FROZEN_BOX_POOL_TOKENS, KvGeometry, StepCost, TierCapacity, TierCost,
};
use lh_sim::aiperf::{ConvKind, PlanOptions, reference};
use lh_sim::corpus::Corpus;
use lh_sim::engine::Engine;
use lh_sim::replay::{Pauses, ReplayConfig, Service};
use lh_sim::replay_log;
use lh_sim::run::{self, Arm, ArmResult, RunConfig};
use lh_sim::scheduler::{Scheduler, StepBudget};
use lh_sim::{metrics, mrc, regime};
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "lh-sim",
    version,
    about = "Replays agent traces through a model of vLLM's two-tier KV cache"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Runs cache-policy arms over an AIPerf agentic replay and reports where prompt tokens came from.
    Run(RunArgs),
    /// Miss-ratio curves of the GPU tier alone (LRU from Mattson stack distances, MIN at chosen
    /// capacities) on a reference run, at the corpus's scale and at its original token counts.
    Mrc(MrcArgs),
    /// Replays a measured AIPerf run (its profile_export.jsonl) through the engine model alone.
    ReplayLog(LogArgs),
    /// Runs the E1 information ladder over a grid of session counts, CPU tier sizes and pause
    /// sources: a CSV of every arm in every cell, and a Markdown regime map.
    Regime(RegimeArgs),
    /// Dumps AIPerf's replay structure and lane plan as the simulator builds it, in the format of
    /// scripts/aiperf_reference.py, to check the port against AIPerf.
    #[command(hide = true)]
    AiperfPlan(PlanArgs),
}

#[derive(Args)]
struct ReplayArgs {
    /// Weka traces: a directory of <trace id>.json (as `lh-bench scale-traces` writes them) or a JSONL
    /// file.
    traces: PathBuf,
    /// Live session trees (AIPerf's lanes).
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// Profiling duration in seconds.
    #[arg(long, default_value_t = 600.0)]
    duration: f64,
    /// Grace period after the duration, in seconds.
    #[arg(long, default_value_t = 60.0)]
    grace: f64,
    /// Cap on every think-time delay in seconds (`interTurnDelayCapSeconds`); 0 for none.
    #[arg(long, default_value_t = 120.0)]
    think_cap: f64,
    /// AIPerf's `randomSeed`.
    #[arg(long, default_value_t = 11)]
    seed: u64,
    /// `trajectoryStartMinRatio`.
    #[arg(long, default_value_t = 0.0)]
    start_min: f64,
    /// `trajectoryStartMaxRatio`.
    #[arg(long, default_value_t = 1.0)]
    start_max: f64,
    /// Align each lane's first requests on its snapshot instant instead of sending the earliest at
    /// once (`burstPhaseStarts: false`).
    #[arg(long)]
    spread_starts: bool,
    /// `systemIdleGapCapSeconds`; 0 for none.
    #[arg(long, default_value_t = 10.0)]
    idle_cap: f64,
    /// Start at most this many session plays, the initial lanes included.
    #[arg(long)]
    max_sessions: Option<u64>,
    /// Let requests to different models share KV, as an AIPerf replay against one served model does.
    #[arg(long)]
    shared_models: bool,
    /// Multiply hash-block and output tokens: 32 replays a 32x-scaled corpus at its original token
    /// counts.
    #[arg(long, default_value_t = 1)]
    token_scale: u32,
    /// The pauses between a session's turns: the traces' delays, or TraceLab's tool waits in their
    /// place (capped by --think-cap).
    #[arg(long, value_enum, default_value_t = PausesArg::Trace)]
    pauses: PausesArg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PausesArg {
    Trace,
    Tracelab,
}

impl ReplayArgs {
    fn pauses(&self, source: PausesArg) -> Pauses {
        match source {
            PausesArg::Trace => Pauses::Trace,
            PausesArg::Tracelab => Pauses::TraceLab {
                cap_s: (self.think_cap > 0.0).then_some(self.think_cap),
            },
        }
    }

    fn plan_options(&self) -> PlanOptions {
        PlanOptions {
            think_cap_s: (self.think_cap > 0.0).then_some(self.think_cap),
            ..PlanOptions::default()
        }
    }

    fn config(&self) -> ReplayConfig {
        ReplayConfig {
            concurrency: self.concurrency,
            duration_s: self.duration,
            grace_s: self.grace,
            start_min_ratio: self.start_min,
            start_max_ratio: self.start_max,
            burst_starts: !self.spread_starts,
            system_idle_cap_s: (self.idle_cap > 0.0).then_some(self.idle_cap),
            seed: self.seed,
            max_sessions: self.max_sessions,
            model_namespaces: !self.shared_models,
            token_scale: self.token_scale,
            pauses: self.pauses(self.pauses),
            ..ReplayConfig::default()
        }
    }

    fn corpus(&self) -> Result<Corpus, String> {
        Corpus::open(&self.traces, self.plan_options())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ServiceArg {
    /// vLLM's step loop with the fitted step-time model.
    Contention,
    /// v0: E0's idle-box cost per request, with no queueing behind other requests.
    Idle,
}

#[derive(Args)]
struct CostArgs {
    /// How the engine serves requests.
    #[arg(long, value_enum, default_value_t = ServiceArg::Contention)]
    service: ServiceArg,
    /// The E0 tier fits: the summary.json that `lh-bench tiers` writes. Default: the frozen box.
    #[arg(long)]
    costs: Option<PathBuf>,
    /// The step-time model, a JSON object with `StepCost`'s fields. Default: the frozen box's fit.
    #[arg(long)]
    step_costs: Option<PathBuf>,
    /// Decode inter-token latency in ms, for the idle service. Default: 15.6 (the W0 replay's p50).
    #[arg(long)]
    itl_ms: Option<f64>,
}

impl CostArgs {
    fn service(&self) -> Result<Service, String> {
        Ok(match self.service {
            ServiceArg::Idle => Service::Idle,
            ServiceArg::Contention => Service::Contention {
                budget: StepBudget::FROZEN_BOX,
                steps: step_cost(self.step_costs.as_deref())?,
            },
        })
    }

    fn cost(&self) -> Result<TierCost, String> {
        let mut cost = match &self.costs {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                TierCost::from_e0_summary(&text).map_err(|e| format!("{}: {e}", path.display()))?
            }
            None => TierCost::frozen_box(),
        };
        if let Some(itl) = self.itl_ms {
            cost = cost.with_inter_token_ms(itl);
        }
        Ok(cost)
    }
}

/// The step-time model in `path`, or the frozen box's.
fn step_cost(path: Option<&Path>) -> Result<StepCost, String> {
    let Some(path) = path else {
        return Ok(StepCost::frozen_box());
    };
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

#[derive(Args)]
struct CapacityArgs {
    /// GPU pool in tokens.
    #[arg(long, default_value_t = FROZEN_BOX_POOL_TOKENS)]
    pool_tokens: u64,
    /// CPU tier sizes as multiples of the pool, e.g. `--cpu-ratio 2,4,8`. Default: the frozen box's
    /// 20 GiB.
    #[arg(long = "cpu-ratio", value_delimiter = ',')]
    cpu_ratios: Vec<f64>,
    /// CPU tier size in GiB, instead of a ratio.
    #[arg(long, conflicts_with = "cpu_ratios")]
    cpu_gib: Option<f64>,
}

impl CapacityArgs {
    fn capacities(&self, geometry: &KvGeometry, scale: u64) -> Vec<TierCapacity> {
        let pool = self.pool_tokens * scale;
        if !self.cpu_ratios.is_empty() {
            return self
                .cpu_ratios
                .iter()
                .map(|&r| TierCapacity::with_cpu_ratio(geometry, pool, r))
                .collect();
        }
        let bytes = self
            .cpu_gib
            .map_or(FROZEN_BOX_CPU_BYTES, |g| (g * (1u64 << 30) as f64) as u64)
            * scale;
        vec![TierCapacity {
            gpu_blocks: pool / geometry.block_tokens,
            cpu_blocks: geometry.blocks_in_bytes(bytes),
        }]
    }
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    replay: ReplayArgs,
    #[command(flatten)]
    capacity: CapacityArgs,
    #[command(flatten)]
    cost: CostArgs,
    /// Arms to run, comma separated: gpu-lru, lru-cpu, s3fifo-cpu, clairvoyant, or the E1 arms
    /// (ttl-5, ttl-30, ttl-120, history-prefetch, oracle-prefetch, tokencake, mori, infercept).
    /// Default: the stock arms and the clairvoyant reference.
    #[arg(long = "arms", value_delimiter = ',')]
    arms: Vec<Arm>,
    /// The arm whose timeline the clairvoyant reference replays.
    #[arg(long, default_value_t = Arm::LruCpu)]
    reference: Arm,
    /// Write the JSON summary to this file.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Write one CSV row per arm and capacity to this file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write one CSV row per profiling request of every arm to this file.
    #[arg(long)]
    requests: Option<PathBuf>,
}

#[derive(Args)]
struct MrcArgs {
    #[command(flatten)]
    replay: ReplayArgs,
    #[command(flatten)]
    capacity: CapacityArgs,
    #[command(flatten)]
    cost: CostArgs,
    /// The arm whose run gives the access sequence.
    #[arg(long, default_value_t = Arm::LruCpu)]
    reference: Arm,
    /// Capacities to run MIN at, as multiples of the GPU pool.
    #[arg(long, value_delimiter = ',', default_value = "0.25,0.5,1,2,4,8")]
    min_at: Vec<f64>,
    /// How much larger the original token counts are. Default: 64 over the corpus's hash block size
    /// (32 for a 32x corpus); 1 skips the unscaled curves.
    #[arg(long)]
    unscale: Option<u32>,
    /// Write every curve point as CSV to this file.
    #[arg(long)]
    csv: Option<PathBuf>,
}

#[derive(Args)]
struct RegimeArgs {
    /// The workload; --concurrency and --pauses give way to --sessions and --pause-sources.
    #[command(flatten)]
    replay: ReplayArgs,
    #[command(flatten)]
    cost: CostArgs,
    /// Session counts (AIPerf's lanes): the KV pressure grows with them.
    #[arg(long, value_delimiter = ',', default_value = "5,8,12,16,24,32,40")]
    sessions: Vec<usize>,
    /// CPU tier sizes as multiples of the GPU pool.
    #[arg(long = "cpu-ratio", value_delimiter = ',', default_value = "2,4,8")]
    cpu_ratios: Vec<f64>,
    /// Pause sources: trace, tracelab.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "trace,tracelab"
    )]
    pause_sources: Vec<PausesArg>,
    /// Cells to run at once. Default: the machine's cores.
    #[arg(long)]
    jobs: Option<usize>,
    /// Write one row per cell and arm to this CSV file.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Write the regime map to this Markdown file.
    #[arg(long)]
    markdown: Option<PathBuf>,
}

#[derive(Args)]
struct LogArgs {
    /// The Weka traces the run replayed.
    traces: PathBuf,
    /// The run's per-request records: AIPerf's profile_export.jsonl.
    log: PathBuf,
    #[command(flatten)]
    capacity: CapacityArgs,
    /// The engine policy: gpu-lru, lru-cpu or s3fifo-cpu.
    #[arg(long, default_value_t = Arm::LruCpu)]
    arm: Arm,
    /// Send each request at its measured start and serve it with the step loop, instead of holding
    /// it in the engine from its measured start to its measured end; reports time to first token.
    #[arg(long)]
    timed: bool,
    /// The step-time model for `--timed`, as for `run`.
    #[arg(long, requires = "timed")]
    step_costs: Option<PathBuf>,
    /// Write one CSV row per request to this file.
    #[arg(long)]
    csv: Option<PathBuf>,
}

// The options of scripts/aiperf_reference.py, under run's names.
#[derive(Args)]
struct PlanArgs {
    /// A directory of <trace id>.json or a JSONL file.
    traces: PathBuf,
    /// Lanes to plan (the script's `--concurrency`).
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// AIPerf's `randomSeed` (`--seed`).
    #[arg(long, default_value_t = 11)]
    seed: u64,
    /// `interTurnDelayCapSeconds` (`--cap`). Unlike `run`'s, 0 caps every delay at 0, as AIPerf does.
    #[arg(long, default_value_t = 120.0)]
    think_cap: f64,
    /// `trajectoryStartMinRatio` (`--min-ratio`).
    #[arg(long, default_value_t = 0.0)]
    start_min: f64,
    /// `trajectoryStartMaxRatio` (`--max-ratio`).
    #[arg(long, default_value_t = 1.0)]
    start_max: f64,
    /// Plan only the first N traces.
    #[arg(long)]
    limit: Option<usize>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Run(args) => run(args),
        Command::Mrc(args) => mrc(args),
        Command::ReplayLog(args) => replay_log(args),
        Command::Regime(args) => regime(args),
        Command::AiperfPlan(args) => aiperf_plan(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lh-sim: {e}");
            ExitCode::FAILURE
        }
    }
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn run(args: RunArgs) -> Result<(), String> {
    let mut cfg = RunConfig::frozen_box(args.replay.config());
    cfg.cost = args.cost.cost()?;
    cfg.service = args.cost.service()?;
    cfg.reference = args.reference;
    let capacities = args
        .capacity
        .capacities(&cfg.geometry, cfg.replay.token_scale as u64);
    let arms = if args.arms.is_empty() {
        Arm::STOCK.to_vec()
    } else {
        args.arms.clone()
    };
    let mut corpus = args.replay.corpus()?;
    println!("{}", header());
    let results = run::run(&mut corpus, &cfg, &arms, &capacities, |r| {
        println!("{}", row(r))
    })?;
    if let Some(path) = &args.json {
        let doc = json!({
            "traces": args.replay.traces,
            "config": cfg,
            "plan": corpus.options(),
            "results": results,
        });
        write(
            path,
            &serde_json::to_string_pretty(&doc).expect("serializable"),
        )?;
    }
    if let Some(path) = &args.csv {
        let summaries: Vec<_> = results.iter().map(|r| r.summary.clone()).collect();
        write(path, &metrics::csv(&summaries))?;
    }
    if let Some(path) = &args.requests {
        write(path, &requests_csv(&mut corpus, &results)?)?;
    }
    Ok(())
}

fn requests_csv(corpus: &mut Corpus, results: &[ArmResult]) -> Result<String, String> {
    let mut text = String::from(
        "arm,gpu_blocks,cpu_blocks,trace,conversation,turn,kind,issued_s,admitted_s,done_s,prompt_tokens,\
         output_tokens,gpu_tokens,cpu_tokens,computed_tokens,recomputed_tokens,ttft_ms,delay_ms\n",
    );
    for result in results {
        let s = &result.summary;
        for r in &result.records {
            let plan = corpus.plan(r.trace as usize)?;
            let kind = match r.kind {
                ConvKind::Root => "main",
                ConvKind::Subagent => "subagent",
                ConvKind::Flat => "flat",
            };
            writeln!(
                text,
                "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{},{},{},{},{},{:.3},{}",
                s.arm,
                s.gpu_blocks,
                s.cpu_blocks,
                plan.trace_id,
                plan.convs[r.conv as usize].id,
                r.turn,
                kind,
                r.issued_ns as f64 / 1e9,
                r.admitted_ns as f64 / 1e9,
                r.done_ns as f64 / 1e9,
                r.prompt_tokens,
                r.output_tokens,
                r.sources.gpu,
                r.sources.cpu,
                r.sources.computed,
                r.recomputed,
                r.ttft_ns as f64 / 1e6,
                r.delay_ms.map_or(String::new(), |d| format!("{d:.3}")),
            )
            .expect("writing to a String");
        }
    }
    Ok(text)
}

fn header() -> String {
    format!(
        "{:<12} {:>6} {:>6} {:>5} {:>10} {:>6} {:>6} {:>6} {:>18} {:>18} {:>7} {:>6} {:>7} {:>7}",
        "arm",
        "gpu",
        "cpu",
        "reqs",
        "prompt",
        "gpu%",
        "cpu%",
        "comp%",
        "ttft p50/p90/p99",
        "resume p50/p90/p99",
        "pcie GB",
        "stalls",
        "lanes",
        "wall s"
    )
}

fn row(r: &ArmResult) -> String {
    let s = &r.summary;
    let pct = |x: f64| format!("{:.1}", 100.0 * x);
    let ms = |p: &metrics::Percentiles| format!("{:.0}/{:.0}/{:.0}", p.p50, p.p90, p.p99);
    let mut out = String::new();
    write!(
        out,
        "{:<12} {:>6} {:>6} {:>5} {:>10} {:>6} {:>6} {:>6} {:>18} {:>18} {:>7.2} {:>6} {:>7} {:>7.1}",
        s.arm,
        s.gpu_blocks,
        s.cpu_blocks,
        s.requests,
        s.prompt_tokens,
        pct(s.shares.gpu),
        pct(s.shares.cpu),
        pct(s.shares.computed),
        ms(&s.ttft_ms),
        ms(&s.resume_ttft_ms),
        (s.pcie_restored_bytes + s.pcie_prefetched_bytes) as f64 / 1e9,
        s.stalls,
        s.replay.lanes,
        r.wall_s
    )
    .expect("writing to a String");
    out
}

/// The LRU curve's capacities: 1/16 to 64 times the pool, four points per doubling.
fn lru_grid() -> Vec<f64> {
    (-16..=24).map(|k| 2f64.powf(k as f64 / 4.0)).collect()
}

fn mrc(args: MrcArgs) -> Result<(), String> {
    if args.reference == Arm::Clairvoyant {
        return Err("the access sequence comes from a stock arm".into());
    }
    let mut cfg = RunConfig::frozen_box(args.replay.config());
    if cfg.replay.token_scale != 1 {
        return Err("mrc scales tokens itself; leave --token-scale at 1".into());
    }
    cfg.cost = args.cost.cost()?;
    cfg.service = args.cost.service()?;
    let capacity = args.capacity.capacities(&cfg.geometry, 1)[0];
    let mut corpus = args.replay.corpus()?;
    let factor = match args.unscale {
        Some(f) => f.max(1),
        None => {
            let bs = corpus.plan(0)?.block_size;
            if 64 % bs != 0 {
                return Err(format!(
                    "hash blocks of {bs} tokens do not divide 64; pass --unscale"
                ));
            }
            64 / bs
        }
    };
    let bt = cfg.replay.block_tokens;
    let caps = |pool: u64, multiples: &[f64]| -> Vec<u64> {
        multiples
            .iter()
            .map(|m| (m * pool as f64).round().max(1.0) as u64)
            .collect()
    };
    let curve = |acc: &[mrc::Access<'_>], pool: u64| {
        mrc::Curve::compute(
            acc,
            bt as u64,
            &caps(pool, &lru_grid()),
            &caps(pool, &args.min_at),
        )
    };
    // (sequence, token scale, usable pool blocks, curve)
    let mut curves = Vec::new();
    let (_, out) = run::run_arm(&mut corpus, &cfg, args.reference, capacity, true)?;
    let timeline = out.timeline.as_ref().expect("the run kept its timeline");
    let pool = capacity.gpu_blocks - 1;
    curves.push(("scaled", 1, pool, curve(&mrc::accesses(timeline), pool)));
    if factor > 1 {
        let pool = capacity.gpu_blocks * factor as u64 - 1;
        let prompts = mrc::rescaled(&mut corpus, timeline, factor, bt)?;
        let acc: Vec<mrc::Access<'_>> = prompts
            .iter()
            .zip(&timeline.requests)
            .map(|((prompt, prompt_tokens, output_tokens), r)| mrc::Access {
                prompt,
                prompt_tokens: *prompt_tokens,
                output_tokens: *output_tokens,
                counted: r.record.is_some(),
            })
            .collect();
        curves.push(("unscaled-same-order", factor, pool, curve(&acc, pool)));
        drop(acc);
        drop(prompts);
        drop(out);
        let mut unscaled = cfg.clone();
        unscaled.replay.token_scale = factor;
        let at = TierCapacity {
            gpu_blocks: capacity.gpu_blocks * factor as u64,
            cpu_blocks: capacity.cpu_blocks * factor as u64,
        };
        let (_, out) = run::run_arm(&mut corpus, &unscaled, args.reference, at, true)?;
        let timeline = out.timeline.as_ref().expect("the run kept its timeline");
        curves.push((
            "unscaled-replay",
            factor,
            pool,
            curve(&mrc::accesses(timeline), pool),
        ));
    }
    let miss = |hits: u64, of: u64| 1.0 - hits as f64 / of.max(1) as f64;
    println!(
        "{:<20} {:>6} {:>9} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "sequence",
        "reqs",
        "prompt Mt",
        "pool x",
        "LRU block",
        "LRU prefix",
        "MIN block",
        "MIN prefix"
    );
    let mut text = String::from(
        "sequence,token_scale,requests,prompt_tokens,capacity_blocks,capacity_tokens,pool_multiple,\
         lru_block_miss,lru_prefix_miss,min_block_miss,min_prefix_miss\n",
    );
    for (name, scale, pool, c) in &curves {
        for p in &c.points {
            let opt = |h: Option<u64>| {
                h.map_or(String::new(), |h| {
                    format!("{:.6}", miss(h, c.prompt_tokens))
                })
            };
            writeln!(
                text,
                "{name},{scale},{},{},{},{},{:.4},{:.6},{:.6},{},{}",
                c.requests,
                c.prompt_tokens,
                p.capacity,
                p.capacity * bt as u64,
                p.capacity as f64 / *pool as f64,
                miss(p.lru_block_hits, c.prompt_tokens),
                miss(p.lru_prefix_hits, c.prompt_tokens),
                opt(p.min_block_hits),
                opt(p.min_prefix_hits)
            )
            .expect("writing to a String");
            if let (Some(mb), Some(mp)) = (p.min_block_hits, p.min_prefix_hits) {
                println!(
                    "{:<20} {:>6} {:>9.2} {:>6.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    name,
                    c.requests,
                    c.prompt_tokens as f64 / 1e6,
                    p.capacity as f64 / *pool as f64,
                    miss(p.lru_block_hits, c.prompt_tokens),
                    miss(p.lru_prefix_hits, c.prompt_tokens),
                    miss(mb, c.prompt_tokens),
                    miss(mp, c.prompt_tokens)
                );
            }
        }
    }
    if let Some(path) = &args.csv {
        write(path, &text)?;
    }
    Ok(())
}

fn regime(args: RegimeArgs) -> Result<(), String> {
    let mut cfg = RunConfig::frozen_box(args.replay.config());
    cfg.cost = args.cost.cost()?;
    cfg.service = args.cost.service()?;
    let mut cells = Vec::new();
    for &source in &args.pause_sources {
        for &sessions in &args.sessions {
            for &cpu_ratio in &args.cpu_ratios {
                let pool = FROZEN_BOX_POOL_TOKENS * cfg.replay.token_scale as u64;
                cells.push(regime::Cell {
                    pauses: args.replay.pauses(source),
                    sessions,
                    cpu_ratio,
                    capacity: TierCapacity::with_cpu_ratio(&cfg.geometry, pool, cpu_ratio),
                });
            }
        }
    }
    let jobs = args.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    let start = std::time::Instant::now();
    let results = regime::sweep(
        &args.replay.traces,
        args.replay.plan_options(),
        &cfg,
        &cells,
        jobs,
    )?;
    let map = regime::markdown(&results);
    print!("{map}");
    println!(
        "{} cells x {} arms in {:.1} s on {jobs} threads",
        cells.len(),
        Arm::E1.len(),
        start.elapsed().as_secs_f64()
    );
    if let Some(path) = &args.csv {
        write(path, &regime::csv(&results))?;
    }
    if let Some(path) = &args.markdown {
        write(path, &map)?;
    }
    Ok(())
}

fn replay_log(args: LogArgs) -> Result<(), String> {
    if args.arm == Arm::Clairvoyant {
        return Err("replay-log runs a stock arm".into());
    }
    let cfg = RunConfig::frozen_box(ReplayConfig::default());
    let capacity = args.capacity.capacities(&cfg.geometry, 1)[0];
    let engine = Engine::new(&cfg.engine_config(args.arm, capacity), None);
    let mut corpus = Corpus::open(&args.traces, PlanOptions::default())?;
    let block_tokens = cfg.replay.block_tokens;
    let out = if args.timed {
        let steps = step_cost(args.step_costs.as_deref())?;
        let scheduler = Scheduler::new(StepBudget::FROZEN_BOX, steps, cfg.cost.clone());
        replay_log::replay_timed(&mut corpus, &args.log, engine, block_tokens, scheduler)?
    } else {
        replay_log::replay(&mut corpus, &args.log, engine, block_tokens)?
    };
    let prompt = out.prompt_tokens.max(1) as f64;
    let pct = |x: u64, of: f64| 100.0 * x as f64 / of;
    println!(
        "arm {} at {} GPU / {} CPU blocks",
        args.arm,
        capacity.gpu_blocks,
        cfg.effective(args.arm, capacity).cpu_blocks
    );
    println!("profiling requests        {}", out.profiling_requests);
    println!(
        "prompt tokens             measured {}  corpus {}",
        out.measured_prompt_tokens, out.prompt_tokens
    );
    println!(
        "cached share              measured {:.2}%  simulated {:.2}% (GPU {:.2}%, CPU {:.2}%)",
        pct(
            out.measured_cached_tokens,
            out.measured_prompt_tokens.max(1) as f64
        ),
        pct(out.sources.gpu + out.sources.cpu, prompt),
        pct(out.sources.gpu, prompt),
        pct(out.sources.cpu, prompt)
    );
    println!(
        "mean |cached share error| {:.2}%",
        100.0 * out.mean_abs_cached_share_error
    );
    println!("stalls                    {}", out.stalls);
    if args.timed {
        let resumed = |f: &dyn Fn(&replay_log::LoggedRequest) -> f64| {
            let values = out.requests.iter().filter(|r| !r.warmup && r.turn > 0);
            metrics::Percentiles::of(values.map(f).collect())
        };
        let measured = resumed(&|r| r.measured_ttft_ms);
        let simulated = resumed(&|r| r.ttft_ms.unwrap_or(f64::NAN));
        println!(
            "resume ttft p50/p90/p99   measured {:.0}/{:.0}/{:.0} ms  simulated {:.0}/{:.0}/{:.0} ms",
            measured.p50, measured.p90, measured.p99, simulated.p50, simulated.p90, simulated.p99
        );
    }
    if let Some(path) = &args.csv {
        let mut text = String::from(
            "conversation,turn,warmup,measured_prompt_tokens,measured_cached_tokens,prompt_tokens,output_tokens,gpu_tokens,cpu_tokens,computed_tokens",
        );
        text.push_str(if args.timed {
            ",measured_ttft_ms,ttft_ms\n"
        } else {
            "\n"
        });
        for r in &out.requests {
            write!(
                text,
                "{},{},{},{},{},{},{},{},{},{}",
                r.conversation,
                r.turn,
                r.warmup,
                r.measured_prompt_tokens,
                r.measured_cached_tokens,
                r.prompt_tokens,
                r.output_tokens,
                r.sources.gpu,
                r.sources.cpu,
                r.sources.computed
            )
            .expect("writing to a String");
            if args.timed {
                let ttft = r.ttft_ms.map_or(String::new(), |t| format!("{t:.3}"));
                write!(text, ",{:.3},{ttft}", r.measured_ttft_ms).expect("writing to a String");
            }
            text.push('\n');
        }
        write(path, &text)?;
    }
    Ok(())
}

fn aiperf_plan(args: PlanArgs) -> Result<(), String> {
    let opts = PlanOptions {
        think_cap_s: Some(args.think_cap),
        ..PlanOptions::default()
    };
    let mut corpus = Corpus::open(&args.traces, opts)?;
    let n = args.limit.unwrap_or(corpus.len()).min(corpus.len());
    let plans = (0..n)
        .map(|i| corpus.plan(i))
        .collect::<Result<Vec<_>, _>>()?;
    let refs: Vec<&_> = plans.iter().map(|p| p.as_ref()).collect();
    let dump = reference::dump(
        &refs,
        args.concurrency,
        args.seed,
        args.start_min,
        args.start_max,
    );
    println!("{dump}");
    Ok(())
}
