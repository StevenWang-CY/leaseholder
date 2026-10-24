//! The E1 regime sweep: the information ladder ([`Arm::E1`]) at each KV pressure (session count),
//! CPU tier size and pause source, over seeds, as a CSV and a Markdown regime map.
//!
//! The map gives each cell's resume-TTFT gains over `lru-cpu` as the mean of the per-seed gains. With
//! several seeds, a table per pause source and CPU tier gives their spread: the oracle's median and
//! range, the gain of the seeds' mean latencies, and the best arm of the history level.
//!
//! In each cell, the share of the clairvoyant gap an arm closes is `(lru − arm) / (lru − clairvoyant)`
//! on a resume-TTFT percentile: 0 for `lru-cpu` (no information), 1 for the clairvoyant reference.
//! It is left empty where the reference is less than 5% faster than `lru-cpu`, as a share of so
//! small a gap says little. The reference keeps `lru-cpu`'s queueing and re-prices only each
//! request's own service (see [`crate::run`]), so an arm that shortens queues can close more than all
//! of it.

use std::fmt::Write as _;
use std::path::Path;

use lh_core::TierCapacity;

use crate::aiperf::PlanOptions;
use crate::corpus::Corpus;
use crate::metrics::Summary;
use crate::replay::Pauses;
use crate::run::{self, Arm, RunConfig};

/// One cell of the sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub pauses: Pauses,
    pub sessions: usize,
    /// The CPU tier over the GPU pool.
    pub cpu_ratio: f64,
    pub capacity: TierCapacity,
}

/// A cell, the seed it ran at, and the summaries of [`Arm::E1`] in it, in that order.
#[derive(Clone, Debug)]
pub struct CellResult {
    pub cell: Cell,
    pub seed: u64,
    pub arms: Vec<Summary>,
}

impl CellResult {
    pub fn arm(&self, arm: Arm) -> &Summary {
        self.arms
            .iter()
            .find(|s| s.arm == arm.name())
            .expect("every E1 arm ran")
    }

    /// `arm`'s resume TTFT percentile `p` (50, 90 or 99), in ms.
    pub fn resume(&self, arm: Arm, p: u32) -> f64 {
        let t = &self.arm(arm).resume_ttft_ms;
        match p {
            50 => t.p50,
            90 => t.p90,
            _ => t.p99,
        }
    }

    /// The share of the clairvoyant gap `arm` closes on resume TTFT percentile `p`.
    pub fn gap_closed(&self, arm: Arm, p: u32) -> Option<f64> {
        let lru = self.resume(Arm::LruCpu, p);
        let gap = lru - self.resume(Arm::Clairvoyant, p);
        (gap >= 0.05 * lru).then(|| (lru - self.resume(arm, p)) / gap)
    }

    /// How much lower `arm`'s resume TTFT percentile `p` is than `lru-cpu`'s, as a share of it.
    pub fn gain(&self, arm: Arm, p: u32) -> f64 {
        1.0 - self.resume(arm, p) / self.resume(Arm::LruCpu, p)
    }
}

/// Runs every cell of `cells` at every seed of `seeds` on `jobs` threads, each opening `traces`
/// itself. The results are in the cells' order, each cell's seeds in theirs.
pub fn sweep(
    traces: &Path,
    plan: PlanOptions,
    base: &RunConfig,
    cells: &[Cell],
    seeds: &[u64],
    jobs: usize,
) -> Result<Vec<CellResult>, String> {
    let runs: Vec<(Cell, u64)> = cells
        .iter()
        .flat_map(|&cell| seeds.iter().map(move |&seed| (cell, seed)))
        .collect();
    par_map(traces, plan, &runs, jobs, |corpus, &(cell, seed)| {
        let mut cfg = base.clone();
        cfg.replay.concurrency = cell.sessions;
        cfg.replay.pauses = cell.pauses;
        cfg.replay.seed = seed;
        run_cell(corpus, &cfg, cell)
    })
}

/// Runs [`Arm::E1`] in `cell` with `cfg`, whose concurrency and pauses are the cell's.
pub(crate) fn run_cell(
    corpus: &mut Corpus,
    cfg: &RunConfig,
    cell: Cell,
) -> Result<CellResult, String> {
    let arms = run::run(corpus, cfg, &Arm::E1, &[cell.capacity], |_| {})?
        .into_iter()
        .map(|r| r.summary)
        .collect();
    Ok(CellResult {
        cell,
        seed: cfg.replay.seed,
        arms,
    })
}

/// Maps `f` over `items` on `jobs` threads, each with its own opening of `traces`, keeping their order.
pub(crate) fn par_map<C: Sync, R: Send>(
    traces: &Path,
    plan: PlanOptions,
    items: &[C],
    jobs: usize,
    f: impl Fn(&mut Corpus, &C) -> Result<R, String> + Sync,
) -> Result<Vec<R>, String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs.clamp(1, items.len().max(1)))
            .map(|_| {
                scope.spawn(|| -> Result<Vec<(usize, R)>, String> {
                    let mut corpus = Corpus::open(traces, plan)?;
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else {
                            return Ok(done);
                        };
                        done.push((i, f(&mut corpus, item)?));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("a sweep worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut out: Vec<Option<R>> = items.iter().map(|_| None).collect();
    for r in results {
        for (i, x) in r? {
            out[i] = Some(x);
        }
    }
    Ok(out
        .into_iter()
        .map(|x| x.expect("every cell ran"))
        .collect())
}

pub(crate) fn pauses_name(p: Pauses) -> &'static str {
    match p {
        Pauses::Trace => "trace",
        Pauses::TraceLab { .. } => "tracelab",
    }
}

fn opt(x: Option<f64>) -> String {
    x.map_or(String::new(), |v| format!("{v:.3}"))
}

/// The columns of [`csv`].
pub(crate) const CSV_COLUMNS: &str = "seed,pauses,sessions,cpu_ratio,arm,requests,resume_ttft_p50_ms,\
     resume_ttft_p90_ms,resume_ttft_p99_ms,gap_closed_p50,gap_closed_p90,gpu_share,cpu_share,\
     computed_share,recomputed_share,pcie_restored_bytes,pcie_prefetched_bytes,pcie_stored_bytes,\
     kv_pressure_mean,pi,rho";

/// One row per cell and arm.
pub fn csv(results: &[CellResult]) -> String {
    let mut text = format!("{CSV_COLUMNS}\n");
    for r in results {
        for (s, &arm) in r.arms.iter().zip(&Arm::E1) {
            csv_row(&mut text, r, s, arm);
            text.push('\n');
        }
    }
    text
}

/// Appends the [`CSV_COLUMNS`] of `arm` in `r` (its summary `s`), without a line end.
pub(crate) fn csv_row(text: &mut String, r: &CellResult, s: &Summary, arm: Arm) {
    let c = &r.cell;
    write!(
        text,
        "{},{},{},{},{},{},{:.1},{:.1},{:.1},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.3},{:.3},{:.4}",
        r.seed,
        pauses_name(c.pauses),
        c.sessions,
        c.cpu_ratio,
        s.arm,
        s.requests,
        s.resume_ttft_ms.p50,
        s.resume_ttft_ms.p90,
        s.resume_ttft_ms.p99,
        opt(r.gap_closed(arm, 50)),
        opt(r.gap_closed(arm, 90)),
        s.shares.gpu,
        s.shares.cpu,
        s.shares.computed,
        s.shares.recomputed,
        s.pcie_restored_bytes,
        s.pcie_prefetched_bytes,
        s.pcie_stored_bytes,
        s.regime.kv_pressure_mean,
        s.regime.pi,
        s.regime.rho,
    )
    .expect("writing to a String");
}

/// The regime map's arms, in its columns' order.
const MAP_ARMS: [Arm; 5] = [
    Arm::OraclePrefetch,
    Arm::HistoryPrefetch,
    Arm::Tokencake,
    Arm::Ttl30,
    Arm::Clairvoyant,
];

/// The arms at the history information level: they act on each session's own past pauses.
const HISTORY_LEVEL: [Arm; 4] = [
    Arm::HistoryPrefetch,
    Arm::Tokencake,
    Arm::Mori,
    Arm::InferCept,
];

/// `x` with its sign, to `decimals` places, and a minus sign rather than a hyphen.
pub(crate) fn signed(x: f64, decimals: usize) -> String {
    format!("{x:+.decimals$}").replace('-', "−")
}

/// `n` with thousands separators.
pub(crate) fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The seeds of a group of results: a run of consecutive seeds as its first and last.
fn seed_label(group: &[&CellResult]) -> String {
    let seeds: Vec<u64> = group.iter().map(|r| r.seed).collect();
    match &seeds[..] {
        [one] => format!("seed {one}"),
        [first, .., last] if seeds.windows(2).all(|w| w[1] == w[0] + 1) => {
            format!("seeds {first}–{last}")
        }
        all => format!(
            "seeds {}",
            all.iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `results` by cell, in the order the cells first appear, each with the results of its seeds.
fn by_cell(results: &[CellResult]) -> Vec<Vec<&CellResult>> {
    let mut groups: Vec<Vec<&CellResult>> = Vec::new();
    for r in results {
        match groups.iter_mut().find(|g| g[0].cell == r.cell) {
            Some(g) => g.push(r),
            None => groups.push(vec![r]),
        }
    }
    groups
}

fn mean(group: &[&CellResult], f: impl Fn(&CellResult) -> f64) -> f64 {
    group.iter().map(|r| f(r)).sum::<f64>() / group.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// The regime map, one table per CPU tier size: each cell's coordinates (from `lru-cpu`) and the
/// mean over its seeds of each arm's resume-TTFT gain over `lru-cpu`. With several seeds, one table
/// per pause source and CPU tier gives their spread.
pub fn markdown(results: &[CellResult]) -> String {
    let groups = by_cell(results);
    let mut ratios: Vec<f64> = Vec::new();
    for g in &groups {
        if !ratios.contains(&g[0].cell.cpu_ratio) {
            ratios.push(g[0].cell.cpu_ratio);
        }
    }
    let mut text = String::new();
    for &ratio in &ratios {
        let rows: Vec<_> = groups
            .iter()
            .filter(|g| g[0].cell.cpu_ratio == ratio)
            .collect();
        let rho: Vec<f64> = rows
            .iter()
            .map(|g| mean(g, |r| r.arm(Arm::LruCpu).regime.rho))
            .collect();
        let (lo, hi) = rho
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        let rho = if format!("{lo:.2}") == format!("{hi:.2}") {
            format!("{lo:.2}")
        } else {
            format!("{lo:.2}–{hi:.2}")
        };
        writeln!(
            text,
            "**The CPU tier {ratio}× the pool; {}.** Resume TTFT gain over `lru-cpu` (1 − arm / \
             `lru-cpu`) at p50 / p90, the mean over the seeds. ρ is {rho} in every cell.\n\n\
             | Pauses | Sessions | Pressure | π | {} |\n\
             |---|---|---|---|{}",
            seed_label(rows[0]),
            MAP_ARMS.map(|a| format!("`{}`", a.name())).join(" | "),
            "---|".repeat(MAP_ARMS.len()),
        )
        .expect("writing to a String");
        for g in &rows {
            let gains = MAP_ARMS.map(|a| {
                format!(
                    "{} / {}%",
                    signed(100.0 * mean(g, |r| r.gain(a, 50)), 0),
                    signed(100.0 * mean(g, |r| r.gain(a, 90)), 0)
                )
            });
            writeln!(
                text,
                "| {} | {} | {:.1} | {:.1} | {} |",
                pauses_name(g[0].cell.pauses),
                g[0].cell.sessions,
                mean(g, |r| r.arm(Arm::LruCpu).regime.kv_pressure_mean),
                mean(g, |r| r.arm(Arm::LruCpu).regime.pi),
                gains.join(" | "),
            )
            .expect("writing to a String");
        }
        text.push('\n');
    }
    if groups.iter().any(|g| g.len() > 1) {
        for &ratio in &ratios {
            for pauses in ["trace", "tracelab"] {
                let rows: Vec<_> = groups
                    .iter()
                    .filter(|g| {
                        g[0].cell.cpu_ratio == ratio && pauses_name(g[0].cell.pauses) == pauses
                    })
                    .collect();
                if !rows.is_empty() {
                    spread(&mut text, pauses, ratio, &rows);
                }
            }
        }
    }
    text
}

/// The spread over the seeds of the cells `rows`, which share a pause source and a CPU tier size.
fn spread(text: &mut String, pauses: &str, ratio: f64, rows: &[&Vec<&CellResult>]) {
    writeln!(
        text,
        "**Pauses: {pauses}; the CPU tier {ratio}× the pool; {}.** Resume TTFT gain over `lru-cpu`: \
         the mean of the per-seed gains, their median (in parentheses) and range [in brackets]; the \
         oracle's gain on the seeds' mean latencies; and the history-level arm with the best mean \
         gain at p90.\n\n\
         | Sessions | Pressure | π | `lru-cpu` resume p50 / p90 | `oracle-prefetch` p50 | p90 \
         | `oracle-prefetch` on mean latencies, p50 / p90 | `clairvoyant` p50 / p90 \
         | Best history arm, p90 |\n\
         |---|---|---|---|---|---|---|---|---|",
        seed_label(rows[0]),
    )
    .expect("writing to a String");
    for g in rows {
        let lru = |p| mean(g, |r| r.resume(Arm::LruCpu, p));
        let oracle = |p: u32| {
            let gains: Vec<f64> = g
                .iter()
                .map(|r| 100.0 * r.gain(Arm::OraclePrefetch, p))
                .collect();
            let (lo, hi) = gains
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                    (lo.min(x), hi.max(x))
                });
            format!(
                "{}% ({}) [{}, {}]",
                signed(gains.iter().sum::<f64>() / gains.len() as f64, 1),
                signed(median(gains), 1),
                signed(lo, 0),
                signed(hi, 0)
            )
        };
        let of_means = |p| 100.0 * (1.0 - mean(g, |r| r.resume(Arm::OraclePrefetch, p)) / lru(p));
        let gain = |a: Arm, p| 100.0 * mean(g, |r| r.gain(a, p));
        // The first arm with the highest mean gain, as the history level's best.
        let best = HISTORY_LEVEL
            .into_iter()
            .fold(None, |best: Option<(Arm, f64)>, a| {
                let x = gain(a, 90);
                match best {
                    Some((_, b)) if b >= x => best,
                    _ => Some((a, x)),
                }
            })
            .expect("the history level has arms");
        writeln!(
            text,
            "| {} | {:.1} | {:.1} | {} / {} ms | {} | {} | {} / {}% | {} / {}% | `{}` {}% |",
            g[0].cell.sessions,
            mean(g, |r| r.arm(Arm::LruCpu).regime.kv_pressure_mean),
            mean(g, |r| r.arm(Arm::LruCpu).regime.pi),
            thousands(lru(50).round() as u64),
            thousands(lru(90).round() as u64),
            oracle(50),
            oracle(90),
            signed(of_means(50), 1),
            signed(of_means(90), 1),
            signed(gain(Arm::Clairvoyant, 50), 1),
            signed(gain(Arm::Clairvoyant, 90), 1),
            best.0.name(),
            signed(best.1, 1),
        )
        .expect("writing to a String");
    }
    text.push('\n');
}

#[cfg(test)]
mod tests {
    use lh_core::KvGeometry;

    use super::*;
    use crate::metrics::Percentiles;

    /// A cell at `seed` whose arms have resume TTFT p50 100 ms and p90 `lru` ms, except those in
    /// `p90`.
    fn cell(seed: u64, lru: f64, p90: &[(Arm, f64)]) -> CellResult {
        let arms = Arm::E1
            .iter()
            .map(|&arm| {
                let p90 = p90.iter().find(|x| x.0 == arm).map_or(lru, |x| x.1);
                Summary {
                    arm: arm.name().into(),
                    resume_ttft_ms: Percentiles {
                        p50: 100.0,
                        p90,
                        ..Percentiles::default()
                    },
                    ..Summary::default()
                }
            })
            .collect();
        CellResult {
            cell: Cell {
                pauses: Pauses::Trace,
                sessions: 16,
                cpu_ratio: 4.0,
                capacity: TierCapacity::frozen_box(&KvGeometry::QWEN3_4B_BF16),
            },
            seed,
            arms,
        }
    }

    #[test]
    fn the_gap_closed_runs_from_lru_to_the_reference() {
        let r = cell(
            11,
            1_000.0,
            &[
                (Arm::Clairvoyant, 600.0),
                (Arm::OraclePrefetch, 700.0),
                (Arm::Tokencake, 1_100.0),
            ],
        );
        assert_eq!(r.gap_closed(Arm::LruCpu, 90), Some(0.0));
        assert_eq!(r.gap_closed(Arm::Clairvoyant, 90), Some(1.0));
        assert_eq!(r.gap_closed(Arm::OraclePrefetch, 90), Some(0.75));
        assert_eq!(r.gap_closed(Arm::Tokencake, 90), Some(-0.25));
        assert!((r.gain(Arm::OraclePrefetch, 90) - 0.3).abs() < 1e-12);
        // No gap at p50, and one under 5% of lru-cpu's is too small to share out.
        assert_eq!(r.gap_closed(Arm::OraclePrefetch, 50), None);
        let small = cell(11, 1_000.0, &[(Arm::Clairvoyant, 960.0)]);
        assert_eq!(small.gap_closed(Arm::OraclePrefetch, 90), None);
        let table = csv(std::slice::from_ref(&r));
        assert_eq!(table.lines().count(), 1 + Arm::E1.len());
        assert!(
            table
                .lines()
                .any(|l| l.starts_with("11,trace,16,4,oracle-prefetch,"))
        );
        // One seed: the map alone, its gains that seed's.
        let map = markdown(&[r]);
        assert!(map.contains("seed 11.") && !map.contains("Spread"), "{map}");
        assert!(
            map.contains("| trace | 16 | 0.0 | 0.0 | +0 / +30% | +0 / +0% | +0 / −10% |"),
            "{map}"
        );
    }

    #[test]
    fn the_map_takes_the_mean_over_seeds_and_a_table_their_spread() {
        // The oracle gains 30% and 10% at p90; the mean of the latencies gains 1 - 1,250 / 1,500.
        let results = [
            cell(
                11,
                1_000.0,
                &[
                    (Arm::OraclePrefetch, 700.0),
                    (Arm::Tokencake, 900.0),
                    (Arm::Ttl30, 1_100.0),
                ],
            ),
            cell(
                12,
                2_000.0,
                &[
                    (Arm::OraclePrefetch, 1_800.0),
                    (Arm::Tokencake, 1_800.0),
                    (Arm::Ttl30, 2_200.0),
                ],
            ),
        ];
        let map = markdown(&results);
        assert!(map.contains("seeds 11–12."), "{map}");
        let rows: Vec<_> = map.lines().filter(|l| l.starts_with("| trace")).collect();
        assert_eq!(
            rows,
            [
                "| trace | 16 | 0.0 | 0.0 | +0 / +20% | +0 / +0% | +0 / +10% | +0 / −10% | +0 / +0% |"
            ]
        );
        let spread: Vec<_> = map.lines().filter(|l| l.starts_with("| 16 |")).collect();
        assert_eq!(
            spread,
            ["| 16 | 0.0 | 0.0 | 100 / 1,500 ms | +0.0% (+0.0) [+0, +0] \
              | +20.0% (+20.0) [+10, +30] | +0.0 / +16.7% | +0.0 / +0.0% | `tokencake` +10.0% |"]
        );
        // Negative numbers carry a minus sign, not a hyphen.
        assert!([" -", "(-", "[-"].iter().all(|h| !map.contains(h)), "{map}");
        let gapped = [cell(11, 1_000.0, &[]), cell(13, 1_000.0, &[])];
        assert!(markdown(&gapped).contains("seeds 11, 13."));
    }

    #[test]
    fn a_median_of_an_even_count_is_the_mean_of_the_middle_two() {
        assert_eq!(median(vec![4.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(signed(-0.3, 0), "−0");
        assert_eq!(signed(2.04, 1), "+2.0");
    }
}
