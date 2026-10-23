//! The E1 regime sweep: the information ladder ([`Arm::E1`]) at each KV pressure (session count),
//! CPU tier size and pause source, as a CSV and a Markdown regime map.
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

/// A cell and the summaries of [`Arm::E1`] in it, in that order.
#[derive(Clone, Debug)]
pub struct CellResult {
    pub cell: Cell,
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

/// Runs every cell of `cells` on `jobs` threads, each opening `traces` itself.
pub fn sweep(
    traces: &Path,
    plan: PlanOptions,
    base: &RunConfig,
    cells: &[Cell],
    jobs: usize,
) -> Result<Vec<CellResult>, String> {
    par_map(traces, plan, cells, jobs, |corpus, &cell| {
        let mut cfg = base.clone();
        cfg.replay.concurrency = cell.sessions;
        cfg.replay.pauses = cell.pauses;
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
    Ok(CellResult { cell, arms })
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
pub(crate) const CSV_COLUMNS: &str = "pauses,sessions,cpu_ratio,arm,requests,resume_ttft_p50_ms,\
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
        "{},{},{},{},{},{:.1},{:.1},{:.1},{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.3},{:.3},{:.4}",
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

fn pct(x: Option<f64>) -> String {
    x.map_or("–".into(), |v| format!("{:+.0}%", 100.0 * v))
}

/// The regime map: per pause source, one row per cell with its coordinates (from `lru-cpu`), the
/// reference's and the oracle's resume TTFT against `lru-cpu`, and the share of the clairvoyant gap
/// each information level closes at p90.
pub fn markdown(results: &[CellResult]) -> String {
    let mut text = String::new();
    for pauses in ["trace", "tracelab"] {
        let rows: Vec<_> = results
            .iter()
            .filter(|r| pauses_name(r.cell.pauses) == pauses)
            .collect();
        if rows.is_empty() {
            continue;
        }
        writeln!(
            text,
            "**Pauses: {pauses}.** Resume TTFT p50/p90 in ms; gains against `lru-cpu`; gap closed at p90.\n\n\
             | sessions | pressure | π | ρ | CPU | lru-cpu | clairvoyant | oracle-prefetch | oracle gain p50/p90 \
             | gap closed: history-prefetch | tokencake | oracle-prefetch |\n\
             |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        .expect("writing to a String");
        for r in rows {
            let lru = r.arm(Arm::LruCpu);
            let t = |a: Arm| format!("{:.0}/{:.0}", r.resume(a, 50), r.resume(a, 90));
            writeln!(
                text,
                "| {} | {:.1} | {:.1} | {:.2} | {}× | {} | {} | {} | {}/{} | {} | {} | {} |",
                r.cell.sessions,
                lru.regime.kv_pressure_mean,
                lru.regime.pi,
                lru.regime.rho,
                r.cell.cpu_ratio,
                t(Arm::LruCpu),
                t(Arm::Clairvoyant),
                t(Arm::OraclePrefetch),
                pct(Some(r.gain(Arm::OraclePrefetch, 50))),
                pct(Some(r.gain(Arm::OraclePrefetch, 90))),
                pct(r.gap_closed(Arm::HistoryPrefetch, 90)),
                pct(r.gap_closed(Arm::Tokencake, 90)),
                pct(r.gap_closed(Arm::OraclePrefetch, 90)),
            )
            .expect("writing to a String");
        }
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use lh_core::KvGeometry;

    use super::*;
    use crate::metrics::Percentiles;

    fn cell(p90: &[(Arm, f64)]) -> CellResult {
        let arms = Arm::E1
            .iter()
            .map(|&arm| {
                let p90 = p90.iter().find(|x| x.0 == arm).map_or(1_000.0, |x| x.1);
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
            arms,
        }
    }

    #[test]
    fn the_gap_closed_runs_from_lru_to_the_reference() {
        let r = cell(&[
            (Arm::Clairvoyant, 600.0),
            (Arm::OraclePrefetch, 700.0),
            (Arm::Tokencake, 1_100.0),
        ]);
        assert_eq!(r.gap_closed(Arm::LruCpu, 90), Some(0.0));
        assert_eq!(r.gap_closed(Arm::Clairvoyant, 90), Some(1.0));
        assert_eq!(r.gap_closed(Arm::OraclePrefetch, 90), Some(0.75));
        assert_eq!(r.gap_closed(Arm::Tokencake, 90), Some(-0.25));
        assert!((r.gain(Arm::OraclePrefetch, 90) - 0.3).abs() < 1e-12);
        // No gap at p50, and one under 5% of lru-cpu's is too small to share out.
        assert_eq!(r.gap_closed(Arm::OraclePrefetch, 50), None);
        let small = cell(&[(Arm::Clairvoyant, 960.0)]);
        assert_eq!(small.gap_closed(Arm::OraclePrefetch, 90), None);
        let table = csv(std::slice::from_ref(&r));
        assert_eq!(table.lines().count(), 1 + Arm::E1.len());
        assert!(
            table
                .lines()
                .any(|l| l.starts_with("trace,16,4,oracle-prefetch,"))
        );
        let map = markdown(&[r]);
        assert!(map.contains("| 16 |") && map.contains("+30%"), "{map}");
    }
}
