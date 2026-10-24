//! Pauses drawn from TraceLab: how long a coding agent waits for its tool calls.
//!
//! `data/tracelab-pauses.tsv` (derived by `scripts/tracelab_pauses.py` from the TraceLab coding-agent
//! trace v0.0.2, CC BY 4.0) holds, per tool, the share of tool rounds and the 0th to 100th
//! percentiles of the wait from a round's last tool call to its last tool result. A pause draws a
//! tool by its share and a wait from that tool's percentiles, interpolating linearly between them.
//! Draws are independent: a session's pauses keep TraceLab's mix of tools but not its sequence.

use std::sync::OnceLock;

const TABLE: &str = include_str!("../data/tracelab-pauses.tsv");

/// The TraceLab table.
#[derive(Debug)]
pub struct ToolPauses {
    /// Each tool's cumulative share of rounds, its name and its percentiles in ms.
    tools: Vec<(f64, String, Vec<f64>)>,
}

impl ToolPauses {
    /// The table shipped with lh-sim.
    pub fn tracelab() -> &'static Self {
        static TRACELAB: OnceLock<ToolPauses> = OnceLock::new();
        TRACELAB.get_or_init(|| Self::parse(TABLE).expect("the shipped TraceLab table parses"))
    }

    fn parse(text: &str) -> Result<Self, String> {
        let mut rows = Vec::new();
        for line in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
            let mut cells = line.split('\t');
            let name = cells.next().ok_or("an empty row")?.to_owned();
            let rounds: f64 = cells
                .next()
                .and_then(|c| c.parse().ok())
                .ok_or_else(|| format!("{name}: no round count"))?;
            let q: Vec<f64> = cells
                .map(|c| c.parse().map_err(|e| format!("{name}: {e}")))
                .collect::<Result<_, _>>()?;
            if q.len() != 101 {
                return Err(format!("{name}: {} percentiles, not 101", q.len()));
            }
            rows.push((rounds, name, q));
        }
        let total: f64 = rows.iter().map(|r| r.0).sum();
        let mut cumulative = 0.0;
        let tools = rows
            .into_iter()
            .map(|(rounds, name, q)| {
                cumulative += rounds / total;
                (cumulative, name, q)
            })
            .collect();
        Ok(Self { tools })
    }

    /// The tool and the pause in ms that `draw` picks: a tool by its share, a pause from its
    /// percentiles.
    pub fn draw(&self, draw: u64) -> (u16, f64) {
        let mut rng = SplitMix64(draw);
        let pick = rng.unit();
        let tool = self
            .tools
            .iter()
            .position(|t| pick < t.0)
            .unwrap_or(self.tools.len() - 1);
        let q = &self.tools[tool].2;
        let rank = rng.unit() * 100.0;
        let lo = rank.floor() as usize;
        let hi = (lo + 1).min(100);
        (tool as u16, q[lo] + (rank - lo as f64) * (q[hi] - q[lo]))
    }

    /// The name of tool `tool`.
    #[cfg(test)]
    pub fn name(&self, tool: u16) -> &str {
        &self.tools[tool as usize].1
    }
}

/// SplitMix64 (Steele, Lea and Flood): a small, well-mixed generator seeded by the draw's key.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_table_parses_and_draws_its_mix() {
        let t = ToolPauses::tracelab();
        assert_eq!(t.name(0), "Bash");
        assert_eq!(t.tools.last().map(|r| r.1.as_str()), Some("other"));
        assert!((t.tools.last().unwrap().0 - 1.0).abs() < 1e-9);
        let draws: Vec<(u16, f64)> = (0..20_000).map(|k| t.draw(k)).collect();
        assert_eq!(draws, (0..20_000).map(|k| t.draw(k)).collect::<Vec<_>>());
        // Bash is a quarter of TraceLab's tool rounds; the median pause is a few hundred ms.
        let bash = draws.iter().filter(|d| d.0 == 0).count() as f64 / draws.len() as f64;
        assert!((bash - 0.254).abs() < 0.02, "{bash}");
        let mut ms: Vec<f64> = draws.iter().map(|d| d.1).collect();
        ms.sort_by(f64::total_cmp);
        let median = ms[ms.len() / 2];
        assert!((150.0..400.0).contains(&median), "{median}");
    }

    #[test]
    fn a_draw_interpolates_between_percentiles() {
        let mut q: Vec<f64> = (0..=100).map(|p| p as f64 * 10.0).collect();
        q[100] = 2_000.0;
        let t = ToolPauses {
            tools: vec![(1.0, "x".into(), q)],
        };
        for k in 0..1_000 {
            let (tool, ms) = t.draw(k);
            assert_eq!(tool, 0);
            assert!((0.0..2_000.0).contains(&ms));
        }
    }
}
