//! A lane's trajectory (`trajectory_source.py`): the instant `t*` it starts from and the streams live
//! at that instant.
//!
//! `t*` is drawn uniformly between the trace's first and last request start, scaled by the start
//! ratios. Every stream (the root and each child) splits at `t*`: its turns before `t*` are history,
//! primed by one warmup request, and the rest are replayed. A child is live at `t*` if its branch had
//! been spawned by then; a root whose next turn joins live children waits for them.

use super::plan::ReplayPlan;
use super::rng;

/// One stream live at `t*`.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub conv: u32,
    /// The first turn at or after `t*`.
    pub next_turn: u32,
    /// Recorded time from `t*` to that turn, in ms.
    pub offset_ms: f64,
    /// 0 for the root.
    pub depth: u8,
    /// The root is gated on live children before its next turn.
    pub waiting: bool,
    /// The root turn this child's branch joins.
    pub join_turn: Option<u32>,
    /// The branch (index into the root's branches) of a child.
    pub branch: Option<u32>,
}

impl State {
    /// The turn the warmup sends: the stream's last request before `t*`.
    pub fn warmup_turn(&self) -> Option<u32> {
        self.next_turn.checked_sub(1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub t_star_ms: f64,
    /// The root first (when it has turns left), then live children.
    pub states: Vec<State>,
    /// `(conversation, turns completed by t*)` for every conversation with history.
    pub boundaries: Vec<(u32, u32)>,
}

/// `t*` for `lane` (`_build_timestamped_trajectory`), or `None` for a trace without timestamps.
pub fn t_star(
    plan: &ReplayPlan,
    seed: u64,
    lane: usize,
    min_ratio: f64,
    max_ratio: f64,
) -> Option<f64> {
    let (start, end) = plan.time_bounds_ms?;
    let duration = end - start;
    let lo = start + min_ratio * duration;
    let hi = (start + max_ratio * duration).max(lo);
    Some(if hi == lo {
        lo
    } else {
        rng::uniform(rng::lane_seed(seed, &plan.trace_id, lane), lo, hi)
    })
}

fn next_at_or_after(plan: &ReplayPlan, conv: u32, t_star: f64) -> Option<u32> {
    plan.convs[conv as usize]
        .turns
        .iter()
        .position(|t| t.ts_ms >= t_star)
        .map(|i| i as u32)
}

/// The streams live at `t_star` (`_snapshot_for`). `None` when nothing is live, which a timestamped
/// trace never produces: the request at the trace's last timestamp is always live.
pub fn snapshot(plan: &ReplayPlan, t_star: f64) -> Option<Snapshot> {
    let root = plan.root();
    let root_next = next_at_or_after(plan, 0, t_star);
    let mut children = Vec::new();
    let mut join_targets = Vec::new();
    for (b, branch) in root.branches.iter().enumerate() {
        let start = branch.start_ms.or_else(|| {
            branch
                .children
                .iter()
                .filter_map(|&c| plan.convs[c as usize].turns.first().map(|t| t.ts_ms))
                .reduce(f64::min)
        });
        if let Some(start) = start
            && t_star < start
        {
            let spawn = branch.spawn_turn;
            let spawn_ts = root.turns.get(spawn as usize).map(|t| t.ts_ms);
            let not_spawned = spawn_ts.is_none_or(|ts| t_star < ts)
                || root_next.is_some_and(|next| next <= spawn);
            if not_spawned {
                continue;
            }
        }
        let join = if branch.background {
            None
        } else {
            branch.join_turn
        };
        let mut any = false;
        for &child in &branch.children {
            let Some(next) = next_at_or_after(plan, child, t_star) else {
                continue;
            };
            let turns = &plan.convs[child as usize].turns;
            if start.is_none() && turns.first().is_some_and(|t| t_star < t.ts_ms) {
                continue;
            }
            children.push(State {
                conv: child,
                next_turn: next,
                offset_ms: (turns[next as usize].ts_ms - t_star).max(0.0),
                depth: 1,
                waiting: false,
                join_turn: join,
                branch: Some(b as u32),
            });
            any = true;
        }
        if any && let Some(j) = join {
            join_targets.push(j);
        }
    }
    let mut states = Vec::with_capacity(children.len() + 1);
    if let Some(next) = root_next {
        let waiting = join_targets.contains(&next);
        states.push(State {
            conv: 0,
            next_turn: next,
            offset_ms: (root.turns[next as usize].ts_ms - t_star).max(0.0),
            depth: 0,
            waiting,
            join_turn: waiting.then_some(next),
            branch: None,
        });
    }
    states.extend(children);
    if states.iter().all(|s| s.waiting) {
        return None;
    }
    let mut boundaries: Vec<(u32, u32)> = (0..plan.convs.len() as u32)
        .filter_map(|c| {
            let done = next_at_or_after(plan, c, t_star)
                .unwrap_or(plan.convs[c as usize].turns.len() as u32);
            (done > 0).then_some((c, done))
        })
        .collect();
    boundaries.sort_by_key(|&(c, _)| plan.id_rank[c as usize]);
    Some(Snapshot {
        t_star_ms: t_star,
        states,
        boundaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aiperf::PlanOptions;
    use crate::trace::Trace;

    fn plan() -> ReplayPlan {
        let json = r#"{"id": "t", "block_size": 2, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
            {"t": 2, "type": "subagent", "agent_id": "a", "duration_ms": 6000, "requests": [
                {"t": 2, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [9], "api_time": 1.0},
                {"t": 5, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [9, 8], "think_time": 1.0}
            ]},
            {"t": 10, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "think_time": 1.0},
            {"t": 20, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [1, 2, 3], "think_time": 1.0}
        ]}"#;
        ReplayPlan::build(
            Trace::from_json(json.as_bytes()).unwrap(),
            &PlanOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn a_live_child_gates_the_root() {
        let p = plan();
        let s = snapshot(&p, 3_000.0).unwrap();
        assert_eq!(s.states.len(), 2);
        let (root, child) = (&s.states[0], &s.states[1]);
        assert_eq!(
            (root.conv, root.next_turn, root.waiting, root.join_turn),
            (0, 1, true, Some(1))
        );
        assert_eq!(
            (child.conv, child.next_turn, child.offset_ms),
            (1, 1, 2_000.0)
        );
        assert_eq!(child.warmup_turn(), Some(0));
        assert_eq!(s.boundaries, vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn a_child_spawned_by_a_sent_turn_is_live_before_its_first_request() {
        // The spawning root turn 0 was sent before t*, so the subagent starts at its recorded offset
        // and the root waits for it. (AIPerf 0.13.0's _snapshot_for gives the same states.)
        let p = plan();
        let s = snapshot(&p, 500.0).unwrap();
        let got: Vec<_> = s
            .states
            .iter()
            .map(|x| (x.conv, x.next_turn, x.offset_ms, x.waiting))
            .collect();
        assert_eq!(got, vec![(0, 1, 9_500.0, true), (1, 0, 1_500.0, false)]);
        assert_eq!(
            (s.states[1].warmup_turn(), s.boundaries.clone()),
            (None, vec![(0, 1)])
        );
    }

    #[test]
    fn after_the_last_root_turn_only_the_end_remains() {
        let p = plan();
        let s = snapshot(&p, 20_000.0).unwrap();
        assert_eq!(s.states.len(), 1);
        assert_eq!((s.states[0].next_turn, s.states[0].offset_ms), (2, 0.0));
        assert_eq!(s.boundaries, vec![(0, 2), (1, 2)]);
    }

    #[test]
    fn t_star_spans_the_trace() {
        let p = plan();
        assert_eq!(t_star(&p, 11, 0, 0.0, 0.0), Some(0.0));
        assert_eq!(t_star(&p, 11, 0, 1.0, 1.0), Some(20_000.0));
        let t = t_star(&p, 11, 3, 0.0, 1.0).unwrap();
        assert!((0.0..20_000.0).contains(&t));
        assert_eq!(t_star(&p, 11, 3, 0.0, 1.0), Some(t), "deterministic");
        assert_ne!(t_star(&p, 11, 4, 0.0, 1.0), Some(t), "seeded per lane");
    }
}
