//! The replay structure in the JSON shape of `scripts/aiperf_reference.py`, which runs AIPerf 0.13.0's
//! own loader and `TrajectorySource` over the same traces; comparing the two checks the port.

use serde_json::{Value, json};

use super::plan::ReplayPlan;
use super::snapshot;

/// `{"conversations": [...], "lanes": [...], "recycle": [...]}` for `plans` (in sampler order) and
/// `concurrency` lanes, as AIPerf builds them with `seed` and the start ratios.
pub fn dump(
    plans: &[&ReplayPlan],
    concurrency: usize,
    seed: u64,
    min_ratio: f64,
    max_ratio: f64,
) -> Value {
    let mut conversations = Vec::new();
    for plan in plans {
        let name = |c: u32| plan.convs[c as usize].id.clone();
        let root = plan.root();
        let branch_name = |b: u32| root.branches[b as usize].id.clone();
        for conv in &plan.convs {
            let turns: Vec<Value> = conv
                .turns
                .iter()
                .map(|t| {
                    json!({
                        "outer": t.outer,
                        "inner": t.inner,
                        "ts_ms": t.ts_ms,
                        "delay_ms": t.delay_ms,
                        "api_ms": t.api_ms,
                        "max_tokens": t.max_tokens,
                        "branches": t.spawns.iter().map(|&b| branch_name(b)).collect::<Vec<_>>(),
                        "joins": t.joins.iter().map(|&b| branch_name(b)).collect::<Vec<_>>(),
                        "preds": t.preds.iter().map(|&(c, k)| json!([name(c), k])).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let branches: Vec<Value> = conv
                .branches
                .iter()
                .map(|b| {
                    json!({
                        "id": b.id,
                        "children": b.children.iter().map(|&c| name(c)).collect::<Vec<_>>(),
                        "background": b.background,
                        "start_ms": b.start_ms,
                    })
                })
                .collect();
            conversations.push(json!({
                "id": conv.id,
                "is_root": conv.kind == super::ConvKind::Root,
                "scope": plan.scopes[conv.scope as usize],
                "turns": turns,
                "branches": branches,
            }));
        }
    }
    let mut lanes = Vec::new();
    let mut next = 0;
    let spawnable: Vec<&&ReplayPlan> = plans
        .iter()
        .filter(|p| !p.root().turns.is_empty())
        .collect();
    while lanes.len() < concurrency && !spawnable.is_empty() {
        let plan = spawnable[next % spawnable.len()];
        next += 1;
        let lane = lanes.len();
        let t_star = snapshot::t_star(plan, seed, lane, min_ratio, max_ratio);
        let snap = t_star.and_then(|t| snapshot::snapshot(plan, t));
        let name = |c: u32| plan.convs[c as usize].id.clone();
        lanes.push(json!({
            "trace": plan.trace_id,
            "t_star_ms": t_star,
            "states": snap.iter().flat_map(|s| s.states.iter()).map(|s| json!({
                "conv": name(s.conv),
                "next": s.next_turn,
                "offset_ms": s.offset_ms,
                "depth": s.depth,
                "waiting": s.waiting,
                "join": s.join_turn,
                "branch": s.branch.map(|b| plan.root().branches[b as usize].id.clone()),
            })).collect::<Vec<_>>(),
            "boundaries": snap.iter().flat_map(|s| s.boundaries.iter()).map(|&(c, n)| json!([name(c), n])).collect::<Vec<_>>(),
        }));
    }
    let recycle: Vec<Value> = (0..32)
        .map(|i| json!(spawnable[(next + i) % spawnable.len()].trace_id))
        .collect();
    json!({ "conversations": conversations, "lanes": lanes, "recycle": recycle })
}
