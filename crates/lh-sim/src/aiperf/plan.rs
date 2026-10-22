//! The replay structure of one trace, as AIPerf 0.13.0's Weka loader (`weka_trace.py`) builds it
//! with `useThinkTimeOnly` and an inter-turn delay cap, and no idle-gap warp.
//!
//! - **Conversations.** The main agent's requests, minus the flattened agent chains that chain
//!   detection splits off, form the root. Each subagent entry becomes one child per detected inner
//!   chain; each flattened chain becomes a child too. A subagent with no main-agent request before it
//!   is dropped.
//! - **Turn delays.** A turn's delay is its recorded `think_time`, or else the idle gap since the
//!   previous turn of its conversation ended; capped, and floored at zero for root turns.
//! - **Branches.** Children spawn from the last root turn before them and join the first later root
//!   turn that starts after they ended (grouped by spawn and join turn); children that outlive every
//!   root turn run in the background.
//! - **Replay predecessors.** Within a scope (the root with its flattened chains, or one subagent's
//!   chains), a turn waits for the latest request of every other stream that had finished before the
//!   turn started in the recording.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::Req;
use super::chains::{self, Item};
use crate::prefix::{NodeId, PrefixTree, PrefixTreeBuilder};
use crate::trace::{Entry, Trace};

const JOIN_EPS: f64 = 1e-6;

/// Subagents or flattened chains grouped into one branch by (spawning root turn, joining root
/// turn).
type Group = ((usize, Option<usize>), Vec<usize>);

/// How the loader times turns.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PlanOptions {
    /// `interTurnDelayCapSeconds`: no turn delay exceeds this.
    pub think_cap_s: Option<f64>,
    /// `useThinkTimeOnly`: a turn's delay is its recorded think time when there is one.
    pub think_time_only: bool,
    /// `WEKA_SPLIT_FLATTENED_AGENTS`: run chain detection at both layers.
    pub split_chains: bool,
}

impl Default for PlanOptions {
    /// The repository's `configs/agentic_replay.yaml` with the W0 think cap (120 s).
    fn default() -> Self {
        Self {
            think_cap_s: Some(120.0),
            think_time_only: true,
            split_chains: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKind {
    Root,
    Subagent,
    Flat,
}

/// One replayed request.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    /// The prompt: a path in [`ReplayPlan::prefixes`] of `hash_blocks` hash blocks.
    pub prompt_end: NodeId,
    pub hash_blocks: u32,
    /// Index into [`ReplayPlan::models`].
    pub model: u8,
    /// `max_tokens`: the recorded output length, at least 1.
    pub max_tokens: u32,
    /// Recorded start, in ms on the trace clock.
    pub ts_ms: f64,
    /// Wait after the previous turn of the conversation returns; `None` for its first turn.
    pub delay_ms: Option<f64>,
    /// Recorded duration in ms.
    pub api_ms: Option<f64>,
    /// Branches (indices into the root's [`Conversation::branches`]) spawned by this turn.
    pub spawns: Vec<u32>,
    /// Branches whose children must all finish before this turn is sent.
    pub joins: Vec<u32>,
    /// Replay predecessors: `(conversation, turn)` pairs that must have completed first.
    pub preds: Vec<(u32, u32)>,
    /// Position of the request in the trace's top-level list.
    pub outer: u32,
    /// Position inside its subagent entry, for subagent turns.
    pub inner: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    /// AIPerf's branch id, e.g. `<trace>:spawn:<agent id>`.
    pub id: String,
    /// Child conversations.
    pub children: Vec<u32>,
    /// No root turn waits for it.
    pub background: bool,
    /// Recorded spawn time in ms.
    pub start_ms: Option<f64>,
    /// The root turn that spawns it.
    pub spawn_turn: u32,
    /// The root turn that waits for it.
    pub join_turn: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Conversation {
    /// AIPerf's conversation (session) id: the trace id for the root, with a `::sa:`, `::fa:`,
    /// `::aux:` or `::wg:` suffix for children.
    pub id: String,
    pub kind: ConvKind,
    /// Replay scope: barriers apply between conversations of one scope.
    pub scope: u32,
    pub turns: Vec<Turn>,
    /// Branches this conversation spawns (only the root spawns).
    pub branches: Vec<Branch>,
}

/// Everything the replay needs about one trace.
#[derive(Clone, Debug)]
pub struct ReplayPlan {
    pub trace_id: String,
    /// Tokens per hash block.
    pub block_size: u32,
    pub prefixes: PrefixTree,
    pub models: Vec<String>,
    /// `convs[0]` is the root, then subagent children, then flattened chains (AIPerf's order).
    pub convs: Vec<Conversation>,
    /// Earliest and latest turn start of the trace, in ms.
    pub time_bounds_ms: Option<(f64, f64)>,
    /// Rank of each conversation id in string order: AIPerf orders replay keys by
    /// `(conversation id, turn)`.
    pub id_rank: Vec<u32>,
    /// AIPerf's replay scope ids, indexed by [`Conversation::scope`].
    pub scopes: Vec<String>,
}

impl ReplayPlan {
    pub fn root(&self) -> &Conversation {
        &self.convs[0]
    }

    pub fn conv_index(&self, id: &str) -> Option<u32> {
        self.convs.iter().position(|c| c.id == id).map(|i| i as u32)
    }

    pub fn requests(&self) -> usize {
        self.convs.iter().map(|c| c.turns.len()).sum()
    }

    /// Builds the plan of `trace`.
    pub fn build(trace: Trace, opts: &PlanOptions) -> Result<Self, String> {
        Builder::new(trace, opts).build()
    }
}

struct SubagentInfo {
    outer: u32,
    t: f64,
    agent_id: String,
    duration_ms: Option<f64>,
    items: Vec<Item>,
}

struct ChildPlan {
    session_id: String,
    subagent: usize,
    items: Vec<Item>,
}

struct FlatPlan {
    session_id: String,
    chain_index: usize,
    items: Vec<Item>,
}

struct Builder<'a> {
    trace_id: String,
    block_size: u32,
    opts: &'a PlanOptions,
    reqs: Vec<Req>,
    models: Vec<String>,
    normals: Vec<Item>,
    subagents: Vec<SubagentInfo>,
}

impl<'a> Builder<'a> {
    fn new(trace: Trace, opts: &'a PlanOptions) -> Self {
        let mut b = Self {
            trace_id: trace.id,
            block_size: trace.block_size,
            opts,
            reqs: Vec::new(),
            models: Vec::new(),
            normals: Vec::new(),
            subagents: Vec::new(),
        };
        for (outer, entry) in trace.entries.into_iter().enumerate() {
            let outer = outer as u32;
            match entry {
                Entry::Request(r) => {
                    let ri = b.push(r.t, r);
                    b.normals.push((outer, ri));
                }
                Entry::Subagent(sa) => {
                    let mut items = Vec::with_capacity(sa.requests.len());
                    for (inner, r) in sa.requests.into_iter().enumerate() {
                        // Older captures record inner requests relative to the spawn.
                        let t = if r.t + JOIN_EPS < sa.t {
                            sa.t + r.t
                        } else {
                            r.t
                        };
                        items.push((inner as u32, b.push(t, r)));
                    }
                    b.subagents.push(SubagentInfo {
                        outer,
                        t: sa.t,
                        agent_id: sa.agent_id,
                        duration_ms: sa.duration_ms,
                        items,
                    });
                }
            }
        }
        b
    }

    fn push(&mut self, t: f64, r: crate::trace::Request) -> usize {
        let model = match self.models.iter().position(|m| *m == r.model) {
            Some(i) => i,
            None => {
                self.models.push(r.model);
                self.models.len() - 1
            }
        };
        self.reqs.push(Req {
            t,
            api: r.api_time,
            think: r.think_time,
            model: model.min(u8::MAX as usize) as u8,
            input: r.input,
            out: r.out,
            hash: r.hash_ids,
        });
        self.reqs.len() - 1
    }

    fn sorted_by_time(&self, items: &[Item]) -> Vec<Item> {
        let mut v = items.to_vec();
        v.sort_by(|a, b| {
            self.reqs[a.1]
                .t
                .total_cmp(&self.reqs[b.1].t)
                .then(a.0.cmp(&b.0))
        });
        v
    }

    /// `_expand_subagent_to_child_plans`.
    fn expand_subagent(&self, index: usize) -> Vec<ChildPlan> {
        let sa = &self.subagents[index];
        let base = format!("{}::sa:{}", self.trace_id, sa.agent_id);
        if !self.opts.split_chains || sa.items.is_empty() {
            return vec![ChildPlan {
                session_id: base,
                subagent: index,
                items: self.sorted_by_time(&sa.items),
            }];
        }
        let (preamble, rest) = chains::split_off_preamble(&self.reqs, &sa.items);
        let d = chains::detect(&self.reqs, &rest);
        let detected_main = d.chains[d.main].items.clone();
        let mut main_items = detected_main.clone();
        if !preamble.is_empty() {
            main_items.extend(preamble);
            main_items = self.sorted_by_time(&main_items);
        }
        let groups = chains::worker_groups(&self.reqs, &d);
        let peak = detected_main
            .iter()
            .map(|&(_, r)| self.reqs[r].input)
            .max()
            .unwrap_or(0);
        let main_model = detected_main.first().map(|&(_, r)| self.reqs[r].model);
        let mut plans = vec![ChildPlan {
            session_id: base.clone(),
            subagent: index,
            items: main_items,
        }];
        for (n, &ci) in d.workers.iter().enumerate() {
            let items = d.chains[ci].items.clone();
            let suffix = chains::worker_suffix(
                &self.reqs,
                &items,
                n,
                peak,
                main_model,
                groups.get(&ci).copied(),
            );
            plans.push(ChildPlan {
                session_id: format!("{base}:{suffix}"),
                subagent: index,
                items,
            });
        }
        plans
    }

    /// `_detect_and_split_flat_chains`: returns the main chain's normals.
    fn split_flat(&self, flat: &mut Vec<FlatPlan>) -> Vec<Item> {
        let (preamble, rest) = chains::split_off_preamble(&self.reqs, &self.normals);
        let d = chains::detect(&self.reqs, &rest);
        if d.workers.is_empty() {
            return self.normals.clone();
        }
        let main = &d.chains[d.main].items;
        let peak = main
            .iter()
            .map(|&(_, r)| self.reqs[r].input)
            .max()
            .unwrap_or(0);
        let main_model = main.first().map(|&(_, r)| self.reqs[r].model);
        let groups = chains::worker_groups(&self.reqs, &d);
        for (n, &ci) in d.workers.iter().enumerate() {
            let items = d.chains[ci].items.clone();
            let suffix = chains::worker_suffix(
                &self.reqs,
                &items,
                n,
                peak,
                main_model,
                groups.get(&ci).copied(),
            );
            flat.push(FlatPlan {
                session_id: format!("{}::{suffix}", self.trace_id),
                chain_index: n,
                items,
            });
        }
        let mut normals = main.clone();
        if !preamble.is_empty() {
            normals.extend(preamble);
            normals = self.sorted_by_time(&normals);
        }
        normals
    }

    fn clamp_delay(&self, delay_ms: Option<f64>) -> Option<f64> {
        let d = delay_ms?;
        if !d.is_finite() {
            return None;
        }
        Some(match self.opts.think_cap_s {
            Some(cap) if d >= 0.0 && d > cap * 1e3 => cap * 1e3,
            _ => d,
        })
    }

    /// The delay before turn `k` of a conversation whose requests are `items`.
    fn turn_delay(&self, items: &[Item], k: usize) -> Option<f64> {
        if k == 0 {
            return None;
        }
        let req = &self.reqs[items[k].1];
        let delay = match req.think {
            Some(think) if self.opts.think_time_only => think * 1e3,
            _ => {
                let prev = &self.reqs[items[k - 1].1];
                let api_ms = prev.api.filter(|a| a.is_finite()).map_or(0.0, |a| a * 1e3);
                (req.t * 1e3 - prev.t * 1e3 - api_ms).max(0.0)
            }
        };
        self.clamp_delay(Some(delay))
    }

    fn turn(&self, items: &[Item], k: usize, ends: &[NodeId], floor_delay: bool) -> Turn {
        let (outer, ri) = items[k];
        let req = &self.reqs[ri];
        let mut delay_ms = self.turn_delay(items, k);
        if floor_delay {
            delay_ms = delay_ms.map(|d| d.max(0.0));
        }
        Turn {
            prompt_end: ends[ri],
            hash_blocks: req.hash.len() as u32,
            model: req.model,
            max_tokens: req.out.clamp(1, u32::MAX as u64) as u32,
            ts_ms: req.t * 1e3,
            delay_ms,
            api_ms: req.api.filter(|a| a.is_finite()).map(|a| a.max(0.0) * 1e3),
            spawns: Vec::new(),
            joins: Vec::new(),
            preds: Vec::new(),
            outer,
            inner: None,
        }
    }

    fn build(self) -> Result<ReplayPlan, String> {
        let mut child_plans = Vec::new();
        for i in 0..self.subagents.len() {
            child_plans.extend(self.expand_subagent(i));
        }
        let mut flat_plans = Vec::new();
        let normals = if self.opts.split_chains && self.normals.len() > 1 {
            self.split_flat(&mut flat_plans)
        } else {
            self.normals.clone()
        };

        // The prefix tree, in trace order so each prompt walks from its predecessor's path.
        let mut tree = PrefixTreeBuilder::default();
        let ends: Vec<NodeId> = self.reqs.iter().map(|r| tree.insert(&r.hash)).collect();

        let mut root_turns: Vec<Turn> = (0..normals.len())
            .map(|k| self.turn(&normals, k, &ends, true))
            .collect();
        // Root turn position by outer index, in outer order.
        let mut by_outer: Vec<(u32, usize)> = normals
            .iter()
            .enumerate()
            .map(|(pos, &(o, _))| (o, pos))
            .collect();
        by_outer.sort_unstable();
        let preceding = |outer: u32| {
            by_outer
                .iter()
                .filter(|&&(o, _)| o < outer)
                .map(|&(_, p)| p)
                .max()
        };
        let join_after = |outer: u32, end_s: f64| {
            by_outer
                .iter()
                .filter(|&&(o, _)| o > outer)
                .find(|&&(_, pos)| self.reqs[normals[pos].1].t + JOIN_EPS >= end_s)
                .map(|&(_, pos)| pos)
        };

        // Subagent branches, grouped by (spawning turn, join turn).
        let mut dropped = HashSet::new();
        let mut groups: Vec<Group> = Vec::new();
        for (i, sa) in self.subagents.iter().enumerate() {
            let Some(spawn) = preceding(sa.outer) else {
                dropped.insert(i);
                continue;
            };
            let end = match sa.duration_ms {
                Some(ms) => {
                    sa.t + if ms.is_finite() && ms >= 0.0 {
                        ms / 1e3
                    } else {
                        0.0
                    }
                }
                None if !sa.items.is_empty() => sa
                    .items
                    .iter()
                    .map(|&(_, r)| self.reqs[r].end())
                    .fold(f64::NEG_INFINITY, f64::max),
                None => sa.t,
            };
            let key = (spawn, join_after(sa.outer, end));
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, members)) => members.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        let mut seen_agents = HashSet::new();
        for (i, sa) in self.subagents.iter().enumerate() {
            if !dropped.contains(&i) && !seen_agents.insert(sa.agent_id.as_str()) {
                return Err(format!(
                    "trace {}: duplicate subagent agent_id {:?} among retained subagents",
                    self.trace_id, sa.agent_id
                ));
            }
        }

        // Conversations: root, retained subagent children, flat chains.
        let mut convs = vec![Conversation {
            id: self.trace_id.clone(),
            kind: ConvKind::Root,
            scope: 0,
            turns: Vec::new(),
            branches: Vec::new(),
        }];
        let mut scopes: Vec<String> = vec![self.trace_id.clone()];
        let mut children_of_subagent: HashMap<usize, Vec<u32>> = HashMap::new();
        for cp in &child_plans {
            if dropped.contains(&cp.subagent) {
                continue;
            }
            let scope_name = replay_scope(&cp.session_id, &self.trace_id);
            let scope = match scopes.iter().position(|s| *s == scope_name) {
                Some(s) => s,
                None => {
                    scopes.push(scope_name);
                    scopes.len() - 1
                }
            };
            let mut turns: Vec<Turn> = (0..cp.items.len())
                .map(|k| self.turn(&cp.items, k, &ends, false))
                .collect();
            let outer = self.subagents[cp.subagent].outer;
            for (t, &(inner, _)) in turns.iter_mut().zip(&cp.items) {
                t.outer = outer;
                t.inner = Some(inner);
            }
            children_of_subagent
                .entry(cp.subagent)
                .or_default()
                .push(convs.len() as u32);
            convs.push(Conversation {
                id: cp.session_id.clone(),
                kind: ConvKind::Subagent,
                scope: scope as u32,
                turns,
                branches: Vec::new(),
            });
        }
        let mut flat_convs = Vec::with_capacity(flat_plans.len());
        for fp in &flat_plans {
            flat_convs.push(convs.len() as u32);
            convs.push(Conversation {
                id: fp.session_id.clone(),
                kind: ConvKind::Flat,
                scope: 0,
                turns: (0..fp.items.len())
                    .map(|k| self.turn(&fp.items, k, &ends, false))
                    .collect(),
                branches: Vec::new(),
            });
        }

        let mut branches = Vec::new();
        for ((spawn, join), members) in &groups {
            let first = &self.subagents[members[0]];
            let children: Vec<u32> = members
                .iter()
                .flat_map(|m| children_of_subagent.get(m).cloned().unwrap_or_default())
                .collect();
            let start_s = members
                .iter()
                .map(|&m| self.subagents[m].t)
                .fold(f64::INFINITY, f64::min);
            branches.push(Branch {
                id: format!("{}:spawn:{}", self.trace_id, first.agent_id),
                children,
                background: join.is_none(),
                start_ms: Some(start_s * 1e3),
                spawn_turn: *spawn as u32,
                join_turn: join.map(|j| j as u32),
            });
        }
        // Flattened chains: spawned off the last root turn before their first request (turn 0 when
        // none), joined like subagents.
        let mut flat_groups: Vec<Group> = Vec::new();
        for (i, fp) in flat_plans.iter().enumerate() {
            let first_outer = fp.items[0].0;
            let spawn = preceding(first_outer).unwrap_or(0);
            let end = fp
                .items
                .iter()
                .map(|&(_, r)| self.reqs[r].end())
                .fold(f64::NEG_INFINITY, f64::max);
            let key = (spawn, join_after(first_outer, end));
            match flat_groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, members)) => members.push(i),
                None => flat_groups.push((key, vec![i])),
            }
        }
        for ((spawn, join), members) in &flat_groups {
            let start_s = members
                .iter()
                .map(|&m| self.reqs[flat_plans[m].items[0].1].t)
                .fold(f64::INFINITY, f64::min);
            branches.push(Branch {
                id: format!(
                    "{}:flatspawn:{}",
                    self.trace_id, flat_plans[members[0]].chain_index
                ),
                children: members.iter().map(|&m| flat_convs[m]).collect(),
                background: join.is_none(),
                start_ms: Some(start_s * 1e3),
                spawn_turn: *spawn as u32,
                join_turn: join.map(|j| j as u32),
            });
        }
        for (b, branch) in branches.iter().enumerate() {
            root_turns[branch.spawn_turn as usize].spawns.push(b as u32);
            if let Some(j) = branch.join_turn {
                root_turns[j as usize].joins.push(b as u32);
            }
        }
        convs[0].turns = root_turns;
        convs[0].branches = branches;

        let mut order: Vec<u32> = (0..convs.len() as u32).collect();
        order.sort_by(|&a, &b| convs[a as usize].id.cmp(&convs[b as usize].id));
        let mut id_rank = vec![0u32; convs.len()];
        for (rank, &c) in order.iter().enumerate() {
            id_rank[c as usize] = rank as u32;
        }
        install_replay_predecessors(&mut convs, &id_rank);

        let mut bounds: Option<(f64, f64)> = None;
        for t in convs.iter().flat_map(|c| &c.turns) {
            if t.ts_ms.is_finite() {
                bounds = Some(match bounds {
                    None => (t.ts_ms, t.ts_ms),
                    Some((lo, hi)) => (lo.min(t.ts_ms), hi.max(t.ts_ms)),
                });
            }
        }
        Ok(ReplayPlan {
            trace_id: self.trace_id,
            block_size: self.block_size,
            prefixes: tree.finish(),
            models: self.models,
            convs,
            time_bounds_ms: bounds,
            id_rank,
            scopes,
        })
    }
}

/// `_replay_scope_for_session`: every chain of one subagent shares the subagent's scope; flattened
/// chains share the root's.
fn replay_scope(session_id: &str, trace_id: &str) -> String {
    let Some((trace, agent)) = session_id.split_once("::sa:") else {
        return trace_id.to_owned();
    };
    let mut agent = agent;
    for marker in [":aux:", ":fa:", ":wg:"] {
        if let Some(pos) = agent.rfind(marker) {
            agent = &agent[..pos];
        }
    }
    format!("{trace}::sa:{agent}")
}

/// `infer_cross_stream_predecessors`, per scope: for every other stream of the scope, a turn depends on
/// that stream's latest request that had completed by the turn's recorded start; a predecessor that
/// finished before another predecessor started is implied and dropped.
fn install_replay_predecessors(convs: &mut [Conversation], id_rank: &[u32]) {
    let n_scopes = convs
        .iter()
        .map(|c| c.scope)
        .max()
        .map_or(0, |s| s as usize + 1);
    let mut by_scope: Vec<Vec<u32>> = vec![Vec::new(); n_scopes];
    for (i, c) in convs.iter().enumerate() {
        by_scope[c.scope as usize].push(i as u32);
    }
    // (end, start, id rank, turn) per stream, sorted ascending.
    let interval = |convs: &[Conversation], c: u32, k: usize| {
        let t = &convs[c as usize].turns[k];
        let start = t.ts_ms;
        (
            start + t.api_ms.unwrap_or(0.0),
            start,
            id_rank[c as usize],
            k as u32,
        )
    };
    for streams in by_scope {
        if streams.len() < 2 {
            continue;
        }
        let sorted: Vec<Vec<(f64, f64, u32, u32)>> = streams
            .iter()
            .map(|&c| {
                let mut v: Vec<_> = (0..convs[c as usize].turns.len())
                    .map(|k| interval(convs, c, k))
                    .collect();
                v.sort_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then(a.1.total_cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                        .then(a.3.cmp(&b.3))
                });
                v
            })
            .collect();
        let mut preds_of: Vec<Vec<Vec<(u32, u32)>>> = streams
            .iter()
            .map(|&c| vec![Vec::new(); convs[c as usize].turns.len()])
            .collect();
        for (si, &c) in streams.iter().enumerate() {
            let turns = &convs[c as usize].turns;
            for (k, slot) in preds_of[si].iter_mut().enumerate() {
                let target = turns[k].ts_ms;
                let mut frontier: Vec<(f64, f64, u32, u32, u32)> = Vec::new();
                for (sj, &other) in streams.iter().enumerate() {
                    if sj == si {
                        continue;
                    }
                    let cands = &sorted[sj];
                    let mut idx = cands.partition_point(|x| x.0 <= target);
                    while idx > 0 && cands[idx - 1].1 >= target {
                        idx -= 1;
                    }
                    if idx > 0 {
                        let (end, start, rank, turn) = cands[idx - 1];
                        frontier.push((end, start, rank, turn, other));
                    }
                }
                let mut preds: Vec<(u32, u32, u32)> = frontier
                    .iter()
                    .filter(|a| {
                        !frontier
                            .iter()
                            .any(|b| !std::ptr::eq(*a, b) && a.1 < b.1 && a.0 <= b.1)
                    })
                    .map(|&(_, _, rank, turn, conv)| (rank, turn, conv))
                    .collect();
                preds.sort_unstable();
                *slot = preds
                    .into_iter()
                    .map(|(_, turn, conv)| (conv, turn))
                    .collect();
            }
        }
        for (si, &c) in streams.iter().enumerate() {
            for (k, preds) in std::mem::take(&mut preds_of[si]).into_iter().enumerate() {
                convs[c as usize].turns[k].preds = preds;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(json: &str) -> ReplayPlan {
        ReplayPlan::build(
            Trace::from_json(json.as_bytes()).unwrap(),
            &PlanOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn delays_are_think_times_capped_and_floored() {
        let p = plan(
            r#"{"id": "t", "block_size": 2, "requests": [
                {"t": 0, "type": "n", "model": "m", "in": 2, "out": 0, "hash_ids": [1], "api_time": 2.0},
                {"t": 10, "type": "n", "model": "m", "in": 4, "out": 5, "hash_ids": [1, 2], "think_time": 3.5},
                {"t": 500, "type": "n", "model": "m", "in": 6, "out": 5, "hash_ids": [1, 2, 3], "think_time": 400.0},
                {"t": 600, "type": "n", "model": "m", "in": 8, "out": 5, "hash_ids": [1, 2, 3, 4], "api_time": 1.0}
            ]}"#,
        );
        let root = p.root();
        let delays: Vec<_> = root.turns.iter().map(|t| t.delay_ms).collect();
        // No think time on the last turn: the idle gap since the previous turn ended (no api_time: 0).
        assert_eq!(
            delays,
            vec![None, Some(3500.0), Some(120_000.0), Some(100_000.0)]
        );
        assert_eq!(root.turns[0].max_tokens, 1, "max_tokens is at least 1");
        assert_eq!(p.convs.len(), 1);
    }

    #[test]
    fn subagents_branch_and_join() {
        let p = plan(
            r#"{"id": "t", "block_size": 2, "requests": [
                {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
                {"t": 2, "type": "subagent", "agent_id": "a", "duration_ms": 5000, "requests": [
                    {"t": 2, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [9], "api_time": 1.0},
                    {"t": 4, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [9, 8], "think_time": 1.0}
                ]},
                {"t": 8, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "think_time": 1.0},
                {"t": 9, "type": "subagent", "agent_id": "b", "duration_ms": 99000, "requests": [
                    {"t": 9, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [7]}
                ]}
            ]}"#,
        );
        let ids: Vec<&str> = p.convs.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["t", "t::sa:a", "t::sa:b"]);
        let root = p.root();
        assert_eq!(root.branches.len(), 2);
        let a = &root.branches[0];
        assert_eq!(
            (a.spawn_turn, a.join_turn, a.background),
            (0, Some(1), false)
        );
        assert_eq!(a.children, vec![1]);
        let b = &root.branches[1];
        assert_eq!((b.spawn_turn, b.join_turn, b.background), (1, None, true));
        assert_eq!(
            (root.turns[0].spawns.clone(), root.turns[1].joins.clone()),
            (vec![0], vec![0])
        );
        assert_eq!(p.convs[1].turns[1].delay_ms, Some(1000.0));
        assert_eq!(p.convs[1].scope, 1);
    }

    #[test]
    fn an_orphan_subagent_is_dropped() {
        let p = plan(
            r#"{"id": "t", "block_size": 2, "requests": [
                {"t": 0, "type": "subagent", "agent_id": "a", "requests": [
                    {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [9]}]},
                {"t": 1, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1]}
            ]}"#,
        );
        assert_eq!(p.convs.len(), 1);
    }

    #[test]
    fn a_flattened_agent_becomes_a_flat_child() {
        let p = plan(
            r#"{"id": "t", "block_size": 2, "requests": [
                {"t": 0, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "api_time": 1.0},
                {"t": 2, "type": "n", "model": "m", "in": 8, "out": 1, "hash_ids": [1, 2, 7, 8], "api_time": 10.0},
                {"t": 3, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [1, 2, 3], "api_time": 1.0},
                {"t": 20, "type": "n", "model": "m", "in": 10, "out": 1, "hash_ids": [1, 2, 7, 8, 9], "api_time": 1.0},
                {"t": 30, "type": "n", "model": "m", "in": 8, "out": 1, "hash_ids": [1, 2, 3, 4], "api_time": 1.0}
            ]}"#,
        );
        // Expected values are what AIPerf 0.13.0's loader produces for this trace.
        let ids: Vec<&str> = p.convs.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["t", "t::fa:000"]);
        let outers: Vec<u32> = p.root().turns.iter().map(|t| t.outer).collect();
        assert_eq!(outers, vec![0, 1, 3]);
        let flat = &p.convs[1];
        assert_eq!(
            flat.turns.iter().map(|t| t.outer).collect::<Vec<_>>(),
            vec![2, 4]
        );
        assert_eq!(flat.scope, 0, "flattened chains share the root's scope");
        // Spawned off root turn 1 (the last before it); it outlives every root turn: background.
        let b = &p.root().branches[0];
        assert_eq!(
            (b.id.as_str(), b.spawn_turn, b.join_turn, b.background),
            ("t:flatspawn:0", 1, None, true)
        );
        assert_eq!(b.start_ms, Some(3000.0));
        // Root turn 2 (t=20) follows the flat chain's turn 0 (ended t=4); flat turn 1 (t=30) follows
        // root turn 2 (ended t=21); flat turn 0 (t=3) follows root turn 0 (ended t=1).
        assert_eq!(p.root().turns[2].preds, vec![(1, 0)]);
        assert_eq!(flat.turns[0].preds, vec![(0, 0)]);
        assert_eq!(flat.turns[1].preds, vec![(0, 2)]);
    }

    #[test]
    fn scopes_strip_worker_markers() {
        assert_eq!(replay_scope("t::sa:agent_1:fa:002", "t"), "t::sa:agent_1");
        assert_eq!(replay_scope("t::sa:agent_1", "t"), "t::sa:agent_1");
        assert_eq!(replay_scope("t::fa:001", "t"), "t");
    }
}
