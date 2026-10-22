//! AIPerf 0.13.0's `agentic_replay` timing (`strategies/agentic_replay.py`, `branch_orchestrator.py`,
//! `replay_dependencies.py`, `session_tree.py`), run over the engine model as a discrete-event
//! simulation.
//!
//! - **Lanes.** `concurrency` lanes take traces from a sequential sampler over the corpus (AIPerf's
//!   Weka loader samples sequentially); each lane snapshots its trace at its own `t*`.
//! - **Warmup.** Every stream live at `t*` first sends its last request before `t*` with
//!   `max_tokens = 1`, all at once (burst starts) or aligned on `t*` (spread). Profiling starts when the
//!   last warmup request returns, and the duration counts from there.
//! - **Dispatch.** Each live stream sends its next turn at its recorded offset from `t*`; with burst
//!   starts the lane's earliest one goes at once. A turn's successor goes its recorded delay after it
//!   returns. A root turn spawns its branches' children when it returns, or when it is sent if the
//!   recorded spawn overlapped the request; each child starts at its recorded offset from the spawn. A
//!   root turn gated on children waits until they finish and then goes at once.
//! - **Barriers.** A turn whose replay predecessors (in its scope) have not completed is held until
//!   they have.
//! - **Recycling.** When a lane's session tree (root and every child) has drained, the lane starts the
//!   next trace from turn 0 under a fresh namespace (AIPerf's cache-bust marker).
//! - **Stop.** No request is sent after the duration; requests still running at the end of the grace
//!   period are dropped from the results. When no request is in flight, pending scheduler timers are
//!   pulled forward so the engine idles at most `system_idle_cap` seconds.
//!
//! The engine serves requests without mutual queueing: a request's time to first token is the cost
//! model's for its hit pattern, then it decodes at a constant inter-token latency. It waits only for
//! free GPU blocks.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::rc::Rc;

use lh_core::{PromptSources, TierCost};
use serde::Serialize;

use crate::aiperf::{ConvKind, ReplayPlan, snapshot};
use crate::corpus::Corpus;
use crate::engine::{Admission, Engine, EngineRequest};
use crate::prefix::{BlockId, IdMap, IdSet, MAX_MODELS, MAX_PLAYS, NodeId, Tokens};

/// How the replay is driven: the `phases` section of `configs/agentic_replay.yaml`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReplayConfig {
    /// Live session trees.
    pub concurrency: usize,
    pub duration_s: f64,
    pub grace_s: f64,
    pub start_min_ratio: f64,
    pub start_max_ratio: f64,
    /// `burstPhaseStarts`.
    pub burst_starts: bool,
    /// `systemIdleGapCapSeconds`.
    pub system_idle_cap_s: Option<f64>,
    /// `randomSeed`.
    pub seed: u64,
    /// Stop starting sessions after this many (the initial lanes count). `None` for no limit.
    pub max_sessions: Option<u64>,
    /// Requests to different models never share KV. Off reproduces an AIPerf replay, which sends
    /// every request to one served model.
    pub model_namespaces: bool,
    /// Tokens per engine block.
    pub block_tokens: u32,
    /// Multiplies the tokens of every hash block and every output: 32 replays a 32x-scaled corpus at
    /// its original token counts (output lengths come back rounded up to a multiple of 32).
    pub token_scale: u32,
}

impl Default for ReplayConfig {
    /// The W0 run: 16 session trees, 600 s plus 60 s grace, burst starts, seed 11.
    fn default() -> Self {
        Self {
            concurrency: 16,
            duration_s: 600.0,
            grace_s: 60.0,
            start_min_ratio: 0.0,
            start_max_ratio: 1.0,
            burst_starts: true,
            system_idle_cap_s: Some(10.0),
            seed: 11,
            max_sessions: None,
            model_namespaces: true,
            block_tokens: 16,
            token_scale: 1,
        }
    }
}

/// One completed request.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub trace: u32,
    pub conv: u32,
    pub turn: u32,
    pub kind: ConvKind,
    pub warmup: bool,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub sources: PromptSources,
    /// Computed tokens of full blocks this run had already seen (evicted from every tier, or the
    /// recomputed last block of a fully cached prompt).
    pub recomputed: u64,
    /// The think-time delay that preceded this turn, in ms.
    pub delay_ms: Option<f64>,
    pub issued_ns: u64,
    pub admitted_ns: u64,
    pub done_ns: u64,
    /// Time to first token as the client sees it: waiting for blocks plus the modeled prefill.
    pub ttft_ns: u64,
}

/// The engine-side order of a run, for replaying it under another policy.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    /// The admitted requests, in admission order.
    pub requests: Vec<TimelineRequest>,
    /// Admissions and finishes, in the order the engine saw them.
    pub events: Vec<TimelineEvent>,
}

/// One admitted request of a [`Timeline`].
#[derive(Clone, Debug)]
pub struct TimelineRequest {
    pub prompt: Vec<BlockId>,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Index into the run's records; `usize::MAX` for warmup requests and requests that did not
    /// complete in time.
    pub record: usize,
    pub admitted_ns: u64,
    pub warmup: bool,
    /// Where the prompt comes from: the corpus trace, its conversation and turn, and the KV
    /// namespace (session play and model) its blocks were cut in.
    pub trace: u32,
    pub conv: u32,
    pub turn: u32,
    pub play: u64,
    pub model: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineEvent {
    Admit(usize),
    Finish(usize),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReplayStats {
    pub lanes: usize,
    pub sessions_started: u64,
    pub warmup_requests: u64,
    pub warmup_s: f64,
    pub profiling_start_s: f64,
    /// Requests sent during profiling but still running when the grace period ended.
    pub cut_off_requests: u64,
    /// Turns held by replay barriers at some point.
    pub barrier_holds: u64,
    pub idle_cap_jumps: u64,
    pub idle_cap_skipped_s: f64,
    /// Time-weighted mean and peak of the tokens in the latest prompt of every live stream (distinct
    /// full blocks), over the profiling window.
    pub working_set_mean_tokens: f64,
    pub working_set_peak_tokens: u64,
}

#[derive(Debug)]
pub struct ReplayOutput {
    pub records: Vec<RequestRecord>,
    pub stats: ReplayStats,
    pub engine: crate::engine::EngineStats,
    pub timeline: Option<Timeline>,
    pub traces: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    SessionStart,
    RootNext,
    Join,
    ChildFirst,
    ChildNext,
}

impl Kind {
    fn is_child(self) -> bool {
        matches!(self, Kind::ChildFirst | Kind::ChildNext)
    }
}

#[derive(Clone, Copy, Debug)]
enum Event {
    Dispatch {
        stream: u32,
        turn: u32,
        kind: Kind,
        delay_ms: Option<f64>,
    },
    Warm {
        stream: u32,
        turn: u32,
    },
    Complete(u64),
    Recycle(usize),
}

#[derive(Debug)]
struct Entry {
    at: u64,
    seq: u64,
    event: Event,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        (self.at, self.seq) == (other.at, other.seq)
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    // Reversed: BinaryHeap pops the earliest.
    fn cmp(&self, other: &Self) -> Ordering {
        (other.at, other.seq).cmp(&(self.at, self.seq))
    }
}

#[derive(Debug)]
struct Stream {
    play: u32,
    conv: u32,
    depth: u8,
    /// `(gated root turn, branch)` this child must satisfy; empty for background children.
    joins: Vec<(u32, u32)>,
    done: bool,
    context: Vec<BlockId>,
}

#[derive(Debug)]
struct Prereq {
    branch: u32,
    expected: u32,
    completed: u32,
    registered: bool,
}

#[derive(Debug)]
struct Gate {
    turn: u32,
    prereqs: Vec<Prereq>,
}

impl Gate {
    fn satisfied(&self) -> bool {
        self.prereqs
            .iter()
            .all(|p| p.registered && p.completed >= p.expected)
    }

    fn prereq(&mut self, branch: u32) -> Option<&mut Prereq> {
        self.prereqs.iter_mut().find(|p| p.branch == branch)
    }
}

#[derive(Debug)]
struct Play {
    plan: Rc<ReplayPlan>,
    trace: u32,
    lane: usize,
    namespace: u64,
    root: Option<u32>,
    root_pending: bool,
    descendants: u32,
    drained: bool,
    completed: HashSet<(u32, u32)>,
    held: BTreeMap<(u32, u32, u32), (u32, Kind)>,
    future_joins: BTreeMap<u32, Gate>,
    active_join: Option<Gate>,
    overlap_spawned: Vec<u32>,
}

#[derive(Debug)]
struct InFlight {
    stream: u32,
    turn: u32,
    warmup: bool,
    issued: u64,
    admitted: u64,
    prompt: Vec<BlockId>,
    prompt_tokens: u64,
    output_tokens: u64,
    delay_ms: Option<f64>,
    admission: Option<Admission>,
    recomputed: u64,
    timeline: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Warmup,
    Profiling,
}

/// A lane's first session: its play, its snapshot, and one stream per live state.
#[derive(Debug)]
struct Lane {
    play: u32,
    snapshot: crate::aiperf::Snapshot,
    streams: Vec<u32>,
}

/// Computed tokens of an admitted prompt in full blocks that `seen` holds: blocks some earlier
/// request had, now evicted from every tier (or the always-recomputed last block). Adds the prompt's
/// blocks to `seen`.
pub(crate) fn recomputed_tokens(
    seen: &mut IdSet,
    prompt: &[BlockId],
    adm: Admission,
    block_tokens: u64,
) -> u64 {
    let start = (adm.gpu_blocks + adm.cpu_blocks) as usize;
    let again = prompt[start..].iter().filter(|b| seen.contains(b)).count() as u64;
    seen.extend(prompt.iter().copied());
    again * block_tokens
}

fn ns(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1e6).round() as u64
    } else {
        0
    }
}

/// Runs one replay of `corpus` through `engine`.
#[derive(Debug)]
pub struct Replay<'a> {
    cfg: ReplayConfig,
    corpus: &'a mut Corpus,
    engine: Engine,
    cost: TierCost,
    now: u64,
    seq: u64,
    shift: u64,
    timers: BinaryHeap<Entry>,
    fixed: BinaryHeap<Entry>,
    phase: Phase,
    profiling_start: u64,
    profiling_end: u64,
    grace_end: u64,
    plays: Vec<Play>,
    streams: Vec<Stream>,
    lanes: Vec<Lane>,
    sampler: usize,
    in_flight: IdMap<InFlight>,
    warmup_outstanding: usize,
    idle_since: Option<u64>,
    next_request: u64,
    records: Vec<RequestRecord>,
    timeline: Option<Timeline>,
    seen: IdSet,
    working_set: IdMap<u32>,
    ws_last_change: u64,
    ws_integral: f64,
    stats: ReplayStats,
    scratch: Vec<NodeId>,
}

impl<'a> Replay<'a> {
    pub fn new(
        cfg: ReplayConfig,
        corpus: &'a mut Corpus,
        engine: Engine,
        cost: TierCost,
        record: bool,
    ) -> Self {
        Self {
            cfg,
            corpus,
            engine,
            cost,
            now: 0,
            seq: 0,
            shift: 0,
            timers: BinaryHeap::new(),
            fixed: BinaryHeap::new(),
            phase: Phase::Warmup,
            profiling_start: 0,
            profiling_end: u64::MAX,
            grace_end: u64::MAX,
            plays: Vec::new(),
            streams: Vec::new(),
            lanes: Vec::new(),
            sampler: 0,
            in_flight: IdMap::default(),
            warmup_outstanding: 0,
            idle_since: None,
            next_request: 0,
            records: Vec::new(),
            timeline: record.then(Timeline::default),
            seen: IdSet::default(),
            working_set: IdMap::default(),
            ws_last_change: 0,
            ws_integral: 0.0,
            stats: ReplayStats::default(),
            scratch: Vec::new(),
        }
    }

    pub fn run(mut self) -> Result<ReplayOutput, String> {
        self.start_lanes()?;
        loop {
            let timer = self.timers.peek().map(|e| (e.at - self.shift, e.seq));
            let fixed = self.fixed.peek().map(|e| (e.at, e.seq));
            let from_timers = match (timer, fixed) {
                (None, None) => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(t), Some(f)) => t <= f,
            };
            let entry = if from_timers {
                self.timers.pop()
            } else {
                self.fixed.pop()
            }
            .expect("peeked");
            let at = if from_timers {
                entry.at - self.shift
            } else {
                entry.at
            };
            if self.phase == Phase::Profiling && at > self.grace_end {
                break;
            }
            self.advance(at);
            match entry.event {
                Event::Dispatch {
                    stream,
                    turn,
                    kind,
                    delay_ms,
                } => self.submit(stream, turn, kind, delay_ms),
                Event::Warm { stream, turn } => self.issue(stream, turn, true, None),
                Event::Complete(id) => self.complete(id)?,
                Event::Recycle(lane) => self.recycle(lane)?,
            }
            if from_timers {
                self.enforce_idle_cap(false);
            }
        }
        if self.engine.stats.oversized > 0 {
            return Err(format!(
                "{} requests need more KV blocks than the GPU pool's {}",
                self.engine.stats.oversized,
                self.engine.gpu_capacity()
            ));
        }
        // The working-set mean covers the duration, or the part of it before a run ran out of work.
        let window_end = if self.now >= self.profiling_end {
            self.profiling_end
        } else {
            self.now
        };
        self.advance(window_end.max(self.now));
        self.stats.cut_off_requests = self
            .in_flight
            .values()
            .filter(|f| !f.warmup && f.issued >= self.profiling_start)
            .count() as u64;
        let window = window_end.saturating_sub(self.profiling_start).max(1) as f64;
        self.stats.working_set_mean_tokens =
            self.ws_integral / window * self.cfg.block_tokens as f64;
        self.stats.lanes = self.lanes.len();
        let traces = self
            .plays
            .iter()
            .map(|p| p.plan.trace_id.clone())
            .collect::<Vec<_>>();
        let mut unique = HashSet::new();
        let traces = traces
            .into_iter()
            .filter(|t| unique.insert(t.clone()))
            .collect();
        Ok(ReplayOutput {
            records: self.records,
            stats: self.stats,
            engine: self.engine.stats.clone(),
            timeline: self.timeline,
            traces,
        })
    }

    /// Moves the clock to `at`, integrating the live working set over the profiling duration.
    fn advance(&mut self, at: u64) {
        if self.phase == Phase::Profiling && at > self.ws_last_change {
            let from = self.ws_last_change.max(self.profiling_start);
            let to = at.min(self.profiling_end);
            if to > from {
                self.ws_integral += self.working_set.len() as f64 * (to - from) as f64;
            }
            self.ws_last_change = at;
        }
        self.now = self.now.max(at);
    }

    fn push_timer(&mut self, at: u64, event: Event) {
        self.seq += 1;
        self.timers.push(Entry {
            at: at + self.shift,
            seq: self.seq,
            event,
        });
    }

    fn push_fixed(&mut self, at: u64, event: Event) {
        self.seq += 1;
        self.fixed.push(Entry {
            at,
            seq: self.seq,
            event,
        });
    }

    fn sending_open(&self) -> bool {
        self.phase == Phase::Profiling && self.now < self.profiling_end
    }

    fn can_start_session(&self) -> bool {
        self.sending_open()
            && self
                .cfg
                .max_sessions
                .is_none_or(|m| self.stats.sessions_started < m)
    }

    /// The next spawnable trace from the sequential sampler.
    fn next_trace(&mut self) -> Result<(usize, Rc<ReplayPlan>), String> {
        for _ in 0..self.corpus.len() {
            let idx = self.sampler % self.corpus.len();
            self.sampler += 1;
            let plan = self.corpus.plan(idx)?;
            if !plan.root().turns.is_empty() {
                return Ok((idx, plan));
            }
        }
        Err("no trace has a main-agent request".into())
    }

    fn new_play(&mut self, trace: usize, plan: Rc<ReplayPlan>, lane: usize) -> Result<u32, String> {
        let namespace = self.plays.len() as u64;
        if namespace >= MAX_PLAYS {
            return Err("too many session plays for the block namespace".into());
        }
        if self.cfg.model_namespaces && plan.models.len() > MAX_MODELS {
            return Err(format!(
                "trace {} has more than {MAX_MODELS} models",
                plan.trace_id
            ));
        }
        self.plays.push(Play {
            plan,
            trace: trace as u32,
            lane,
            namespace,
            root: None,
            root_pending: false,
            descendants: 0,
            drained: false,
            completed: HashSet::new(),
            held: BTreeMap::new(),
            future_joins: BTreeMap::new(),
            active_join: None,
            overlap_spawned: Vec::new(),
        });
        self.stats.sessions_started += 1;
        Ok(namespace as u32)
    }

    fn new_stream(&mut self, play: u32, conv: u32, depth: u8) -> u32 {
        self.streams.push(Stream {
            play,
            conv,
            depth,
            joins: Vec::new(),
            done: false,
            context: Vec::new(),
        });
        self.streams.len() as u32 - 1
    }

    /// Builds the lanes (`TrajectorySource`) and sends the warmup.
    fn start_lanes(&mut self) -> Result<(), String> {
        let budget = self.cfg.concurrency + 2 * self.corpus.len().max(1);
        let mut attempts = 0;
        while self.lanes.len() < self.cfg.concurrency && attempts < budget {
            attempts += 1;
            let (trace, plan) = self.next_trace()?;
            let lane = self.lanes.len();
            let t_star = snapshot::t_star(
                &plan,
                self.cfg.seed,
                lane,
                self.cfg.start_min_ratio,
                self.cfg.start_max_ratio,
            )
            .ok_or_else(|| format!("trace {} has no timestamps", plan.trace_id))?;
            let snap = snapshot::snapshot(&plan, t_star).ok_or_else(|| {
                format!("trace {}: nothing live at t* = {t_star} ms", plan.trace_id)
            })?;
            let play = self.new_play(trace, plan.clone(), lane)?;
            // One stream per live state; the warmup primes their history.
            let mut streams = Vec::with_capacity(snap.states.len());
            for state in &snap.states {
                let stream = self.new_stream(play, state.conv, state.depth);
                if state.depth == 0 {
                    self.plays[play as usize].root = Some(stream);
                }
                streams.push(stream);
            }
            self.lanes.push(Lane {
                play,
                snapshot: snap,
                streams,
            });
        }
        let mut warm: Vec<(u32, u32, f64)> = Vec::new();
        for lane in &self.lanes {
            let plan = &self.plays[lane.play as usize].plan;
            for (state, &stream) in lane.snapshot.states.iter().zip(&lane.streams) {
                if let Some(turn) = state.warmup_turn() {
                    let ts = plan.convs[state.conv as usize].turns[turn as usize].ts_ms;
                    warm.push((stream, turn, lane.snapshot.t_star_ms - ts));
                }
            }
        }
        self.warmup_outstanding = warm.len();
        self.stats.warmup_requests = warm.len() as u64;
        if warm.is_empty() {
            return self.start_profiling();
        }
        let max_lead = warm.iter().map(|w| w.2).fold(0.0, f64::max);
        for (stream, turn, lead) in warm {
            if self.cfg.burst_starts || max_lead - lead <= 0.0 {
                self.issue(stream, turn, true, None);
            } else {
                self.push_timer(ns(max_lead - lead), Event::Warm { stream, turn });
            }
        }
        Ok(())
    }

    fn start_profiling(&mut self) -> Result<(), String> {
        self.phase = Phase::Profiling;
        self.profiling_start = self.now;
        self.ws_last_change = self.now;
        self.profiling_end = self.now + ns(self.cfg.duration_s * 1e3);
        self.grace_end = self.profiling_end + ns(self.cfg.grace_s * 1e3);
        self.stats.warmup_s = self.now as f64 / 1e9;
        self.stats.profiling_start_s = self.now as f64 / 1e9;
        self.stats.working_set_peak_tokens =
            self.working_set.len() as u64 * self.cfg.block_tokens as u64;
        // Seed every lane's barrier history and live-children joins before any lane dispatches.
        let mut dispatches = Vec::new();
        for lane in 0..self.lanes.len() {
            let play = self.lanes[lane].play;
            let snap = self.lanes[lane].snapshot.clone();
            let streams = self.lanes[lane].streams.clone();
            let plan = self.plays[play as usize].plan.clone();
            for &(conv, done) in &snap.boundaries {
                for turn in 0..done {
                    self.plays[play as usize].completed.insert((conv, turn));
                }
            }
            let root_state = snap.states.iter().find(|s| s.depth == 0).cloned();
            self.plays[play as usize].root_pending = root_state.is_some();
            for (state, &stream) in snap.states.iter().zip(&streams) {
                if state.depth == 0 {
                    continue;
                }
                self.plays[play as usize].descendants += 1;
                let (Some(root), Some(gated), Some(branch)) =
                    (&root_state, state.join_turn, state.branch)
                else {
                    continue;
                };
                let gate = self.seeded_gate(play, root, gated, &plan);
                if let Some(p) = gate.prereq(branch) {
                    p.expected += 1;
                    p.registered = true;
                }
                self.streams[stream as usize].joins.push((gated, branch));
            }
            let dispatchable: Vec<(usize, f64)> = snap
                .states
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.waiting)
                .map(|(i, s)| (i, s.offset_ms))
                .collect();
            let t0 = if self.cfg.burst_starts {
                dispatchable
                    .iter()
                    .map(|d| d.1)
                    .fold(f64::INFINITY, f64::min)
            } else {
                0.0
            };
            for (i, offset) in dispatchable {
                let state = &snap.states[i];
                let kind = if state.depth == 0 {
                    Kind::SessionStart
                } else {
                    Kind::ChildNext
                };
                dispatches.push((streams[i], state.next_turn, kind, offset - t0));
            }
        }
        for (stream, turn, kind, delay) in dispatches {
            if delay > 0.0 {
                self.push_timer(
                    self.now + ns(delay),
                    Event::Dispatch {
                        stream,
                        turn,
                        kind,
                        delay_ms: None,
                    },
                );
            } else {
                self.submit(stream, turn, kind, None);
            }
        }
        self.enforce_idle_cap(false);
        Ok(())
    }

    /// `_ensure_seeded_join`: the join on `gated` for a snapshot lane, created with the prerequisites
    /// of every branch the turn joins; branches spawned before `t*` count as registered.
    fn seeded_gate(
        &mut self,
        play: u32,
        root: &snapshot::State,
        gated: u32,
        plan: &ReplayPlan,
    ) -> &mut Gate {
        let p = &mut self.plays[play as usize];
        if p.active_join.as_ref().is_some_and(|g| g.turn == gated) {
            return p.active_join.as_mut().expect("checked");
        }
        if !p.future_joins.contains_key(&gated) {
            let prereqs = plan.root().turns[gated as usize]
                .joins
                .iter()
                .map(|&b| Prereq {
                    branch: b,
                    expected: 0,
                    completed: 0,
                    registered: plan.root().branches[b as usize].spawn_turn < root.next_turn,
                })
                .collect();
            let gate = Gate {
                turn: gated,
                prereqs,
            };
            if root.waiting && root.join_turn == Some(gated) {
                p.active_join = Some(gate);
                return p.active_join.as_mut().expect("just set");
            }
            p.future_joins.insert(gated, gate);
        }
        p.future_joins.get_mut(&gated).expect("present")
    }

    /// `_ensure_future_join`.
    fn future_gate(&mut self, play: u32, gated: u32) -> &mut Gate {
        let p = &mut self.plays[play as usize];
        let plan = p.plan.clone();
        p.future_joins.entry(gated).or_insert_with(|| Gate {
            turn: gated,
            prereqs: plan.root().turns[gated as usize]
                .joins
                .iter()
                .map(|&b| Prereq {
                    branch: b,
                    expected: 0,
                    completed: 0,
                    registered: false,
                })
                .collect(),
        })
    }

    fn branch_start(plan: &ReplayPlan, branch: u32) -> Option<f64> {
        let b = &plan.root().branches[branch as usize];
        b.start_ms.or_else(|| {
            b.children
                .iter()
                .filter_map(|&c| plan.convs[c as usize].turns.first().map(|t| t.ts_ms))
                .reduce(f64::min)
        })
    }

    /// `_spawn_children_and_register_gates`.
    fn spawn_children(&mut self, play: u32, branches: &[u32], origin: Option<f64>) {
        let plan = self.plays[play as usize].plan.clone();
        let mut immediate = Vec::new();
        for &b in branches {
            if origin.is_none() && self.plays[play as usize].overlap_spawned.contains(&b) {
                continue;
            }
            let branch = &plan.root().branches[b as usize];
            let gated = if branch.background {
                None
            } else {
                branch.join_turn
            };
            let base = origin.or_else(|| Self::branch_start(&plan, b));
            for &child in &branch.children {
                let stream = self.new_stream(play, child, 1);
                if let Some(g) = gated {
                    if let Some(p) = self.future_gate(play, g).prereq(b) {
                        p.expected += 1;
                        p.registered = true;
                    }
                    self.streams[stream as usize].joins.push((g, b));
                }
                self.plays[play as usize].descendants += 1;
                let first = plan.convs[child as usize].turns.first().map(|t| t.ts_ms);
                let offset = match (base, first) {
                    (Some(base), Some(t)) => (t - base).max(0.0),
                    _ => 0.0,
                };
                if offset > 0.0 {
                    let at = self.now + ns(offset);
                    self.push_fixed(
                        at,
                        Event::Dispatch {
                            stream,
                            turn: 0,
                            kind: Kind::ChildFirst,
                            delay_ms: None,
                        },
                    );
                } else {
                    immediate.push(stream);
                }
            }
        }
        for stream in immediate {
            self.submit(stream, 0, Kind::ChildFirst, None);
        }
    }

    /// Sends a turn through the replay barrier: held until its predecessors have completed.
    fn submit(&mut self, stream: u32, turn: u32, kind: Kind, delay_ms: Option<f64>) {
        let s = &self.streams[stream as usize];
        let play = &mut self.plays[s.play as usize];
        if self.phase == Phase::Profiling {
            let t = &play.plan.convs[s.conv as usize].turns[turn as usize];
            if !t.preds.iter().all(|p| play.completed.contains(p)) {
                let key = (play.plan.id_rank[s.conv as usize], turn, s.conv);
                play.held.insert(key, (stream, kind));
                self.stats.barrier_holds += 1;
                return;
            }
        }
        self.issue_ready(stream, turn, kind, delay_ms);
    }

    /// Sends a turn whose predecessors have completed, unless the duration is over. A refused child
    /// can never continue, so its joins are released (`on_child_stopped`).
    fn issue_ready(&mut self, stream: u32, turn: u32, kind: Kind, delay_ms: Option<f64>) {
        if self.sending_open() {
            self.issue(stream, turn, false, delay_ms);
        } else if kind.is_child() {
            self.child_done(stream);
        }
    }

    fn issue(&mut self, stream: u32, turn: u32, warmup: bool, delay_ms: Option<f64>) {
        let (play_id, conv, depth) = {
            let s = &self.streams[stream as usize];
            (s.play, s.conv, s.depth)
        };
        let play = &self.plays[play_id as usize];
        let plan = play.plan.clone();
        let t = &plan.convs[conv as usize].turns[turn as usize];
        let scale = self.cfg.token_scale.max(1);
        let tokens = Tokens::new(plan.block_size * scale, self.cfg.block_tokens);
        let model = if self.cfg.model_namespaces {
            t.model
        } else {
            0
        };
        let prompt_tokens = tokens.prompt_tokens(t.hash_blocks);
        let mut prompt =
            Vec::with_capacity((prompt_tokens / self.cfg.block_tokens as u64) as usize);
        plan.prefixes.engine_blocks(
            t.prompt_end,
            tokens,
            play.namespace,
            model,
            &mut self.scratch,
            &mut prompt,
        );
        let output_tokens = if warmup {
            1
        } else {
            t.max_tokens as u64 * scale as u64
        };
        self.set_context(stream, &prompt);
        let id = self.next_request;
        self.next_request += 1;
        self.idle_since = None;
        self.in_flight.insert(
            id,
            InFlight {
                stream,
                turn,
                warmup,
                issued: self.now,
                admitted: 0,
                prompt: prompt.clone(),
                prompt_tokens,
                output_tokens,
                delay_ms,
                admission: None,
                recomputed: 0,
                timeline: usize::MAX,
            },
        );
        if let Some(adm) = self.engine.submit(
            id,
            EngineRequest {
                prompt,
                prompt_tokens,
                output_tokens,
            },
        ) {
            self.admitted(id, adm);
        }
        if !warmup && depth == 0 {
            self.on_root_issued(play_id, turn);
        }
    }

    fn admitted(&mut self, id: u64, adm: Admission) {
        let now = self.now;
        let f = self
            .in_flight
            .get_mut(&id)
            .expect("admitted request is in flight");
        f.admitted = now;
        f.admission = Some(adm);
        f.recomputed =
            recomputed_tokens(&mut self.seen, &f.prompt, adm, self.cfg.block_tokens as u64);
        let done = now + self.cost.service(adm.sources, f.output_tokens).as_nanos() as u64;
        if let Some(tl) = &mut self.timeline {
            let stream = &self.streams[f.stream as usize];
            let play = &self.plays[stream.play as usize];
            let model = play.plan.convs[stream.conv as usize].turns[f.turn as usize].model;
            f.timeline = tl.requests.len();
            tl.events.push(TimelineEvent::Admit(tl.requests.len()));
            tl.requests.push(TimelineRequest {
                prompt: f.prompt.clone(),
                prompt_tokens: f.prompt_tokens,
                output_tokens: f.output_tokens,
                record: usize::MAX,
                admitted_ns: now,
                warmup: f.warmup,
                trace: play.trace,
                conv: stream.conv,
                turn: f.turn,
                play: play.namespace,
                model: if self.cfg.model_namespaces { model } else { 0 },
            });
        }
        self.push_fixed(done, Event::Complete(id));
    }

    fn complete(&mut self, id: u64) -> Result<(), String> {
        let f = self
            .in_flight
            .remove(&id)
            .expect("completion of a request in flight");
        if let Some(tl) = &mut self.timeline {
            tl.events.push(TimelineEvent::Finish(f.timeline));
        }
        for (qid, adm) in self.engine.finish(id) {
            self.admitted(qid, adm);
        }
        let s = &self.streams[f.stream as usize];
        let plan = self.plays[s.play as usize].plan.clone();
        let conv = &plan.convs[s.conv as usize];
        let adm = f.admission.expect("a completed request was admitted");
        let counted =
            !f.warmup && self.phase == Phase::Profiling && f.issued >= self.profiling_start;
        if counted {
            if let Some(tl) = &mut self.timeline {
                tl.requests[f.timeline].record = self.records.len();
            }
            let ttft = f.admitted - f.issued + self.cost.ttft(adm.sources).as_nanos() as u64;
            self.records.push(RequestRecord {
                trace: self.plays[s.play as usize].trace,
                conv: s.conv,
                turn: f.turn,
                kind: conv.kind,
                warmup: false,
                prompt_tokens: f.prompt_tokens,
                output_tokens: f.output_tokens,
                sources: adm.sources,
                recomputed: f.recomputed,
                delay_ms: f.delay_ms,
                issued_ns: f.issued - self.profiling_start,
                admitted_ns: f.admitted - self.profiling_start,
                done_ns: self.now - self.profiling_start,
                ttft_ns: ttft,
            });
        }
        if f.warmup {
            self.warmup_outstanding -= 1;
            if self.warmup_outstanding == 0 {
                self.start_profiling()?;
            }
            return Ok(());
        }
        self.credit_return(f.stream, f.turn);
        self.enforce_idle_cap(true);
        Ok(())
    }

    /// The return path of one request (`CreditCallbackHandler.on_credit_return`).
    fn credit_return(&mut self, stream: u32, turn: u32) {
        let (play, conv, depth) = {
            let s = &self.streams[stream as usize];
            (s.play, s.conv, s.depth)
        };
        let plan = self.plays[play as usize].plan.clone();
        let is_final = turn as usize + 1 == plan.convs[conv as usize].turns.len();
        if depth > 0 && is_final {
            self.child_done(stream);
        }
        self.plays[play as usize].completed.insert((conv, turn));
        self.release_held(play);
        let mut intercepted = false;
        if depth == 0 {
            let spawns = plan.convs[conv as usize].turns[turn as usize]
                .spawns
                .clone();
            if !spawns.is_empty() {
                self.spawn_children(play, &spawns, None);
            }
            intercepted = self.suspend_if_gated(play, turn + 1);
            if is_final {
                self.plays[play as usize].root_pending = false;
                self.maybe_drain(play);
            }
        }
        if intercepted || is_final {
            return;
        }
        if !self.sending_open() {
            if depth > 0 {
                self.child_done(stream);
            }
            return;
        }
        let next = turn + 1;
        let delay = plan.convs[conv as usize].turns[next as usize].delay_ms;
        let kind = if depth == 0 {
            Kind::RootNext
        } else {
            Kind::ChildNext
        };
        match delay {
            Some(d) if d > 0.0 => {
                let at = self.now + ns(d);
                self.push_timer(
                    at,
                    Event::Dispatch {
                        stream,
                        turn: next,
                        kind,
                        delay_ms: Some(d),
                    },
                );
            }
            _ => self.submit(stream, next, kind, delay),
        }
    }

    /// `_maybe_suspend_parent`: whether the root must wait for children before `next`.
    fn suspend_if_gated(&mut self, play: u32, next: u32) -> bool {
        let p = &mut self.plays[play as usize];
        if let Some(active) = &p.active_join
            && active.turn == next
            && !active.satisfied()
        {
            return true;
        }
        match p.future_joins.get(&next) {
            None => false,
            Some(g) if g.satisfied() => {
                p.future_joins.remove(&next);
                false
            }
            Some(_) => {
                p.active_join = p.future_joins.remove(&next);
                true
            }
        }
    }

    /// A child stream finished (or can never continue): satisfy its gates and drain.
    fn child_done(&mut self, stream: u32) {
        let s = &mut self.streams[stream as usize];
        if s.done {
            return;
        }
        s.done = true;
        let play = s.play;
        let joins = std::mem::take(&mut s.joins);
        self.clear_context(stream);
        for (gated, branch) in joins {
            if let Some(root) = self.satisfy(play, gated, branch) {
                self.submit(root, gated, Kind::Join, None);
            }
        }
        let p = &mut self.plays[play as usize];
        p.descendants = p.descendants.saturating_sub(1);
        self.maybe_drain(play);
    }

    /// `_satisfy_prerequisite`: counts one finished child against the join on `gated`. Returns the
    /// root stream when the root was blocked on that join and may now send it.
    fn satisfy(&mut self, play: u32, gated: u32, branch: u32) -> Option<u32> {
        let p = &mut self.plays[play as usize];
        let active = p.active_join.as_ref().is_some_and(|g| g.turn == gated);
        let gate = if active {
            p.active_join.as_mut()?
        } else {
            p.future_joins.get_mut(&gated)?
        };
        gate.prereq(branch)?.completed += 1;
        if !gate.satisfied() {
            return None;
        }
        if active {
            p.active_join = None;
            p.root
        } else {
            p.future_joins.remove(&gated);
            None
        }
    }

    fn maybe_drain(&mut self, play: u32) {
        let p = &mut self.plays[play as usize];
        if p.drained || p.root_pending || p.descendants > 0 {
            return;
        }
        p.drained = true;
        p.completed.clear();
        p.held.clear();
        let lane = p.lane;
        if let Some(root) = p.root {
            self.clear_context(root);
        }
        self.push_timer(self.now, Event::Recycle(lane));
    }

    fn recycle(&mut self, lane: usize) -> Result<(), String> {
        if !self.can_start_session() {
            return Ok(());
        }
        let (trace, plan) = self.next_trace()?;
        let play = self.new_play(trace, plan, lane)?;
        let root = self.new_stream(play, 0, 0);
        let p = &mut self.plays[play as usize];
        p.root = Some(root);
        p.root_pending = true;
        self.submit(root, 0, Kind::SessionStart, None);
        Ok(())
    }

    /// `on_credit_issued`: a root turn whose recorded spawn overlapped the request starts those
    /// branches now, offset from the turn's recorded start.
    fn on_root_issued(&mut self, play: u32, turn: u32) {
        let plan = self.plays[play as usize].plan.clone();
        let t = &plan.root().turns[turn as usize];
        let Some(api) = t.api_ms.filter(|a| *a > 0.0) else {
            return;
        };
        let end = t.ts_ms + api;
        let overlapping: Vec<u32> = t
            .spawns
            .iter()
            .copied()
            .filter(|&b| Self::branch_start(&plan, b).is_some_and(|s| s < end))
            .collect();
        if overlapping.is_empty() {
            return;
        }
        self.spawn_children(play, &overlapping, Some(t.ts_ms));
        self.plays[play as usize]
            .overlap_spawned
            .extend(overlapping);
    }

    fn release_held(&mut self, play: u32) {
        let p = &self.plays[play as usize];
        if p.held.is_empty() {
            return;
        }
        let ready: Vec<(u32, u32, u32)> = p
            .held
            .iter()
            .filter(|&(&(_, turn, conv), _)| {
                p.plan.convs[conv as usize].turns[turn as usize]
                    .preds
                    .iter()
                    .all(|x| p.completed.contains(x))
            })
            .map(|(k, _)| *k)
            .collect();
        for key in ready {
            let (stream, kind) = self.plays[play as usize]
                .held
                .remove(&key)
                .expect("collected above");
            self.issue_ready(stream, key.1, kind, None);
        }
    }

    /// `enforce_system_idle_cap`: with nothing in flight, pull every pending scheduler timer forward
    /// by the same amount so the next one fires within the cap.
    fn enforce_idle_cap(&mut self, follows_return: bool) {
        let Some(cap) = self.cfg.system_idle_cap_s else {
            return;
        };
        if !self.in_flight.is_empty() {
            self.idle_since = None;
            return;
        }
        if follows_return || self.idle_since.is_none() {
            self.idle_since = Some(self.now);
        }
        let elapsed = self.now - self.idle_since.expect("set above");
        let remaining = ns(cap * 1e3).saturating_sub(elapsed);
        if let Some(e) = self.timers.peek() {
            let earliest = e.at - self.shift;
            let limit = self.now + remaining;
            if earliest > limit {
                let jump = earliest - limit;
                self.shift += jump;
                self.stats.idle_cap_jumps += 1;
                self.stats.idle_cap_skipped_s += jump as f64 / 1e9;
            }
        }
    }

    /// Replaces a stream's context (its latest prompt) in the live working set.
    fn set_context(&mut self, stream: u32, prompt: &[BlockId]) {
        for &b in prompt {
            *self.working_set.entry(b).or_insert(0) += 1;
        }
        self.clear_context(stream);
        self.streams[stream as usize].context = prompt.to_vec();
        if self.phase == Phase::Profiling {
            let tokens = self.working_set.len() as u64 * self.cfg.block_tokens as u64;
            self.stats.working_set_peak_tokens = self.stats.working_set_peak_tokens.max(tokens);
        }
    }

    fn clear_context(&mut self, stream: u32) {
        for b in std::mem::take(&mut self.streams[stream as usize].context) {
            if let Some(n) = self.working_set.get_mut(&b) {
                *n -= 1;
                if *n == 0 {
                    self.working_set.remove(&b);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aiperf::PlanOptions;
    use crate::engine::{CpuPolicyKind, EngineConfig, GpuPolicyKind};
    use crate::trace::{Trace, TraceSource};

    const S: u64 = 1_000_000_000;

    fn run(traces: &[&str], cfg: ReplayConfig) -> ReplayOutput {
        let traces = traces
            .iter()
            .map(|t| Trace::from_json(t.as_bytes()).unwrap())
            .collect();
        let mut corpus = Corpus::new(TraceSource::from_traces(traces), PlanOptions::default());
        let engine = Engine::new(
            &EngineConfig {
                gpu_blocks: 4096,
                cpu_blocks: 0,
                block_tokens: 16,
                gpu_policy: GpuPolicyKind::Lru,
                cpu_policy: CpuPolicyKind::Lru,
            },
            None,
        );
        Replay::new(cfg, &mut corpus, engine, TierCost::frozen_box(), false)
            .run()
            .unwrap()
    }

    /// Lanes that start at their traces' first requests and play each trace once, with nothing
    /// pulled forward.
    fn from_start(concurrency: usize) -> ReplayConfig {
        ReplayConfig {
            concurrency,
            start_min_ratio: 0.0,
            start_max_ratio: 0.0,
            system_idle_cap_s: None,
            max_sessions: Some(concurrency as u64),
            duration_s: 3_600.0,
            grace_s: 600.0,
            ..ReplayConfig::default()
        }
    }

    fn find(out: &ReplayOutput, conv: u32, turn: u32) -> &RequestRecord {
        out.records
            .iter()
            .find(|r| (r.conv, r.turn) == (conv, turn))
            .expect("the request ran")
    }

    const PAUSES: &str = r#"{"id": "pauses", "block_size": 2, "requests": [
        {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
        {"t": 600, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "api_time": 1.0, "think_time": 599.0},
        {"t": 640, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [1, 2, 3], "think_time": 30.0}
    ]}"#;

    #[test]
    fn think_times_are_capped() {
        let out = run(&[PAUSES], from_start(1));
        assert_eq!(out.records.len(), 3);
        let (t0, t1, t2) = (find(&out, 0, 0), find(&out, 0, 1), find(&out, 0, 2));
        assert_eq!(t0.issued_ns, 0);
        assert_eq!(
            t1.issued_ns - t0.done_ns,
            120 * S,
            "599 s is capped at 120 s"
        );
        assert_eq!(t2.issued_ns - t1.done_ns, 30 * S);
        assert_eq!(
            (t1.delay_ms, t2.delay_ms),
            (Some(120_000.0), Some(30_000.0))
        );
    }

    #[test]
    fn the_idle_cap_pulls_pauses_forward() {
        let out = run(
            &[PAUSES],
            ReplayConfig {
                system_idle_cap_s: Some(10.0),
                ..from_start(1)
            },
        );
        let (t0, t1, t2) = (find(&out, 0, 0), find(&out, 0, 1), find(&out, 0, 2));
        assert_eq!(t1.issued_ns - t0.done_ns, 10 * S);
        assert_eq!(t2.issued_ns - t1.done_ns, 10 * S);
        assert_eq!(out.stats.idle_cap_jumps, 2);
        assert!((out.stats.idle_cap_skipped_s - 130.0).abs() < 1e-6);
    }

    #[test]
    fn subagents_run_together_and_the_parent_waits_for_both() {
        let trace = r#"{"id": "fanout", "block_size": 2, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 2, "out": 1, "hash_ids": [1], "api_time": 1.0},
            {"t": 2, "type": "subagent", "agent_id": "a", "duration_ms": 8000, "requests": [
                {"t": 2, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [9, 10], "api_time": 1.0},
                {"t": 6, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [9, 10, 11], "think_time": 3.0}
            ]},
            {"t": 2, "type": "subagent", "agent_id": "b", "duration_ms": 8000, "requests": [
                {"t": 2, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [20, 21], "api_time": 1.0},
                {"t": 7, "type": "n", "model": "m", "in": 6, "out": 1, "hash_ids": [20, 21, 22], "think_time": 4.0}
            ]},
            {"t": 20, "type": "n", "model": "m", "in": 4, "out": 1, "hash_ids": [1, 2], "think_time": 1.0}
        ]}"#;
        let out = run(&[trace], from_start(1));
        assert_eq!(out.records.len(), 6);
        let root0 = find(&out, 0, 0);
        let (a0, a1, b0, b1) = (
            find(&out, 1, 0),
            find(&out, 1, 1),
            find(&out, 2, 0),
            find(&out, 2, 1),
        );
        // Both children start when the spawning turn returns, and overlap.
        assert_eq!((a0.issued_ns, b0.issued_ns), (root0.done_ns, root0.done_ns));
        assert!(a0.issued_ns < b0.done_ns && b0.issued_ns < a0.done_ns);
        assert_eq!(
            (a1.issued_ns - a0.done_ns, b1.issued_ns - b0.done_ns),
            (3 * S, 4 * S)
        );
        assert_eq!(a0.kind, ConvKind::Subagent);
        // The parent's next turn joins both children: it goes when the last one finishes.
        assert_eq!(find(&out, 0, 1).issued_ns, a1.done_ns.max(b1.done_ns));
    }

    #[test]
    fn a_drained_lane_recycles_under_a_fresh_namespace() {
        let ids: Vec<String> = (1..=16).map(|i| i.to_string()).collect();
        let trace = format!(
            r#"{{"id": "loop", "block_size": 2, "requests": [
                {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [{}], "api_time": 1.0}},
                {{"t": 5, "type": "n", "model": "m", "in": 32, "out": 1, "hash_ids": [{}], "think_time": 1.0}}
            ]}}"#,
            ids[..8].join(","),
            ids.join(",")
        );
        let out = run(
            &[&trace],
            ReplayConfig {
                max_sessions: Some(3),
                ..from_start(1)
            },
        );
        assert_eq!((out.stats.sessions_started, out.records.len()), (3, 6));
        let mut last_done = 0;
        for play in out.records.chunks(2) {
            let (first, second) = (&play[0], &play[1]);
            assert_eq!((first.turn, second.turn), (0, 1));
            // The prompt the previous play cached is computed again: its blocks are new.
            assert_eq!(
                (first.sources.gpu, first.sources.computed, first.recomputed),
                (0, 16, 0)
            );
            // The second turn hits the first turn's block within the play.
            assert_eq!(second.sources.gpu, 16);
            assert!(
                first.issued_ns >= last_done,
                "a lane recycles only after its tree drains"
            );
            last_done = second.done_ns;
        }
    }
}
