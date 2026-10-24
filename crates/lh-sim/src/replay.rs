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
//! The engine serves requests in one of two ways ([`Service`]). Under contention, a request reaches
//! vLLM's step loop ([`crate::scheduler`]) after its host time and shares every step with the
//! requests running beside it. Idle (v0), its time to first token is E0's cost for its hit pattern
//! after any wait for free blocks, then it decodes at a constant inter-token latency, with no
//! queueing behind other requests.
//!
//! A session pauses between turns for the trace's delay, or for a TraceLab tool wait in its place
//! ([`Pauses`]); a [`Retention`] arm acts on its KV meanwhile ([`crate::retention`]).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::rc::Rc;

use lh_core::{PromptSources, StepCost, TierCost};
use serde::Serialize;

use crate::aiperf::{ConvKind, ReplayPlan, snapshot};
use crate::corpus::Corpus;
use crate::engine::{Admission, Engine, EngineRequest};
use crate::pauses::ToolPauses;
use crate::prefix::{BlockId, IdMap, IdSet, MAX_PLAYS, NodeId, Tokens};
use crate::retention::{Interception, PauseHistory, Retention, intercept};
use crate::scheduler::{Outcome, Scheduler, StepBudget};

/// How the engine serves the requests it admits.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Service {
    /// v0: E0's idle-box cost per request ([`TierCost`]), with no queueing behind other requests.
    Idle,
    /// vLLM's step loop with the step-time model.
    Contention { budget: StepBudget, steps: StepCost },
}

impl Service {
    /// The step loop on the frozen box: its 512/256 step budget and the fitted step-time model.
    pub const fn frozen_box() -> Self {
        Self::Contention {
            budget: StepBudget::FROZEN_BOX,
            steps: StepCost::frozen_box(),
        }
    }
}

/// Where the pauses between a session's turns come from.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pauses {
    /// The traces' delays (AIPerf's think times).
    Trace,
    /// A TraceLab tool wait ([`ToolPauses`]) in place of every delay, capped at `cap_s`.
    TraceLab { cap_s: Option<f64> },
}

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
    /// Tokens per engine block.
    pub block_tokens: u32,
    /// Multiplies the tokens of every hash block and every output: 32 replays a 32x-scaled corpus at
    /// its original token counts (output lengths come back rounded up to a multiple of 32).
    pub token_scale: u32,
    pub pauses: Pauses,
    /// KV a realtime tier needs free beside the replay (E2's voice calls), in tokens: the replay
    /// measures how long the background leaves less than this of the pool free. 0 measures nothing.
    #[serde(skip_serializing_if = "is_zero")]
    pub realtime_tokens: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
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
            block_tokens: 16,
            token_scale: 1,
            pauses: Pauses::Trace,
            realtime_tokens: 0,
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
    /// When the engine took its blocks.
    pub admitted_ns: u64,
    /// How long it waited for free blocks.
    pub blocked_ns: u64,
    pub done_ns: u64,
    /// Time to first token as the client sees it.
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
    /// Index into the run's records; `None` for warmup requests and requests that did not complete
    /// in time.
    pub record: Option<usize>,
    pub warmup: bool,
    /// Where the prompt comes from: the corpus trace, its conversation and turn, and the KV
    /// namespace (the session play) its blocks were cut in.
    pub trace: u32,
    pub conv: u32,
    pub turn: u32,
    pub play: u64,
}

/// An admission or a finish of a [`Timeline`] request, by its index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineEvent {
    Admit(usize),
    Finish(usize),
}

/// How the replay itself went: lanes and plays, warmup, barriers, the idle cap, the working set.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ReplayStats {
    pub lanes: usize,
    pub sessions_started: u64,
    pub warmup_requests: u64,
    /// When profiling started, in seconds from the run's start: how long the warmup took.
    pub profiling_start_s: f64,
    /// Requests sent during profiling but still running when the grace period ended.
    pub cut_off_requests: u64,
    /// Turns held by replay barriers at some point.
    pub barrier_holds: u64,
    /// The idle cap's jumps, and the idle time they skipped, until sending closes.
    pub idle_cap_jumps: u64,
    pub idle_cap_skipped_s: f64,
    /// Time-weighted mean and peak of the tokens in the latest prompt of every live stream (distinct
    /// full blocks), over the profiling window.
    pub working_set_mean_tokens: f64,
    pub working_set_peak_tokens: u64,
    /// The share of the profiling window in which the live working set left less than
    /// [`ReplayConfig::realtime_tokens`] of the pool free: the realtime tier's KV then displaces
    /// cached background KV, which vLLM evicts for a request that needs blocks. `None`, and left out,
    /// when nothing is measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime_short_share: Option<f64>,
    /// The share in which the blocks running requests pin left less than that free: a realtime
    /// request then waits for blocks, as vLLM preempts no running request for a waiting one (#40004).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime_blocked_share: Option<f64>,
}

/// What one replay produced.
#[derive(Debug)]
pub struct ReplayOutput {
    pub records: Vec<RequestRecord>,
    pub stats: ReplayStats,
    pub engine: crate::engine::EngineStats,
    pub timeline: Option<Timeline>,
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
    /// Idle service: a request's last token.
    Complete(u64),
    /// Contention: a request reaches the scheduler after its host time.
    Arrive(u64),
    /// Contention: the step loop runs (a step ends, or a restore finishes on an idle engine).
    Step,
    Recycle(usize),
    /// A retention timer of pause `pause` of `stream`.
    Retain {
        stream: u32,
        pause: u64,
        step: RetainStep,
    },
}

#[derive(Clone, Copy, Debug)]
enum RetainStep {
    /// The pause's hold ends: a TTL, or the session is late.
    Release,
    /// Start loading the session's blocks from the CPU tier.
    Prefetch,
    /// The load has ended: the blocks are in the pool, held for the session.
    Prefetched,
}

/// A session between its turns, until its next turn is admitted.
#[derive(Debug)]
struct Paused {
    id: u64,
    since: u64,
    /// The tool it waits for, when the pauses name one.
    tool: Option<u16>,
    /// The engine's hold on its KV.
    hold: Option<u64>,
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

/// One conversation of a play as it is replayed: the root, or a child spawned by a branch.
#[derive(Debug)]
struct Stream {
    play: u32,
    conv: u32,
    /// 0 for the root, 1 for a child.
    depth: u8,
    /// `(gated root turn, branch)` this child must satisfy; empty for background children.
    joins: Vec<(u32, u32)>,
    /// A child that finished, or that can never continue once sending stops.
    done: bool,
    /// The blocks of its latest prompt, counted in the live working set.
    context: Vec<BlockId>,
    /// Between turns, with a retention arm.
    paused: Option<Paused>,
}

/// One branch a gated root turn waits for.
#[derive(Debug)]
struct Prereq {
    branch: u32,
    /// Children spawned so far.
    expected: u32,
    /// Children finished so far.
    completed: u32,
    /// The branch has been spawned, so `expected` is final.
    registered: bool,
}

/// A root turn that waits for the children of the branches it joins (AIPerf's SPAWN_JOIN).
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

/// One session play: a trace replayed by a lane, under its own KV namespace.
#[derive(Debug)]
struct Play {
    plan: Rc<ReplayPlan>,
    /// The trace's index in the corpus.
    trace: u32,
    lane: usize,
    /// The KV namespace its blocks are cut in: the play's index.
    namespace: u64,
    root: Option<u32>,
    /// The root has turns left to send.
    root_pending: bool,
    /// Child streams not yet done.
    descendants: u32,
    /// The session tree has finished and its lane has moved on.
    drained: bool,
    /// `(conversation, turn)` pairs completed: the replay barriers' history.
    completed: HashSet<(u32, u32)>,
    /// Turns held by replay barriers, in AIPerf's `(conversation id, turn)` order, with their stream and
    /// kind.
    held: BTreeMap<(u32, u32, u32), (u32, Kind)>,
    /// Gates of later root turns.
    future_joins: BTreeMap<u32, Gate>,
    /// The gate the root is blocked on now.
    active_join: Option<Gate>,
    /// Branches spawned when their spawning turn was sent, so its return does not spawn them again.
    overlap_spawned: Vec<u32>,
}

/// A request on its way to the engine, in it, or waiting for its blocks.
#[derive(Debug)]
struct InFlight {
    stream: u32,
    turn: u32,
    warmup: bool,
    issued: u64,
    admitted: u64,
    blocked: u64,
    /// When its first token came out, under contention.
    first_token: u64,
    prompt: Vec<BlockId>,
    prompt_tokens: u64,
    output_tokens: u64,
    delay_ms: Option<f64>,
    admission: Option<Admission>,
    recomputed: u64,
    /// Its index in the timeline, once admitted into a run that records one.
    timeline: Option<usize>,
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
    /// The step loop under contention; `None` for the idle service.
    scheduler: Option<Scheduler>,
    /// When the step loop runs next: a `Step` event due at another time is stale.
    step_at: Option<u64>,
    /// A reused buffer for the step loop's outcomes.
    outcomes: Vec<Outcome>,
    retention: Retention,
    /// The share of a restore for which the step loop stalls: the step model's, 1 when idle.
    restore_stall: f64,
    /// Every session's past pauses: the history information level.
    history: PauseHistory,
    next_pause: u64,
    now: u64,
    /// Insertion counter: breaks ties between events due at the same time.
    seq: u64,
    /// How far the idle cap has pulled `timers` and `following` forward: an entry stored at `at` fires
    /// at `at - shift`.
    shift: u64,
    /// The scheduler's timers (spread warmup, first dispatches, think times, recycling), which the idle
    /// cap pulls forward.
    timers: BinaryHeap<Entry>,
    /// Timers the idle cap pulls forward with the scheduler's but leaves out of its test for the next
    /// one: the oracle's prefetches, which follow the dispatch they foresee.
    following: BinaryHeap<Entry>,
    /// Events the idle cap leaves in place: request arrivals, steps and completions, children's first
    /// turns, and the other retention timers, on the engine's clock.
    fixed: BinaryHeap<Entry>,
    phase: Phase,
    profiling_start: u64,
    profiling_end: u64,
    grace_end: u64,
    plays: Vec<Play>,
    streams: Vec<Stream>,
    lanes: Vec<Lane>,
    /// Draws so far from the sequential trace sampler.
    sampler: usize,
    in_flight: IdMap<InFlight>,
    warmup_outstanding: usize,
    /// The start of the current idle gap (nothing in flight since the last return), for the idle cap.
    idle_since: Option<u64>,
    next_request: u64,
    records: Vec<RequestRecord>,
    timeline: Option<Timeline>,
    /// Every block an admitted prompt has had, for attributing recompute.
    seen: IdSet,
    /// The blocks of the live streams' latest prompts, each with the number of streams holding it.
    working_set: IdMap<u32>,
    /// Where `ws_integral` has reached.
    ws_last_change: u64,
    /// The working set's size integrated over the profiling window, in blocks × ns.
    ws_integral: f64,
    /// [`ReplayConfig::realtime_tokens`] in blocks, and the profiling time (ns) in which the working
    /// set, and the pinned blocks, left fewer than that free.
    realtime_blocks: usize,
    realtime_short_ns: u64,
    realtime_blocked_ns: u64,
    stats: ReplayStats,
    /// A reused buffer for prefix-tree paths.
    scratch: Vec<NodeId>,
}

impl<'a> Replay<'a> {
    /// A replay of `corpus` through `engine` with `retention` acting on pausing sessions; with
    /// `record`, the output keeps the engine's timeline.
    pub fn new(
        cfg: ReplayConfig,
        corpus: &'a mut Corpus,
        engine: Engine,
        cost: TierCost,
        service: Service,
        retention: Retention,
        record: bool,
    ) -> Self {
        let (scheduler, restore_stall) = match service {
            Service::Idle => (None, 1.0),
            Service::Contention { budget, steps } => {
                let stall = steps.restore_stall;
                (Some(Scheduler::new(budget, steps, cost.clone())), stall)
            }
        };
        let realtime_blocks = cfg.realtime_tokens.div_ceil(cfg.block_tokens as u64) as usize;
        Self {
            cfg,
            corpus,
            engine,
            cost,
            scheduler,
            step_at: None,
            outcomes: Vec::new(),
            retention,
            restore_stall,
            history: PauseHistory::default(),
            next_pause: 0,
            now: 0,
            seq: 0,
            shift: 0,
            timers: BinaryHeap::new(),
            following: BinaryHeap::new(),
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
            realtime_blocks,
            realtime_short_ns: 0,
            realtime_blocked_ns: 0,
            stats: ReplayStats::default(),
            scratch: Vec::new(),
        }
    }

    /// Runs the replay to the end of its grace period, or until it runs out of work.
    pub fn run(mut self) -> Result<ReplayOutput, String> {
        self.start_lanes()?;
        loop {
            let (now, shift) = (self.now, self.shift);
            let heads = [
                self.timers.peek().map(|e| (e.at - shift, e.seq)),
                // A prefetch the cap pulls into the past starts at once.
                self.following
                    .peek()
                    .map(|e| (e.at.saturating_sub(shift).max(now), e.seq)),
                self.fixed.peek().map(|e| (e.at, e.seq)),
            ];
            let Some((heap, (at, _))) = heads
                .into_iter()
                .enumerate()
                .filter_map(|(i, head)| Some((i, head?)))
                .min_by_key(|&(_, head)| head)
            else {
                break;
            };
            let from_timers = heap == 0;
            let entry = match heap {
                0 => self.timers.pop(),
                1 => self.following.pop(),
                _ => self.fixed.pop(),
            }
            .expect("peeked");
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
                Event::Arrive(id) => self.arrive(id)?,
                Event::Step if self.step_at == Some(at) => self.run_steps()?,
                Event::Step => {}
                Event::Recycle(lane) => self.recycle(lane)?,
                Event::Retain {
                    stream,
                    pause,
                    step,
                } => self.retain(stream, pause, step)?,
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
        // The working-set mean covers the duration, or the part of it before the run ran out of work;
        // `advance` has integrated up to the last event.
        let window_end = self.now.min(self.profiling_end);
        self.stats.cut_off_requests = self
            .in_flight
            .values()
            .filter(|f| !f.warmup && f.issued >= self.profiling_start)
            .count() as u64;
        let window = window_end.saturating_sub(self.profiling_start).max(1) as f64;
        self.stats.working_set_mean_tokens =
            self.ws_integral / window * self.cfg.block_tokens as f64;
        if self.realtime_blocks > 0 {
            self.stats.realtime_short_share = Some(self.realtime_short_ns as f64 / window);
            self.stats.realtime_blocked_share = Some(self.realtime_blocked_ns as f64 / window);
        }
        self.stats.lanes = self.lanes.len();
        Ok(ReplayOutput {
            records: self.records,
            stats: self.stats,
            engine: self.engine.stats,
            timeline: self.timeline,
        })
    }

    /// Moves the clock to `at`, integrating the live working set, and the time it and the pinned
    /// blocks leave the realtime tier short, over the profiling duration.
    fn advance(&mut self, at: u64) {
        if self.phase == Phase::Profiling && at > self.ws_last_change {
            let from = self.ws_last_change.max(self.profiling_start);
            let to = at.min(self.profiling_end);
            if to > from {
                self.ws_integral += self.working_set.len() as f64 * (to - from) as f64;
                if self.realtime_blocks > 0 {
                    let pool = self.engine.gpu_capacity();
                    if self.working_set.len() + self.realtime_blocks > pool {
                        self.realtime_short_ns += to - from;
                    }
                    if self.engine.pinned_blocks() + self.realtime_blocks > pool {
                        self.realtime_blocked_ns += to - from;
                    }
                }
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

    fn push_following(&mut self, at: u64, event: Event) {
        self.seq += 1;
        self.following.push(Entry {
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
            paused: None,
        });
        self.streams.len() as u32 - 1
    }

    /// Builds the lanes (`TrajectorySource`) and sends the warmup.
    fn start_lanes(&mut self) -> Result<(), String> {
        for lane in 0..self.cfg.concurrency {
            let (trace, plan) = self.next_trace()?;
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
            let base = origin.unwrap_or(branch.start_ms);
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
                let first = plan.convs[child as usize].turns.first();
                let offset = first.map_or(0.0, |t| (t.ts_ms - base).max(0.0));
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
        let prompt_tokens = tokens.prompt_tokens(t.hash_blocks);
        let mut prompt =
            Vec::with_capacity((prompt_tokens / self.cfg.block_tokens as u64) as usize);
        plan.prefixes.engine_blocks(
            t.prompt_end,
            tokens,
            play.namespace,
            &mut self.scratch,
            &mut prompt,
        );
        let output_tokens = if warmup {
            1
        } else {
            t.max_tokens as u64 * scale as u64
        };
        self.pause_ends(stream);
        self.set_context(stream, &prompt);
        let id = self.next_request;
        self.next_request += 1;
        self.idle_since = None;
        let req = EngineRequest {
            prompt: prompt.clone(),
            prompt_tokens,
            output_tokens,
        };
        self.in_flight.insert(
            id,
            InFlight {
                stream,
                turn,
                warmup,
                issued: self.now,
                admitted: 0,
                blocked: 0,
                first_token: 0,
                prompt,
                prompt_tokens,
                output_tokens,
                delay_ms,
                admission: None,
                recomputed: 0,
                timeline: None,
            },
        );
        match &mut self.scheduler {
            None => {
                if let Some(adm) = self.engine.submit(id, req) {
                    self.admitted(id, adm, 0);
                }
            }
            Some(s) => {
                if !self.engine.reject_oversized(&req) {
                    let at = self.now + s.host_ns(prompt_tokens, self.now);
                    self.push_fixed(at, Event::Arrive(id));
                }
            }
        }
        if !warmup && depth == 0 {
            self.on_root_issued(play_id, turn);
        }
    }

    /// Contention: request `id` reaches the scheduler, which runs at once if it is idle.
    fn arrive(&mut self, id: u64) -> Result<(), String> {
        let f = &self.in_flight[&id];
        let req = EngineRequest {
            prompt: f.prompt.clone(),
            prompt_tokens: f.prompt_tokens,
            output_tokens: f.output_tokens,
        };
        let s = self
            .scheduler
            .as_mut()
            .expect("arrivals happen under contention");
        s.arrive(id, req);
        if s.busy() {
            return Ok(());
        }
        self.run_steps()
    }

    /// Contention: runs the step loop now and applies what happened.
    fn run_steps(&mut self) -> Result<(), String> {
        let mut out = std::mem::take(&mut self.outcomes);
        let next = self
            .scheduler
            .as_mut()
            .expect("steps run under contention")
            .run(self.now, &mut self.engine, &mut out);
        for outcome in out.drain(..) {
            match outcome {
                Outcome::Admitted {
                    id,
                    admission,
                    blocked_ns,
                } => self.admitted(id, admission, blocked_ns),
                Outcome::FirstToken(id) => {
                    self.in_flight
                        .get_mut(&id)
                        .expect("a request with a first token is in flight")
                        .first_token = self.now;
                }
                Outcome::Finished(id) => self.complete(id)?,
            }
        }
        self.outcomes = out;
        self.step_at = next;
        if let Some(at) = next {
            self.push_fixed(at, Event::Step);
        }
        Ok(())
    }

    fn admitted(&mut self, id: u64, adm: Admission, blocked: u64) {
        let now = self.now;
        let f = self
            .in_flight
            .get_mut(&id)
            .expect("admitted request is in flight");
        f.admitted = now;
        f.blocked = blocked;
        f.admission = Some(adm);
        f.recomputed =
            recomputed_tokens(&mut self.seen, &f.prompt, adm, self.cfg.block_tokens as u64);
        let done = self
            .scheduler
            .is_none()
            .then(|| now + self.cost.service(adm.sources, f.output_tokens).as_nanos() as u64);
        if let Some(tl) = &mut self.timeline {
            let stream = &self.streams[f.stream as usize];
            let play = &self.plays[stream.play as usize];
            f.timeline = Some(tl.requests.len());
            tl.events.push(TimelineEvent::Admit(tl.requests.len()));
            tl.requests.push(TimelineRequest {
                prompt: f.prompt.clone(),
                prompt_tokens: f.prompt_tokens,
                output_tokens: f.output_tokens,
                record: None,
                warmup: f.warmup,
                trace: play.trace,
                conv: stream.conv,
                turn: f.turn,
                play: play.namespace,
            });
        }
        let (stream, turn) = (f.stream, f.turn);
        if let Some(done) = done {
            self.push_fixed(done, Event::Complete(id));
        }
        if self.retention == Retention::Mori
            && let Some(pause) = self.predicted_pause(stream, turn + 1)
        {
            self.engine.expect_pause(id, ns(pause));
        }
        // Its hits are pinned now; the pause's hold can go.
        if let Some(hold) = self.streams[stream as usize]
            .paused
            .take()
            .and_then(|p| p.hold)
        {
            self.release(hold);
        }
    }

    /// Request `id` returned its last token. Under contention the scheduler has already finished it
    /// in the engine.
    fn complete(&mut self, id: u64) -> Result<(), String> {
        let f = self
            .in_flight
            .remove(&id)
            .expect("completion of a request in flight");
        if let (Some(tl), Some(i)) = (&mut self.timeline, f.timeline) {
            tl.events.push(TimelineEvent::Finish(i));
        }
        if self.scheduler.is_none() {
            // The idle engine queues a request only while it waits for blocks.
            for (qid, adm) in self.engine.finish(id, self.now) {
                let blocked = self.now - self.in_flight[&qid].issued;
                self.admitted(qid, adm, blocked);
            }
        }
        let s = &self.streams[f.stream as usize];
        let plan = self.plays[s.play as usize].plan.clone();
        let conv = &plan.convs[s.conv as usize];
        let adm = f.admission.expect("a completed request was admitted");
        let counted =
            !f.warmup && self.phase == Phase::Profiling && f.issued >= self.profiling_start;
        if counted {
            if let (Some(tl), Some(i)) = (&mut self.timeline, f.timeline) {
                tl.requests[i].record = Some(self.records.len());
            }
            let ttft = match self.scheduler {
                None => f.admitted - f.issued + self.cost.ttft(adm.sources).as_nanos() as u64,
                Some(_) => f.first_token - f.issued,
            };
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
                blocked_ns: f.blocked,
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
        let (delay, tool) = self.delay(stream, next);
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
                self.pause(stream, at, tool);
            }
            _ => self.submit(stream, next, kind, delay),
        }
    }

    /// The delay before turn `turn` of `stream` and the tool it waits for: the trace's, or a
    /// TraceLab draw in its place (the same for every arm).
    fn delay(&self, stream: u32, turn: u32) -> (Option<f64>, Option<u16>) {
        let s = &self.streams[stream as usize];
        let play = &self.plays[s.play as usize];
        let delay = play.plan.convs[s.conv as usize].turns[turn as usize].delay_ms;
        match (self.cfg.pauses, delay) {
            (Pauses::TraceLab { cap_s }, Some(d)) if d > 0.0 => {
                let key = self.cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (u64::from(play.trace) << 40)
                    ^ (u64::from(s.conv) << 20)
                    ^ u64::from(turn);
                let (tool, ms) = ToolPauses::tracelab().draw(key);
                (Some(cap_s.map_or(ms, |cap| ms.min(cap * 1e3))), Some(tool))
            }
            _ => (delay, None),
        }
    }

    /// What the history level predicts for the pause before turn `turn` of `stream`, in ms.
    fn predicted_pause(&self, stream: u32, turn: u32) -> Option<f64> {
        let s = &self.streams[stream as usize];
        let turns = self.plays[s.play as usize].plan.convs[s.conv as usize]
            .turns
            .len();
        let tool = if (turn as usize) < turns {
            self.delay(stream, turn).1
        } else {
            None
        };
        self.history.predict(stream, tool)
    }

    /// `stream` pauses until its next turn, due at `at`, after calling `tool`: the retention arm
    /// acts on its KV, its latest prompt.
    fn pause(&mut self, stream: u32, at: u64, tool: Option<u16>) {
        // A turn that was never admitted (larger than the pool) leaves its pause behind.
        if let Some(hold) = self.streams[stream as usize]
            .paused
            .take()
            .and_then(|p| p.hold)
        {
            self.release(hold);
        }
        self.next_pause += 1;
        let id = self.next_pause;
        let context = self.streams[stream as usize].context.clone();
        let tokens = context.len() as u64 * self.cfg.block_tokens as u64;
        let predicted = self.history.predict(stream, tool).map(ns);
        let mut hold = None;
        let mut back = None;
        match self.retention {
            Retention::None | Retention::Mori => {}
            Retention::Ttl(ttl_s) => {
                hold = self.engine.hold(&context);
                let end = self.now + ns(ttl_s * 1e3);
                self.retain_at(end, stream, id, RetainStep::Release);
            }
            Retention::HistoryPrefetch => back = predicted.map(|p| self.now + p),
            Retention::OraclePrefetch => back = Some(at.max(self.now)),
            Retention::Tokencake => {
                self.engine.demote(&context);
                back = predicted.map(|p| self.now + p);
            }
            Retention::InferCept => {
                let restore = self.cost.restore_ms(tokens);
                let choice = predicted.map(|p| {
                    intercept(
                        tokens,
                        p as f64 / 1e6,
                        self.engine.running_tokens(),
                        self.restore_stall * restore,
                        self.cost.recompute_ms(tokens),
                        self.engine.has_cpu_tier(),
                    )
                });
                if choice == Some(Interception::Preserve) {
                    hold = self.engine.hold(&context);
                    back = predicted.map(|p| self.now + p);
                } else {
                    self.engine.demote(&context);
                }
            }
        }
        if let Some(back) = back {
            if self.retention != Retention::InferCept {
                // Start the load so that it ends just before the session returns. The oracle knows
                // when the turn is sent, so its load moves with the dispatch when the idle cap pulls
                // that forward; the history level predicts the engine's time.
                let lead = ns(self.cost.restore_ms(tokens) + 1.0);
                let start = back.saturating_sub(lead).max(self.now);
                let prefetch = Event::Retain {
                    stream,
                    pause: id,
                    step: RetainStep::Prefetch,
                };
                if self.retention == Retention::OraclePrefetch {
                    self.push_following(start, prefetch);
                } else {
                    self.push_fixed(start, prefetch);
                }
            }
            // A session later than twice its expected pause is not coming back soon.
            let late = back + (back - self.now);
            self.retain_at(late, stream, id, RetainStep::Release);
        }
        self.streams[stream as usize].paused = Some(Paused {
            id,
            since: self.now,
            tool,
            hold,
        });
    }

    /// `stream` sends its next turn: the history learns how long it paused. A prefetch that lands
    /// before the turn is admitted still serves it.
    fn pause_ends(&mut self, stream: u32) {
        if let Some(p) = &self.streams[stream as usize].paused {
            let ms = (self.now - p.since) as f64 / 1e6;
            self.history.observe(stream, p.tool, ms);
        }
    }

    fn retain_at(&mut self, at: u64, stream: u32, pause: u64, step: RetainStep) {
        self.push_fixed(
            at,
            Event::Retain {
                stream,
                pause,
                step,
            },
        );
    }

    /// A retention timer; stale once the turn after its pause has been admitted.
    fn retain(&mut self, stream: u32, pause: u64, step: RetainStep) -> Result<(), String> {
        let s = stream as usize;
        if self.streams[s]
            .paused
            .as_ref()
            .is_none_or(|p| p.id != pause)
        {
            return Ok(());
        }
        match step {
            RetainStep::Release => {
                let hold = self.streams[s].paused.as_mut().and_then(|p| p.hold.take());
                if let Some(hold) = hold {
                    self.release(hold);
                    self.kick()?;
                }
            }
            RetainStep::Prefetch => {
                let context = &self.streams[s].context;
                let tokens =
                    self.engine.prefetchable(context) as u64 * self.cfg.block_tokens as u64;
                let end = match &mut self.scheduler {
                    Some(s) if tokens > 0 => s.load(self.now, tokens),
                    _ => self.now + ns(self.cost.restore_ms(tokens)),
                };
                self.retain_at(end, stream, pause, RetainStep::Prefetched);
            }
            RetainStep::Prefetched => {
                let stream = &mut self.streams[s];
                let hold = self.engine.prefetch(&stream.context);
                let p = stream.paused.as_mut().expect("checked above");
                debug_assert!(p.hold.is_none());
                p.hold = hold;
            }
        }
        Ok(())
    }

    /// Ends an engine hold; the idle engine admits what now fits.
    fn release(&mut self, hold: u64) {
        let now = self.now;
        for (qid, adm) in self.engine.release(hold) {
            let blocked = now - self.in_flight[&qid].issued;
            self.admitted(qid, adm, blocked);
        }
    }

    /// Contention: runs an idle step loop, which may now admit what waits for blocks.
    fn kick(&mut self) -> Result<(), String> {
        match &self.scheduler {
            Some(s) if !s.busy() => self.run_steps(),
            _ => Ok(()),
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
            .filter(|&b| plan.root().branches[b as usize].start_ms < end)
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

    /// `enforce_system_idle_cap`: with nothing in flight, pull every pending scheduler timer, and the
    /// timers that follow them, forward by the same amount so the next scheduler timer fires within
    /// the cap.
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
                // The grace period sends nothing, so what the cap skips then is not counted.
                if self.now < self.profiling_end {
                    self.stats.idle_cap_jumps += 1;
                    self.stats.idle_cap_skipped_s += jump as f64 / 1e9;
                }
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
        let engine = EngineConfig {
            gpu_blocks: 4096,
            cpu_blocks: 0,
            block_tokens: 16,
            gpu_policy: GpuPolicyKind::Lru,
            cpu_policy: CpuPolicyKind::Lru,
        };
        run_with(traces, cfg, &engine, Retention::None)
    }

    fn run_with(
        traces: &[&str],
        cfg: ReplayConfig,
        engine: &EngineConfig,
        retention: Retention,
    ) -> ReplayOutput {
        let traces = traces
            .iter()
            .map(|t| Trace::from_json(t.as_bytes()).unwrap())
            .collect();
        let mut corpus = Corpus::new(TraceSource::from_traces(traces), PlanOptions::default());
        Replay::new(
            cfg,
            &mut corpus,
            Engine::new(engine, None),
            TierCost::frozen_box(),
            Service::Idle,
            retention,
            false,
        )
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
    fn the_idle_cap_counters_stop_when_sending_closes() {
        const LONG: &str = r#"{"id": "long", "block_size": 2, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 2, "out": 2000, "hash_ids": [7], "api_time": 1.0}
        ]}"#;
        let capped = ReplayConfig {
            system_idle_cap_s: Some(10.0),
            ..from_start(2)
        };
        let all = run(&[PAUSES, LONG], capped.clone());
        let long = all
            .records
            .iter()
            .find(|r| r.trace == 1)
            .expect("the long request");
        // Sending closes while the long request runs and `pauses` thinks before its turn 1. When the
        // long request returns, in the grace period, the cap still pulls turn 1's dispatch forward,
        // but turn 1 is never sent: that jump is not counted.
        let close_s = long.done_ns as f64 / 2e9 - all.stats.profiling_start_s;
        let out = run(
            &[PAUSES, LONG],
            ReplayConfig {
                duration_s: close_s,
                ..capped
            },
        );
        assert_eq!(out.records.len(), 2);
        assert_eq!(
            (out.stats.idle_cap_jumps, out.stats.idle_cap_skipped_s),
            (0, 0.0)
        );
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

    /// Session A pauses 5 s twice. During its second pause session C sends a request (leaving cached
    /// blocks newer than A's), then session B one of `b_blocks` blocks, in an 11-block pool with a
    /// 64-block CPU tier. Returns A's third turn under `retention`.
    fn third_turn_after_pauses(retention: Retention, b_blocks: u64) -> RequestRecord {
        let a = r#"{"id": "a", "block_size": 16, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 64, "out": 1, "hash_ids": [1, 2, 3, 4], "api_time": 0.1},
            {"t": 5, "type": "n", "model": "m", "in": 80, "out": 1, "hash_ids": [1, 2, 3, 4, 5], "think_time": 5.0},
            {"t": 10, "type": "n", "model": "m", "in": 96, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6], "think_time": 5.0}
        ]}"#;
        let ids: Vec<String> = (101..101 + b_blocks).map(|i| i.to_string()).collect();
        let b = format!(
            r#"{{"id": "b", "block_size": 16, "requests": [
                {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [100], "api_time": 0.1}},
                {{"t": 8, "type": "n", "model": "m", "in": {}, "out": 1, "hash_ids": [{}], "think_time": 8.0}}
            ]}}"#,
            16 * b_blocks,
            ids.join(",")
        );
        let c = r#"{"id": "c", "block_size": 16, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [200], "api_time": 0.1},
            {"t": 6.5, "type": "n", "model": "m", "in": 48, "out": 1, "hash_ids": [200, 201, 202], "think_time": 6.5}
        ]}"#;
        let engine = EngineConfig {
            gpu_blocks: 12,
            cpu_blocks: 64,
            block_tokens: 16,
            gpu_policy: GpuPolicyKind::Lru,
            cpu_policy: CpuPolicyKind::Lru,
        };
        let out = run_with(&[a, &b, c], from_start(3), &engine, retention);
        assert_eq!(out.records.len(), 7);
        out.records
            .iter()
            .find(|r| (r.trace, r.turn) == (0, 2))
            .expect("A's third turn ran")
            .clone()
    }

    fn sources(r: &RequestRecord) -> (u64, u64) {
        (r.sources.gpu, r.sources.cpu)
    }

    #[test]
    fn a_ttl_hold_keeps_a_paused_session_over_newer_cached_blocks() {
        // LRU evicts A's newest block for B's 4 blocks, before C's newer ones; the hold keeps it.
        assert_eq!(
            sources(&third_turn_after_pauses(Retention::None, 4)),
            (64, 16)
        );
        assert_eq!(
            sources(&third_turn_after_pauses(Retention::Ttl(30.0), 4)),
            (80, 0)
        );
        // A hold shorter than the pause has ended by then.
        assert_eq!(
            sources(&third_turn_after_pauses(Retention::Ttl(1.0), 4)),
            (64, 16)
        );
    }

    #[test]
    fn a_prefetch_brings_an_evicted_session_back_before_it_returns() {
        // B's 11 blocks evict everything; A's blocks come back from the CPU tier.
        assert_eq!(
            sources(&third_turn_after_pauses(Retention::None, 11)),
            (0, 80)
        );
        for retention in [
            Retention::OraclePrefetch,
            // A's first pause predicts its second exactly.
            Retention::HistoryPrefetch,
            Retention::Tokencake,
        ] {
            let r = third_turn_after_pauses(retention, 11);
            assert_eq!(sources(&r), (80, 0), "{retention:?}");
        }
    }

    #[test]
    fn the_oracle_prefetch_moves_with_a_turn_the_idle_cap_pulls_forward() {
        // A pauses 60 s. B's second turn evicts A's blocks from an 11-block pool; then nothing is in
        // flight, and with the cap A's next turn goes 10 s after B's returns.
        let a = r#"{"id": "a", "block_size": 16, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 64, "out": 1, "hash_ids": [1, 2, 3, 4], "api_time": 0.1},
            {"t": 60, "type": "n", "model": "m", "in": 80, "out": 1, "hash_ids": [1, 2, 3, 4, 5], "think_time": 60.0}
        ]}"#;
        let ids: Vec<String> = (101..112).map(|i| i.to_string()).collect();
        let b = format!(
            r#"{{"id": "b", "block_size": 16, "requests": [
                {{"t": 0, "type": "n", "model": "m", "in": 16, "out": 1, "hash_ids": [100], "api_time": 0.1}},
                {{"t": 1, "type": "n", "model": "m", "in": 176, "out": 1, "hash_ids": [{}], "think_time": 1.0}}
            ]}}"#,
            ids.join(",")
        );
        let engine = EngineConfig {
            gpu_blocks: 12,
            cpu_blocks: 64,
            block_tokens: 16,
            gpu_policy: GpuPolicyKind::Lru,
            cpu_policy: CpuPolicyKind::Lru,
        };
        for cap in [None, Some(10.0)] {
            let cfg = ReplayConfig {
                system_idle_cap_s: cap,
                ..from_start(2)
            };
            for (retention, expected) in [
                (Retention::None, (0, 64)),
                (Retention::OraclePrefetch, (64, 0)),
            ] {
                let out = run_with(&[a, &b], cfg.clone(), &engine, retention);
                let turn = |trace, turn| {
                    out.records
                        .iter()
                        .find(|r| (r.trace, r.turn) == (trace, turn))
                        .expect("the turn ran")
                };
                let (a0, a1, b1) = (turn(0, 0), turn(0, 1), turn(1, 1));
                let pause = match cap {
                    None => 60 * S,
                    Some(_) => b1.done_ns + 10 * S - a0.done_ns,
                };
                assert_eq!(a1.issued_ns - a0.done_ns, pause, "{cap:?}");
                assert_eq!(sources(a1), expected, "{retention:?}, cap {cap:?}");
            }
        }
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

    #[test]
    fn the_realtime_tier_is_short_while_too_little_of_the_pool_is_free() {
        // 11 usable blocks: an 8-block prompt, a 1 s pause, then a 9-block prompt over it.
        let trace = r#"{"id": "rt", "block_size": 16, "requests": [
            {"t": 0, "type": "n", "model": "m", "in": 128, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8], "api_time": 0.1},
            {"t": 2, "type": "n", "model": "m", "in": 144, "out": 1, "hash_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "think_time": 1.0}
        ]}"#;
        let engine = EngineConfig {
            gpu_blocks: 12,
            cpu_blocks: 0,
            block_tokens: 16,
            gpu_policy: GpuPolicyKind::Lru,
            cpu_policy: CpuPolicyKind::Lru,
        };
        let run_rt = |realtime_tokens| {
            let cfg = ReplayConfig {
                realtime_tokens,
                ..from_start(1)
            };
            run_with(&[trace], cfg, &engine, Retention::None)
        };
        let share = |ns: u64, end: u64| ns as f64 / end as f64;
        // 4 blocks: the working set leaves fewer free throughout, the pinned prompt while each runs.
        let out = run_rt(64);
        let (first, second) = (&out.records[0], &out.records[1]);
        let end = second.done_ns;
        let running = first.done_ns - first.admitted_ns + second.done_ns - second.admitted_ns;
        assert_eq!(out.stats.realtime_short_share, Some(1.0));
        let blocked = out.stats.realtime_blocked_share.unwrap();
        assert!((blocked - share(running, end)).abs() < 1e-12, "{blocked}");
        // 3 blocks fit beside the 8-block prompt, pinned or not, but not beside the 9-block one.
        let out = run_rt(48);
        let second = &out.records[1];
        assert_eq!(second.done_ns, end);
        let short = out.stats.realtime_short_share.unwrap();
        assert!((short - share(end - second.issued_ns, end)).abs() < 1e-12);
        let blocked = out.stats.realtime_blocked_share.unwrap();
        assert!((blocked - share(second.done_ns - second.admitted_ns, end)).abs() < 1e-12);
        let none = run_rt(0).stats;
        assert_eq!(
            (none.realtime_short_share, none.realtime_blocked_share),
            (None, None)
        );
    }
}
