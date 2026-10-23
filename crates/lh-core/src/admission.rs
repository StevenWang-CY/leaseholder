//! Admission: whether a request goes upstream now, waits, or is refused, and which admitted requests give way
//! to it.
//!
//! An [`Admission`] holds each request from its arrival at the admission point to the end of its response, and
//! decides at three events: an arrival, a response's first byte, a departure. Its decisions are [`Action`]s on
//! held requests, which the caller carries out: the caller owns the waiting, its timeouts and the cancellation
//! of upstream requests. The decisions depend only on the order of events, so there is no clock, and a
//! request's tokens are an estimate the caller passes in ([`estimate_tokens`]).
//!
//! The first four policies are experiment E2's baselines that `lh-gateway` emulates; its README says what each
//! one reproduces of the system it stands for and what it leaves out.
//! - [`Policy::StaticCap`]: at most a fixed number of background requests in flight.
//! - [`Policy::TokenReservation`]: background requests in flight while their tokens and a reservation fit in
//!   the KV pool.
//! - [`Policy::FlowControl`]: llm-d's flow control. Each class is a priority band, dispatched strictly; each
//!   band has a holdback ceiling on the requests of all bands in flight; a blocked realtime or interactive
//!   request evicts the newest background requests.
//! - [`Policy::PreemptResubmit`]: abort-and-resubmit. A realtime arrival preempts the background requests that
//!   have no first byte yet, and background waits until the realtime request has its own.
//! - [`Policy::Leaseholder`]: Leaseholder's own (E2 arm 9). Realtime requests go upstream at once and are never
//!   refused. Background requests yield to the realtime sessions' turns, which the caller tracks from lifecycle
//!   events and reports as counts ([`Admission::set_turns`]): they leave KV headroom free while a turn is
//!   imminent, and share a budget of predicted prefill time in each critical window. A realtime request that
//!   finds the headroom not kept has its reply capped. A request the caller has let wait long enough is admitted
//!   regardless ([`Admission::age`]).
//!
//! In every policy, the waiting requests of a class are admitted in arrival order, and none goes ahead of a
//! waiting request of a stricter class.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use lh_proto::SloClass;

use crate::step::StepCost;

/// A held request, numbered in arrival order.
pub type RequestId = u64;

/// Bytes of request body per prompt token, where nothing better is known. The W2 agent replay sends 3.9 to 4.0
/// per token.
pub const BYTES_PER_TOKEN: u64 = 4;

/// What a background request's prefill costs the engine, for policy leaseholder's critical-window throttle: the
/// steps a lone prefill of its new tokens takes after the rest of its prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillCost {
    /// The step model of the engine and GPU, which `lh-sim`'s `scripts/fit_step_cost.py` fits.
    pub steps: StepCost,
    /// The engine's step budget, `--max-num-batched-tokens`: a lone prefill runs in steps of this many tokens.
    pub step_tokens: u64,
}

impl PrefillCost {
    /// The predicted prefill time of a request with this estimate.
    pub fn of(&self, estimate: Estimate) -> Duration {
        // vLLM computes at least the last prompt token.
        let new = estimate.new.max(1);
        let cached = estimate.tokens.saturating_sub(new);
        let ms = self.steps.prefill_ms(new, cached, self.step_tokens);
        Duration::from_secs_f64(ms.max(0.0) / 1e3)
    }
}

/// An admission policy with its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// At most `background_max_in_flight` background requests in flight.
    StaticCap { background_max_in_flight: u32 },
    /// A background request goes upstream while the tokens of the background requests in flight, its own and
    /// `reserved_tokens` fit in `pool_tokens`. A request that cannot fit beside the reservation goes alone,
    /// so it cannot hold up the queue for good.
    TokenReservation {
        pool_tokens: u64,
        reserved_tokens: u64,
    },
    /// A request goes upstream while the requests in flight, of every class, are fewer than its class's
    /// ceiling. A blocked realtime or interactive request sheds the newest background requests in flight, if
    /// that is enough to let it in.
    FlowControl { ceilings: Ceilings },
    /// A realtime arrival preempts the background requests in flight that have no first byte yet: they wait
    /// at the front of the queue, and background requests wait while any realtime request has none.
    PreemptResubmit,
    /// Realtime requests go upstream at once and are never refused; background requests yield to the
    /// realtime sessions' turns ([`Leaseholder`]). Interactive requests are not held.
    Leaseholder(Leaseholder),
}

/// Policy leaseholder's mechanisms. Each one can be switched off on its own, for E2's ablations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Leaseholder {
    /// The KV pool, in tokens.
    pub pool_tokens: u64,
    /// The KV of one realtime turn, its prompt and its reply, in tokens.
    pub headroom_tokens: u64,
    /// Turn-imminent KV headroom: while realtime sessions are armed, a background request goes upstream only
    /// if the estimated tokens of the background requests in flight and its own leave `headroom_tokens` of
    /// the pool free for each armed session. The trigger says which sessions are armed; `None` turns the
    /// headroom off.
    pub headroom: Option<Trigger>,
    /// The critical-window prefill throttle: while any realtime session is in its critical window, the
    /// background requests admitted in the window may take this much predicted prefill time in all, each
    /// priced by the admission point's [`PrefillCost`]. `None` turns the throttle off.
    pub critical_prefill_budget: Option<Duration>,
    /// Degrade: a realtime request that arrives while the background requests in flight leave less than
    /// `headroom_tokens` of the pool free has its reply capped at this many tokens. `None` turns it off.
    pub degrade_max_tokens: Option<u32>,
}

/// Which realtime sessions arm the headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// From the user's speech onset to the reply's first audio.
    Onset,
    /// The critical window alone: from the end of the user's turn, or the realtime request if it comes
    /// first, to the reply's first audio.
    Request,
}

impl Trigger {
    /// How many of the sessions `turns` counts are armed.
    pub const fn armed(self, turns: Turns) -> u32 {
        match self {
            Self::Onset => turns.speaking + turns.critical,
            Self::Request => turns.critical,
        }
    }
}

/// How many realtime sessions are in the phases of a turn that policy leaseholder acts on. The caller tracks
/// each session from lifecycle events and from its realtime requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Turns {
    /// The user is speaking.
    pub speaking: u32,
    /// The critical window: from the end of the user's turn, or the realtime request if it comes first, to
    /// the reply's first audio.
    pub critical: u32,
}

/// Why policy leaseholder holds a background request back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deferral {
    /// The headroom of the armed realtime sessions.
    Headroom,
    /// The critical window's prefill budget.
    Critical,
}

impl Deferral {
    pub const ALL: [Self; 2] = [Self::Headroom, Self::Critical];

    /// The name, also a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Headroom => "headroom",
            Self::Critical => "critical",
        }
    }
}

/// Flow control's holdback ceilings: a class is dispatched while the requests in flight, of every class, are
/// fewer than its ceiling. They do not increase from realtime to background, so as the engine fills up, the
/// lower classes are held back first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ceilings {
    pub realtime: u32,
    pub interactive: u32,
    pub background: u32,
}

impl Ceilings {
    /// The ceiling of `class`.
    pub const fn of(&self, class: SloClass) -> u32 {
        match class {
            SloClass::Realtime => self.realtime,
            SloClass::Interactive => self.interactive,
            SloClass::Background => self.background,
        }
    }
}

impl Policy {
    /// Whether the policy decides for requests of `class`. Requests of the other classes need not arrive: they
    /// go upstream at once and count towards nothing.
    pub const fn holds(&self, class: SloClass) -> bool {
        match self {
            Self::StaticCap { .. } | Self::TokenReservation { .. } => {
                matches!(class, SloClass::Background)
            }
            Self::FlowControl { .. } => true,
            Self::PreemptResubmit | Self::Leaseholder(_) => !matches!(class, SloClass::Interactive),
        }
    }
}

/// A decision about a held request, for the caller to carry out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// The waiting request goes upstream.
    Admit(RequestId),
    /// The admitted request is preempted: its upstream request is cancelled, and it waits to go again.
    Preempt(RequestId),
    /// The admitted request is shed: its upstream request is cancelled and it fails. It is no longer held.
    Shed(RequestId),
}

/// A request that arrived and is now held.
#[derive(Debug, PartialEq, Eq)]
pub struct Arrival {
    pub id: RequestId,
    /// Admitted at once; otherwise it waits for an [`Action::Admit`].
    pub admitted: bool,
    /// What the arrival did to other held requests.
    pub actions: Vec<Action>,
    /// Why policy leaseholder holds a waiting request back: what holds the first of the waiting requests of
    /// its class, which may be the request itself.
    pub deferred: Option<Deferral>,
    /// The reply cap of a realtime request policy leaseholder degraded.
    pub reply_cap: Option<u32>,
}

/// A request refused on arrival because its class's queue is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueFull;

/// The requests at the admission point and the policy that decides for them.
#[derive(Debug)]
pub struct Admission {
    policy: Policy,
    queue_limit: usize,
    next_id: RequestId,
    held: HashMap<RequestId, Held>,
    /// Waiting requests by class, in arrival order; a preempted request goes back to the front.
    waiting: [VecDeque<RequestId>; 3],
    /// The tokens of the waiting background requests.
    waiting_tokens: u64,
    /// Admitted requests by class.
    admitted: [u32; 3],
    /// The admitted background requests, oldest first.
    background: Vec<RequestId>,
    /// The tokens of the admitted background requests.
    background_tokens: u64,
    /// Admitted realtime requests without a first byte (preempt-resubmit).
    realtime_pending: u32,
    /// What policy leaseholder's throttle prices a background prefill by.
    prefill_cost: Option<PrefillCost>,
    /// The realtime sessions' turns (policy leaseholder).
    turns: Turns,
    /// The predicted prefill time of the background requests admitted in the current critical window.
    window_prefill: Duration,
}

#[derive(Debug)]
struct Held {
    class: SloClass,
    tokens: u64,
    /// The predicted prefill time of a background request that policy leaseholder's throttle prices.
    prefill: Duration,
    admitted: bool,
    first_byte: bool,
}

impl Admission {
    /// An admission point holding nothing, whose class queues hold at most `queue_limit` requests each.
    ///
    /// # Panics
    /// Under policy leaseholder with its throttle on, which prices prefills: that admission point is made with
    /// [`Admission::with_prefill_cost`].
    pub fn new(policy: Policy, queue_limit: usize) -> Self {
        let throttle = matches!(
            policy,
            Policy::Leaseholder(Leaseholder {
                critical_prefill_budget: Some(_),
                ..
            })
        );
        assert!(
            !throttle,
            "policy leaseholder's throttle prices prefills: use Admission::with_prefill_cost"
        );
        Self::build(policy, queue_limit, None)
    }

    /// An admission point holding nothing, whose class queues hold at most `queue_limit` requests each, and
    /// where policy leaseholder's throttle prices a background request's prefill by `cost`.
    pub fn with_prefill_cost(policy: Policy, queue_limit: usize, cost: PrefillCost) -> Self {
        Self::build(policy, queue_limit, Some(cost))
    }

    fn build(policy: Policy, queue_limit: usize, prefill_cost: Option<PrefillCost>) -> Self {
        Self {
            policy,
            queue_limit,
            next_id: 0,
            held: HashMap::new(),
            waiting: Default::default(),
            waiting_tokens: 0,
            admitted: [0; 3],
            background: Vec::new(),
            background_tokens: 0,
            realtime_pending: 0,
            prefill_cost,
            turns: Turns::default(),
            window_prefill: Duration::ZERO,
        }
    }

    pub const fn policy(&self) -> Policy {
        self.policy
    }

    /// A request of `class` with an estimated prompt arrives. It is admitted at once if it fits and no request
    /// of its class or a stricter one waits; otherwise it waits, unless its class's queue is full.
    pub fn arrive(&mut self, class: SloClass, estimate: Estimate) -> Result<Arrival, QueueFull> {
        let mut actions = Vec::new();
        let held = Held {
            class,
            tokens: estimate.tokens,
            prefill: self.prefill(class, estimate),
            admitted: false,
            first_byte: false,
        };
        let behind = SloClass::ALL[..=index(class)]
            .iter()
            .any(|&c| !self.waiting[index(c)].is_empty());
        let admitted = !behind && (self.fits(&held) || self.make_room(class, &mut actions));
        if !admitted && self.waiting[index(class)].len() >= self.queue_limit {
            return Err(QueueFull);
        }
        let deferred = if admitted { None } else { self.deferral(&held) };
        let reply_cap = self.reply_cap(class);
        let id = self.next_id;
        self.next_id += 1;
        let tokens = held.tokens;
        self.held.insert(id, held);
        if admitted {
            self.admit(id);
            if class == SloClass::Realtime && self.policy == Policy::PreemptResubmit {
                self.preempt(&mut actions);
            }
        } else {
            self.waiting[index(class)].push_back(id);
            if class == SloClass::Background {
                self.waiting_tokens += tokens;
            }
        }
        Ok(Arrival {
            id,
            admitted,
            actions,
            deferred,
            reply_cap,
        })
    }

    /// The realtime sessions' turns changed (policy leaseholder). A critical window opens with its whole
    /// prefill budget, and what the change frees goes upstream.
    pub fn set_turns(&mut self, turns: Turns) -> Vec<Action> {
        if self.turns.critical == 0 && turns.critical > 0 {
            self.window_prefill = Duration::ZERO;
        }
        self.turns = turns;
        let mut actions = Vec::new();
        self.dispatch(&mut actions);
        actions
    }

    /// The waiting request has waited as long as the caller lets a request be deferred: it is admitted
    /// regardless, so that background work is never starved. Admitting it frees nothing, so no other request
    /// follows it. Returns whether it was `oversize`, its estimate alone over what policy leaseholder's
    /// headroom leaves the background requests while the sessions armed now stay armed, so that only aging
    /// could admit it meanwhile; `None` if it is not waiting.
    pub fn age(&mut self, id: RequestId) -> Option<bool> {
        let held = self.held.get(&id).filter(|held| !held.admitted)?;
        let (class, tokens) = (held.class, held.tokens);
        let oversize = match self.policy {
            Policy::Leaseholder(lh) => self.headroom_limit(&lh).is_some_and(|limit| tokens > limit),
            _ => false,
        };
        self.waiting[index(class)].retain(|&waiting| waiting != id);
        if class == SloClass::Background {
            self.waiting_tokens -= tokens;
        }
        self.admit(id);
        Some(oversize)
    }

    /// The estimated tokens of the background requests policy leaseholder's headroom holds back now: every
    /// waiting one, while the first of them waits for the headroom.
    pub fn held_back(&self) -> u64 {
        let Policy::Leaseholder(lh) = self.policy else {
            return 0;
        };
        match self.waiting[index(SloClass::Background)].front() {
            Some(id) if self.blocking(&lh, &self.held[id]) == Some(Deferral::Headroom) => {
                self.waiting_tokens
            }
            _ => 0,
        }
    }

    /// The admitted request's response produced its first byte. `None` if the request is not admitted (it was
    /// preempted first): the byte belongs to a cancelled upstream request and must not be relayed.
    pub fn first_byte(&mut self, id: RequestId) -> Option<Vec<Action>> {
        let held = self.held.get_mut(&id).filter(|held| held.admitted)?;
        let mut actions = Vec::new();
        let first = !std::mem::replace(&mut held.first_byte, true);
        if first && held.class == SloClass::Realtime && self.policy == Policy::PreemptResubmit {
            self.realtime_pending -= 1;
            self.dispatch(&mut actions);
        }
        Some(actions)
    }

    /// The request leaves, waiting or admitted: its response ended, or it gave up. What it held is freed,
    /// which may admit others. Leaving again, or after being shed, does nothing.
    pub fn leave(&mut self, id: RequestId) -> Vec<Action> {
        let mut actions = Vec::new();
        if let Some(held) = self.held.remove(&id) {
            if held.admitted {
                self.release(id, &held);
            } else {
                self.waiting[index(held.class)].retain(|&waiting| waiting != id);
                if held.class == SloClass::Background {
                    self.waiting_tokens -= held.tokens;
                }
            }
            self.dispatch(&mut actions);
        }
        actions
    }

    /// Whether the request is held and waiting.
    pub fn is_waiting(&self, id: RequestId) -> bool {
        self.held.get(&id).is_some_and(|held| !held.admitted)
    }

    /// The admitted requests of `class`.
    pub fn admitted(&self, class: SloClass) -> u32 {
        self.admitted[index(class)]
    }

    /// The waiting requests of `class`.
    pub fn waiting(&self, class: SloClass) -> usize {
        self.waiting[index(class)].len()
    }

    /// The estimated tokens of the admitted background requests.
    pub const fn background_tokens(&self) -> u64 {
        self.background_tokens
    }

    /// The estimated tokens of the waiting background requests.
    pub const fn waiting_tokens(&self) -> u64 {
        self.waiting_tokens
    }

    /// Whether a held request may go upstream beside the admitted ones.
    fn fits(&self, held: &Held) -> bool {
        let (class, tokens) = (held.class, held.tokens);
        let background = class == SloClass::Background;
        match self.policy {
            Policy::StaticCap {
                background_max_in_flight,
            } => !background || self.admitted(SloClass::Background) < background_max_in_flight,
            Policy::TokenReservation {
                pool_tokens,
                reserved_tokens,
            } => {
                !background
                    || self.background.is_empty()
                    || self.background_tokens + tokens + reserved_tokens <= pool_tokens
            }
            Policy::FlowControl { ceilings } => self.in_flight() < ceilings.of(class),
            Policy::PreemptResubmit => !background || self.realtime_pending == 0,
            Policy::Leaseholder(lh) => !background || self.blocking(&lh, held).is_none(),
        }
    }

    /// What holds a background request back under policy leaseholder, if anything: the headroom first, then
    /// the critical window's prefill budget.
    fn blocking(&self, lh: &Leaseholder, held: &Held) -> Option<Deferral> {
        if self
            .headroom_limit(lh)
            .is_some_and(|limit| self.background_tokens + held.tokens > limit)
        {
            return Some(Deferral::Headroom);
        }
        if let Some(budget) = lh.critical_prefill_budget
            && self.turns.critical > 0
            && self.window_prefill + held.prefill > budget
        {
            return Some(Deferral::Critical);
        }
        None
    }

    /// The tokens the background requests in flight may hold while the headroom keeps room for the armed
    /// sessions; `None` while none is armed, or with the headroom off.
    fn headroom_limit(&self, lh: &Leaseholder) -> Option<u64> {
        let armed = lh.headroom?.armed(self.turns);
        (armed > 0).then(|| {
            lh.pool_tokens
                .saturating_sub(lh.headroom_tokens.saturating_mul(u64::from(armed)))
        })
    }

    /// Why a request that is to wait waits under policy leaseholder: what holds the first waiting request of
    /// its class back, or the request itself when none waits.
    fn deferral(&self, held: &Held) -> Option<Deferral> {
        let Policy::Leaseholder(lh) = self.policy else {
            return None;
        };
        let first = self.waiting[index(held.class)]
            .front()
            .map_or(held, |id| &self.held[id]);
        self.blocking(&lh, first)
    }

    /// The reply cap of a realtime request arriving now: policy leaseholder degrades it when the background
    /// requests in flight leave less than one turn's headroom free.
    fn reply_cap(&self, class: SloClass) -> Option<u32> {
        match self.policy {
            Policy::Leaseholder(lh) if class == SloClass::Realtime => lh
                .degrade_max_tokens
                .filter(|_| self.background_tokens + lh.headroom_tokens > lh.pool_tokens),
            _ => None,
        }
    }

    /// The predicted prefill time of a request that policy leaseholder's throttle prices.
    fn prefill(&self, class: SloClass, estimate: Estimate) -> Duration {
        let throttle = matches!(
            self.policy,
            Policy::Leaseholder(Leaseholder {
                critical_prefill_budget: Some(_),
                ..
            })
        );
        match &self.prefill_cost {
            Some(cost) if throttle && class == SloClass::Background => cost.of(estimate),
            _ => Duration::ZERO,
        }
    }

    /// Flow control's eviction: sheds the newest background requests in flight that keep a realtime or
    /// interactive request above its ceiling. Sheds nothing unless shedding is enough.
    fn make_room(&mut self, class: SloClass, actions: &mut Vec<Action>) -> bool {
        let Policy::FlowControl { ceilings } = self.policy else {
            return false;
        };
        let excess = (self.in_flight() + 1).saturating_sub(ceilings.of(class)) as usize;
        if class == SloClass::Background || excess > self.background.len() {
            return false;
        }
        let newest = self.background.len() - excess;
        for id in self.background.split_off(newest).into_iter().rev() {
            let held = self.held.remove(&id).expect("admitted requests are held");
            self.admitted[index(SloClass::Background)] -= 1;
            self.background_tokens -= held.tokens;
            actions.push(Action::Shed(id));
        }
        true
    }

    /// Preempt-resubmit, as a realtime request is admitted: the admitted background requests without a first
    /// byte go back to the front of the queue, oldest first.
    fn preempt(&mut self, actions: &mut Vec<Action>) {
        let held = &self.held;
        let preempted: Vec<RequestId> = self
            .background
            .extract_if(.., |id| !held[id].first_byte)
            .collect();
        for &id in preempted.iter().rev() {
            let held = self.held.get_mut(&id).expect("admitted requests are held");
            held.admitted = false;
            self.admitted[index(SloClass::Background)] -= 1;
            self.background_tokens -= held.tokens;
            self.waiting[index(SloClass::Background)].push_front(id);
            self.waiting_tokens += held.tokens;
        }
        actions.extend(preempted.into_iter().map(Action::Preempt));
    }

    /// Admits waiting requests, the strictest class first and each class in arrival order, until one does not
    /// fit: nothing goes ahead of a waiting request of a stricter class.
    fn dispatch(&mut self, actions: &mut Vec<Action>) {
        for class in SloClass::ALL {
            while let Some(&id) = self.waiting[index(class)].front() {
                let held = &self.held[&id];
                let tokens = held.tokens;
                if !self.fits(held) && !self.make_room(class, actions) {
                    return;
                }
                self.waiting[index(class)].pop_front();
                if class == SloClass::Background {
                    self.waiting_tokens -= tokens;
                }
                self.admit(id);
                actions.push(Action::Admit(id));
            }
        }
    }

    fn admit(&mut self, id: RequestId) {
        let held = self
            .held
            .get_mut(&id)
            .expect("only held requests are admitted");
        held.admitted = true;
        self.admitted[index(held.class)] += 1;
        match held.class {
            SloClass::Background => {
                self.background.push(id);
                self.background_tokens += held.tokens;
                if self.turns.critical > 0 {
                    self.window_prefill += held.prefill;
                }
            }
            SloClass::Realtime if self.policy == Policy::PreemptResubmit => {
                self.realtime_pending += 1;
            }
            _ => {}
        }
    }

    /// Frees what an admitted request held.
    fn release(&mut self, id: RequestId, held: &Held) {
        self.admitted[index(held.class)] -= 1;
        match held.class {
            SloClass::Background => {
                self.background.retain(|&admitted| admitted != id);
                self.background_tokens -= held.tokens;
            }
            SloClass::Realtime if self.policy == Policy::PreemptResubmit && !held.first_byte => {
                self.realtime_pending -= 1;
            }
            _ => {}
        }
    }

    fn in_flight(&self) -> u32 {
        self.admitted.iter().sum()
    }
}

/// The index of a class in per-class arrays, strictest first.
const fn index(class: SloClass) -> usize {
    class as usize
}

/// What the engine reported for the last request of a session: its prompt tokens, and the size of its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionUsage {
    pub prompt_tokens: u64,
    pub body_bytes: u64,
}

/// A request's prompt, as [`estimate_tokens`] estimates it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Estimate {
    /// The prompt tokens.
    pub tokens: u64,
    /// Those beyond the session's last reported prompt: all of them when no report anchors the estimate.
    pub new: u64,
}

/// A request's prompt tokens, estimated without a tokenizer. An agent's next turn resends the conversation with
/// more added, so a body at least as large as its session's last is that request's prompt tokens plus the
/// growth, at [`BYTES_PER_TOKEN`]. A smaller body is another conversation (a subagent shares its parent's
/// session) or a compacted one, and it is estimated, like a session without a usage, as the whole body at that
/// rate. At least one token. `examples/estimate_error.rs` measures the estimate on an AIPerf replay.
pub fn estimate_tokens(body_bytes: u64, last: Option<SessionUsage>) -> Estimate {
    match last {
        Some(last) if body_bytes >= last.body_bytes => {
            let new = (body_bytes - last.body_bytes) / BYTES_PER_TOKEN;
            Estimate {
                tokens: last.prompt_tokens.saturating_add(new).max(1),
                new,
            }
        }
        _ => {
            let tokens = (body_bytes / BYTES_PER_TOKEN).max(1);
            Estimate {
                tokens,
                new: tokens,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SloClass::*;

    /// A prompt of `tokens` tokens, all new.
    fn estimate(tokens: u64) -> Estimate {
        Estimate {
            tokens,
            new: tokens,
        }
    }

    fn admitted(admission: &mut Admission, class: SloClass, tokens: u64) -> RequestId {
        let arrival = admission.arrive(class, estimate(tokens)).unwrap();
        assert!(arrival.admitted, "{class} {tokens} waits");
        assert_eq!(arrival.actions, []);
        arrival.id
    }

    fn waits(admission: &mut Admission, class: SloClass, tokens: u64) -> RequestId {
        let arrival = admission.arrive(class, estimate(tokens)).unwrap();
        assert!(!arrival.admitted, "{class} {tokens} is admitted");
        assert_eq!(arrival.actions, []);
        arrival.id
    }

    #[test]
    fn a_static_cap_holds_background_requests_and_queues_the_rest_in_order() {
        let policy = Policy::StaticCap {
            background_max_in_flight: 2,
        };
        assert!(policy.holds(Background) && !policy.holds(Realtime) && !policy.holds(Interactive));
        let mut a = Admission::new(policy, 2);
        let first = admitted(&mut a, Background, 0);
        let second = admitted(&mut a, Background, 0);
        let third = waits(&mut a, Background, 0);
        let fourth = waits(&mut a, Background, 0);
        assert_eq!(a.arrive(Background, estimate(0)), Err(QueueFull));
        assert_eq!((a.admitted(Background), a.waiting(Background)), (2, 2));

        assert_eq!(a.leave(second), [Action::Admit(third)]);
        // A waiting request that leaves frees its place in the queue, and admits nothing.
        assert_eq!(a.leave(fourth), []);
        assert!(!a.is_waiting(fourth));
        let fifth = waits(&mut a, Background, 0);
        assert_eq!(a.leave(first), [Action::Admit(fifth)]);
        assert_eq!(a.leave(first), [], "leaving twice does nothing");
        assert_eq!((a.admitted(Background), a.waiting(Background)), (2, 0));
    }

    #[test]
    fn token_reservation_admits_background_while_its_tokens_fit_beside_the_reservation() {
        let policy = Policy::TokenReservation {
            pool_tokens: 1_000,
            reserved_tokens: 200,
        };
        let mut a = Admission::new(policy, 8);
        let small = admitted(&mut a, Background, 300);
        let medium = admitted(&mut a, Background, 500);
        // 300 + 500 + 300 + 200 > 1,000. The 50 behind it waits too: the queue is first in, first out.
        let large = waits(&mut a, Background, 300);
        let tiny = waits(&mut a, Background, 50);
        admitted(&mut a, Realtime, 5_000);
        assert_eq!(a.leave(small), [Action::Admit(large)]);
        assert_eq!(a.leave(medium), [Action::Admit(tiny)]);
        assert_eq!(a.leave(large), []);
        assert_eq!(a.leave(tiny), []);

        // A request too large to fit beside the reservation goes alone, and then holds everything back.
        let huge = admitted(&mut a, Background, 900);
        let next = waits(&mut a, Background, 1);
        assert_eq!(a.leave(huge), [Action::Admit(next)]);
    }

    #[test]
    fn a_waiting_head_that_leaves_lets_the_requests_behind_it_in() {
        let policy = Policy::TokenReservation {
            pool_tokens: 1_000,
            reserved_tokens: 0,
        };
        let mut a = Admission::new(policy, 8);
        admitted(&mut a, Background, 600);
        let head = waits(&mut a, Background, 600);
        let behind = waits(&mut a, Background, 100);
        assert_eq!(a.leave(head), [Action::Admit(behind)]);
    }

    #[test]
    fn flow_control_dispatches_bands_strictly_and_each_in_arrival_order() {
        let ceilings = Ceilings {
            realtime: 2,
            interactive: 2,
            background: 1,
        };
        let mut a = Admission::new(Policy::FlowControl { ceilings }, 8);
        let interactive = admitted(&mut a, Interactive, 0);
        let early_background = waits(&mut a, Background, 0);
        let second_interactive = admitted(&mut a, Interactive, 0);
        // Two in flight and no background request among them to shed: realtime waits too.
        let realtime = waits(&mut a, Realtime, 0);
        let late_background = waits(&mut a, Background, 0);
        let late_interactive = waits(&mut a, Interactive, 0);

        assert_eq!(a.leave(interactive), [Action::Admit(realtime)]);
        assert_eq!(
            a.leave(second_interactive),
            [Action::Admit(late_interactive)]
        );
        assert_eq!(a.leave(realtime), []);
        assert_eq!(a.leave(late_interactive), [Action::Admit(early_background)]);
        assert_eq!(a.leave(early_background), [Action::Admit(late_background)]);
    }

    #[test]
    fn flow_control_sheds_the_newest_background_requests_for_a_blocked_stricter_one() {
        let ceilings = Ceilings {
            realtime: 2,
            interactive: 2,
            background: 2,
        };
        let mut a = Admission::new(Policy::FlowControl { ceilings }, 8);
        let older = admitted(&mut a, Background, 0);
        let newer = admitted(&mut a, Background, 0);
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert!(realtime.admitted);
        assert_eq!(realtime.actions, [Action::Shed(newer)]);
        assert_eq!(a.leave(newer), [], "a shed request is no longer held");
        let second = a.arrive(Realtime, estimate(0)).unwrap();
        assert_eq!(second.actions, [Action::Shed(older)]);
        // Nothing left to shed.
        let third = waits(&mut a, Realtime, 0);
        let background = waits(&mut a, Background, 0);
        assert_eq!(a.leave(realtime.id), [Action::Admit(third)]);
        assert_eq!(a.leave(second.id), [Action::Admit(background)]);
    }

    #[test]
    fn flow_control_sheds_only_when_shedding_is_enough() {
        let ceilings = Ceilings {
            realtime: 4,
            interactive: 2,
            background: 2,
        };
        let mut a = Admission::new(Policy::FlowControl { ceilings }, 8);
        let older = admitted(&mut a, Background, 0);
        let newer = admitted(&mut a, Background, 0);
        let realtime = admitted(&mut a, Realtime, 0);
        admitted(&mut a, Realtime, 0);
        // Four in flight against a ceiling of two: three would have to go, and only two can be shed.
        let interactive = waits(&mut a, Interactive, 0);
        assert_eq!(
            a.leave(realtime),
            [
                Action::Shed(newer),
                Action::Shed(older),
                Action::Admit(interactive)
            ]
        );
    }

    #[test]
    fn a_realtime_arrival_preempts_background_requests_without_a_first_byte() {
        let policy = Policy::PreemptResubmit;
        assert!(policy.holds(Realtime) && policy.holds(Background) && !policy.holds(Interactive));
        let mut a = Admission::new(policy, 8);
        let streaming = admitted(&mut a, Background, 0);
        let prefilling = admitted(&mut a, Background, 0);
        assert_eq!(a.first_byte(streaming), Some(vec![]));
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert!(realtime.admitted);
        assert_eq!(realtime.actions, [Action::Preempt(prefilling)]);
        assert!(a.is_waiting(prefilling));
        assert_eq!(
            a.first_byte(prefilling),
            None,
            "a byte from a preempted request is not relayed"
        );
        // Background that arrives while the realtime request waits for its first byte waits behind it.
        let arriving = waits(&mut a, Background, 0);
        assert_eq!(
            a.first_byte(realtime.id),
            Some(vec![Action::Admit(prefilling), Action::Admit(arriving)])
        );
        assert_eq!(a.admitted(Background), 3);
        // Once the realtime request has its first byte, it preempts nothing more.
        assert_eq!(a.first_byte(realtime.id), Some(vec![]));
        assert_eq!(a.leave(realtime.id), []);
    }

    #[test]
    fn background_waits_until_every_realtime_request_has_a_first_byte_or_has_left() {
        let mut a = Admission::new(Policy::PreemptResubmit, 1);
        let first = admitted(&mut a, Realtime, 0);
        let second = admitted(&mut a, Realtime, 0);
        let background = waits(&mut a, Background, 0);
        assert_eq!(a.arrive(Background, estimate(0)), Err(QueueFull));
        assert_eq!(a.first_byte(first), Some(vec![]));
        // A realtime request whose response failed before its first byte ends its wait too.
        assert_eq!(a.leave(second), [Action::Admit(background)]);
    }

    #[test]
    fn a_preempted_request_that_leaves_is_not_admitted_again() {
        let mut a = Admission::new(Policy::PreemptResubmit, 8);
        let background = admitted(&mut a, Background, 0);
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert_eq!(realtime.actions, [Action::Preempt(background)]);
        assert_eq!(a.leave(background), []);
        assert_eq!(a.first_byte(realtime.id), Some(vec![]));
        assert_eq!(a.admitted(Background), 0);
    }

    #[test]
    fn tokens_are_estimated_from_the_last_usage_of_the_session() {
        assert_eq!(estimate_tokens(4_000, None), estimate(1_000));
        assert_eq!(estimate_tokens(3, None), estimate(1));
        let last = SessionUsage {
            prompt_tokens: 18_342,
            body_bytes: 64_000,
        };
        let grown = |tokens, new| Estimate { tokens, new };
        assert_eq!(estimate_tokens(64_000, Some(last)), grown(18_342, 0));
        assert_eq!(estimate_tokens(66_000, Some(last)), grown(18_842, 500));
        // A smaller body is another conversation, such as a subagent's in its parent's session.
        assert_eq!(estimate_tokens(4_000, Some(last)), estimate(1_000));
    }

    const POOL: u64 = 10_000;
    const TURN: u64 = 1_000;

    /// Policy leaseholder with every mechanism on: the headroom armed from speech onset, 300 ms of prefill in a
    /// critical window, replies capped at 16 tokens.
    fn leaseholder() -> Leaseholder {
        Leaseholder {
            pool_tokens: POOL,
            headroom_tokens: TURN,
            headroom: Some(Trigger::Onset),
            critical_prefill_budget: Some(Duration::from_millis(300)),
            degrade_max_tokens: Some(16),
        }
    }

    fn turns(speaking: u32, critical: u32) -> Turns {
        Turns { speaking, critical }
    }

    /// The frozen box: its step model and its 512-token steps.
    fn frozen_box() -> PrefillCost {
        PrefillCost {
            steps: StepCost::frozen_box(),
            step_tokens: 512,
        }
    }

    /// An admission point that prices prefills as the frozen box runs them.
    fn priced(policy: Policy, queue_limit: usize) -> Admission {
        Admission::with_prefill_cost(policy, queue_limit, frozen_box())
    }

    #[test]
    fn a_prefill_is_priced_by_the_engine_it_runs_on() {
        let cost = frozen_box();
        let ms = |estimate| cost.of(estimate).as_secs_f64() * 1e3;
        assert!((ms(chunk(0)) - 38.2).abs() < 0.1);
        assert!((ms(chunk(30_000)) - 268.7).abs() < 0.1);
        // A cached prompt still costs the step that computes its last token.
        let resent = Estimate {
            tokens: 30_000,
            new: 0,
        };
        assert!((ms(resent) - cost.steps.prefill_ms(1, 29_999, 512)).abs() < 1e-6);
        // Another engine's step budget: 1,024 tokens in one step, not two.
        let wide = PrefillCost {
            step_tokens: 1_024,
            ..frozen_box()
        };
        let cold = estimate(1_024);
        assert!(wide.of(cold) < cost.of(cold));
    }

    #[test]
    #[should_panic(expected = "use Admission::with_prefill_cost")]
    fn a_throttle_needs_a_prefill_cost() {
        Admission::new(Policy::Leaseholder(leaseholder()), 8);
    }

    /// 512 new tokens after `context` cached ones: 38 ms of prefill on an empty context, 269 ms after 30,000.
    fn chunk(context: u64) -> Estimate {
        Estimate {
            tokens: context + 512,
            new: 512,
        }
    }

    #[test]
    fn realtime_goes_upstream_at_once_whatever_waits() {
        let policy = Policy::Leaseholder(leaseholder());
        assert!(policy.holds(Realtime) && policy.holds(Background) && !policy.holds(Interactive));
        let mut a = priced(policy, 1);
        assert_eq!(a.set_turns(turns(0, 1)), []);
        let deferred = waits(&mut a, Background, 9_500);
        assert_eq!(a.arrive(Background, estimate(1)), Err(QueueFull));
        // The background queue is full and its head waits: a realtime request still goes, and takes nothing
        // from the background requests.
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert!(realtime.admitted);
        assert_eq!((realtime.actions, realtime.deferred), (vec![], None));
        assert_eq!(a.leave(realtime.id), []);
        assert!(a.is_waiting(deferred));
    }

    #[test]
    fn the_headroom_defers_background_while_a_turn_is_imminent() {
        let policy = Policy::Leaseholder(Leaseholder {
            critical_prefill_budget: None,
            degrade_max_tokens: None,
            ..leaseholder()
        });
        let mut a = priced(policy, 8);
        // With no session armed, background holds what it will.
        let early = admitted(&mut a, Background, 8_500);
        assert_eq!(a.set_turns(turns(1, 0)), []);
        // 8,500 + 400 leaves one turn's 1,000 free; 200 more would not.
        let small = admitted(&mut a, Background, 400);
        let arrival = a.arrive(Background, estimate(200)).unwrap();
        assert!(!arrival.admitted);
        assert_eq!(arrival.deferred, Some(Deferral::Headroom));
        assert_eq!(a.held_back(), 200);
        // Each armed session has its own headroom.
        assert_eq!(a.set_turns(turns(1, 1)), []);
        assert_eq!(a.leave(small), []);
        // The early request ends, its KV parks, and the deferred one fits beside the headroom of both.
        assert_eq!(a.leave(early), [Action::Admit(arrival.id)]);
        assert_eq!(a.held_back(), 0);
        assert_eq!(a.set_turns(Turns::default()), []);
        admitted(&mut a, Background, 9_800);
    }

    #[test]
    fn the_request_trigger_arms_only_the_critical_window() {
        let policy = Policy::Leaseholder(Leaseholder {
            headroom: Some(Trigger::Request),
            critical_prefill_budget: None,
            ..leaseholder()
        });
        let mut a = priced(policy, 8);
        assert_eq!(a.set_turns(turns(2, 0)), []);
        admitted(&mut a, Background, 9_000);
        assert_eq!(a.set_turns(turns(1, 1)), []);
        let deferred = waits(&mut a, Background, 1);
        // The reply's first audio ends the window; the other user still speaking arms nothing.
        assert_eq!(a.set_turns(turns(1, 0)), [Action::Admit(deferred)]);
    }

    #[test]
    fn the_critical_window_spends_a_budget_of_predicted_prefill_time() {
        let policy = Policy::Leaseholder(Leaseholder {
            headroom: None,
            degrade_max_tokens: None,
            ..leaseholder()
        });
        let mut a = priced(policy, 8);
        // Admitted before the window, so not charged to it.
        a.arrive(Background, chunk(30_000)).unwrap();
        assert_eq!(a.set_turns(turns(0, 1)), []);
        // 512 tokens after 30,000 take 269 ms of the 300 ms budget, and 512 after none 38 ms more.
        assert!(a.arrive(Background, chunk(30_000)).unwrap().admitted);
        let shallow = a.arrive(Background, chunk(0)).unwrap();
        assert_eq!(
            (shallow.admitted, shallow.deferred),
            (false, Some(Deferral::Critical))
        );
        assert_eq!(
            a.held_back(),
            0,
            "the throttle, not the headroom, holds it back"
        );
        // Another session's window overlaps this one: still one window.
        assert_eq!(a.set_turns(turns(0, 2)), []);
        assert_eq!(a.set_turns(turns(0, 0)), [Action::Admit(shallow.id)]);

        // A new window has the whole budget again: seven shallow chunks fit, an eighth does not.
        assert_eq!(a.set_turns(turns(0, 1)), []);
        for _ in 0..7 {
            assert!(a.arrive(Background, chunk(0)).unwrap().admitted);
        }
        assert!(!a.arrive(Background, chunk(0)).unwrap().admitted);
    }

    #[test]
    fn aging_admits_a_deferred_request_regardless() {
        let mut a = priced(Policy::Leaseholder(leaseholder()), 8);
        // Four armed sessions leave background 6,000 tokens: 7,000 cannot go even alone.
        assert_eq!(a.set_turns(turns(4, 0)), []);
        let oversize = waits(&mut a, Background, 7_000);
        let behind = a.arrive(Background, estimate(100)).unwrap();
        assert_eq!(behind.deferred, Some(Deferral::Headroom));
        assert_eq!(a.held_back(), 7_100);
        assert_eq!(a.age(oversize), Some(true));
        assert_eq!(a.age(oversize), None, "it is no longer waiting");
        assert_eq!(a.admitted(Background), 1);
        // 100 tokens would fit alone; behind the aged request they do not.
        assert!(a.is_waiting(behind.id));
        assert_eq!(a.age(behind.id), Some(false));
        assert_eq!(a.held_back(), 0);
    }

    #[test]
    fn an_aged_prefill_spends_the_critical_window_budget() {
        let mut a = priced(Policy::Leaseholder(leaseholder()), 8);
        assert_eq!(a.set_turns(turns(0, 1)), []);
        // A cold 4,000-token prefill takes more than the window's 300 ms; a 100-token one would fit.
        let cold = a.arrive(Background, estimate(4_000)).unwrap();
        assert_eq!(cold.deferred, Some(Deferral::Critical));
        let behind = waits(&mut a, Background, 100);
        assert_eq!(a.age(cold.id), Some(false));
        assert!(a.is_waiting(behind));
        assert_eq!(a.set_turns(turns(0, 0)), [Action::Admit(behind)]);
    }

    #[test]
    fn degrade_caps_a_realtime_reply_when_the_headroom_was_not_kept() {
        // Degrade alone: nothing keeps background from filling the pool while a turn is imminent.
        let policy = Policy::Leaseholder(Leaseholder {
            headroom: None,
            critical_prefill_budget: None,
            ..leaseholder()
        });
        let mut a = priced(policy, 8);
        assert_eq!(a.set_turns(turns(1, 0)), []);
        let background = admitted(&mut a, Background, 9_001);
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert!(realtime.admitted);
        assert_eq!(realtime.reply_cap, Some(16));
        assert_eq!(a.leave(background), []);
        admitted(&mut a, Background, 9_000);
        let full_turn = a.arrive(Realtime, estimate(0)).unwrap();
        assert_eq!(
            full_turn.reply_cap, None,
            "one turn's 1,000 tokens are free"
        );

        let off = Policy::Leaseholder(Leaseholder {
            degrade_max_tokens: None,
            ..leaseholder()
        });
        let mut a = priced(off, 8);
        admitted(&mut a, Background, POOL);
        assert_eq!(a.arrive(Realtime, estimate(0)).unwrap().reply_cap, None);
    }

    #[test]
    fn with_every_mechanism_off_background_is_never_held() {
        let policy = Policy::Leaseholder(Leaseholder {
            headroom: None,
            critical_prefill_budget: None,
            degrade_max_tokens: None,
            ..leaseholder()
        });
        let mut a = priced(policy, 8);
        assert_eq!(a.set_turns(turns(3, 3)), []);
        for _ in 0..3 {
            admitted(&mut a, Background, POOL);
        }
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert_eq!((realtime.admitted, realtime.reply_cap), (true, None));
    }

    #[test]
    fn the_headroom_and_the_throttle_together() {
        let mut a = priced(Policy::Leaseholder(leaseholder()), 8);
        assert_eq!(a.set_turns(turns(0, 1)), []);
        // 8,000 cold tokens fit beside one turn's headroom, but their prefill does not fit the window.
        let cold = a.arrive(Background, estimate(8_000)).unwrap();
        assert_eq!(cold.deferred, Some(Deferral::Critical));
        let behind = a.arrive(Background, estimate(2_000)).unwrap();
        assert_eq!(behind.deferred, Some(Deferral::Critical), "it waits behind");
        assert_eq!(a.held_back(), 0);
        // The reply plays and the user speaks again: no window, but the headroom is armed.
        assert_eq!(a.set_turns(turns(1, 0)), [Action::Admit(cold.id)]);
        assert!(a.is_waiting(behind.id));
        assert_eq!(a.held_back(), 2_000);
        assert_eq!(a.set_turns(turns(0, 0)), [Action::Admit(behind.id)]);
    }

    #[test]
    fn the_baselines_hold_nothing_back_for_turns() {
        let policy = Policy::StaticCap {
            background_max_in_flight: 1,
        };
        let mut a = priced(policy, 8);
        admitted(&mut a, Background, 0);
        let waiting = waits(&mut a, Background, 0);
        assert_eq!(a.set_turns(turns(1, 1)), []);
        assert_eq!(a.held_back(), 0);
        let realtime = a.arrive(Realtime, estimate(0)).unwrap();
        assert_eq!((realtime.deferred, realtime.reply_cap), (None, None));
        assert!(a.is_waiting(waiting));
    }

    #[test]
    fn the_background_tokens_held_are_reported_in_flight_and_waiting() {
        let mut a = priced(Policy::Leaseholder(leaseholder()), 8);
        let running = admitted(&mut a, Background, 6_000);
        assert_eq!(a.set_turns(turns(1, 0)), []);
        // 6,000 + 4,000 leave less than one turn's 1,000 free: it waits.
        let deferred = waits(&mut a, Background, 4_000);
        admitted(&mut a, Realtime, 700);
        assert_eq!((a.background_tokens(), a.waiting_tokens()), (6_000, 4_000));
        assert_eq!(a.leave(running), [Action::Admit(deferred)]);
        assert_eq!((a.background_tokens(), a.waiting_tokens()), (4_000, 0));
    }
}
