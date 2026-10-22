//! Admission: whether a request goes upstream now, waits, or is refused, and which admitted requests give way
//! to it.
//!
//! An [`Admission`] holds each request from its arrival at the admission point to the end of its response, and
//! decides at three events: an arrival, a response's first byte, a departure. Its decisions are [`Action`]s on
//! held requests, which the caller carries out: the caller owns the waiting, its timeouts and the cancellation
//! of upstream requests. The decisions depend only on the order of events, so there is no clock, and a
//! request's tokens are an estimate the caller passes in ([`estimate_tokens`]).
//!
//! The policies are experiment E2's baselines that `lh-gateway` emulates; its README says what each one
//! reproduces of the system it stands for and what it leaves out.
//! - [`Policy::StaticCap`]: at most a fixed number of background requests in flight.
//! - [`Policy::TokenReservation`]: background requests in flight while their tokens and a reservation fit in
//!   the KV pool.
//! - [`Policy::FlowControl`]: llm-d's flow control. Each class is a priority band, dispatched strictly; each
//!   band has a holdback ceiling on the requests of all bands in flight; a blocked realtime or interactive
//!   request evicts the newest background requests.
//! - [`Policy::PreemptResubmit`]: abort-and-resubmit. A realtime arrival preempts the background requests that
//!   have no first byte yet, and background waits until the realtime request has its own.
//!
//! In every policy, the waiting requests of a class are admitted in arrival order, and none goes ahead of a
//! waiting request of a stricter class.

use std::collections::{HashMap, VecDeque};

use lh_proto::SloClass;

/// A held request, numbered in arrival order.
pub type RequestId = u64;

/// Bytes of request body per prompt token, where nothing better is known. The W2 agent replay sends 3.9 to 4.0
/// per token.
pub const BYTES_PER_TOKEN: u64 = 4;

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
            Self::PreemptResubmit => !matches!(class, SloClass::Interactive),
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
    /// Admitted requests by class.
    admitted: [u32; 3],
    /// The admitted background requests, oldest first.
    background: Vec<RequestId>,
    /// The tokens of the admitted background requests.
    background_tokens: u64,
    /// Admitted realtime requests without a first byte (preempt-resubmit).
    realtime_pending: u32,
}

#[derive(Debug)]
struct Held {
    class: SloClass,
    tokens: u64,
    admitted: bool,
    first_byte: bool,
}

impl Admission {
    /// An admission point holding nothing, whose class queues hold at most `queue_limit` requests each.
    pub fn new(policy: Policy, queue_limit: usize) -> Self {
        Self {
            policy,
            queue_limit,
            next_id: 0,
            held: HashMap::new(),
            waiting: Default::default(),
            admitted: [0; 3],
            background: Vec::new(),
            background_tokens: 0,
            realtime_pending: 0,
        }
    }

    pub const fn policy(&self) -> Policy {
        self.policy
    }

    /// A request of `class` with an estimated `tokens` arrives. It is admitted at once if it fits and no
    /// request of its class or a stricter one waits; otherwise it waits, unless its class's queue is full.
    pub fn arrive(&mut self, class: SloClass, tokens: u64) -> Result<Arrival, QueueFull> {
        let mut actions = Vec::new();
        let behind = SloClass::ALL[..=index(class)]
            .iter()
            .any(|&c| !self.waiting[index(c)].is_empty());
        let admitted = !behind && (self.fits(class, tokens) || self.make_room(class, &mut actions));
        if !admitted && self.waiting[index(class)].len() >= self.queue_limit {
            return Err(QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        let held = Held {
            class,
            tokens,
            admitted: false,
            first_byte: false,
        };
        self.held.insert(id, held);
        if admitted {
            self.admit(id);
            if class == SloClass::Realtime && self.policy == Policy::PreemptResubmit {
                self.preempt(&mut actions);
            }
        } else {
            self.waiting[index(class)].push_back(id);
        }
        Ok(Arrival {
            id,
            admitted,
            actions,
        })
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

    /// Whether a request of `class` with `tokens` may go upstream beside the admitted ones.
    fn fits(&self, class: SloClass, tokens: u64) -> bool {
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
        }
        actions.extend(preempted.into_iter().map(Action::Preempt));
    }

    /// Admits waiting requests, the strictest class first and each class in arrival order, until one does not
    /// fit: nothing goes ahead of a waiting request of a stricter class.
    fn dispatch(&mut self, actions: &mut Vec<Action>) {
        for class in SloClass::ALL {
            while let Some(&id) = self.waiting[index(class)].front() {
                let tokens = self.held[&id].tokens;
                if !self.fits(class, tokens) && !self.make_room(class, actions) {
                    return;
                }
                self.waiting[index(class)].pop_front();
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

/// A request's prompt tokens, estimated without a tokenizer. An agent's next turn resends the conversation with
/// more added, so a body at least as large as its session's last is that request's prompt tokens plus the
/// growth, at [`BYTES_PER_TOKEN`]. A smaller body is another conversation (a subagent shares its parent's
/// session) or a compacted one, and it is estimated, like a session without a usage, as the whole body at that
/// rate. At least one token. `examples/estimate_error.rs` measures the estimate on an AIPerf replay.
pub fn estimate_tokens(body_bytes: u64, last: Option<SessionUsage>) -> u64 {
    let tokens = match last {
        Some(last) if body_bytes >= last.body_bytes => last
            .prompt_tokens
            .saturating_add((body_bytes - last.body_bytes) / BYTES_PER_TOKEN),
        _ => body_bytes / BYTES_PER_TOKEN,
    };
    tokens.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use SloClass::*;

    fn admitted(admission: &mut Admission, class: SloClass, tokens: u64) -> RequestId {
        let arrival = admission.arrive(class, tokens).unwrap();
        assert!(arrival.admitted, "{class} {tokens} waits");
        assert_eq!(arrival.actions, []);
        arrival.id
    }

    fn waits(admission: &mut Admission, class: SloClass, tokens: u64) -> RequestId {
        let arrival = admission.arrive(class, tokens).unwrap();
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
        assert_eq!(a.arrive(Background, 0), Err(QueueFull));
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
        let realtime = a.arrive(Realtime, 0).unwrap();
        assert!(realtime.admitted);
        assert_eq!(realtime.actions, [Action::Shed(newer)]);
        assert_eq!(a.leave(newer), [], "a shed request is no longer held");
        let second = a.arrive(Realtime, 0).unwrap();
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
        let realtime = a.arrive(Realtime, 0).unwrap();
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
        assert_eq!(a.arrive(Background, 0), Err(QueueFull));
        assert_eq!(a.first_byte(first), Some(vec![]));
        // A realtime request whose response failed before its first byte ends its wait too.
        assert_eq!(a.leave(second), [Action::Admit(background)]);
    }

    #[test]
    fn a_preempted_request_that_leaves_is_not_admitted_again() {
        let mut a = Admission::new(Policy::PreemptResubmit, 8);
        let background = admitted(&mut a, Background, 0);
        let realtime = a.arrive(Realtime, 0).unwrap();
        assert_eq!(realtime.actions, [Action::Preempt(background)]);
        assert_eq!(a.leave(background), []);
        assert_eq!(a.first_byte(realtime.id), Some(vec![]));
        assert_eq!(a.admitted(Background), 0);
    }

    #[test]
    fn tokens_are_estimated_from_the_last_usage_of_the_session() {
        assert_eq!(estimate_tokens(4_000, None), 1_000);
        assert_eq!(estimate_tokens(3, None), 1);
        let last = SessionUsage {
            prompt_tokens: 18_342,
            body_bytes: 64_000,
        };
        assert_eq!(estimate_tokens(64_000, Some(last)), 18_342);
        assert_eq!(estimate_tokens(66_000, Some(last)), 18_842);
        // A smaller body is another conversation, such as a subagent's in its parent's session.
        assert_eq!(estimate_tokens(4_000, Some(last)), 1_000);
    }
}
