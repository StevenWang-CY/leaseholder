//! The admission point, between resolving a request and sending it upstream. `lh_core::admission` decides
//! whether a request goes upstream now, waits or is refused, and which admitted requests give way to it; this
//! module carries the decisions out. Each held request has a watch channel on which it learns what was decided
//! for it, so waiting never polls, and a wait ends after `queue_timeout_s`.
//!
//! A held request owns a [`Ticket`] from its arrival to the end of its response, and dropping the ticket is
//! leaving: a client that goes away while it waits, or before its response ends, frees its place at once.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use lh_core::admission::{self, Action, Policy, RequestId, SessionUsage};
use lh_proto::{SessionId, SloClass};
use tokio::sync::watch;

use crate::config;
use crate::metrics::{Metrics, Refusal};

/// Sessions whose last usage is kept for token estimates.
const SESSIONS: usize = 4096;

/// The gateway's admission point.
#[derive(Debug)]
pub(crate) struct Admission {
    state: Mutex<State>,
    policy: Policy,
    queue_timeout: Duration,
    metrics: Arc<Metrics>,
}

#[derive(Debug)]
struct State {
    core: admission::Admission,
    /// Where each held request learns what was decided for it.
    signals: HashMap<RequestId, watch::Sender<Signal>>,
    sessions: Sessions,
}

/// What a held request was last told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Waiting,
    Admitted,
    /// Shed to make room: the request is no longer held, and its response must fail.
    Shed,
}

impl Admission {
    pub fn new(config: &config::Admission, metrics: Arc<Metrics>) -> Self {
        let state = State {
            core: admission::Admission::new(config.policy, config.queue_limit),
            signals: HashMap::new(),
            sessions: Sessions::default(),
        };
        Self {
            state: Mutex::new(state),
            policy: config.policy,
            queue_timeout: config.queue_timeout,
            metrics,
        }
    }

    /// Takes a request of `class` through the admission point. Returns its ticket once it is admitted, `None`
    /// for a class the policy does not hold (it goes upstream at once), or why it was refused.
    pub async fn admit(
        self: &Arc<Self>,
        class: SloClass,
        session: &SessionId,
        body_bytes: usize,
    ) -> Result<Option<Ticket>, Refusal> {
        if !self.policy.holds(class) {
            self.metrics.admitted(class, Duration::ZERO);
            return Ok(None);
        }
        // Background requests are priced where a policy counts their tokens: token-reservation admits by them,
        // and preempt-resubmit counts those its preemptions discard.
        let priced = matches!(
            self.policy,
            Policy::TokenReservation { .. } | Policy::PreemptResubmit
        );
        let estimated =
            (priced && class == SloClass::Background).then(|| (session.clone(), body_bytes as u64));
        // Every held request takes the lock, so what can be done before it is. Most requests are admitted at
        // once, so only one that waits has its signal changed under the lock.
        let (sender, signal) = watch::channel(Signal::Admitted);
        let (id, tokens) = {
            let mut state = self.lock();
            let tokens = estimated.as_ref().map_or(0, |(session, body_bytes)| {
                state.sessions.estimate(session, *body_bytes)
            });
            let Ok(arrival) = state.core.arrive(class, tokens) else {
                drop(state);
                self.metrics.refused(class, Refusal::QueueFull);
                return Err(Refusal::QueueFull);
            };
            if !arrival.admitted {
                sender.send_replace(Signal::Waiting);
            }
            state.signals.insert(arrival.id, sender);
            state.apply(arrival.actions);
            (arrival.id, tokens)
        };
        let mut ticket = Ticket {
            admission: Arc::clone(self),
            id,
            class,
            signal,
            tokens,
            estimated,
            waited: Duration::ZERO,
            prompt_tokens: None,
            first_byte: false,
            left: false,
        };
        let queued = Instant::now();
        ticket.wait().await?;
        ticket.waited = queued.elapsed();
        self.metrics.admitted(class, ticket.waited);
        Ok(Some(ticket))
    }

    /// The policy; `Debug` shows its parameters.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl State {
    /// Tells each request what the core decided for it.
    fn apply(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Admit(id) => self.signal(id, Signal::Admitted),
                Action::Preempt(id) => self.signal(id, Signal::Waiting),
                Action::Shed(id) => {
                    if let Some(signal) = self.signals.remove(&id) {
                        signal.send_replace(Signal::Shed);
                    }
                }
            }
        }
    }

    fn signal(&self, id: RequestId, signal: Signal) {
        if let Some(sender) = self.signals.get(&id) {
            sender.send_replace(signal);
        }
    }

    /// A request leaves the admission point, which may admit others. Returns the request's signal, for the
    /// caller to drop once it has let go of the lock.
    fn leave(&mut self, id: RequestId) -> Option<watch::Sender<Signal>> {
        let actions = self.core.leave(id);
        self.apply(actions);
        self.signals.remove(&id)
    }
}

/// A request's place at the admission point, from its arrival to the end of its response.
#[derive(Debug)]
pub(crate) struct Ticket {
    admission: Arc<Admission>,
    id: RequestId,
    class: SloClass,
    signal: watch::Receiver<Signal>,
    /// The request's estimated prompt tokens; 0 where its policy does not price it.
    tokens: u64,
    /// The session and body size of a request whose tokens were estimated: its usage informs the next
    /// estimate for the session.
    estimated: Option<(SessionId, u64)>,
    /// How long the request waited to be admitted.
    waited: Duration,
    /// The prompt tokens the response reported.
    pub prompt_tokens: Option<u64>,
    /// Whether the response's first byte has been recorded.
    first_byte: bool,
    /// Set when the request has already left, so dropping the ticket does nothing.
    left: bool,
}

impl Ticket {
    /// How long the request waited to be admitted, after the decision that made it wait.
    pub fn waited(&self) -> Duration {
        self.waited
    }

    /// Whether the policy may shed the request after admitting it: flow control sheds background requests.
    pub fn sheddable(&self) -> bool {
        self.class == SloClass::Background
            && matches!(self.admission.policy, Policy::FlowControl { .. })
    }

    /// Completes when the request is shed.
    pub async fn shed(&mut self) {
        // The sender goes only after it sent Shed, or when the request leaves, which an admitted request does
        // by dropping this ticket.
        let _ = self.signal.wait_for(|signal| *signal == Signal::Shed).await;
    }

    /// Whether the policy may preempt the request before its first byte: preempt-resubmit preempts
    /// background requests.
    pub fn preemptible(&self) -> bool {
        self.class == SloClass::Background && self.admission.policy == Policy::PreemptResubmit
    }

    /// Completes when the request is preempted: it waits again, and its upstream attempt is to be dropped.
    pub async fn preempted(&mut self) {
        let _ = self
            .signal
            .wait_for(|signal| *signal != Signal::Admitted)
            .await;
    }

    /// Counts the upstream attempt a preemption dropped, and waits until the request is admitted again, for at
    /// most `queue_timeout_s`.
    pub async fn readmitted(&mut self) -> Result<(), Refusal> {
        self.admission.metrics.preempted(self.class, self.tokens);
        self.wait().await?;
        self.admission.metrics.resubmitted(self.class);
        Ok(())
    }

    /// Records the response's first byte, which under preempt-resubmit ends a realtime request's hold on
    /// background work and makes a background request safe from preemption. False if the request was
    /// preempted first: the byte belongs to a dropped attempt and must not be relayed. Only a preemptible
    /// request can be.
    pub fn first_byte(&mut self) -> bool {
        if self.first_byte || self.admission.policy != Policy::PreemptResubmit {
            return true;
        }
        let mut state = self.admission.lock();
        let Some(actions) = state.core.first_byte(self.id) else {
            return false;
        };
        state.apply(actions);
        self.first_byte = true;
        true
    }

    /// Waits until the request is admitted, for at most `queue_timeout_s`.
    async fn wait(&mut self) -> Result<(), Refusal> {
        if *self.signal.borrow() == Signal::Admitted {
            return Ok(());
        }
        let _waiting = self.admission.metrics.waiting(self.class);
        let admitted = self.signal.wait_for(|signal| *signal == Signal::Admitted);
        if let Ok(Ok(_)) = tokio::time::timeout(self.admission.queue_timeout, admitted).await {
            return Ok(());
        }
        let mut state = self.admission.lock();
        // The timer and the admission can cross: a request admitted meanwhile goes on.
        if !state.core.is_waiting(self.id) {
            return Ok(());
        }
        let signal = state.leave(self.id);
        drop(state);
        drop(signal);
        self.left = true;
        self.admission.metrics.refused(self.class, Refusal::Timeout);
        Err(Refusal::Timeout)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.left {
            return;
        }
        let mut state = self.admission.lock();
        if let (Some((session, body_bytes)), Some(prompt_tokens)) =
            (&self.estimated, self.prompt_tokens)
        {
            let usage = SessionUsage {
                prompt_tokens,
                body_bytes: *body_bytes,
            };
            state.sessions.observe(session, usage);
        }
        let signal = state.leave(self.id);
        drop(state);
        drop(signal);
    }
}

/// The last usage of recent sessions, for token estimates. It holds at most [`SESSIONS`] sessions, whatever
/// clients send, and forgets the earliest first.
#[derive(Debug, Default)]
struct Sessions {
    usage: HashMap<SessionId, SessionUsage>,
    order: VecDeque<SessionId>,
}

impl Sessions {
    fn estimate(&self, session: &SessionId, body_bytes: u64) -> u64 {
        admission::estimate_tokens(body_bytes, self.usage.get(session).copied())
    }

    fn observe(&mut self, session: &SessionId, usage: SessionUsage) {
        if let Some(last) = self.usage.get_mut(session) {
            *last = usage;
            return;
        }
        if self.order.len() == SESSIONS
            && let Some(earliest) = self.order.pop_front()
        {
            self.usage.remove(&earliest);
        }
        self.order.push_back(session.clone());
        self.usage.insert(session.clone(), usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_are_bounded_and_forget_the_earliest_first() {
        let mut sessions = Sessions::default();
        let usage = |prompt_tokens| SessionUsage {
            prompt_tokens,
            body_bytes: 400,
        };
        for n in 0..=SESSIONS {
            let session = SessionId::new(format!("s-{n}")).unwrap();
            sessions.observe(&session, usage(n as u64 + 1));
        }
        assert_eq!(sessions.usage.len(), SESSIONS);
        let first = SessionId::new("s-0").unwrap();
        let last = SessionId::new(format!("s-{SESSIONS}")).unwrap();
        // Forgotten: a quarter token per byte again.
        assert_eq!(sessions.estimate(&first, 400), 100);
        assert_eq!(sessions.estimate(&last, 400), SESSIONS as u64 + 1);
        sessions.observe(&last, usage(7));
        assert_eq!(sessions.estimate(&last, 800), 107);
    }
}
