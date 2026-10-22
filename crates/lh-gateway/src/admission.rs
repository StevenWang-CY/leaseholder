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

use lh_core::admission::{self, Action, Policy, QueueFull, RequestId, SessionUsage};
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
        let arrived = Instant::now();
        if !self.policy.holds(class) {
            self.metrics.admitted(class, Duration::ZERO);
            return Ok(None);
        }
        let mut ticket = {
            let mut state = self.lock();
            // Only token-reservation prices requests, and only background ones.
            let estimated = (matches!(self.policy, Policy::TokenReservation { .. })
                && class == SloClass::Background)
                .then(|| (session.clone(), body_bytes as u64));
            let tokens = estimated.as_ref().map_or(0, |(session, body_bytes)| {
                state.sessions.estimate(session, *body_bytes)
            });
            let arrival = match state.core.arrive(class, tokens) {
                Ok(arrival) => arrival,
                Err(QueueFull) => {
                    self.metrics.refused(class, Refusal::QueueFull);
                    return Err(Refusal::QueueFull);
                }
            };
            let signal = if arrival.admitted {
                Signal::Admitted
            } else {
                Signal::Waiting
            };
            let (sender, signal) = watch::channel(signal);
            state.signals.insert(arrival.id, sender);
            state.apply(arrival.actions);
            Ticket {
                admission: Arc::clone(self),
                id: arrival.id,
                class,
                signal,
                estimated,
                prompt_tokens: None,
                left: false,
            }
        };
        ticket.wait().await?;
        self.metrics.admitted(class, arrived.elapsed());
        Ok(Some(ticket))
    }

    /// The policy; `Debug` shows its parameters.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    pub fn queue_timeout(&self) -> Duration {
        self.queue_timeout
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

    /// A request leaves the admission point, which may admit others.
    fn leave(&mut self, id: RequestId) {
        self.signals.remove(&id);
        let actions = self.core.leave(id);
        self.apply(actions);
    }
}

/// A request's place at the admission point, from its arrival to the end of its response.
#[derive(Debug)]
pub(crate) struct Ticket {
    admission: Arc<Admission>,
    id: RequestId,
    class: SloClass,
    signal: watch::Receiver<Signal>,
    /// The session and body size of a request whose tokens were estimated: its usage informs the next
    /// estimate for the session.
    estimated: Option<(SessionId, u64)>,
    /// The prompt tokens the response reported.
    pub prompt_tokens: Option<u64>,
    /// Set when the request has already left, so dropping the ticket does nothing.
    left: bool,
}

impl Ticket {
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
        state.leave(self.id);
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
            (self.estimated.take(), self.prompt_tokens)
        {
            let usage = SessionUsage {
                prompt_tokens,
                body_bytes,
            };
            state.sessions.observe(session, usage);
        }
        state.leave(self.id);
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

    fn observe(&mut self, session: SessionId, usage: SessionUsage) {
        if let Some(last) = self.usage.get_mut(&session) {
            *last = usage;
            return;
        }
        if self.order.len() == SESSIONS
            && let Some(earliest) = self.order.pop_front()
        {
            self.usage.remove(&earliest);
        }
        self.order.push_back(session.clone());
        self.usage.insert(session, usage);
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
            sessions.observe(session, usage(n as u64 + 1));
        }
        assert_eq!(sessions.usage.len(), SESSIONS);
        let first = SessionId::new("s-0").unwrap();
        let last = SessionId::new(format!("s-{SESSIONS}")).unwrap();
        // Forgotten: a quarter token per byte again.
        assert_eq!(sessions.estimate(&first, 400), 100);
        assert_eq!(sessions.estimate(&last, 400), SESSIONS as u64 + 1);
        sessions.observe(last.clone(), usage(7));
        assert_eq!(sessions.estimate(&last, 800), 107);
    }
}
