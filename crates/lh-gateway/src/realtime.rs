//! Policy leaseholder's view of the realtime sessions: where each one is in its turn, from the lifecycle events
//! its emitter posts and from its own realtime requests. `lh_core::admission` decides from how many sessions are
//! in each phase ([`Turns`]); this module keeps the phases, ends those that time out, and counts the headroom's
//! false alarms.
//!
//! A session goes `Idle` → `UserSpeaking` (`user_speech_started`) → `Critical` (`user_turn_ended`, or its
//! realtime request if that comes first, as LiveKit's preemptive generation sends it) → `Playback`
//! (`playback_started`, the reply's first audio) → `Idle` (`playback_ended`). A barge-in (`interrupted`) is the
//! user speaking again, and `session_closed` forgets the session.
//! - **Lost or reordered events.** An event moves its session to the event's phase whatever the phase was, in the
//!   order the gateway receives events, so a lost or late event costs the phases until the next one. Every phase
//!   but `Idle` ends after `state_timeout_s` without an event or request, counted.
//! - **No events.** A session that has posted none (its emitter sends none, or events are off) follows its
//!   realtime requests alone: it is `Critical` while one is in flight, and forgotten when the last one ends.
//! - **False alarms.** An arming of the headroom (from speech onset, or only in the critical window) is followed
//!   if a realtime request of the session arrives before the session leaves the armed phases; otherwise it is a
//!   false alarm, charged the token-seconds of background work the headroom held back meanwhile.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lh_core::admission::{Trigger, Turns};
use lh_proto::{EventKind, SessionId};
use prometheus_client::metrics::gauge::Gauge;

use crate::metrics::Metrics;

/// Realtime sessions followed at most, a bound on what emitters can make the gateway hold. A new session that
/// finds the table full first forgets the idle ones; if every session is in a turn, it is not followed.
const SESSIONS: usize = 4096;

/// Where a realtime session is in its turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Idle,
    UserSpeaking,
    /// From the end of the user's turn, or the realtime request if it comes first, to the reply's first audio.
    Critical,
    Playback,
}

impl Phase {
    pub const ALL: [Self; 4] = [
        Self::Idle,
        Self::UserSpeaking,
        Self::Critical,
        Self::Playback,
    ];

    /// The name, also a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::UserSpeaking => "user_speaking",
            Self::Critical => "critical",
            Self::Playback => "playback",
        }
    }

    /// The counts of one session in this phase.
    const fn turns(self) -> Turns {
        Turns {
            speaking: matches!(self, Self::UserSpeaking) as u32,
            critical: matches!(self, Self::Critical) as u32,
        }
    }
}

/// The realtime sessions and their phases.
#[derive(Debug)]
pub(crate) struct Realtime {
    sessions: HashMap<SessionId, Session>,
    /// Sessions in each phase, in [`Phase::ALL`] order, and their gauges.
    counts: [u32; 4],
    gauges: [Gauge; 4],
    /// What arms the headroom; `None` with the headroom off.
    trigger: Option<Trigger>,
    timeout: Duration,
    held: HeldBack,
    metrics: Arc<Metrics>,
}

#[derive(Debug)]
struct Session {
    phase: Phase,
    /// When the session entered its phase, or an event or a request last put it there.
    since: Instant,
    /// Whether the session has posted a lifecycle event; until it does, its realtime requests drive it.
    evented: bool,
    /// Its realtime requests in flight.
    requests: u32,
    /// An arming of the headroom that no realtime request has followed yet: the token-seconds held back when
    /// it began.
    armed: Option<f64>,
}

/// The integral over time of the estimated background tokens the headroom holds back.
#[derive(Debug)]
struct HeldBack {
    token_seconds: f64,
    tokens: u64,
    at: Instant,
}

impl HeldBack {
    fn until(&self, now: Instant) -> f64 {
        let held = now.saturating_duration_since(self.at).as_secs_f64();
        self.token_seconds + self.tokens as f64 * held
    }
}

impl Realtime {
    pub fn new(trigger: Option<Trigger>, timeout: Duration, metrics: Arc<Metrics>) -> Self {
        Self {
            sessions: HashMap::new(),
            counts: [0; 4],
            gauges: Phase::ALL.map(|phase| metrics.realtime_sessions(phase)),
            trigger,
            timeout,
            held: HeldBack {
                token_seconds: 0.0,
                tokens: 0,
                at: Instant::now(),
            },
            metrics,
        }
    }

    /// The sessions in each phase the policy acts on.
    pub fn turns(&self) -> Turns {
        Turns {
            speaking: self.counts[Phase::UserSpeaking as usize],
            critical: self.counts[Phase::Critical as usize],
        }
    }

    /// From `now` on, the headroom holds back `tokens` of background work.
    pub fn held_back(&mut self, now: Instant, tokens: u64) {
        self.held.token_seconds = self.held.until(now);
        self.held.tokens = tokens;
        self.held.at = now;
    }

    /// A lifecycle event of `session`, received now.
    pub fn event(&mut self, session: &SessionId, kind: &EventKind, now: Instant) {
        let phase = match kind {
            EventKind::UserSpeechStarted | EventKind::Interrupted { .. } => Phase::UserSpeaking,
            EventKind::UserTurnEnded => Phase::Critical,
            EventKind::PlaybackStarted => Phase::Playback,
            EventKind::PlaybackEnded { .. } => Phase::Idle,
            EventKind::SessionClosed => return self.forget(session, now),
            // The agents' events.
            EventKind::TurnStarted
            | EventKind::TurnFinished { .. }
            | EventKind::ToolWaitStarted { .. }
            | EventKind::ToolWaitEnded { .. } => return,
        };
        if let Some(state) = self.follow(session, now) {
            state.evented = true;
            self.enter(session, phase, now);
        }
    }

    /// A realtime request of `session` arrived: the session is in its critical window, and an arming of the
    /// headroom is followed.
    pub fn request_started(&mut self, session: &SessionId, now: Instant) {
        if let Some(state) = self.follow(session, now) {
            state.requests += 1;
            self.enter(session, Phase::Critical, now);
            if let Some(state) = self.sessions.get_mut(session) {
                state.armed = None;
            }
        }
    }

    /// A realtime request of `session` ended. A session without events is forgotten with its last request.
    pub fn request_ended(&mut self, session: &SessionId, now: Instant) {
        let Some(state) = self.sessions.get_mut(session) else {
            return;
        };
        state.requests = state.requests.saturating_sub(1);
        if !state.evented && state.requests == 0 {
            self.forget(session, now);
        }
    }

    /// Ends the phases that lasted `state_timeout_s`, and returns when the next one will.
    pub fn expire(&mut self, now: Instant) -> Option<Instant> {
        let timeout = self.timeout;
        let ended: Vec<(SessionId, Phase)> = self
            .sessions
            .iter()
            .filter(|(_, state)| state.phase != Phase::Idle && now >= state.since + timeout)
            .map(|(session, state)| (session.clone(), state.phase))
            .collect();
        for (session, phase) in ended {
            self.metrics.state_timeout(phase);
            self.enter(&session, Phase::Idle, now);
        }
        self.sessions
            .values()
            .filter(|state| state.phase != Phase::Idle)
            .map(|state| state.since + timeout)
            .min()
    }

    /// The session's state, created idle if the gateway does not follow it yet; `None` if the table is full of
    /// sessions in a turn.
    fn follow(&mut self, session: &SessionId, now: Instant) -> Option<&mut Session> {
        if !self.sessions.contains_key(session) {
            if self.sessions.len() >= SESSIONS {
                self.sessions.retain(|_, state| state.phase != Phase::Idle);
                self.counts[Phase::Idle as usize] = 0;
                self.gauges[Phase::Idle as usize].set(0);
                if self.sessions.len() >= SESSIONS {
                    return None;
                }
            }
            let state = Session {
                phase: Phase::Idle,
                since: now,
                evented: false,
                requests: 0,
                armed: None,
            };
            self.sessions.insert(session.clone(), state);
            self.count(Phase::Idle, 1);
        }
        self.sessions.get_mut(session)
    }

    /// Moves a followed session to `phase`, or confirms it there, at `now`. An arming that ends here is a false
    /// alarm.
    fn enter(&mut self, session: &SessionId, phase: Phase, now: Instant) {
        let Some(state) = self.sessions.get_mut(session) else {
            return;
        };
        let was = std::mem::replace(&mut state.phase, phase);
        state.since = now;
        let armed = |phase: Phase| self.trigger.is_some_and(|t| t.armed(phase.turns()) > 0);
        match (armed(was), armed(phase)) {
            (false, true) => state.armed = Some(self.held.until(now)),
            (true, false) => {
                if let Some(at) = state.armed.take() {
                    self.metrics.false_alarm(self.held.until(now) - at);
                }
            }
            _ => {}
        }
        if was != phase {
            self.count(was, -1);
            self.count(phase, 1);
        }
    }

    /// Stops following a session. An arming it leaves unfollowed is a false alarm.
    fn forget(&mut self, session: &SessionId, now: Instant) {
        if let Some(state) = self.sessions.remove(session) {
            if let Some(at) = state.armed {
                self.metrics.false_alarm(self.held.until(now) - at);
            }
            self.count(state.phase, -1);
        }
    }

    fn count(&mut self, phase: Phase, change: i32) {
        let count = &mut self.counts[phase as usize];
        *count = count.saturating_add_signed(change);
        self.gauges[phase as usize].set(i64::from(*count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: &str) -> SessionId {
        SessionId::new(id).unwrap()
    }

    fn tracker(trigger: Option<Trigger>) -> (Realtime, Arc<Metrics>) {
        let metrics = Arc::new(Metrics::new());
        let realtime = Realtime::new(trigger, Duration::from_secs(10), Arc::clone(&metrics));
        (realtime, metrics)
    }

    fn turns(speaking: u32, critical: u32) -> Turns {
        Turns { speaking, critical }
    }

    #[test]
    fn a_turn_goes_through_its_phases_and_back_to_idle() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("voice-RM_1"), Instant::now());
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        assert_eq!(rt.turns(), turns(1, 0));
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.request_started(&s, t0);
        rt.request_ended(&s, t0);
        assert_eq!(
            rt.turns(),
            turns(0, 1),
            "a session with events leaves its window at its first audio"
        );
        rt.event(&s, &EventKind::PlaybackStarted, t0);
        assert_eq!(rt.turns(), turns(0, 0));
        rt.event(&s, &EventKind::Interrupted { played_ms: 900 }, t0);
        assert_eq!(rt.turns(), turns(1, 0), "a barge-in is the user speaking");
        rt.event(&s, &EventKind::SessionClosed, t0);
        assert_eq!((rt.turns(), rt.sessions.len()), (turns(0, 0), 0));
        // The speaking user's arming, which no request followed, ended with the session.
        assert_eq!(metrics.false_alarms().0, 1);
    }

    #[test]
    fn a_session_without_events_follows_its_requests() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("bench-1"), Instant::now());
        rt.request_started(&s, t0);
        rt.request_started(&s, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.request_ended(&s, t0);
        assert_eq!(rt.turns(), turns(0, 1), "one request is still in flight");
        rt.request_ended(&s, t0);
        assert_eq!((rt.turns(), rt.sessions.len()), (turns(0, 0), 0));
        assert_eq!(metrics.false_alarms().0, 0);
    }

    #[test]
    fn every_phase_but_idle_times_out() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        let (a, b) = (session("voice-a"), session("voice-b"));
        rt.event(&a, &EventKind::UserTurnEnded, t0);
        rt.event(&b, &EventKind::PlaybackStarted, t0 + Duration::from_secs(4));
        assert_eq!(rt.expire(t0), Some(t0 + Duration::from_secs(10)));
        // A lost playback_started: the window ends at the timeout, counted, and so does its arming.
        let next = rt.expire(t0 + Duration::from_secs(10));
        assert_eq!(next, Some(t0 + Duration::from_secs(14)));
        assert_eq!(rt.turns(), turns(0, 0));
        assert_eq!(metrics.false_alarms().0, 1);
        assert_eq!(rt.expire(t0 + Duration::from_secs(14)), None);
        let timeouts = |phase| metrics.state_timeouts(phase);
        assert_eq!(
            (timeouts(Phase::Critical), timeouts(Phase::Playback)),
            (1, 1)
        );
        // An event confirms a phase, and its timeout starts again.
        rt.event(
            &a,
            &EventKind::UserSpeechStarted,
            t0 + Duration::from_secs(20),
        );
        rt.event(
            &a,
            &EventKind::UserSpeechStarted,
            t0 + Duration::from_secs(25),
        );
        assert_eq!(
            rt.expire(t0 + Duration::from_secs(30)),
            Some(t0 + Duration::from_secs(35))
        );
    }

    #[test]
    fn events_move_a_session_whatever_its_phase() {
        let (mut rt, _) = tracker(Some(Trigger::Onset));
        let (s, t0) = (session("voice-RM_2"), Instant::now());
        // The onset was lost: the end of the turn still opens the window, and a late playback_ended ends it.
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        assert_eq!(rt.turns(), turns(0, 1));
        rt.event(&s, &EventKind::PlaybackEnded { played_ms: 2_000 }, t0);
        assert_eq!(rt.turns(), turns(0, 0));
        // The agents' events change nothing.
        rt.event(&s, &EventKind::TurnStarted, t0);
        assert_eq!((rt.turns(), rt.counts), (turns(0, 0), [1, 0, 0, 0]));
    }

    #[test]
    fn a_false_alarm_is_charged_what_the_headroom_held_back_meanwhile() {
        let (mut rt, metrics) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        rt.held_back(t0, 0);
        let (noise, turn) = (session("voice-noise"), session("voice-turn"));
        rt.event(&noise, &EventKind::UserSpeechStarted, at(1));
        rt.held_back(at(2), 1_000);
        // A real turn armed meanwhile is followed by its request.
        rt.event(&turn, &EventKind::UserSpeechStarted, at(2));
        rt.request_started(&turn, at(3));
        rt.held_back(at(4), 0);
        // The noise's arming times out: 1,000 tokens held back for 2 s.
        rt.expire(at(11));
        let (alarms, token_seconds) = metrics.false_alarms();
        assert_eq!(alarms, 1);
        assert!((token_seconds - 2_000.0).abs() < 1e-6, "{token_seconds}");
    }

    #[test]
    fn the_request_trigger_arms_only_the_critical_window() {
        let (mut rt, metrics) = tracker(Some(Trigger::Request));
        let (s, t0) = (session("voice-RM_3"), Instant::now());
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        rt.event(&s, &EventKind::PlaybackStarted, t0);
        assert_eq!(metrics.false_alarms().0, 0, "speech alone arms nothing");
        // An end of turn that no request follows is one.
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        rt.event(&s, &EventKind::UserSpeechStarted, t0);
        assert_eq!(metrics.false_alarms().0, 1);
        // Without the headroom nothing is armed.
        let (mut rt, metrics) = tracker(None);
        rt.event(&s, &EventKind::UserTurnEnded, t0);
        rt.event(&s, &EventKind::SessionClosed, t0);
        assert_eq!(metrics.false_alarms().0, 0);
    }

    #[test]
    fn the_table_is_bounded_and_forgets_idle_sessions_first() {
        let (mut rt, _) = tracker(Some(Trigger::Onset));
        let t0 = Instant::now();
        rt.event(
            &session("voice-speaking"),
            &EventKind::UserSpeechStarted,
            t0,
        );
        for n in 1..SESSIONS {
            let s = session(&format!("voice-{n}"));
            rt.event(&s, &EventKind::PlaybackEnded { played_ms: 0 }, t0);
        }
        assert_eq!(rt.sessions.len(), SESSIONS);
        rt.event(&session("voice-new"), &EventKind::UserTurnEnded, t0);
        assert_eq!(rt.sessions.len(), 2);
        assert_eq!((rt.turns(), rt.counts), (turns(1, 1), [0, 1, 1, 0]));
    }
}
