//! The GPU nodes, and the sessions placed on them.
//!
//! A gateway with one `upstream` has one node, without a name, a probe or a placement: its requests go to it as
//! they always have. With `[nodes]`, each node has its engine's connection pool, its own admission point (one
//! `lh_core` admission and, under policy leaseholder, one [`Realtime`](crate::realtime::Realtime): KV estimates
//! are per engine) and a readiness probe, and each session is placed on one of them ([`Placement`]):
//! - **Sticky.** A session is placed by its first request, or by `POST /v1/sessions`, and stays on its node until
//!   it has gone `idle_s` without a request. `lh_core::placement` chooses the node of a session that has none,
//!   from what each node holds.
//! - **Readiness.** Each node's `/health` is probed every `probe_interval_s`. A node that fails three probes in a
//!   row takes no new sessions, and its sessions are placed again by their next request: their KV stays behind,
//!   so each move is counted.
//! - **Events.** A session's lifecycle events go to its node's admission point, and `session_closed` ends its
//!   placement.
//! - **Minted sessions.** A request that names no session is placed on its own and not remembered: no later
//!   request can name the session the gateway minted for it.
//! - **No waiting.** A request of a placed session reads the placements under a shared lock and counts itself on
//!   its node with atomics, and its end takes no lock: requests do not wait on each other here. Placing a
//!   session, the events and the expiry write the placements.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::response::Response;
use http::StatusCode;
use lh_core::placement::{self as policy, Policy};
use lh_proto::{EventKind, LifecycleEvent, SessionId, SloClass};
use prometheus_client::metrics::gauge::Gauge;
use serde::Deserialize;
use tokio::time::MissedTickBehavior;

use crate::admission::Admission;
use crate::config;
use crate::exchange::Exchange;
use crate::metrics::{Metrics, Route};
use crate::proxy::{Upstream, read_body};
use crate::server::{App, Shared};

/// Sessions remembered at most, a bound on what clients can make the gateway hold. A new session that finds the
/// table full is placed but not remembered.
const SESSIONS: usize = 4096;

/// Failed probes in a row after which a node takes no new sessions.
const PROBES_DOWN: u32 = 3;

/// A GPU node: its engine's connection pool and, under an `[admission]` policy, its own admission point.
#[derive(Debug)]
pub(crate) struct Node {
    /// `None` for the one upstream of a gateway without `[nodes]`.
    pub name: Option<Arc<str>>,
    pub upstream: Upstream,
    /// `None` under policy "none": requests go upstream at once.
    pub admission: Option<Arc<Admission>>,
    /// Records the series the node keeps.
    pub metrics: Arc<Metrics>,
    /// Whether it serves each class, strictest first (placement dedicated).
    serves: [bool; 3],
    /// Whether its readiness probe answers. A node starts ready, and only a gateway with `[nodes]` probes.
    ready: AtomicBool,
}

impl Node {
    pub fn new(
        name: Option<Arc<str>>,
        upstream: Upstream,
        admission: Option<&config::Admission>,
        metrics: Arc<Metrics>,
        serves: [bool; 3],
    ) -> Self {
        let admission = admission.map(|admission| Admission::new(admission, Arc::clone(&metrics)));
        if name.is_some() {
            metrics.node_ready().set(1);
        }
        Self {
            name,
            upstream,
            admission,
            metrics,
            serves,
            ready: AtomicBool::new(true),
        }
    }

    /// Whether the node takes new sessions.
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Records whether the node answers its probe, logging a change.
    fn set_ready(&self, ready: bool) {
        self.metrics.node_ready().set(i64::from(ready));
        if self.ready.swap(ready, Ordering::Relaxed) == ready {
            return;
        }
        let node = self.name.as_deref().unwrap_or_default();
        if ready {
            tracing::info!(node, "node {node} is ready again");
        } else {
            tracing::warn!(
                node,
                "node {node} takes no new sessions: its /health failed {PROBES_DOWN} probes in a row"
            );
        }
    }
}

/// Why a session was placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    /// It had no node: a new session, or one that idled out.
    New,
    /// Its node failed readiness, and its KV stayed behind.
    NodeDown,
}

impl Reason {
    pub const ALL: [Self; 2] = [Self::New, Self::NodeDown];

    /// The name, also a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::NodeDown => "node_down",
        }
    }
}

/// The sessions placed on the nodes of a gateway with `[nodes]`.
#[derive(Debug)]
pub(crate) struct Placement {
    policy: Policy,
    probe_interval: Duration,
    /// The placements: a request of a placed session reads them, placing a session writes them.
    table: RwLock<Table>,
    metrics: Arc<Metrics>,
}

impl Placement {
    /// The placement of `nodes`, with the task that ends idle sessions' placements.
    pub fn new(config: &config::Placement, nodes: &[Node], metrics: Arc<Metrics>) -> Arc<Self> {
        let placement = Arc::new(Self {
            policy: config.policy,
            probe_interval: config.probe_interval,
            table: RwLock::new(Table::new(nodes.iter().map(|node| &*node.metrics))),
            metrics,
        });
        tokio::spawn(expire(Arc::downgrade(&placement), config.idle));
        placement
    }

    pub fn policy(&self) -> Policy {
        self.policy
    }

    pub fn probe_interval(&self) -> Duration {
        self.probe_interval
    }

    /// Places a request of `session`, as `class` asks: on the session's node while it holds, otherwise on the
    /// node the policy chooses; `None` when no ready node takes it. A request that does not name its session
    /// (`named` false: the gateway minted it) is placed, but the session is not remembered.
    pub fn request(
        &self,
        nodes: &[Node],
        session: &SessionId,
        class: SloClass,
        named: bool,
    ) -> Option<InFlight> {
        let ready = |node: usize| nodes[node].ready();
        {
            let table = self.read();
            if let Found::On(placed) = table.find(session, ready) {
                return Some(table.start(placed.node, Some(placed)));
            }
        }
        let mut table = self.write();
        let (node, placed) = match table.find(session, ready) {
            // Placed by another of its requests meanwhile.
            Found::On(placed) => (placed.node, Some(placed)),
            found => {
                let node = self.choose(nodes, &table, class)?;
                self.metrics.placed(self.policy, found.reason());
                let placed = named
                    .then(|| table.place(session, class, node, false, Instant::now()))
                    .flatten();
                (node, placed)
            }
        };
        Some(table.start(node, placed))
    }

    /// `POST /v1/sessions`: the node of `session`, placed now as `class` asks if it has none; `None` when no ready
    /// node takes it. A session placed this way counts as a request in flight on its node until its first
    /// request, so that calls which start together are not all placed on the same node.
    pub fn session(&self, nodes: &[Node], session: &SessionId, class: SloClass) -> Option<usize> {
        let mut table = self.write();
        match table.find(session, |node| nodes[node].ready()) {
            Found::On(placed) => Some(placed.node),
            found => {
                let node = self.choose(nodes, &table, class)?;
                self.metrics.placed(self.policy, found.reason());
                table.place(session, class, node, true, Instant::now());
                Some(node)
            }
        }
    }

    /// Lifecycle events, in the order the gateway received them, to the admission point of their session's node.
    /// The events of a session without a node reach none, and `session_closed` ends a session's placement.
    pub fn events(&self, nodes: &[Node], events: &[LifecycleEvent]) {
        let nodes_of: Vec<Option<usize>> = {
            let mut table = self.write();
            events
                .iter()
                .map(|event| {
                    let node = table.sessions.get(&event.session_id).map(|s| s.node);
                    if matches!(event.kind, EventKind::SessionClosed) {
                        table.forget(&event.session_id);
                    }
                    node
                })
                .collect()
        };
        for (index, node) in nodes.iter().enumerate() {
            let Some(admission) = &node.admission else {
                continue;
            };
            let mut own = events
                .iter()
                .zip(&nodes_of)
                .filter(|(_, of)| **of == Some(index))
                .map(|(event, _)| event)
                .peekable();
            if own.peek().is_some() {
                admission.events(own);
            }
        }
    }

    /// The node the policy places a session of `class` on, from what each node holds now.
    fn choose(&self, nodes: &[Node], table: &Table, class: SloClass) -> Option<usize> {
        let leaseholder = self.policy == Policy::Leaseholder;
        let snapshot: Vec<policy::Node> = nodes
            .iter()
            .zip(&table.nodes)
            .zip(table.in_flight.iter())
            .map(|((node, held), in_flight)| {
                let mut snapshot = policy::Node {
                    ready: node.ready(),
                    serves: node.serves,
                    in_flight: in_flight.load(Ordering::Relaxed),
                    realtime_sessions: held.sessions[SloClass::Realtime as usize],
                    ..policy::Node::default()
                };
                if leaseholder && let Some(admission) = &node.admission {
                    let load = admission.load();
                    snapshot.armed = load.armed;
                    snapshot.background_tokens = load.background_tokens;
                    snapshot.free_tokens = load.free_tokens;
                }
                snapshot
            })
            .collect();
        self.policy.place(class, &snapshot)
    }

    fn read(&self) -> RwLockReadGuard<'_, Table> {
        self.table.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Table> {
        self.table.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A request's place on its node, from its placement to its end; dropping it ends the request there, without the
/// table's lock.
#[derive(Debug)]
pub(crate) struct InFlight {
    /// The table's count of each node's requests in flight.
    in_flight: Arc<[AtomicU32]>,
    node: usize,
    /// The request's session, when the table remembers it.
    session: Option<Arc<Session>>,
}

impl InFlight {
    /// The index of the request's node.
    pub fn node(&self) -> usize {
        self.node
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        end(
            &self.in_flight,
            self.session.as_deref(),
            self.node,
            Instant::now(),
        );
    }
}

/// Ends a request on `node`, of `session` when the table remembers it, at `now`. A session placed again meanwhile
/// keeps its new placement, which the request did not count on.
fn end(in_flight: &[AtomicU32], session: Option<&Session>, node: usize, now: Instant) {
    if let Some(placed) = session {
        placed.ended(now);
    }
    in_flight[node].fetch_sub(1, Ordering::Relaxed);
}

/// Ends the placements of the sessions idle for `idle`, until the placement is gone. Every session idles out
/// `idle` after it last went idle, so none that goes idle while this sleeps can idle out before it wakes.
async fn expire(placement: Weak<Placement>, idle: Duration) {
    let mut next = Instant::now() + idle;
    loop {
        tokio::time::sleep_until(next.into()).await;
        let Some(placement) = placement.upgrade() else {
            return;
        };
        let now = Instant::now();
        let soonest = placement.write().expire(now, idle);
        next = soonest.unwrap_or(now + idle);
    }
}

/// Probes the `/health` of the node at `index` every `interval`, for at most `interval` each, while the gateway
/// runs: it takes no new sessions once [`PROBES_DOWN`] probes in a row fail, and again after one that succeeds.
pub(crate) async fn probe(shared: Weak<Shared>, index: usize, interval: Duration) {
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut failed = 0u32;
    loop {
        ticks.tick().await;
        let Some(shared) = shared.upgrade() else {
            return;
        };
        let node = &shared.nodes[index];
        failed = if node.upstream.healthy(interval).await {
            0
        } else {
            failed.saturating_add(1)
        };
        node.set_ready(failed < PROBES_DOWN);
    }
}

/// The body of `POST /v1/sessions`. Unknown fields are ignored.
#[derive(Deserialize)]
struct Asked {
    session_id: SessionId,
    /// The tenant's default class when absent.
    class: Option<SloClass>,
}

/// `POST /v1/sessions` from a trusted tenant, with gateways that have `[nodes]`: `{"session_id": ...,
/// "class": ...}` gets 200 `{"session_id", "class", "node"}`, the node the session is placed on, placed now if it
/// has none. Asking again returns the same node while the placement holds.
pub(crate) async fn post(State(app): State<App>, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Sessions);
    let (parts, body) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));
    if !tenant.trusted {
        return ex.error(
            StatusCode::FORBIDDEN,
            "untrusted_tenant",
            "only trusted tenants may place sessions",
        );
    }
    let body = match read_body(body, shared.max_body_bytes).await {
        Ok(body) => body,
        Err((status, code, message)) => return ex.error(status, code, message),
    };
    let asked: Asked = match serde_json::from_slice(&body) {
        Ok(asked) => asked,
        Err(err) => {
            let message = format!("expected {{\"session_id\": ..., \"class\": ...}}: {err}");
            return ex.error(StatusCode::BAD_REQUEST, "invalid_session", message);
        }
    };
    // As for a request, a class stricter than the tenant's max_class is clamped to it.
    let wanted = asked.class.unwrap_or(tenant.default_class);
    let class = wanted.max(tenant.max_class);
    if class != wanted {
        shared.metrics.class_clamped();
    }
    ex.class = Some(class);
    let placed = shared
        .placement
        .as_ref()
        .and_then(|placement| placement.session(&shared.nodes, &asked.session_id, class));
    let Some(index) = placed else {
        ex.session = Some(asked.session_id);
        let message = format!("no ready node takes {class} sessions");
        return ex.error(StatusCode::SERVICE_UNAVAILABLE, "no_ready_node", message);
    };
    let node = &shared.nodes[index];
    ex.on_node(node);
    let body = serde_json::json!({
        "session_id": asked.session_id.as_str(),
        "class": class,
        "node": node.name.as_deref(),
    });
    ex.session = Some(asked.session_id);
    ex.respond(StatusCode::OK, &body)
}

/// The sessions placed on the nodes, and what each node holds.
#[derive(Debug)]
struct Table {
    sessions: HashMap<SessionId, Arc<Session>>,
    /// Each node's requests in flight, and the sessions placed on it by `POST /v1/sessions` that have sent none
    /// yet, in the order of the nodes. Requests count themselves here under the table's shared lock, and end
    /// without it.
    in_flight: Arc<[AtomicU32]>,
    /// The sessions placed on each node, in the order of the nodes.
    nodes: Vec<Held>,
}

/// The sessions placed on a node.
#[derive(Debug)]
struct Held {
    /// By class, strictest first, and their gauges.
    sessions: [u32; 3],
    gauges: [Gauge; 3],
}

/// A session's placement. Its requests hold it until they end, so a request of a session placed again meanwhile
/// ends on the placement it started on.
#[derive(Debug)]
struct Session {
    node: usize,
    /// The class it was placed with.
    class: SloClass,
    /// When it was placed.
    placed: Instant,
    /// Its requests in flight.
    requests: AtomicU32,
    /// Placed by `POST /v1/sessions` and no request since: it counts as one in flight on its node.
    pending: AtomicBool,
    /// When its last request ended, in nanoseconds after it was placed (0 before any did).
    idle_from: AtomicU64,
}

impl Session {
    /// One of its requests ended at `now`.
    fn ended(&self, now: Instant) {
        let after = u64::try_from(now.saturating_duration_since(self.placed).as_nanos());
        self.idle_from
            .fetch_max(after.unwrap_or(u64::MAX), Ordering::Release);
        self.requests.fetch_sub(1, Ordering::AcqRel);
    }

    /// Whether it has a request in flight.
    fn busy(&self) -> bool {
        self.requests.load(Ordering::Acquire) > 0
    }

    /// When its idle time began: when its last request ended, or when it was placed.
    fn since(&self) -> Instant {
        self.placed + Duration::from_nanos(self.idle_from.load(Ordering::Acquire))
    }
}

/// What became of a session's placement.
#[derive(Debug)]
enum Found {
    /// It holds: the node is ready.
    On(Arc<Session>),
    /// There is none: a new session, or one that idled out.
    None,
    /// Its node failed readiness.
    Lost,
}

impl Found {
    fn reason(&self) -> Reason {
        match self {
            Self::Lost => Reason::NodeDown,
            Self::On(_) | Self::None => Reason::New,
        }
    }
}

impl Table {
    /// An empty table for the nodes whose metric handles are `nodes`, in order.
    fn new<'a>(nodes: impl IntoIterator<Item = &'a Metrics>) -> Self {
        let held = |metrics: &Metrics| Held {
            sessions: [0; 3],
            gauges: SloClass::ALL.map(|class| metrics.sessions(class)),
        };
        let nodes: Vec<Held> = nodes.into_iter().map(held).collect();
        Self {
            sessions: HashMap::new(),
            in_flight: nodes.iter().map(|_| AtomicU32::new(0)).collect(),
            nodes,
        }
    }

    /// The placement of `session`, if it holds on a node that is `ready`.
    fn find(&self, session: &SessionId, ready: impl Fn(usize) -> bool) -> Found {
        match self.sessions.get(session) {
            Some(placed) if ready(placed.node) => Found::On(Arc::clone(placed)),
            Some(_) => Found::Lost,
            None => Found::None,
        }
    }

    /// Places `session` on `node`, replacing any placement it had: by a request, or by `POST /v1/sessions`
    /// (`pending`). A full table does not remember a new session: `None`.
    fn place(
        &mut self,
        session: &SessionId,
        class: SloClass,
        node: usize,
        pending: bool,
        now: Instant,
    ) -> Option<Arc<Session>> {
        if let Some(old) = self.sessions.remove(session) {
            self.release(&old);
        } else if self.sessions.len() >= SESSIONS {
            return None;
        }
        let placed = Arc::new(Session {
            node,
            class,
            placed: now,
            requests: AtomicU32::new(0),
            pending: AtomicBool::new(pending),
            idle_from: AtomicU64::new(0),
        });
        if pending {
            self.in_flight[node].fetch_add(1, Ordering::Relaxed);
        }
        self.nodes[node].count(class, 1);
        self.sessions.insert(session.clone(), Arc::clone(&placed));
        Some(placed)
    }

    /// A request starts on `node`, of `session` when the table remembers it: its place there until it ends. The
    /// table's shared lock is enough.
    fn start(&self, node: usize, session: Option<Arc<Session>>) -> InFlight {
        self.count(session.as_deref(), node);
        InFlight {
            in_flight: Arc::clone(&self.in_flight),
            node,
            session,
        }
    }

    /// Counts a request that starts on `node`, of `session` when the table remembers it. The first request of a
    /// session placed by `POST /v1/sessions` takes over the place the session held.
    fn count(&self, session: Option<&Session>, node: usize) {
        let took_over = session.is_some_and(|placed| {
            placed.requests.fetch_add(1, Ordering::AcqRel);
            placed.pending.swap(false, Ordering::AcqRel)
        });
        if !took_over {
            self.in_flight[node].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Ends the placement of a closed session.
    fn forget(&mut self, session: &SessionId) {
        if let Some(placed) = self.sessions.remove(session) {
            self.release(&placed);
        }
    }

    /// Ends the placements of the sessions that went `idle` without a request, and returns when the next one will
    /// idle out.
    fn expire(&mut self, now: Instant, idle: Duration) -> Option<Instant> {
        let idled = |placed: &Session| !placed.busy() && now >= placed.since() + idle;
        let ended: Vec<SessionId> = self
            .sessions
            .iter()
            .filter(|(_, placed)| idled(placed))
            .map(|(session, _)| session.clone())
            .collect();
        for session in ended {
            self.forget(&session);
        }
        self.sessions
            .values()
            .filter(|placed| !placed.busy())
            .map(|placed| placed.since() + idle)
            .min()
    }

    /// Frees what a placement that ended held on its node.
    fn release(&mut self, placed: &Session) {
        if placed.pending.swap(false, Ordering::AcqRel) {
            self.in_flight[placed.node].fetch_sub(1, Ordering::Relaxed);
        }
        self.nodes[placed.node].count(placed.class, -1);
    }
}

impl Held {
    fn count(&mut self, class: SloClass, change: i32) {
        let count = &mut self.sessions[class as usize];
        *count = count.saturating_add_signed(change);
        self.gauges[class as usize].set(i64::from(*count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use SloClass::*;

    fn session(id: &str) -> SessionId {
        SessionId::new(id).unwrap()
    }

    /// A table of two nodes, `a` and `b`.
    fn table() -> Table {
        let names: [Arc<str>; 2] = ["a".into(), "b".into()];
        let metrics = Metrics::with_nodes(&names, Policy::LeastLoaded);
        let handles = names.map(|name| metrics.on(&name));
        Table::new(&handles)
    }

    /// What the nodes hold: requests in flight, and sessions of each class.
    fn held(table: &Table) -> Vec<(u32, [u32; 3])> {
        table
            .in_flight
            .iter()
            .zip(&table.nodes)
            .map(|(in_flight, held)| (in_flight.load(Ordering::Relaxed), held.sessions))
            .collect()
    }

    /// The node `session` holds on, all nodes ready.
    fn node_of(table: &Table, session: &SessionId) -> Option<usize> {
        match table.find(session, |_| true) {
            Found::On(placed) => Some(placed.node),
            Found::None | Found::Lost => None,
        }
    }

    #[test]
    fn a_session_keeps_its_node_while_the_node_is_ready() {
        let (mut t, t0) = (table(), Instant::now());
        let s = session("agent-1");
        assert!(matches!(t.find(&s, |_| true), Found::None));
        let placed = t.place(&s, Background, 1, false, t0).unwrap();
        t.count(Some(&placed), 1);
        t.count(Some(&placed), 1);
        assert_eq!(held(&t), [(0, [0; 3]), (2, [0, 0, 1])]);
        assert_eq!(node_of(&t, &s), Some(1));
        end(&t.in_flight, Some(&placed), 1, t0);
        end(&t.in_flight, Some(&placed), 1, t0);
        assert_eq!(held(&t), [(0, [0; 3]), (0, [0, 0, 1])]);
        // Its node fails readiness: the placement is lost, and a new one replaces it on the other node.
        let lost = t.find(&s, |node| node == 0);
        assert!(matches!(lost, Found::Lost));
        assert_eq!(lost.reason(), Reason::NodeDown);
        t.place(&s, Background, 0, false, t0);
        assert_eq!(held(&t), [(0, [0, 0, 1]), (0, [0; 3])]);
        assert_eq!(node_of(&t, &s), Some(0));
    }

    #[test]
    fn a_request_on_the_node_a_session_left_does_not_touch_its_new_placement() {
        let (mut t, t0) = (table(), Instant::now());
        let s = session("agent-1");
        let on_zero = t.place(&s, Background, 0, false, t0).unwrap();
        t.count(Some(&on_zero), 0);
        // Node 0 failed while the request ran; the session's next request went to node 1.
        let on_one = t.place(&s, Background, 1, false, t0).unwrap();
        t.count(Some(&on_one), 1);
        end(&t.in_flight, Some(&on_zero), 0, t0 + Duration::from_secs(1));
        assert_eq!(held(&t), [(0, [0; 3]), (1, [0, 0, 1])]);
        assert!(t.sessions[&s].busy(), "the request on node 1 still runs");
    }

    #[test]
    fn a_session_idles_out_after_idle_without_a_request() {
        let (mut t, t0) = (table(), Instant::now());
        let idle = Duration::from_secs(300);
        let at = |s: u64| t0 + Duration::from_secs(s);
        let (busy, quiet) = (session("busy"), session("quiet"));
        let running = t.place(&busy, Background, 0, false, t0).unwrap();
        t.count(Some(&running), 0);
        let done = t.place(&quiet, Interactive, 1, false, t0).unwrap();
        t.count(Some(&done), 1);
        end(&t.in_flight, Some(&done), 1, at(10));
        assert_eq!(t.expire(at(100), idle), Some(at(310)));
        // A request in flight keeps its session however long it runs.
        assert_eq!(t.expire(at(310), idle), None);
        assert_eq!(node_of(&t, &quiet), None);
        assert_eq!(node_of(&t, &busy), Some(0));
        assert_eq!(held(&t), [(1, [0, 0, 1]), (0, [0; 3])]);
        end(&t.in_flight, Some(&running), 0, at(400));
        assert_eq!(t.expire(at(699), idle), Some(at(700)));
        assert_eq!(t.expire(at(700), idle), None);
        assert!(t.sessions.is_empty());
    }

    #[test]
    fn a_session_placed_ahead_of_its_requests_holds_a_place_until_its_first_one() {
        let (mut t, t0) = (table(), Instant::now());
        let call = session("voice-RM_a1");
        let placed = t.place(&call, Realtime, 0, true, t0).unwrap();
        assert_eq!(held(&t), [(1, [1, 0, 0]), (0, [0; 3])]);
        // Its first request takes the place over, and its end frees it.
        t.count(Some(&placed), 0);
        assert_eq!(held(&t)[0], (1, [1, 0, 0]));
        end(&t.in_flight, Some(&placed), 0, t0);
        t.count(Some(&placed), 0);
        assert_eq!(held(&t)[0], (1, [1, 0, 0]));
        end(&t.in_flight, Some(&placed), 0, t0);
        assert_eq!(held(&t)[0], (0, [1, 0, 0]));
        // A call that never sends a request gives its place up as it idles out, or closes.
        let (silent, closed) = (session("voice-RM_b2"), session("voice-RM_c3"));
        t.place(&silent, Realtime, 1, true, t0);
        t.place(&closed, Realtime, 1, true, t0);
        assert_eq!(held(&t)[1], (2, [2, 0, 0]));
        t.forget(&closed);
        assert_eq!(held(&t)[1], (1, [1, 0, 0]));
        t.expire(t0 + Duration::from_secs(60), Duration::from_secs(60));
        assert_eq!(held(&t), [(0, [0; 3]), (0, [0; 3])]);
    }

    #[test]
    fn a_full_table_places_new_sessions_without_remembering_them() {
        let (mut t, t0) = (table(), Instant::now());
        for n in 0..SESSIONS {
            t.place(&session(&format!("s-{n}")), Background, n % 2, false, t0);
        }
        let late = session("late");
        assert!(t.place(&late, Background, 0, false, t0).is_none());
        assert_eq!(node_of(&t, &late), None);
        // Its requests still count on their node.
        t.count(None, 0);
        assert_eq!(held(&t)[0], (1, [0, 0, SESSIONS as u32 / 2]));
        end(&t.in_flight, None, 0, t0);
        assert_eq!(held(&t)[0].0, 0);
        // A session it remembers is placed again in its place.
        let first = session("s-0");
        t.place(&first, Interactive, 1, false, t0);
        assert_eq!(node_of(&t, &first), Some(1));
        assert_eq!(t.sessions.len(), SESSIONS);
    }

    #[test]
    fn requests_from_many_threads_leave_no_count_behind() {
        let table = Arc::new(RwLock::new(table()));
        let t0 = Instant::now();
        let sessions: Vec<SessionId> = (0..8)
            .map(|n| session(&format!("agent-{n}")))
            .chain((0..8).map(|n| session(&format!("call-{n}"))))
            .collect();
        // Four threads send the sessions' requests as the gateway does: a placed session's under the shared lock,
        // the others placed under the exclusive one.
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let (table, sessions) = (Arc::clone(&table), sessions.clone());
                std::thread::spawn(move || {
                    for n in 0..5_000 {
                        let s = &sessions[(worker + n) % sessions.len()];
                        let placed = {
                            let t = table.read().unwrap();
                            match t.find(s, |_| true) {
                                Found::On(placed) => Some(t.start(placed.node, Some(placed))),
                                Found::None | Found::Lost => None,
                            }
                        };
                        let request = placed.unwrap_or_else(|| {
                            let mut t = table.write().unwrap();
                            let placed = t.place(s, Background, n % 2, false, t0);
                            t.start(n % 2, placed)
                        });
                        drop(request);
                    }
                })
            })
            .collect();
        // Meanwhile the calls are placed again and again ahead of their requests, which take their places over,
        // and placements idle for a millisecond end.
        for n in 0..2_000 {
            let mut t = table.write().unwrap();
            t.place(&sessions[8 + n % 8], Realtime, n % 2, true, Instant::now());
            t.expire(Instant::now(), Duration::from_millis(1));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let mut t = table.write().unwrap();
        t.expire(Instant::now() + Duration::from_secs(1), Duration::ZERO);
        assert!(t.sessions.is_empty());
        assert_eq!(held(&t), [(0, [0; 3]), (0, [0; 3])]);
    }
}
