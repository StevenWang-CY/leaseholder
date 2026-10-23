//! The gateway's Prometheus metrics, in a registry it owns (no global state), rendered at `GET /metrics`.
//!
//! Every label takes its values from a fixed set or from the configuration (tenant and node names), so the
//! number of series is bounded. Every series except those of requests (labeled with a status) and lifecycle
//! events is created at start, so it exports zero before traffic.
//!
//! With `[nodes]`, the series each node keeps (its admission point's, its realtime sessions', its upstream errors)
//! carry a `node` label, and a handle for one node ([`Metrics::on`]) records them; the others are the gateway's.
//! A gateway with one upstream exports every series as it always has, without the label.

use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use lh_core::admission::Deferral;
use lh_core::placement;
use lh_proto::SloClass;
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::{EncodeLabelSet, LabelSetEncoder};
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::{Metric, Registry, Unit};

use crate::placement::Reason;
use crate::realtime::Phase;
use crate::usage::Usage;

/// The API routes, as the `route` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    ChatCompletions,
    Completions,
    Responses,
    Models,
    Events,
    /// The placement endpoint, with `[nodes]`.
    Sessions,
    /// Any path or method the gateway does not serve.
    Other,
}

impl Route {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Responses => "/v1/responses",
            Self::Models => "/v1/models",
            Self::Events => "/v1/events",
            Self::Sessions => "/v1/sessions",
            Self::Other => "other",
        }
    }
}

/// How a proxied request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The upstream response was relayed in full (whatever its status).
    Completed,
    /// The client went away first; the gateway closed the upstream request.
    Cancelled,
    /// The client stopped reading for longer than the stall timeout; the gateway aborted both sides.
    Stalled,
    /// The upstream failed: no connection, no response headers in time, or a broken body.
    Error,
    /// The admission policy shed it to make room for a stricter class; the gateway cancelled the upstream
    /// request.
    Shed,
}

impl Outcome {
    const ALL: [Self; 5] = [
        Self::Completed,
        Self::Cancelled,
        Self::Stalled,
        Self::Error,
        Self::Shed,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Stalled => "stalled",
            Self::Error => "error",
            Self::Shed => "shed",
        }
    }
}

/// Why an upstream request failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UpstreamError {
    /// No connection could be made (502).
    Connect,
    /// The connection failed before the response headers arrived (502).
    Request,
    /// No response headers within the first-byte timeout (504).
    Timeout,
    /// The response body broke off.
    Body,
}

impl UpstreamError {
    const ALL: [Self; 4] = [Self::Connect, Self::Request, Self::Timeout, Self::Body];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Request => "request",
            Self::Timeout => "timeout",
            Self::Body => "body",
        }
    }
}

/// Why the admission point refused a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Its class's queue was full (429).
    QueueFull,
    /// It waited `queue_timeout_s` without being admitted (503).
    Timeout,
}

impl Refusal {
    const ALL: [Self; 2] = [Self::QueueFull, Self::Timeout];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::Timeout => "timeout",
        }
    }
}

/// Buckets of the gateway's own overhead, from 50 us. 2 ms, the SLO, is a bucket bound.
const OVERHEAD_BUCKETS: [f64; 12] = [
    50e-6, 100e-6, 200e-6, 300e-6, 500e-6, 750e-6, 1e-3, 1.5e-3, 2e-3, 5e-3, 10e-3, 50e-3,
];

/// Buckets of time to first byte and total duration: from realtime turns to long agent turns.
const LATENCY_BUCKETS: [f64; 16] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RequestLabels {
    /// Empty when the request was not authenticated.
    tenant: Option<Arc<str>>,
    route: &'static str,
    /// Empty for routes without a class.
    class: Option<&'static str>,
    status: u16,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OutcomeLabels {
    class: &'static str,
    outcome: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RouteLabels {
    route: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ClassLabels {
    class: &'static str,
}

/// The `node` label of the series each node keeps: none for the one upstream of a gateway without `[nodes]`.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct NodeLabel(Option<Arc<str>>);

impl EncodeLabelSet for NodeLabel {
    fn encode(&self, encoder: &mut LabelSetEncoder) -> Result<(), std::fmt::Error> {
        match &self.0 {
            Some(node) => [("node", &**node)].encode(encoder),
            None => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct NodeClassLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    class: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct FieldLabels {
    field: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct KindLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct EventLabels {
    r#type: &'static str,
    source: &'static str,
    duplicate: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct UsageLabels {
    class: &'static str,
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RefusalLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    class: &'static str,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ReasonLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct OversizeLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    oversize: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StateLabels {
    #[prometheus(flatten)]
    node: NodeLabel,
    state: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct PlacementLabels {
    policy: &'static str,
    reason: &'static str,
}

type Histograms<L> = Family<L, Histogram, fn() -> Histogram>;

/// A series each node keeps that has no label of its own: unlabeled for a gateway with one upstream, as it has
/// always been exported, and one per node with `[nodes]`.
#[derive(Clone, Debug)]
enum PerNode<M> {
    One(M),
    Nodes(Family<NodeLabel, M>),
}

impl<M: Clone + Default> PerNode<M> {
    fn new(nodes: &[NodeLabel]) -> Self {
        match nodes {
            [NodeLabel(None)] => Self::One(M::default()),
            _ => Self::Nodes(Family::default()),
        }
    }

    fn of(&self, node: &NodeLabel) -> M {
        match self {
            Self::One(metric) => metric.clone(),
            Self::Nodes(family) => family.get_or_create_owned(node),
        }
    }
}

impl<M> PerNode<M>
where
    M: Metric + Clone,
    Family<NodeLabel, M>: Metric,
{
    fn register(&self, registry: &mut Registry, name: &str, help: &str) {
        match self {
            Self::One(metric) => registry.register(name, help, metric.clone()),
            Self::Nodes(family) => registry.register(name, help, family.clone()),
        }
    }
}

/// The gateway's metrics, as one node records them: cloning a handle shares every series.
#[derive(Clone, Debug)]
pub(crate) struct Metrics {
    registry: Arc<Registry>,
    /// The node this handle records the per-node series for.
    node: NodeLabel,
    requests: Family<RequestLabels, Counter>,
    outcomes: Family<OutcomeLabels, Counter>,
    overhead: Histograms<RouteLabels>,
    request_body: Histograms<RouteLabels>,
    first_byte: Histograms<ClassLabels>,
    duration: Histograms<ClassLabels>,
    streams_active: Family<ClassLabels, Gauge>,
    sanitized: Family<FieldLabels, Counter>,
    class_clamped: Counter,
    upstream_errors: Family<KindLabels, Counter>,
    events: Family<EventLabels, Counter>,
    usage_tokens: Family<UsageLabels, Counter>,
    admitted: Family<NodeClassLabels, Counter>,
    waiting: Family<NodeClassLabels, Gauge>,
    admission_wait: Histograms<NodeClassLabels>,
    refused: Family<RefusalLabels, Counter>,
    resubmits: Family<NodeClassLabels, Counter>,
    discarded_tokens: Family<NodeClassLabels, Counter>,
    deferred: Family<ReasonLabels, Counter>,
    deferral: Histograms<ReasonLabels>,
    aged: Family<OversizeLabels, Counter>,
    degraded: PerNode<Counter>,
    false_alarms: PerNode<Counter>,
    false_alarm_token_seconds: PerNode<Counter<f64, AtomicU64>>,
    realtime_sessions: Family<StateLabels, Gauge>,
    state_timeouts: Family<StateLabels, Counter>,
    /// With `[nodes]` only: the sessions placed on each node, the placements, and the nodes' readiness.
    sessions: Family<NodeClassLabels, Gauge>,
    placements: Family<PlacementLabels, Counter>,
    node_ready: Family<NodeLabel, Gauge>,
}

impl Metrics {
    /// The metrics of a gateway with one upstream.
    pub fn new() -> Self {
        Self::build(&[NodeLabel(None)], None)
    }

    /// The metrics of a gateway whose `[nodes]` are `nodes`, placing sessions by `policy`: the gateway's handle,
    /// whose [`Metrics::on`] gives each node's.
    pub fn with_nodes(nodes: &[Arc<str>], policy: placement::Policy) -> Self {
        let nodes: Vec<NodeLabel> = nodes
            .iter()
            .map(|node| NodeLabel(Some(Arc::clone(node))))
            .collect();
        Self::build(&nodes, Some(policy))
    }

    /// The handle with which `node` records the series it keeps.
    pub fn on(&self, node: &Arc<str>) -> Self {
        Self {
            node: NodeLabel(Some(Arc::clone(node))),
            ..self.clone()
        }
    }

    fn build(nodes: &[NodeLabel], policy: Option<placement::Policy>) -> Self {
        let metrics = Self {
            registry: Arc::new(Registry::default()),
            node: NodeLabel(None),
            requests: Family::default(),
            outcomes: Family::default(),
            overhead: Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS)),
            request_body: Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS)),
            first_byte: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            duration: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            streams_active: Family::default(),
            sanitized: Family::default(),
            class_clamped: Counter::default(),
            upstream_errors: Family::default(),
            events: Family::default(),
            usage_tokens: Family::default(),
            admitted: Family::default(),
            waiting: Family::default(),
            admission_wait: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            refused: Family::default(),
            resubmits: Family::default(),
            discarded_tokens: Family::default(),
            deferred: Family::default(),
            deferral: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            aged: Family::default(),
            degraded: PerNode::new(nodes),
            false_alarms: PerNode::new(nodes),
            false_alarm_token_seconds: PerNode::new(nodes),
            realtime_sessions: Family::default(),
            state_timeouts: Family::default(),
            sessions: Family::default(),
            placements: Family::default(),
            node_ready: Family::default(),
        };
        for class in SloClass::ALL.map(SloClass::as_str) {
            for outcome in Outcome::ALL.map(Outcome::as_str) {
                create(&metrics.outcomes, OutcomeLabels { class, outcome });
            }
            for kind in ["prompt", "completion", "cached"] {
                create(&metrics.usage_tokens, UsageLabels { class, kind });
            }
            for node in nodes {
                for reason in Refusal::ALL.map(Refusal::as_str) {
                    let node = node.clone();
                    create(
                        &metrics.refused,
                        RefusalLabels {
                            node,
                            class,
                            reason,
                        },
                    );
                }
            }
            create(&metrics.first_byte, ClassLabels { class });
            create(&metrics.duration, ClassLabels { class });
            create(&metrics.streams_active, ClassLabels { class });
            for node in nodes {
                let labels = NodeClassLabels {
                    node: node.clone(),
                    class,
                };
                create(&metrics.admitted, labels.clone());
                create(&metrics.waiting, labels.clone());
                create(&metrics.admission_wait, labels.clone());
                create(&metrics.resubmits, labels.clone());
                create(&metrics.discarded_tokens, labels.clone());
                if policy.is_some() {
                    create(&metrics.sessions, labels);
                }
            }
        }
        for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
            let route = route.as_str();
            create(&metrics.overhead, RouteLabels { route });
            create(&metrics.request_body, RouteLabels { route });
        }
        for node in nodes {
            for kind in UpstreamError::ALL.map(UpstreamError::as_str) {
                let node = node.clone();
                create(&metrics.upstream_errors, KindLabels { node, kind });
            }
            for reason in Deferral::ALL.map(Deferral::as_str) {
                let labels = ReasonLabels {
                    node: node.clone(),
                    reason,
                };
                create(&metrics.deferred, labels.clone());
                create(&metrics.deferral, labels);
            }
            for oversize in [false, true] {
                let node = node.clone();
                create(&metrics.aged, OversizeLabels { node, oversize });
            }
            // The node's series of the counters without labels of their own.
            metrics.degraded.of(node);
            metrics.false_alarms.of(node);
            metrics.false_alarm_token_seconds.of(node);
            for state in Phase::ALL.map(Phase::as_str) {
                let labels = StateLabels {
                    node: node.clone(),
                    state,
                };
                create(&metrics.realtime_sessions, labels.clone());
                if state != Phase::Idle.as_str() {
                    create(&metrics.state_timeouts, labels);
                }
            }
            if policy.is_some() {
                create(&metrics.node_ready, node.clone());
            }
        }
        if let Some(policy) = policy {
            for reason in Reason::ALL.map(Reason::as_str) {
                let policy = policy.as_str();
                create(&metrics.placements, PlacementLabels { policy, reason });
            }
        }
        for field in crate::sanitize::sanitized_fields() {
            create(&metrics.sanitized, FieldLabels { field });
        }
        metrics.register(policy.is_some())
    }

    /// Registers every series, in the order of the exposition; those of placement only for a gateway with
    /// `[nodes]`.
    fn register(mut self, nodes: bool) -> Self {
        let mut registry = Registry::with_prefix("lh_gateway");
        let r = &mut registry;
        r.register(
            "requests",
            "API requests by tenant, route, class and response status (499: the client left before the response)",
            self.requests.clone(),
        );
        r.register(
            "request_outcomes",
            "Proxied requests by class and how they ended",
            self.outcomes.clone(),
        );
        r.register_with_unit(
            "overhead",
            "Time from the end of a request's body to writing the request to the upstream: parse, sanitize, \
             resolve, connection checkout",
            Unit::Seconds,
            self.overhead.clone(),
        );
        r.register_with_unit(
            "request_body",
            "Time from receiving a request's headers to the end of its body: the client's upload, which a \
             direct call to the engine waits for too",
            Unit::Seconds,
            self.request_body.clone(),
        );
        r.register_with_unit(
            "first_byte",
            "Time from receiving a request to the first response body byte from the upstream",
            Unit::Seconds,
            self.first_byte.clone(),
        );
        r.register_with_unit(
            "duration",
            "Time from receiving a request to the end of the upstream response",
            Unit::Seconds,
            self.duration.clone(),
        );
        r.register(
            "streams_active",
            "Responses being relayed from the upstream",
            self.streams_active.clone(),
        );
        r.register(
            "sanitized",
            "Request headers and body fields removed because clients may not set them",
            self.sanitized.clone(),
        );
        r.register(
            "class_clamped",
            "Requests whose class was stricter than their tenant's max_class",
            self.class_clamped.clone(),
        );
        r.register(
            "upstream_errors",
            "Upstream failures by kind",
            self.upstream_errors.clone(),
        );
        r.register(
            "events",
            "Lifecycle events by type, source and whether they repeated an earlier event",
            self.events.clone(),
        );
        r.register(
            "usage_tokens",
            "Tokens reported in response usage, by class and kind",
            self.usage_tokens.clone(),
        );
        r.register(
            "admission_admitted",
            "Requests the admission point let go upstream, by class (none without an admission policy)",
            self.admitted.clone(),
        );
        r.register(
            "admission_waiting",
            "Requests waiting at the admission point",
            self.waiting.clone(),
        );
        r.register_with_unit(
            "admission_wait",
            "Time from reaching the admission point to being admitted",
            Unit::Seconds,
            self.admission_wait.clone(),
        );
        r.register(
            "admission_refused",
            "Requests the admission point refused, by class and reason",
            self.refused.clone(),
        );
        r.register(
            "admission_resubmits",
            "Requests sent upstream again after a preemption dropped their attempt",
            self.resubmits.clone(),
        );
        r.register(
            "admission_discarded_tokens",
            "Estimated prompt tokens of the upstream attempts preemptions dropped: at most the prefill they \
             discarded",
            self.discarded_tokens.clone(),
        );
        r.register(
            "admission_deferred",
            "Background requests policy leaseholder deferred, by what held them back when they arrived: the \
             headroom of the armed realtime sessions, or the critical window's prefill budget",
            self.deferred.clone(),
        );
        r.register_with_unit(
            "admission_deferral",
            "Time from arrival to admission of the background requests policy leaseholder deferred, by what \
             held them back when they arrived",
            Unit::Seconds,
            self.deferral.clone(),
        );
        r.register(
            "admission_aged",
            "Deferred requests admitted regardless after max_defer_s; oversize: their estimate alone was over \
             what the headroom leaves background requests",
            self.aged.clone(),
        );
        self.degraded.register(
            r,
            "admission_degraded",
            "Realtime requests whose reply policy leaseholder capped, because the background requests in flight \
             left less than one turn's headroom free",
        );
        self.false_alarms.register(
            r,
            "admission_false_alarms",
            "Headroom armings that no realtime request of their session followed: the session left the armed \
             states, timed out or closed first",
        );
        self.false_alarm_token_seconds.register(
            r,
            "admission_false_alarm_token_seconds",
            "Estimated background tokens the headroom held back during false alarms, times how long",
        );
        r.register(
            "realtime_sessions",
            "Realtime sessions policy leaseholder follows, by the state of their turn",
            self.realtime_sessions.clone(),
        );
        r.register(
            "realtime_state_timeouts",
            "Realtime sessions that stayed in a state of their turn for state_timeout_s without an event, by the \
             state: a lost or missing event",
            self.state_timeouts.clone(),
        );
        if nodes {
            r.register(
                "sessions",
                "Sessions placed on each node, by the class they were placed with",
                self.sessions.clone(),
            );
            r.register(
                "placements",
                "Sessions placed on a node, by policy and reason: new, or placed again because their node failed \
                 readiness (node_down), which leaves their KV behind",
                self.placements.clone(),
            );
            r.register(
                "node_ready",
                "Whether a node answers its readiness probe (1), or takes no new sessions (0)",
                self.node_ready.clone(),
            );
        }
        self.registry = Arc::new(registry);
        self
    }

    /// The OpenMetrics text exposition.
    pub fn encode(&self) -> String {
        let mut text = String::with_capacity(16 << 10);
        encode(&mut text, &self.registry).expect("writing to a String cannot fail");
        text
    }

    pub fn request(
        &self,
        tenant: Option<&Arc<str>>,
        route: Route,
        class: Option<SloClass>,
        status: u16,
    ) {
        let labels = RequestLabels {
            tenant: tenant.cloned(),
            route: route.as_str(),
            class: class.map(SloClass::as_str),
            status,
        };
        self.requests.get_or_create(&labels).inc();
    }

    pub fn outcome(&self, class: SloClass, outcome: Outcome) {
        let labels = OutcomeLabels {
            class: class.as_str(),
            outcome: outcome.as_str(),
        };
        self.outcomes.get_or_create(&labels).inc();
    }

    /// How long a request's body took to arrive after its headers.
    pub fn request_body(&self, route: Route, elapsed: Duration) {
        let labels = RouteLabels {
            route: route.as_str(),
        };
        self.request_body
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    /// The overhead histogram of a route, observed when the upstream request is written.
    pub fn overhead(&self, route: Route) -> Histogram {
        self.overhead.get_or_create_owned(&RouteLabels {
            route: route.as_str(),
        })
    }

    pub fn first_byte(&self, class: SloClass, elapsed: Duration) {
        let labels = ClassLabels {
            class: class.as_str(),
        };
        self.first_byte
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    pub fn duration(&self, class: SloClass, elapsed: Duration) {
        let labels = ClassLabels {
            class: class.as_str(),
        };
        self.duration
            .get_or_create(&labels)
            .observe(elapsed.as_secs_f64());
    }

    /// Counts a stream as active until the returned guard drops.
    pub fn stream_started(&self, class: SloClass) -> GaugeGuard {
        GaugeGuard::new(self.streams_active.get_or_create_owned(&ClassLabels {
            class: class.as_str(),
        }))
    }

    /// Counts a request as waiting at the admission point until the returned guard drops.
    pub fn waiting(&self, class: SloClass) -> GaugeGuard {
        GaugeGuard::new(self.waiting.get_or_create_owned(&self.on_class(class)))
    }

    /// A request was admitted after waiting `wait`.
    pub fn admitted(&self, class: SloClass, wait: Duration) {
        let labels = self.on_class(class);
        self.admitted.get_or_create(&labels).inc();
        self.admission_wait
            .get_or_create(&labels)
            .observe(wait.as_secs_f64());
    }

    /// A preemption dropped an upstream attempt of `tokens` estimated prompt tokens.
    pub fn preempted(&self, class: SloClass, tokens: u64) {
        let labels = self.on_class(class);
        self.discarded_tokens.get_or_create(&labels).inc_by(tokens);
    }

    /// A preempted request was sent upstream again.
    pub fn resubmitted(&self, class: SloClass) {
        let labels = self.on_class(class);
        self.resubmits.get_or_create(&labels).inc();
    }

    /// Policy leaseholder deferred a background request.
    pub fn deferred(&self, reason: Deferral) {
        let labels = ReasonLabels {
            node: self.node.clone(),
            reason: reason.as_str(),
        };
        self.deferred.get_or_create(&labels).inc();
    }

    /// A deferred background request was admitted after waiting `wait`.
    pub fn deferral(&self, reason: Deferral, wait: Duration) {
        let labels = ReasonLabels {
            node: self.node.clone(),
            reason: reason.as_str(),
        };
        self.deferral
            .get_or_create(&labels)
            .observe(wait.as_secs_f64());
    }

    /// A deferred request was admitted regardless.
    pub fn aged(&self, oversize: bool) {
        let labels = OversizeLabels {
            node: self.node.clone(),
            oversize,
        };
        self.aged.get_or_create(&labels).inc();
    }

    /// A realtime request's reply was capped.
    pub fn degraded(&self) {
        self.degraded.of(&self.node).inc();
    }

    /// A headroom arming ended with no realtime request, having held back `token_seconds` of background work.
    pub fn false_alarm(&self, token_seconds: f64) {
        self.false_alarms.of(&self.node).inc();
        self.false_alarm_token_seconds
            .of(&self.node)
            .inc_by(token_seconds.max(0.0));
    }

    /// The gauge of the realtime sessions in `phase`.
    pub fn realtime_sessions(&self, phase: Phase) -> Gauge {
        self.realtime_sessions.get_or_create_owned(&StateLabels {
            node: self.node.clone(),
            state: phase.as_str(),
        })
    }

    /// A realtime session's `phase` timed out.
    pub fn state_timeout(&self, phase: Phase) {
        let labels = StateLabels {
            node: self.node.clone(),
            state: phase.as_str(),
        };
        self.state_timeouts.get_or_create(&labels).inc();
    }

    pub fn refused(&self, class: SloClass, reason: Refusal) {
        let labels = RefusalLabels {
            node: self.node.clone(),
            class: class.as_str(),
            reason: reason.as_str(),
        };
        self.refused.get_or_create(&labels).inc();
    }

    /// The gauge of the sessions of `class` placed on this handle's node.
    pub fn sessions(&self, class: SloClass) -> Gauge {
        self.sessions.get_or_create_owned(&self.on_class(class))
    }

    /// The gauge of whether this handle's node answers its readiness probe.
    pub fn node_ready(&self) -> Gauge {
        self.node_ready.get_or_create_owned(&self.node)
    }

    /// A session was placed by `policy`, for `reason`.
    pub fn placed(&self, policy: placement::Policy, reason: Reason) {
        let labels = PlacementLabels {
            policy: policy.as_str(),
            reason: reason.as_str(),
        };
        self.placements.get_or_create(&labels).inc();
    }

    /// The labels of this handle's node and `class`.
    fn on_class(&self, class: SloClass) -> NodeClassLabels {
        NodeClassLabels {
            node: self.node.clone(),
            class: class.as_str(),
        }
    }

    pub fn sanitized(&self, field: &'static str) {
        self.sanitized.get_or_create(&FieldLabels { field }).inc();
    }

    pub fn class_clamped(&self) {
        self.class_clamped.inc();
    }

    pub fn upstream_error(&self, kind: UpstreamError) {
        let labels = KindLabels {
            node: self.node.clone(),
            kind: kind.as_str(),
        };
        self.upstream_errors.get_or_create(&labels).inc();
    }

    pub fn event(&self, r#type: &'static str, source: &'static str, duplicate: bool) {
        let labels = EventLabels {
            r#type,
            source,
            duplicate,
        };
        self.events.get_or_create(&labels).inc();
    }

    pub fn usage(&self, class: SloClass, usage: &Usage) {
        let class = class.as_str();
        for (kind, tokens) in [
            ("prompt", usage.prompt),
            ("completion", usage.completion),
            ("cached", usage.cached),
        ] {
            self.usage_tokens
                .get_or_create(&UsageLabels { class, kind })
                .inc_by(tokens);
        }
    }

    #[cfg(test)]
    pub fn sanitized_count(&self, field: &'static str) -> u64 {
        self.sanitized
            .get(&FieldLabels { field })
            .map_or(0, |counter| counter.get())
    }

    /// The false alarms, and their token-seconds.
    #[cfg(test)]
    pub fn false_alarms(&self) -> (u64, f64) {
        (
            self.false_alarms.of(&self.node).get(),
            self.false_alarm_token_seconds.of(&self.node).get(),
        )
    }

    #[cfg(test)]
    pub fn state_timeouts(&self, phase: Phase) -> u64 {
        let labels = StateLabels {
            node: self.node.clone(),
            state: phase.as_str(),
        };
        self.state_timeouts
            .get(&labels)
            .map_or(0, |counter| counter.get())
    }
}

/// Creates a series at zero, so it is exported before its first event.
fn create<S, M, C>(family: &Family<S, M, C>, labels: S)
where
    S: Clone + Hash + Eq,
    C: MetricConstructor<M>,
{
    drop(family.get_or_create(&labels));
}

/// Counts one in a gauge while it lives: an active stream, or a waiting request.
#[derive(Debug)]
pub(crate) struct GaugeGuard(Gauge);

impl GaugeGuard {
    fn new(gauge: Gauge) -> Self {
        gauge.inc();
        Self(gauge)
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}
