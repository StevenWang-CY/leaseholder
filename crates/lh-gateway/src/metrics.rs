//! The gateway's Prometheus metrics, in a registry it owns (no global state), rendered at `GET /metrics`.
//!
//! Every label takes its values from a fixed set or from the configuration (tenant names), so the number of
//! series is bounded. Series with fixed label sets are created at start, so they export zero before traffic.

use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

use lh_proto::{SloClass, headers};
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::{Registry, Unit};

use crate::usage::Usage;

/// The API routes, as the `route` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    ChatCompletions,
    Completions,
    Responses,
    Models,
    Events,
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
}

impl Outcome {
    const ALL: [Self; 4] = [Self::Completed, Self::Cancelled, Self::Stalled, Self::Error];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Stalled => "stalled",
            Self::Error => "error",
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

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct FieldLabels {
    field: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct KindLabels {
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

type Histograms<L> = Family<L, Histogram, fn() -> Histogram>;

#[derive(Debug)]
pub(crate) struct Metrics {
    registry: Registry,
    requests: Family<RequestLabels, Counter>,
    outcomes: Family<OutcomeLabels, Counter>,
    overhead: Histograms<RouteLabels>,
    first_byte: Histograms<ClassLabels>,
    duration: Histograms<ClassLabels>,
    streams_active: Family<ClassLabels, Gauge>,
    sanitized: Family<FieldLabels, Counter>,
    class_clamped: Counter,
    upstream_errors: Family<KindLabels, Counter>,
    events: Family<EventLabels, Counter>,
    usage_tokens: Family<UsageLabels, Counter>,
}

impl Metrics {
    pub fn new() -> Self {
        let metrics = Self {
            registry: Registry::with_prefix("lh_gateway"),
            requests: Family::default(),
            outcomes: Family::default(),
            overhead: Family::new_with_constructor(|| Histogram::new(OVERHEAD_BUCKETS)),
            first_byte: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            duration: Family::new_with_constructor(|| Histogram::new(LATENCY_BUCKETS)),
            streams_active: Family::default(),
            sanitized: Family::default(),
            class_clamped: Counter::default(),
            upstream_errors: Family::default(),
            events: Family::default(),
            usage_tokens: Family::default(),
        };
        for class in SloClass::ALL.map(SloClass::as_str) {
            for outcome in Outcome::ALL.map(Outcome::as_str) {
                create(&metrics.outcomes, OutcomeLabels { class, outcome });
            }
            for kind in ["prompt", "completion", "cached"] {
                create(&metrics.usage_tokens, UsageLabels { class, kind });
            }
            create(&metrics.first_byte, ClassLabels { class });
            create(&metrics.duration, ClassLabels { class });
            create(&metrics.streams_active, ClassLabels { class });
        }
        for route in [Route::ChatCompletions, Route::Completions, Route::Responses] {
            let route = route.as_str();
            create(&metrics.overhead, RouteLabels { route });
        }
        for kind in UpstreamError::ALL.map(UpstreamError::as_str) {
            create(&metrics.upstream_errors, KindLabels { kind });
        }
        let fields = headers::ENGINE_CONTROL_HEADERS
            .into_iter()
            .chain([headers::REQUEST_ID, headers::CLASS, headers::SYNTHETIC])
            .chain(headers::ENGINE_CONTROL_BODY_FIELDS);
        for field in fields {
            create(&metrics.sanitized, FieldLabels { field });
        }
        metrics.register()
    }

    fn register(mut self) -> Self {
        let r = &mut self.registry;
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
            "Time from receiving a request to writing it to the upstream: body read, parse, sanitize, resolve, \
             connection checkout",
            Unit::Seconds,
            self.overhead.clone(),
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
    pub fn stream_started(&self, class: SloClass) -> ActiveStream {
        let gauge = self.streams_active.get_or_create_owned(&ClassLabels {
            class: class.as_str(),
        });
        gauge.inc();
        ActiveStream(gauge)
    }

    pub fn sanitized(&self, field: &'static str) {
        self.sanitized.get_or_create(&FieldLabels { field }).inc();
    }

    pub fn class_clamped(&self) {
        self.class_clamped.inc();
    }

    pub fn upstream_error(&self, kind: UpstreamError) {
        let labels = KindLabels {
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
}

/// Creates a series at zero, so it is exported before its first event.
fn create<S, M, C>(family: &Family<S, M, C>, labels: S)
where
    S: Clone + Hash + Eq,
    C: MetricConstructor<M>,
{
    drop(family.get_or_create(&labels));
}

/// Decrements `lh_gateway_streams_active` when dropped.
#[derive(Debug)]
pub(crate) struct ActiveStream(Gauge);

impl Drop for ActiveStream {
    fn drop(&mut self) {
        self.0.dec();
    }
}
