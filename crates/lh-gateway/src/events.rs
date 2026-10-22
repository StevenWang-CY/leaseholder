//! Lifecycle events: `POST /v1/events` from trusted tenants, deduplicated by `(source, event_id)` within a
//! bounded window, then handed to an [`EventSink`]. v0 records them ([`LogSink`]); the policy core will consume
//! them through the same trait.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::response::Response;
use http::StatusCode;
use lh_proto::LifecycleEvent;
use serde_json::value::RawValue;

use crate::exchange::Exchange;
use crate::metrics::{Metrics, Route};
use crate::proxy::read_body;
use crate::server::App;

/// An event the gateway accepted: not a repeat of one seen within the deduplication window.
#[derive(Clone, Debug)]
pub struct AcceptedEvent {
    /// The tenant that posted it.
    pub tenant: Arc<str>,
    /// The gateway's receipt time, which orders events (emitters' clocks differ).
    pub received: Instant,
    pub event: LifecycleEvent,
}

/// Where accepted events go.
pub trait EventSink: Send + Sync + std::fmt::Debug + 'static {
    /// Called once per accepted event, in receipt order within a request. It runs on the request path, so it
    /// must not block: a consumer that does real work should queue the event.
    fn accept(&self, event: AcceptedEvent);
}

/// The v0 sink: one structured log line per event (target `lh_gateway::events`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LogSink;

impl EventSink for LogSink {
    fn accept(&self, accepted: AcceptedEvent) {
        let event = &accepted.event;
        tracing::info!(
            target: "lh_gateway::events",
            tenant = &*accepted.tenant,
            event_id = event.event_id.as_str(),
            session_id = event.session_id.as_str(),
            source = event.source.as_str(),
            event_type = event.kind.label(),
            at_ms = event.at_ms,
            kind = ?event.kind,
            "lifecycle event"
        );
    }
}

/// Deduplication and dispatch of incoming events.
#[derive(Debug)]
pub(crate) struct Events {
    seen: Mutex<Seen>,
    capacity: usize,
    ttl: Duration,
    sink: Arc<dyn EventSink>,
}

/// Keys seen within the window, in arrival order. Every key has the same lifetime, so arrival order is also
/// expiry order and the oldest key is always at the front.
#[derive(Debug, Default)]
struct Seen {
    keys: HashSet<Arc<str>>,
    order: VecDeque<(Instant, Arc<str>)>,
}

impl Events {
    pub fn new(capacity: usize, ttl: Duration, sink: Arc<dyn EventSink>) -> Self {
        Self {
            seen: Mutex::default(),
            capacity,
            ttl,
            sink,
        }
    }

    /// Records a batch, returning how many events were new and how many were repeats.
    fn ingest(
        &self,
        tenant: &Arc<str>,
        events: Vec<LifecycleEvent>,
        metrics: &Metrics,
    ) -> (usize, usize) {
        let total = events.len();
        let received = Instant::now();
        let fresh: Vec<LifecycleEvent> = {
            let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            seen.expire(received, self.ttl);
            events
                .into_iter()
                .filter(|event| {
                    let key = format!("{}\0{}", event.source.as_str(), event.event_id);
                    let fresh = seen.insert(key.into(), received, self.capacity);
                    metrics.event(event.kind.label(), event.source.as_str(), !fresh);
                    fresh
                })
                .collect()
        };
        let accepted = fresh.len();
        for event in fresh {
            self.sink.accept(AcceptedEvent {
                tenant: Arc::clone(tenant),
                received,
                event,
            });
        }
        (accepted, total - accepted)
    }
}

impl Seen {
    fn expire(&mut self, now: Instant, ttl: Duration) {
        while let Some((at, _)) = self.order.front() {
            if now.duration_since(*at) < ttl {
                break;
            }
            if let Some((_, key)) = self.order.pop_front() {
                self.keys.remove(&key);
            }
        }
    }

    /// Remembers `key`, evicting the oldest key when full. Returns false if it was already present.
    fn insert(&mut self, key: Arc<str>, now: Instant, capacity: usize) -> bool {
        if self.keys.contains(&key) {
            return false;
        }
        if self.order.len() >= capacity
            && let Some((_, oldest)) = self.order.pop_front()
        {
            self.keys.remove(&oldest);
        }
        self.keys.insert(Arc::clone(&key));
        self.order.push_back((now, key));
        true
    }
}

/// `POST /v1/events`: one event object or an array of them. Answers 202 `{"accepted": n, "duplicates": m}`.
/// A batch is validated in full before any of it is recorded.
pub(crate) async fn post(State(app): State<App>, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Events);
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
            "only trusted tenants may post lifecycle events",
        );
    }
    let body = match read_body(body, shared.max_body_bytes).await {
        Ok(body) => body,
        Err((status, code, message)) => return ex.error(status, code, message),
    };
    let events = match parse(&body) {
        Ok(events) => events,
        Err(message) => return ex.error(StatusCode::BAD_REQUEST, "invalid_event", message),
    };
    let (accepted, duplicates) = shared.events.ingest(&tenant.name, events, &shared.metrics);
    let body = serde_json::json!({"accepted": accepted, "duplicates": duplicates});
    ex.respond(StatusCode::ACCEPTED, &body)
}

fn parse(body: &[u8]) -> Result<Vec<LifecycleEvent>, String> {
    let events = if body.trim_ascii_start().starts_with(b"[") {
        let items: Vec<&RawValue> =
            serde_json::from_slice(body).map_err(|e| format!("invalid JSON array: {e}"))?;
        items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                serde_json::from_str(item.get()).map_err(|e| format!("event {i}: {e}"))
            })
            .collect::<Result<Vec<LifecycleEvent>, _>>()?
    } else {
        vec![serde_json::from_slice(body).map_err(|e| format!("invalid event: {e}"))?]
    };
    for (i, event) in events.iter().enumerate() {
        if event.event_id.is_empty() || event.event_id.len() > LifecycleEvent::MAX_ID_LEN {
            return Err(format!(
                "event {i}: event_id must be 1 to {} bytes",
                LifecycleEvent::MAX_ID_LEN
            ));
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_forgets_by_age_and_by_capacity() {
        let mut seen = Seen::default();
        let start = Instant::now();
        let ttl = Duration::from_secs(600);
        assert!(seen.insert("a".into(), start, 2));
        assert!(!seen.insert("a".into(), start, 2));
        assert!(seen.insert("b".into(), start + Duration::from_secs(1), 2));
        // Full: "c" evicts the oldest key, so "a" counts as new again.
        assert!(seen.insert("c".into(), start + Duration::from_secs(2), 2));
        assert!(seen.insert("a".into(), start + Duration::from_secs(3), 2));
        assert_eq!(seen.keys.len(), 2);

        seen.expire(start + Duration::from_secs(602), ttl);
        assert_eq!(
            seen.order.len(),
            1,
            "only the key inserted at +3 s is inside the window"
        );
        assert!(seen.insert("b".into(), start + Duration::from_secs(602), 2));
    }

    #[test]
    fn errors_name_the_problem() {
        let unknown = parse(br#"{"event_id":"e","session_id":"s","at_ms":1,"source":"client","type":"user_laughed"}"#)
            .unwrap_err();
        assert!(unknown.contains("user_laughed"), "{unknown}");
        let second = parse(
            br#"[{"event_id":"e","session_id":"s","at_ms":1,"source":"client","type":"turn_started"},
                 {"event_id":"f","session_id":"s","at_ms":1,"source":"client"}]"#,
        )
        .unwrap_err();
        assert!(
            second.starts_with("event 1:") && second.contains("type"),
            "{second}"
        );
        let long = format!(
            r#"{{"event_id":"{}","session_id":"s","at_ms":1,"source":"client","type":"turn_started"}}"#,
            "e".repeat(129)
        );
        assert!(
            parse(long.as_bytes())
                .unwrap_err()
                .contains("1 to 128 bytes")
        );
    }
}
