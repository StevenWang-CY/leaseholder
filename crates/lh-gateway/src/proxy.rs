//! The proxy path: authenticate, sanitize, resolve, forward, and relay the response through a bounded buffer.
//!
//! A response body is relayed by a pump task that reads the upstream body into a bounded channel; the client's
//! response body reads the channel. The channel is the backpressure: when it is full, the pump stops reading the
//! upstream, and if it stays full for the stall timeout the pump aborts both sides. When the client goes away,
//! its body (the receiver) is dropped, the pump sees the channel close at once and drops the upstream body,
//! which closes the upstream connection; vLLM aborts a request whose client disconnects.

use std::convert::Infallible;
use std::error::Error as _;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use bytes::Bytes;
use http::header::{CONTENT_TYPE, HeaderValue};
use http::uri::{self, Authority, PathAndQuery, Scheme, Uri};
use http::{StatusCode, Version};
use http_body::{Body as _, Frame, SizeHint};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use hyper::body::Incoming;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use prometheus_client::metrics::histogram::Histogram;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use crate::admission::Ticket;
use crate::exchange::Exchange;
use crate::metrics::{Outcome, Refusal, Route, UpstreamError};
use crate::sanitize::{
    REQUEST_ID, cap_reply, resolve_class, resolve_session, rewrite_body, sanitize_headers,
    set_engine_headers, strip_hop_by_hop,
};
use crate::server::{App, Shared};
use crate::usage::Scanner;

/// The pooled HTTP/1.1 client to the engine.
#[derive(Debug)]
pub(crate) struct Upstream {
    client: Client<HttpConnector, UpstreamBody>,
    authority: Authority,
}

impl Upstream {
    /// A client that keeps idle connections for `idle_timeout`. The engine closes idle connections after its
    /// keep-alive, and a request sent on a connection it is closing fails, so the gateway must let go first.
    pub fn new(authority: Authority, idle_timeout: Duration) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        let client = Client::builder(TokioExecutor::new())
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(idle_timeout)
            .build(connector);
        Self { client, authority }
    }

    pub fn authority(&self) -> &Authority {
        &self.authority
    }

    fn uri(&self, path_and_query: PathAndQuery) -> Uri {
        let mut parts = uri::Parts::default();
        parts.scheme = Some(Scheme::HTTP);
        parts.authority = Some(self.authority.clone());
        parts.path_and_query = Some(path_and_query);
        Uri::from_parts(parts).expect("a scheme, an authority and a path form a valid URI")
    }

    /// Whether the engine's `/health` answers with success within `timeout`.
    pub async fn healthy(&self, timeout: Duration) -> bool {
        let uri = self.uri(PathAndQuery::from_static("/health"));
        let request = http::Request::get(uri)
            .body(UpstreamBody::empty())
            .expect("a GET with a valid URI is a valid request");
        matches!(
            tokio::time::timeout(timeout, self.client.request(request)).await,
            Ok(Ok(response)) if response.status().is_success()
        )
    }
}

/// `POST /v1/chat/completions`, `/v1/completions` and `/v1/responses`.
pub(crate) async fn inference(app: App, route: Route, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, route);
    let (mut parts, body) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));

    let inputs = sanitize_headers(&mut parts.headers, tenant.trusted, &shared.metrics);
    let class = match resolve_class(tenant, inputs.class.as_ref()) {
        Ok((class, clamped)) => {
            if clamped {
                shared.metrics.class_clamped();
            }
            class
        }
        Err(err) => return ex.error(StatusCode::BAD_REQUEST, "invalid_class", err.to_string()),
    };
    ex.class = Some(class);

    let body = match read_body(body, shared.max_body_bytes).await {
        Ok(body) => body,
        Err((status, code, message)) => return ex.error(status, code, message),
    };
    // The overhead starts once the body is in: until then the gateway waits on the client, as the engine
    // would.
    let read = Instant::now();
    shared.metrics.request_body(route, read - ex.received);
    let signals = shared.upstream_priority;
    let priority = signals.body.then(|| class.engine_priority());
    let mut body = match rewrite_body(&body, &tenant.cache_salt, priority, &shared.metrics) {
        Ok(body) => body,
        Err(err) => return ex.error(StatusCode::BAD_REQUEST, "invalid_json", err.to_string()),
    };
    let session = resolve_session(&tenant.name, &inputs, body.prompt_cache_key.as_deref());
    set_engine_headers(&mut parts.headers, &session, class, &ex.id, signals);
    parts
        .headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let session = &*ex.session.insert(session);

    // The admission point. Its decisions are the gateway's overhead; the time a request waits there is not.
    let mut start = read;
    let ticket = match &shared.admission {
        None => None,
        Some(admission) => {
            let arrived = Instant::now();
            match admission.admit(class, session, body.json.len()).await {
                Ok(ticket) => {
                    let waited = ticket.as_ref().map_or(Duration::ZERO, Ticket::waited);
                    ex.admission_wait = Some(waited);
                    if let Some(ticket) = &ticket {
                        ex.deferred = ticket.deferred();
                        ex.aged = ticket.aged();
                        if let Some(cap) = ticket.reply_cap() {
                            body.json = cap_reply(&body.json, route, cap);
                            ex.degraded = true;
                        }
                    }
                    start += waited;
                    ticket
                }
                Err(refusal) => {
                    ex.admission_wait = Some(arrived.elapsed());
                    return ex.refused(refusal);
                }
            }
        }
    };

    let sent = (start, shared.metrics.overhead(route));
    let body = UpstreamBody {
        data: Some(body.json),
        sent: Some(sent),
    };
    forward(&app, ex, parts, body, ticket).await
}

/// `GET /v1/models`: authenticated and sanitized, with no session or class.
pub(crate) async fn models(app: App, request: Request) -> Response {
    let shared = &app.shared;
    let mut ex = Exchange::new(&shared.metrics, Route::Models);
    let (mut parts, _) = request.into_parts();
    let Some(tenant) = shared
        .tenants
        .authenticate(&parts.headers, app.listener_tenant)
    else {
        return ex.unauthorized();
    };
    ex.tenant = Some(Arc::clone(&tenant.name));
    sanitize_headers(&mut parts.headers, tenant.trusted, &shared.metrics);
    parts.headers.insert(REQUEST_ID.clone(), ex.id.clone());
    forward(&app, ex, parts, UpstreamBody::empty(), None).await
}

/// Reads a request body of at most `limit` bytes.
pub(crate) async fn read_body(
    body: Body,
    limit: usize,
) -> Result<Bytes, (StatusCode, &'static str, String)> {
    let too_large = || {
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            format!("the request body exceeds {limit} bytes"),
        )
    };
    // A declared length over the limit is refused before reading anything.
    if body.size_hint().lower() > limit as u64 {
        return Err(too_large());
    }
    match Limited::new(body, limit).collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(err) if err.is::<LengthLimitError>() => Err(too_large()),
        Err(err) => Err((
            StatusCode::BAD_REQUEST,
            "unreadable_body",
            format!("cannot read the request body: {err}"),
        )),
    }
}

/// Sends the request upstream and relays the response. An admitted request's ticket goes with the relay and
/// is given back when the response ends. A request shed before its response answers 429; one preempted
/// before its first byte is sent again.
async fn forward(
    app: &App,
    mut ex: Exchange,
    mut parts: http::request::Parts,
    body: UpstreamBody,
    mut ticket: Option<Ticket>,
) -> Response {
    let shared = &app.shared;
    let path = parts.uri.path_and_query().cloned();
    parts.uri = shared
        .upstream
        .uri(path.unwrap_or_else(|| PathAndQuery::from_static("/")));
    parts.version = Version::HTTP_11;
    parts.extensions.clear();

    ex.forwarded = true;
    let sent = match &mut ticket {
        Some(ticket) if ticket.preemptible() => {
            resubmitting(shared, &mut ex, parts, body, ticket).await
        }
        Some(ticket) if ticket.sheddable() => {
            let request = http::Request::from_parts(parts, body);
            tokio::select! {
                sent = send(shared, request) => sent.map(|response| (response, None)),
                // Dropping the request future closes its connection, which aborts the engine request.
                () = ticket.shed() => return ex.shed(),
            }
        }
        _ => send(shared, http::Request::from_parts(parts, body))
            .await
            .map(|response| (response, None)),
    };
    let (response, first) = match sent {
        Ok(sent) => sent,
        Err(failure) => return failure.answer(ex, shared),
    };

    let (mut parts, upstream_body) = response.into_parts();
    strip_hop_by_hop(&mut parts.headers);
    ex.decorate(&mut parts.headers);
    ex.status = Some(parts.status);
    let mut scanner = Scanner::for_content_type(parts.headers.get(CONTENT_TYPE));
    let (frames, receiver) = mpsc::channel(shared.stream_buffer_frames);
    // The first frame of a preemptible response, read before the client was answered.
    if let Some(first) = first {
        scanner.feed(&first);
        frames
            .try_send(first)
            .expect("a new channel has room for a frame");
    }
    let aborted = Arc::new(AtomicBool::new(false));
    tokio::spawn(pump(
        upstream_body,
        frames,
        Arc::clone(&aborted),
        ex,
        scanner,
        shared.stall_timeout,
        ticket,
    ));
    let body = ClientBody {
        frames: receiver,
        aborted,
    };
    Response::from_parts(parts, Body::new(body))
}

/// Why a request has no response to relay.
#[derive(Debug)]
enum Failure {
    /// The connection failed, or closed before the response headers (502).
    Upstream(hyper_util::client::legacy::Error),
    /// Nothing within the first-byte timeout (504): no response headers, or for a preemptible request, no
    /// first body frame.
    Timeout(&'static str),
    /// A preemptible request's response broke off before its first body frame (502).
    Body,
    /// A preempted request was not admitted again in time (503).
    Refused(Refusal),
}

impl Failure {
    fn answer(self, ex: Exchange, shared: &Shared) -> Response {
        let metrics = &shared.metrics;
        match self {
            Self::Upstream(err) => {
                let kind = if err.is_connect() {
                    UpstreamError::Connect
                } else {
                    UpstreamError::Request
                };
                metrics.upstream_error(kind);
                let mut message = format!("upstream request failed: {err}");
                let mut source = err.source();
                while let Some(cause) = source {
                    message.push_str(&format!(": {cause}"));
                    source = cause.source();
                }
                ex.error(StatusCode::BAD_GATEWAY, "upstream_unavailable", message)
            }
            Self::Timeout(awaited) => {
                metrics.upstream_error(UpstreamError::Timeout);
                let timeout = shared.first_byte_timeout;
                let message = format!("no {awaited} from the upstream within {timeout:?}");
                ex.error(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", message)
            }
            Self::Body => {
                metrics.upstream_error(UpstreamError::Body);
                let message = "the upstream response broke off before its first byte";
                ex.error(StatusCode::BAD_GATEWAY, "upstream_unavailable", message)
            }
            Self::Refused(refusal) => ex.refused(refusal),
        }
    }
}

/// Sends a request upstream and waits for its response headers, for at most the first-byte timeout.
async fn send(
    shared: &Shared,
    request: http::Request<UpstreamBody>,
) -> Result<http::Response<Incoming>, Failure> {
    let response = shared.upstream.client.request(request);
    match tokio::time::timeout(shared.first_byte_timeout, response).await {
        Ok(response) => response.map_err(Failure::Upstream),
        Err(_) => Err(Failure::Timeout("response headers")),
    }
}

/// Sends a preemptible request and holds its response back until the first body frame. A preemption before
/// that frame drops the attempt, which closes its connection and aborts the engine request, and the request
/// goes again once it is admitted again. The client has seen nothing of the dropped attempt, so it gets one
/// response whose first byte comes later. Each attempt has the first-byte timeout to its first frame.
async fn resubmitting(
    shared: &Shared,
    ex: &mut Exchange,
    parts: http::request::Parts,
    body: UpstreamBody,
    ticket: &mut Ticket,
) -> Result<(http::Response<Incoming>, Option<Bytes>), Failure> {
    let again = (parts.clone(), body.data.clone());
    let mut request = http::Request::from_parts(parts, body);
    loop {
        tokio::select! {
            attempt = first_frame(shared, request) => {
                let (response, first) = attempt?;
                // A frame that arrives as the request is preempted belongs to the dropped attempt.
                if first.is_none() || ticket.first_byte() {
                    if first.is_some() {
                        ex.first_byte = Some(ex.received.elapsed());
                    }
                    return Ok((response, first));
                }
            }
            () = ticket.preempted() => {}
        }
        ticket.readmitted().await.map_err(Failure::Refused)?;
        ex.resubmits += 1;
        let (parts, data) = again.clone();
        request = http::Request::from_parts(parts, UpstreamBody { data, sent: None });
    }
}

/// Sends a request and reads its response to the first body frame, or to the end of a body without one.
async fn first_frame(
    shared: &Shared,
    request: http::Request<UpstreamBody>,
) -> Result<(http::Response<Incoming>, Option<Bytes>), Failure> {
    let attempt = async {
        let response = shared
            .upstream
            .client
            .request(request)
            .await
            .map_err(Failure::Upstream)?;
        let (parts, mut body) = response.into_parts();
        let first = loop {
            match body.frame().await {
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) => break Some(data),
                    Err(_) => continue,
                },
                Some(Err(_)) => return Err(Failure::Body),
                None => break None,
            }
        };
        Ok((http::Response::from_parts(parts, body), first))
    };
    tokio::time::timeout(shared.first_byte_timeout, attempt)
        .await
        .unwrap_or(Err(Failure::Timeout("first byte")))
}

/// Relays an upstream body into the client's channel until either side ends, or until the admission policy
/// sheds the request.
async fn pump(
    mut upstream: Incoming,
    frames: mpsc::Sender<Bytes>,
    aborted: Arc<AtomicBool>,
    mut ex: Exchange,
    mut scanner: Scanner,
    stall_timeout: Duration,
    mut ticket: Option<Ticket>,
) {
    let _active = ex.class.map(|class| ex.metrics().stream_started(class));
    let sheddable = ticket.as_ref().is_some_and(Ticket::sheddable);
    let outcome = loop {
        let frame = tokio::select! {
            biased;
            () = frames.closed() => break Outcome::Cancelled,
            () = shed(&mut ticket), if sheddable => break Outcome::Shed,
            frame = upstream.frame() => frame,
        };
        let data = match frame {
            None => break Outcome::Completed,
            Some(Err(_)) => {
                ex.metrics().upstream_error(UpstreamError::Body);
                break Outcome::Error;
            }
            // Trailers are not relayed; vLLM sends none.
            Some(Ok(frame)) => match frame.into_data() {
                Ok(data) => data,
                Err(_) => continue,
            },
        };
        if ex.first_byte.is_none() {
            ex.first_byte = Some(ex.received.elapsed());
            // A realtime request's first byte frees the background work preempt-resubmit holds back for it.
            if let Some(ticket) = &mut ticket {
                ticket.first_byte();
            }
        }
        let scanned = data.clone();
        if let Err(err) = frames.try_send(data) {
            let data = match err {
                TrySendError::Full(data) => data,
                TrySendError::Closed(_) => break Outcome::Cancelled,
            };
            let sent = tokio::select! {
                biased;
                () = shed(&mut ticket), if sheddable => break Outcome::Shed,
                sent = tokio::time::timeout(stall_timeout, frames.send(data)) => sent,
            };
            match sent {
                Ok(Ok(())) => {}
                Ok(Err(_)) => break Outcome::Cancelled,
                Err(_) => break Outcome::Stalled,
            }
        }
        scanner.feed(&scanned);
    };
    if matches!(outcome, Outcome::Stalled | Outcome::Error | Outcome::Shed) {
        aborted.store(true, Ordering::Release);
    }
    // An unfinished upstream body closes its connection when dropped, which aborts the engine request.
    drop(upstream);
    drop(frames);
    ex.summary = scanner.finish();
    // The request leaves the admission point once its upstream connection is gone.
    if let Some(mut ticket) = ticket {
        ticket.prompt_tokens = ex.summary.usage.map(|usage| usage.prompt);
        drop(ticket);
    }
    ex.finish(outcome);
}

/// Completes when the admission policy sheds the request.
async fn shed(ticket: &mut Option<Ticket>) {
    match ticket {
        Some(ticket) => ticket.shed().await,
        None => std::future::pending().await,
    }
}

/// The body of an upstream request. Its first poll happens when hyper writes the request to a pooled
/// connection, which is where the gateway's overhead for the request ends; it started when the client's body
/// was complete.
#[derive(Debug)]
pub(crate) struct UpstreamBody {
    data: Option<Bytes>,
    sent: Option<(Instant, Histogram)>,
}

impl UpstreamBody {
    fn empty() -> Self {
        Self {
            data: None,
            sent: None,
        }
    }
}

impl http_body::Body for UpstreamBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        if let Some((received, overhead)) = self.sent.take() {
            overhead.observe(received.elapsed().as_secs_f64());
        }
        Poll::Ready(self.data.take().map(|data| Ok(Frame::data(data))))
    }

    fn is_end_stream(&self) -> bool {
        self.data.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::with_exact(self.data.as_ref().map_or(0, |data| data.len() as u64))
    }
}

/// The client's half of a relayed response.
#[derive(Debug)]
struct ClientBody {
    frames: mpsc::Receiver<Bytes>,
    /// Set by the pump before it stops on a stall, an upstream error or a shed, so the client sees an error
    /// instead of a clean end: a truncated response must not look complete.
    aborted: Arc<AtomicBool>,
}

#[derive(Debug)]
struct StreamAborted;

impl std::fmt::Display for StreamAborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the upstream response was aborted")
    }
}

impl std::error::Error for StreamAborted {}

impl http_body::Body for ClientBody {
    type Data = Bytes;
    type Error = StreamAborted;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, StreamAborted>>> {
        match ready!(self.frames.poll_recv(cx)) {
            Some(data) => Poll::Ready(Some(Ok(Frame::data(data)))),
            // The pump stores the flag before it drops the sender, so it is visible once the channel closes.
            None if self.aborted.load(Ordering::Acquire) => Poll::Ready(Some(Err(StreamAborted))),
            None => Poll::Ready(None),
        }
    }
}
