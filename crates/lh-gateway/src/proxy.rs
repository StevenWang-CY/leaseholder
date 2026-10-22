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

use crate::exchange::Exchange;
use crate::metrics::{Outcome, Route, UpstreamError};
use crate::sanitize::{
    REQUEST_ID, resolve_class, resolve_session, rewrite_body, sanitize_headers, set_engine_headers,
    strip_hop_by_hop,
};
use crate::server::App;
use crate::usage::Scanner;

/// Idle upstream connections close after this. vLLM closes them after `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` (5 s by
/// default), and a request sent on a connection the server is closing fails, so the gateway lets go first.
const UPSTREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(4);

/// The pooled HTTP/1.1 client to the engine.
#[derive(Debug)]
pub(crate) struct Upstream {
    client: Client<HttpConnector, UpstreamBody>,
    authority: Authority,
}

impl Upstream {
    pub fn new(authority: Authority) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_nodelay(true);
        let client = Client::builder(TokioExecutor::new())
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(UPSTREAM_IDLE_TIMEOUT)
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
    let body = match rewrite_body(&body, &tenant.cache_salt, &shared.metrics) {
        Ok(body) => body,
        Err(err) => return ex.error(StatusCode::BAD_REQUEST, "invalid_json", err.to_string()),
    };
    let session = resolve_session(&tenant.name, &inputs, body.prompt_cache_key.as_deref());
    set_engine_headers(&mut parts.headers, &session, class, &ex.id);
    parts
        .headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    ex.session = Some(session);

    let sent = (ex.received, shared.metrics.overhead(route));
    let body = UpstreamBody {
        data: Some(body.json),
        sent: Some(sent),
    };
    forward(&app, ex, parts, body).await
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
    forward(&app, ex, parts, UpstreamBody::empty()).await
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

/// Sends the request upstream and relays the response.
async fn forward(
    app: &App,
    mut ex: Exchange,
    mut parts: http::request::Parts,
    body: UpstreamBody,
) -> Response {
    let shared = &app.shared;
    let path = parts.uri.path_and_query().cloned();
    parts.uri = shared
        .upstream
        .uri(path.unwrap_or_else(|| PathAndQuery::from_static("/")));
    parts.version = Version::HTTP_11;
    parts.extensions.clear();
    let request = http::Request::from_parts(parts, body);

    ex.forwarded = true;
    let timeout = shared.first_byte_timeout;
    let response =
        match tokio::time::timeout(timeout, shared.upstream.client.request(request)).await {
            Ok(Ok(response)) => response,
            Ok(Err(err)) => {
                let kind = if err.is_connect() {
                    UpstreamError::Connect
                } else {
                    UpstreamError::Request
                };
                shared.metrics.upstream_error(kind);
                let mut message = format!("upstream request failed: {err}");
                let mut source = err.source();
                while let Some(cause) = source {
                    message.push_str(&format!(": {cause}"));
                    source = cause.source();
                }
                return ex.error(StatusCode::BAD_GATEWAY, "upstream_unavailable", message);
            }
            Err(_) => {
                shared.metrics.upstream_error(UpstreamError::Timeout);
                let message = format!("no response headers from the upstream within {timeout:?}");
                return ex.error(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", message);
            }
        };

    let (mut parts, upstream_body) = response.into_parts();
    strip_hop_by_hop(&mut parts.headers);
    ex.decorate(&mut parts.headers);
    ex.status = Some(parts.status);
    let scanner = Scanner::for_content_type(parts.headers.get(CONTENT_TYPE));
    let (frames, receiver) = mpsc::channel(shared.stream_buffer_frames);
    let aborted = Arc::new(AtomicBool::new(false));
    tokio::spawn(pump(
        upstream_body,
        frames,
        Arc::clone(&aborted),
        ex,
        scanner,
        shared.stall_timeout,
    ));
    let body = ClientBody {
        frames: receiver,
        aborted,
    };
    Response::from_parts(parts, Body::new(body))
}

/// Relays an upstream body into the client's channel until either side ends.
async fn pump(
    mut upstream: Incoming,
    frames: mpsc::Sender<Bytes>,
    aborted: Arc<AtomicBool>,
    mut ex: Exchange,
    mut scanner: Scanner,
    stall_timeout: Duration,
) {
    let _active = ex.class.map(|class| ex.metrics().stream_started(class));
    let outcome = loop {
        let frame = tokio::select! {
            biased;
            () = frames.closed() => break Outcome::Cancelled,
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
        ex.first_byte.get_or_insert_with(|| ex.received.elapsed());
        let scanned = data.clone();
        if let Err(err) = frames.try_send(data) {
            let data = match err {
                TrySendError::Full(data) => data,
                TrySendError::Closed(_) => break Outcome::Cancelled,
            };
            match tokio::time::timeout(stall_timeout, frames.send(data)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => break Outcome::Cancelled,
                Err(_) => break Outcome::Stalled,
            }
        }
        scanner.feed(&scanned);
    };
    if matches!(outcome, Outcome::Stalled | Outcome::Error) {
        aborted.store(true, Ordering::Release);
    }
    // An unfinished upstream body closes its connection when dropped, which aborts the engine request.
    drop(upstream);
    drop(frames);
    ex.summary = scanner.finish();
    ex.finish(outcome);
}

/// The body of an upstream request. Its first poll happens when hyper writes the request to a pooled
/// connection, which is where the gateway's overhead for the request ends.
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
    /// Set by the pump before it stops on a stall or an upstream error, so the client sees an error instead of
    /// a clean end: a truncated response must not look complete.
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
