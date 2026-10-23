//! The test harness: a mock engine, a gateway on ephemeral listeners, and a raw HTTP/1.1 client.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use bytes::Bytes;
use futures_util::stream;
use http::header::CONTENT_TYPE;
use http::{HeaderMap, Method, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use lh_gateway::{AcceptedEvent, Config, EventSink, ListenAddr, LogSink, Server};
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream, UnixStream};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;

pub const AGENTS_KEY: &str = "agents-key";
pub const BENCH_KEY: &str = "bench-key";

/// A request header naming the request, for [`Mock::tags`] and [`Mock::dropped`].
pub const TAG: &str = "x-mock-tag";
/// A request header that makes the mock engine hold its answer until [`Mock::release`]: a JSON response's
/// headers, or an event stream's first chunk, as an engine holds them through a prefill.
pub const GATE: &str = "x-mock-gate";

/// How the mock engine answers inference requests.
#[derive(Clone, Debug)]
pub enum Script {
    Json(&'static str),
    /// `text/event-stream`: the chunks, with `gap` before each one after the first.
    Sse {
        chunks: Vec<Bytes>,
        gap: Duration,
    },
    /// A stream that never ends.
    Endless {
        chunk: Bytes,
        gap: Duration,
    },
    /// JSON, but the response headers come only after this delay.
    Slow(Duration),
}

/// A request the mock engine received.
#[derive(Clone, Debug)]
pub struct Recorded {
    pub path: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|v| v.to_str().unwrap())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

struct MockState {
    script: Script,
    requests: Mutex<Vec<Recorded>>,
    healthy: AtomicBool,
    /// The permits gated answers wait for, granted in the order they began to wait.
    gate: Semaphore,
    dropped: mpsc::UnboundedSender<(Instant, Option<String>)>,
}

/// An in-process engine on 127.0.0.1.
pub struct Mock {
    pub addr: SocketAddr,
    state: Arc<MockState>,
    dropped: mpsc::UnboundedReceiver<(Instant, Option<String>)>,
}

/// Sends the time it is dropped, with the request's tag: held by a response body, or by a handler that has
/// not answered yet.
struct DropSignal(
    mpsc::UnboundedSender<(Instant, Option<String>)>,
    Option<String>,
);

impl Drop for DropSignal {
    fn drop(&mut self) {
        let _ = self.0.send((Instant::now(), self.1.take()));
    }
}

impl Mock {
    pub async fn start(script: Script) -> Self {
        let (dropped_tx, dropped) = mpsc::unbounded_channel();
        let state = Arc::new(MockState {
            script,
            requests: Mutex::default(),
            healthy: AtomicBool::new(true),
            gate: Semaphore::new(0),
            dropped: dropped_tx,
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(answer))
            .route("/v1/completions", post(answer))
            .route("/v1/responses", post(answer))
            .route("/v1/models", get(models))
            .route(
                "/health",
                get(|State(state): State<Arc<MockState>>| async move {
                    if state.healthy.load(Ordering::SeqCst) {
                        StatusCode::OK
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                }),
            )
            .with_state(Arc::clone(&state));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let listener = listener.tap_io(|tcp| tcp.set_nodelay(true).unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            addr,
            state,
            dropped,
        }
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.state.healthy.store(healthy, Ordering::SeqCst);
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.state.requests.lock().unwrap().clone()
    }

    pub fn last(&self) -> Recorded {
        self.requests()
            .pop()
            .expect("the mock engine received a request")
    }

    /// When the next response body (or pending handler) was dropped.
    pub async fn next_drop(&mut self, within: Duration) -> Instant {
        tokio::time::timeout(within, self.dropped.recv())
            .await
            .expect("the mock engine's response was not dropped in time")
            .unwrap()
            .0
    }

    /// When the response of the request tagged `tag` was dropped; drops of other requests are skipped.
    pub async fn dropped(&mut self, tag: &str, within: Duration) -> Instant {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            let (at, dropped) = tokio::time::timeout_at(deadline, self.dropped.recv())
                .await
                .unwrap_or_else(|_| panic!("the response of {tag} was not dropped in time"))
                .unwrap();
            if dropped.as_deref() == Some(tag) {
                return at;
            }
        }
    }

    /// Lets `n` gated answers go on, the longest waiting first.
    pub fn release(&self, n: usize) {
        self.state.gate.add_permits(n);
    }

    /// The tags of the requests received, in order of arrival.
    pub fn tags(&self) -> Vec<String> {
        self.requests()
            .iter()
            .filter_map(|r| r.header(TAG).map(str::to_owned))
            .collect()
    }

    /// Waits until the mock engine has received `n` requests in all.
    pub async fn await_requests(&self, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.requests().len() < n {
            assert!(
                Instant::now() < deadline,
                "{} requests reached the mock engine, expected {n}: {:?}",
                self.requests().len(),
                self.tags()
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

impl MockState {
    /// Waits for a permit of the gate.
    async fn pass(&self) {
        self.gate.acquire().await.unwrap().forget();
    }

    async fn record(&self, request: Request) {
        let (parts, body) = request.into_parts();
        let body = body.collect().await.unwrap().to_bytes();
        self.requests.lock().unwrap().push(Recorded {
            path: parts.uri.path().to_owned(),
            headers: parts.headers,
            body,
        });
    }
}

async fn models(State(state): State<Arc<MockState>>, request: Request) -> &'static str {
    state.record(request).await;
    r#"{"object":"list","data":[{"id":"qwen","object":"model"}]}"#
}

async fn answer(State(state): State<Arc<MockState>>, request: Request) -> Response {
    let headers = request.headers();
    let tag = headers.get(TAG).map(|v| v.to_str().unwrap().to_owned());
    let gated = headers.contains_key(GATE);
    let guard = DropSignal(state.dropped.clone(), tag);
    state.record(request).await;
    match state.script.clone() {
        Script::Json(json) => {
            if gated {
                state.pass().await;
            }
            ([(CONTENT_TYPE, "application/json")], json).into_response()
        }
        Script::Slow(delay) => {
            tokio::time::sleep(delay).await;
            drop(guard);
            ([(CONTENT_TYPE, "application/json")], "{}").into_response()
        }
        Script::Sse { chunks, gap } => {
            let gate = gated.then_some(state);
            let chunks = stream::unfold(
                (chunks.into_iter(), guard, gate, true),
                move |(mut chunks, guard, gate, first)| async move {
                    let chunk = chunks.next()?;
                    if !first {
                        tokio::time::sleep(gap).await;
                    } else if let Some(state) = &gate {
                        state.pass().await;
                    }
                    Some((Ok::<_, Infallible>(chunk), (chunks, guard, gate, false)))
                },
            );
            event_stream(Body::from_stream(chunks))
        }
        Script::Endless { chunk, gap } => {
            let chunks = stream::unfold(guard, move |guard| {
                let chunk = chunk.clone();
                async move {
                    if !gap.is_zero() {
                        tokio::time::sleep(gap).await;
                    }
                    Some((Ok::<_, Infallible>(chunk), guard))
                }
            });
            event_stream(Body::from_stream(chunks))
        }
    }
}

fn event_stream(body: Body) -> Response {
    ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
}

/// A gateway serving TCP 127.0.0.1:0 and a Unix socket whose tenant is `voice`.
pub struct Gateway {
    pub tcp: Target,
    pub unix: Target,
    shutdown: Option<oneshot::Sender<()>>,
    done: Option<JoinHandle<()>>,
}

/// Where a test client connects.
#[derive(Clone, Debug)]
pub enum Target {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

/// A Unix socket path unique to this test process and call, short enough for `sun_path`.
pub fn socket_path() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lh-gw-{}-{n}.sock", std::process::id()))
}

pub fn key_digest(key: &str) -> String {
    Sha256::digest(key)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The test configuration: `voice` (trusted, keyless, on the Unix socket), `agents` (trusted, default
/// background, max interactive) and `bench` (untrusted, default interactive). `top` holds extra top-level keys.
pub fn gateway_toml(upstream: SocketAddr, socket: &std::path::Path, top: &str) -> String {
    format!(
        "\nupstream = \"http://{upstream}\"\n{}",
        harness_toml(socket, top)
    )
}

/// The test configuration of a gateway on GPU nodes instead of one upstream: `tables` holds `[nodes.<name>]`,
/// `[placement]` and any other table.
pub fn nodes_toml(socket: &std::path::Path, tables: &str) -> String {
    harness_toml(socket, tables)
}

/// The test configuration's listeners and tenants, after `top`.
fn harness_toml(socket: &std::path::Path, top: &str) -> String {
    format!(
        r#"log_level = "debug"
{top}

[[listen]]
tcp = "127.0.0.1:0"

[[listen]]
unix = "{socket}"
tenant = "voice"

[tenants.voice]
trusted = true
default_class = "realtime"

[tenants.agents]
keys_sha256 = ["{agents}"]
trusted = true
default_class = "background"
max_class = "interactive"

[tenants.bench]
keys_sha256 = ["{bench}"]
default_class = "interactive"
"#,
        socket = socket.display(),
        agents = key_digest(AGENTS_KEY),
        bench = key_digest(BENCH_KEY),
    )
}

impl Gateway {
    pub async fn start(upstream: SocketAddr, top: &str) -> Self {
        Self::start_with(upstream, top, Arc::new(LogSink)).await
    }

    pub async fn start_with(upstream: SocketAddr, top: &str, sink: Arc<dyn EventSink>) -> Self {
        let config = Config::from_toml(&gateway_toml(upstream, &socket_path(), top)).unwrap();
        Self::from_config(config, sink).await
    }

    /// A gateway on GPU nodes: `tables` holds `[nodes.<name>]`, `[placement]` and any other table.
    pub async fn start_nodes(tables: &str) -> Self {
        let config = Config::from_toml(&nodes_toml(&socket_path(), tables)).unwrap();
        Self::from_config(config, Arc::new(LogSink)).await
    }

    pub async fn from_config(config: Config, sink: Arc<dyn EventSink>) -> Self {
        let server = Server::bind(config, sink).await.unwrap();
        let mut targets = server
            .local_addrs()
            .unwrap()
            .into_iter()
            .map(|addr| match addr {
                ListenAddr::Tcp(addr) => Target::Tcp(addr),
                ListenAddr::Unix(path) => Target::Unix(path),
            });
        let (tcp, unix) = (targets.next().unwrap(), targets.next().unwrap());
        let (shutdown, stop) = oneshot::channel::<()>();
        let done = tokio::spawn(server.run(async move {
            let _ = stop.await;
        }));
        Self {
            tcp,
            unix,
            shutdown: Some(shutdown),
            done: Some(done),
        }
    }

    /// Starts the shutdown; [`Gateway::stopped`] waits for the drain.
    pub fn shut_down(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    pub async fn stopped(&mut self) {
        self.shut_down();
        if let Some(done) = self.done.take() {
            done.await.unwrap();
        }
    }

    /// The Prometheus exposition.
    pub async fn metrics(&self) -> String {
        let reply = call(&self.tcp, "GET", "/metrics", &[], "").await;
        String::from_utf8(reply.body.to_vec()).unwrap()
    }

    /// Waits until a series reaches `value`: outcomes are recorded when a relay task ends, just after the
    /// client has seen the end of the response.
    pub async fn await_metric(&self, series: &str, value: f64) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let text = self.metrics().await;
            let current = metric(&text, series);
            if current == Some(value) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{series} is {current:?}, expected {value}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.shut_down();
    }
}

/// The value of one series in an exposition.
pub fn metric(text: &str, series: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
        .map(|value| value.parse().unwrap())
}

/// A response read in full.
#[derive(Debug)]
pub struct Reply {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Reply {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|v| v.to_str().unwrap())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

/// Sends one request on a new connection and reads the whole response.
pub async fn call(
    target: &Target,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Reply {
    let (response, _connection) = open(target, method, path, headers, body).await;
    let (parts, body) = response.into_parts();
    Reply {
        status: parts.status,
        headers: parts.headers,
        body: body.collect().await.unwrap().to_bytes(),
    }
}

/// Sends a chat completion on a new connection in the background; the handle gives the whole response.
pub fn spawn_call(
    target: &Target,
    headers: Vec<(&'static str, &'static str)>,
    body: &'static str,
) -> JoinHandle<Reply> {
    let target = target.clone();
    tokio::spawn(async move { call(&target, "POST", "/v1/chat/completions", &headers, body).await })
}

/// Sends one request on a new connection and returns the response head with its connection task. Aborting
/// the task closes the connection, as a client that goes away would.
pub async fn open(
    target: &Target,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (http::Response<Incoming>, JoinHandle<()>) {
    let (mut sender, connection) = connect(target).await;
    let response = sender
        .send_request(request(method, path, headers, body))
        .await
        .unwrap();
    (response, connection)
}

pub fn request(
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> http::Request<Full<Bytes>> {
    let mut request = http::Request::builder()
        .method(Method::from_bytes(method.as_bytes()).unwrap())
        .uri(path)
        .header("host", "gateway");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request
        .body(Full::new(Bytes::copy_from_slice(body.as_bytes())))
        .unwrap()
}

/// A new HTTP/1.1 connection and the task that drives it.
pub async fn connect(target: &Target) -> (http1::SendRequest<Full<Bytes>>, JoinHandle<()>) {
    match target {
        Target::Tcp(addr) => {
            let stream = TcpStream::connect(addr).await.unwrap();
            stream.set_nodelay(true).unwrap();
            handshake(TokioIo::new(stream)).await
        }
        Target::Unix(path) => {
            handshake(TokioIo::new(UnixStream::connect(path).await.unwrap())).await
        }
    }
}

async fn handshake<I>(io: TokioIo<I>) -> (http1::SendRequest<Full<Bytes>>, JoinHandle<()>)
where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (sender, connection) = http1::handshake(io).await.unwrap();
    (
        sender,
        tokio::spawn(async move {
            let _ = connection.await;
        }),
    )
}

pub const AGENTS_AUTH: (&str, &str) = ("authorization", "Bearer agents-key");
pub const BENCH_AUTH: (&str, &str) = ("authorization", "Bearer bench-key");

/// Cache salts, `printf 'leaseholder-cache-salt\0<tenant>' | shasum -a 256 | cut -c1-32`.
pub const AGENTS_SALT: &str = "5aba4e96998029ea96716cc693e1703e";
pub const BENCH_SALT: &str = "c3ecbad17b0b9e1a8d1761a3759441da";
pub const VOICE_SALT: &str = "32e853135c92a8afaf0bf4b3a2afd65c";

/// Records accepted lifecycle events.
#[derive(Debug, Default)]
pub struct RecordingSink(pub Mutex<Vec<AcceptedEvent>>);

impl EventSink for RecordingSink {
    fn accept(&self, event: AcceptedEvent) {
        self.0.lock().unwrap().push(event);
    }
}

pub const CHAT: &str = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
