//! Listeners, routes and the process lifecycle: bind every listener, serve, and drain on shutdown.

use std::fmt::Display;
use std::future::{Future, IntoFuture};
use std::io;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, get, post};
use axum::serve::ListenerExt;
use http::header::{CONTENT_TYPE, HeaderValue};
use http::{Method, StatusCode, Uri};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::watch;
use tokio::task::JoinSet;

use crate::admission::Admission;
use crate::auth::Tenants;
use crate::config::{Config, ListenAddr, PrioritySignals};
use crate::events::{self, EventSink, Events};
use crate::exchange::Exchange;
use crate::metrics::{Metrics, Route};
use crate::proxy::{self, Upstream};

/// `GET /ready` succeeds while the upstream's `/health` has answered within this window.
const READY_WINDOW: Duration = Duration::from_secs(1);

/// State shared by every listener.
#[derive(Debug)]
pub(crate) struct Shared {
    pub tenants: Tenants,
    pub upstream: Upstream,
    pub upstream_priority: PrioritySignals,
    /// `None` under policy "none": requests go upstream at once.
    pub admission: Option<Arc<Admission>>,
    pub metrics: Arc<Metrics>,
    pub events: Events,
    pub max_body_bytes: usize,
    pub stream_buffer_frames: usize,
    pub stall_timeout: Duration,
    pub first_byte_timeout: Duration,
    /// When the upstream's `/health` last succeeded.
    healthy_at: Mutex<Option<Instant>>,
}

/// The state of one listener's routes.
#[derive(Clone, Debug)]
pub(crate) struct App {
    pub shared: Arc<Shared>,
    /// The tenant of requests on this listener that carry no known API key.
    pub listener_tenant: Option<usize>,
}

/// A gateway with every listener bound.
#[derive(Debug)]
pub struct Server {
    shared: Arc<Shared>,
    listeners: Vec<(Listener, Option<usize>)>,
    /// Removed when the server stops.
    sockets: Vec<SocketFile>,
    drain_timeout: Duration,
}

#[derive(Debug)]
enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener, PathBuf),
}

impl Server {
    /// Binds every configured listener; nothing is served until [`Server::run`]. Accepted lifecycle events go
    /// to `sink`.
    pub async fn bind(config: Config, sink: Arc<dyn EventSink>) -> io::Result<Self> {
        let tenants = Tenants::new(&config.tenants);
        let mut listeners = Vec::with_capacity(config.listeners.len());
        let mut sockets = Vec::new();
        for listen in &config.listeners {
            let tenant = listen
                .tenant
                .as_deref()
                .and_then(|name| tenants.index(name));
            let listener = match &listen.addr {
                ListenAddr::Tcp(addr) => Listener::Tcp(
                    TcpListener::bind(addr)
                        .await
                        .map_err(|e| context(e, addr))?,
                ),
                ListenAddr::Unix(path) => {
                    let (listener, file) = bind_unix(path).await?;
                    sockets.push(file);
                    Listener::Unix(listener, path.clone())
                }
            };
            listeners.push((listener, tenant));
        }
        let metrics = Arc::new(Metrics::new());
        let shared = Shared {
            tenants,
            upstream: Upstream::new(config.upstream, config.upstream_idle_timeout),
            upstream_priority: config.upstream_priority,
            admission: config
                .admission
                .map(|admission| Arc::new(Admission::new(&admission, Arc::clone(&metrics)))),
            metrics,
            events: Events::new(config.event_dedup_capacity, config.event_dedup_ttl, sink),
            max_body_bytes: config.max_body_bytes,
            stream_buffer_frames: config.stream_buffer_frames,
            stall_timeout: config.stall_timeout,
            first_byte_timeout: config.first_byte_timeout,
            healthy_at: Mutex::new(None),
        };
        Ok(Self {
            shared: Arc::new(shared),
            listeners,
            sockets,
            drain_timeout: config.drain_timeout,
        })
    }

    /// Where the listeners are bound, in configuration order, with the assigned port for TCP port 0.
    pub fn local_addrs(&self) -> io::Result<Vec<ListenAddr>> {
        self.listeners
            .iter()
            .map(|(listener, _)| match listener {
                Listener::Tcp(listener) => listener.local_addr().map(ListenAddr::Tcp),
                Listener::Unix(_, path) => Ok(ListenAddr::Unix(path.clone())),
            })
            .collect()
    }

    /// Serves until `shutdown` completes, then stops accepting and lets in-flight requests finish for up to the
    /// drain timeout. Unix socket files are removed before it returns.
    pub async fn run(self, shutdown: impl Future<Output = ()>) {
        tracing::info!(
            upstream = %self.shared.upstream.authority(),
            listeners = ?self.local_addrs().unwrap_or_default(),
            admission = ?self.shared.admission.as_ref().map(|admission| admission.policy()),
            "lh-gateway serving"
        );
        // Dropping `stop` tells every listener to shut down.
        let (stop, stopped) = watch::channel(());
        let mut serving = JoinSet::new();
        for (listener, listener_tenant) in self.listeners {
            let router = router(App {
                shared: Arc::clone(&self.shared),
                listener_tenant,
            });
            let mut stopped = stopped.clone();
            let signal = async move {
                let _ = stopped.changed().await;
            };
            match listener {
                Listener::Tcp(listener) => {
                    let listener = listener.tap_io(|stream| {
                        if let Err(err) = stream.set_nodelay(true) {
                            tracing::debug!("cannot set TCP_NODELAY: {err}");
                        }
                    });
                    serving.spawn(serve(listener, router, signal));
                }
                Listener::Unix(listener, _) => {
                    serving.spawn(serve(listener, router, signal));
                }
            }
        }
        shutdown.await;
        tracing::info!(
            drain_timeout_s = self.drain_timeout.as_secs_f64(),
            "shutting down: no new connections; draining in-flight requests"
        );
        drop(stop);
        let drain = async { while serving.join_next().await.is_some() {} };
        if tokio::time::timeout(self.drain_timeout, drain)
            .await
            .is_err()
        {
            tracing::warn!("drain timeout reached; stopping with requests in flight");
        }
        drop(self.sockets);
    }
}

fn serve<L>(
    listener: L,
    router: Router,
    signal: impl Future<Output = ()> + Send + 'static,
) -> impl Future<Output = io::Result<()>> + Send
where
    L: axum::serve::Listener,
    L::Addr: std::fmt::Debug,
{
    axum::serve(listener, router)
        .with_graceful_shutdown(signal)
        .into_future()
}

fn router(app: App) -> Router {
    Router::new()
        .route("/v1/chat/completions", inference(Route::ChatCompletions))
        .route("/v1/completions", inference(Route::Completions))
        .route("/v1/responses", inference(Route::Responses))
        .route(
            "/v1/models",
            get(|State(app): State<App>, request: Request| proxy::models(app, request)),
        )
        .route("/v1/events", post(events::post))
        .route("/metrics", get(metrics))
        .route("/health", get(|| async { "ok\n" }))
        .route("/ready", get(ready))
        .fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::NOT_FOUND,
                    "not_found",
                    format!("no such route: {}", request_line(&method, &uri)),
                )
            },
        )
        .method_not_allowed_fallback(
            |State(app): State<App>, method: Method, uri: Uri| async move {
                Exchange::new(&app.shared.metrics, Route::Other).error(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method_not_allowed",
                    format!(
                        "the route does not accept this method: {}",
                        request_line(&method, &uri)
                    ),
                )
            },
        )
        .with_state(app)
}

/// The method and path of a request that matched no route, for its error and log line. The `route` label of
/// such a request is "other", so this is the only record of what was asked for; long paths are cut.
fn request_line(method: &Method, uri: &Uri) -> String {
    const MAX_PATH: usize = 256;
    let path = uri.path();
    let end = (0..=MAX_PATH.min(path.len()))
        .rev()
        .find(|&i| path.is_char_boundary(i))
        .unwrap_or(0);
    let cut = if end < path.len() { "…" } else { "" };
    format!("{method} {}{cut}", &path[..end])
}

fn inference(route: Route) -> MethodRouter<App> {
    post(move |State(app): State<App>, request: Request| proxy::inference(app, route, request))
}

async fn metrics(State(app): State<App>) -> Response {
    let content_type =
        HeaderValue::from_static("application/openmetrics-text; version=1.0.0; charset=utf-8");
    ([(CONTENT_TYPE, content_type)], app.shared.metrics.encode()).into_response()
}

/// 200 while the upstream's `/health` has answered within [`READY_WINDOW`]; otherwise it asks once more.
async fn ready(State(app): State<App>) -> Response {
    let shared = &app.shared;
    let last = *shared
        .healthy_at
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let fresh = last.is_some_and(|at| at.elapsed() < READY_WINDOW);
    if fresh || shared.upstream.healthy(READY_WINDOW).await {
        if !fresh {
            *shared
                .healthy_at
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        }
        (StatusCode::OK, "ready\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "upstream not ready\n").into_response()
    }
}

/// A Unix socket file the server created, removed when dropped.
#[derive(Debug)]
struct SocketFile(PathBuf);

impl Drop for SocketFile {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.0) {
            tracing::warn!("cannot remove {}: {err}", self.0.display());
        }
    }
}

/// Binds a Unix socket with mode 0600. A socket file nobody accepts on is stale (its process died) and is
/// replaced; a live socket, or a file that is not a socket, is an error.
async fn bind_unix(path: &Path) -> io::Result<(UnixListener, SocketFile)> {
    if UnixStream::connect(path).await.is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AddrInUse,
            format!(
                "{}: another process is accepting on this socket",
                path.display()
            ),
        ));
    }
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_socket()) {
        std::fs::remove_file(path).map_err(|e| context(e, path.display()))?;
    }
    let listener = UnixListener::bind(path).map_err(|e| context(e, path.display()))?;
    let file = SocketFile(path.to_owned());
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| context(e, path.display()))?;
    Ok((listener, file))
}

fn context(err: io::Error, what: impl Display) -> io::Error {
    io::Error::new(err.kind(), format!("{what}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_lines_name_the_method_and_cut_long_paths() {
        let uri: Uri = "/prometheus/metrics?x=1".parse().unwrap();
        assert_eq!(request_line(&Method::GET, &uri), "GET /prometheus/metrics");
        let long: Uri = format!("/{}", "a".repeat(300)).parse().unwrap();
        let line = request_line(&Method::POST, &long);
        assert_eq!(line, format!("POST /{}…", "a".repeat(255)));
    }
}
