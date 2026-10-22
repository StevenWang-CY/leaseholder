//! The proxy tax: the latency `lh-gateway` adds over talking to the engine directly, and its CPU time.
//!
//! `cargo run --release -p lh-gateway --example overhead`
//!
//! A mock engine that answers at once, the client and two proxies run in this process, each on its own tokio
//! runtime with 4 worker threads, over loopback TCP with pooled keep-alive connections. Every scenario runs three
//! times: direct to the engine; through a bare reverse proxy on the same stack (axum server, hyper-util client,
//! bodies passed through untouched), which is the cost of the HTTP hop itself; and through `lh-gateway`.
//! "Added" is the gateway's quantile minus the direct one. The gateway formats its per-request log line, into
//! a sink, as it does in production.
//! - Non-streaming: request sent to full response received, with a 2 KiB chat body and with a 64 KiB one.
//! - Streaming, first byte: request sent to the first SSE chunk received.
//! - Streaming, per chunk: the engine stamps each chunk as it writes it; the latency is stamp to receipt.
//!   Chunks are 1 ms apart, so they are not coalesced.
//! - CPU: busy time of the proxy's runtime workers per request (per stream for streaming rows), which includes
//!   the system calls the proxy makes.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::serve::ListenerExt;
use bytes::Bytes;
use futures_util::stream;
use http::header::{AUTHORIZATION, CONTENT_TYPE, HOST};
use http_body_util::{BodyExt, Full};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use lh_gateway::{Config, ListenAddr, LogSink, Server};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::runtime::{Handle, Runtime};

const KEY: &str = "overhead-bench-key";
const CHUNKS: usize = 8;
const CHUNK_GAP: Duration = Duration::from_millis(1);
const WORKERS: usize = 4;
const RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1792650000,"model":"qwen",
"choices":[{"index":0,"message":{"role":"assistant","content":"The table for four is booked for seven tonight. We will hold it for fifteen minutes."},"finish_reason":"stop"}],
"usage":{"prompt_tokens":512,"completion_tokens":21,"total_tokens":533,"prompt_tokens_details":{"cached_tokens":448}}}"#;

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn now_ns() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Where requests go, and the runtime whose workers do the proxy's work (none when direct).
#[derive(Clone)]
struct Target {
    addr: SocketAddr,
    runtime: Option<Handle>,
}

impl Target {
    fn busy(&self) -> Duration {
        self.runtime.as_ref().map_or(Duration::ZERO, |handle| {
            let metrics = handle.metrics();
            (0..metrics.num_workers())
                .map(|worker| metrics.worker_total_busy_duration(worker))
                .sum()
        })
    }
}

fn main() {
    let (engine_rt, bare_rt, gateway_rt, client_rt) = (runtime(), runtime(), runtime(), runtime());
    let engine = engine_rt.block_on(start_engine());
    let direct = Target {
        addr: engine,
        runtime: None,
    };
    let bare = Target {
        addr: bare_rt.block_on(start_bare_proxy(engine)),
        runtime: Some(bare_rt.handle().clone()),
    };
    let gateway = Target {
        addr: gateway_rt.block_on(start_gateway(engine)),
        runtime: Some(gateway_rt.handle().clone()),
    };
    client_rt.block_on(measure([direct, bare, gateway]));
}

fn runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKERS)
        .enable_all()
        .build()
        .unwrap()
}

async fn serve(app: Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let listener = listener.tap_io(|tcp| tcp.set_nodelay(true).unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// The mock engine: JSON at once, or `CHUNKS` stamped SSE chunks `CHUNK_GAP` apart (the first at once).
async fn start_engine() -> SocketAddr {
    serve(Router::new().route(
        "/v1/chat/completions",
        post(|body: Bytes| async move {
            if memchr::memmem::find(&body, b"\"stream\":true").is_none() {
                return ([(CONTENT_TYPE, "application/json")], RESPONSE).into_response();
            }
            let chunks = stream::unfold(0, |i| async move {
                if i == CHUNKS {
                    return None;
                }
                if i > 0 {
                    tokio::time::sleep(CHUNK_GAP).await;
                }
                let chunk = format!(
                    "data: {{\"t\":{},\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\
                     \"delta\":{{\"content\":\" token\"}},\"finish_reason\":null}}]}}\n\n",
                    now_ns()
                );
                Some((Ok::<_, Infallible>(Bytes::from(chunk)), i + 1))
            });
            ([(CONTENT_TYPE, "text/event-stream")], Body::from_stream(chunks)).into_response()
        }),
    ))
    .await
}

/// A bare reverse proxy on the gateway's stack that forwards both bodies untouched: the HTTP hop alone.
async fn start_bare_proxy(engine: SocketAddr) -> SocketAddr {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    let client: Client<HttpConnector, Body> =
        Client::builder(TokioExecutor::new()).build(connector);
    serve(Router::new().fallback(move |request: Request| {
        let client = client.clone();
        async move {
            let (mut parts, body) = request.into_parts();
            parts.uri = format!("http://{engine}{}", parts.uri.path())
                .parse()
                .unwrap();
            let response = client.request(http::Request::from_parts(parts, body)).await;
            response.unwrap().map(Body::new)
        }
    }))
    .await
}

async fn start_gateway(engine: SocketAddr) -> SocketAddr {
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_writer(std::io::sink)
        .init();
    let digest: String = Sha256::digest(KEY)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let config = Config::from_toml(&format!(
        "upstream = \"http://{engine}\"\n[[listen]]\ntcp = \"127.0.0.1:0\"\n\
         [tenants.agents]\nkeys_sha256 = [\"{digest}\"]\ntrusted = true\ndefault_class = \"background\"\n\
         max_class = \"interactive\""
    ))
    .unwrap();
    let server = Server::bind(config, Arc::new(LogSink)).await.unwrap();
    let addrs = server.local_addrs().unwrap();
    let [ListenAddr::Tcp(addr)] = addrs[..] else {
        unreachable!("the configuration has one TCP listener")
    };
    tokio::spawn(server.run(std::future::pending()));
    addr
}

#[derive(Clone, Copy)]
enum Kind {
    Json,
    Stream,
}

/// Sorted latency samples of one run, in nanoseconds, and the proxy's busy time.
#[derive(Default)]
struct Samples {
    requests: Vec<u64>,
    first_bytes: Vec<u64>,
    chunks: Vec<u64>,
    busy: Duration,
}

type HttpClient = Client<HttpConnector, Full<Bytes>>;

async fn measure(targets: [Target; 3]) {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    let client: HttpClient = Client::builder(TokioExecutor::new()).build(connector);
    let small = chat_body(2 << 10, false);
    let large = chat_body(64 << 10, false);
    let streaming = chat_body(2 << 10, true);
    // Warm every path: connections, allocator, metric series.
    for target in &targets {
        run(&client, target, &small, Kind::Json, 64, 2_000).await;
        run(&client, target, &streaming, Kind::Stream, 16, 200).await;
    }

    println!(
        "lh-gateway proxy tax: {} {}, {WORKERS} worker threads per runtime. Microseconds; p50 / p99.\n",
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    println!(
        "| Scenario | Concurrency | Samples | Direct | Bare proxy | lh-gateway | Added by lh-gateway | CPU per request, bare / lh-gateway |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    let scenarios: [(&str, &Bytes, Kind, &[usize], usize); 3] = [
        (
            "Non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            &[1, 8, 64],
            5_000,
        ),
        (
            "Non-streaming, 64 KiB body",
            &large,
            Kind::Json,
            &[1, 8],
            1_000,
        ),
        ("Streaming", &streaming, Kind::Stream, &[1, 8, 64], 500),
    ];
    for (name, body, kind, concurrencies, per_task) in scenarios {
        for &concurrency in concurrencies {
            let count = per_task * concurrency.min(8);
            let mut runs = Vec::with_capacity(3);
            for target in &targets {
                runs.push(run(&client, target, body, kind, concurrency, count).await);
            }
            match kind {
                Kind::Json => row(name, concurrency, &runs, |s| &s.requests, count),
                Kind::Stream => {
                    row(
                        "Streaming, first byte",
                        concurrency,
                        &runs,
                        |s| &s.first_bytes,
                        count,
                    );
                    row(
                        "Streaming, per chunk",
                        concurrency,
                        &runs,
                        |s| &s.chunks,
                        count,
                    );
                }
            }
        }
    }
    report_gateway_metric(&client, &targets[2]).await;
}

fn row(
    scenario: &str,
    concurrency: usize,
    runs: &[Samples],
    samples: fn(&Samples) -> &Vec<u64>,
    count: usize,
) {
    let us = |ns: u64| ns as f64 / 1e3;
    let pair = |s: &Samples| {
        let v = samples(s);
        format!("{:.0} / {:.0}", us(quantile(v, 0.5)), us(quantile(v, 0.99)))
    };
    let (direct, bare, gateway) = (&runs[0], &runs[1], &runs[2]);
    let added =
        |q: f64| (us(quantile(samples(gateway), q)) - us(quantile(samples(direct), q))).max(0.0);
    let cpu = |s: &Samples| s.busy.as_secs_f64() * 1e6 / count as f64;
    println!(
        "| {scenario} | {concurrency} | {} | {} | {} | {} | {:.0} / {:.0} | {:.0} / {:.0} |",
        samples(gateway).len(),
        pair(direct),
        pair(bare),
        pair(gateway),
        added(0.5),
        added(0.99),
        cpu(bare),
        cpu(gateway),
    );
}

fn quantile(sorted: &[u64], q: f64) -> u64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Runs `total` requests from `concurrency` tasks. Streaming tasks start staggered over one stream's length,
/// so equal-length streams do not start in lockstep.
async fn run(
    client: &HttpClient,
    target: &Target,
    body: &Bytes,
    kind: Kind,
    concurrency: usize,
    total: usize,
) -> Samples {
    let busy_before = target.busy();
    let tasks: Vec<_> = (0..concurrency)
        .map(|task| {
            let (client, body, addr) = (client.clone(), body.clone(), target.addr);
            let count = total / concurrency + usize::from(task < total % concurrency);
            let stagger = match kind {
                Kind::Json => Duration::ZERO,
                Kind::Stream => CHUNK_GAP * (CHUNKS * task / concurrency) as u32,
            };
            tokio::spawn(async move {
                tokio::time::sleep(stagger).await;
                let mut samples = Samples::default();
                for _ in 0..count {
                    one(&client, addr, body.clone(), kind, &mut samples).await;
                }
                samples
            })
        })
        .collect();
    let mut all = Samples::default();
    for task in tasks {
        let samples = task.await.unwrap();
        all.requests.extend(samples.requests);
        all.first_bytes.extend(samples.first_bytes);
        all.chunks.extend(samples.chunks);
    }
    all.busy = target.busy() - busy_before;
    for samples in [&mut all.requests, &mut all.first_bytes, &mut all.chunks] {
        samples.sort_unstable();
    }
    all
}

async fn one(
    client: &HttpClient,
    addr: SocketAddr,
    body: Bytes,
    kind: Kind,
    samples: &mut Samples,
) {
    let request = http::Request::post(format!("http://{addr}/v1/chat/completions"))
        .header(HOST, "gateway")
        .header(AUTHORIZATION, format!("Bearer {KEY}"))
        .header(CONTENT_TYPE, "application/json")
        .header("x-session-id", "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41")
        .header("x-lh-class", "interactive")
        .body(Full::new(body))
        .unwrap();
    let started = Instant::now();
    let response = client.request(request).await.unwrap();
    assert!(response.status().is_success(), "{}", response.status());
    let mut body = response.into_body();
    match kind {
        Kind::Json => {
            while body.frame().await.is_some() {}
            samples.requests.push(started.elapsed().as_nanos() as u64);
        }
        Kind::Stream => {
            let mut chunk = 0;
            while let Some(frame) = body.frame().await {
                let received = now_ns();
                if chunk == 0 {
                    samples
                        .first_bytes
                        .push(started.elapsed().as_nanos() as u64);
                }
                let data = frame.unwrap().into_data().unwrap();
                for stamp in stamps(&data) {
                    // The first chunk is the first byte; the later ones measure the per-chunk path.
                    if chunk > 0 {
                        samples.chunks.push(received.saturating_sub(stamp));
                    }
                    chunk += 1;
                }
            }
            assert_eq!(chunk, CHUNKS);
        }
    }
}

/// The engine's write stamps in an SSE frame, which may hold more than one event.
fn stamps(data: &[u8]) -> impl Iterator<Item = u64> + '_ {
    memchr::memmem::find_iter(data, b"\"t\":").map(move |at| {
        data[at + 4..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .fold(0, |n, b| n * 10 + u64::from(b - b'0'))
    })
}

/// A chat request of about `size` bytes, with a session key as Codex sends it.
fn chat_body(size: usize, stream: bool) -> Bytes {
    let text = "Please check the reservation book for Friday evening and tell me what is free. "
        .repeat(size / 80);
    Bytes::from(
        serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": text}],
            "max_tokens": 64,
            "stream": stream,
            "prompt_cache_key": "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41",
        })
        .to_string(),
    )
}

/// The gateway's own view of its overhead, from `lh_gateway_overhead_seconds`.
async fn report_gateway_metric(client: &HttpClient, gateway: &Target) {
    let request = http::Request::get(format!("http://{}/metrics", gateway.addr))
        .header(HOST, "gateway")
        .body(Full::default())
        .unwrap();
    let response = client.request(request).await.unwrap();
    let text = String::from_utf8(response.collect().await.unwrap().to_bytes().to_vec()).unwrap();
    let series = |name: &str| -> f64 {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
            .unwrap_or(0.0)
    };
    let route = "route=\"/v1/chat/completions\"";
    let count = series(&format!("lh_gateway_overhead_seconds_count{{{route}}}"));
    let sum = series(&format!("lh_gateway_overhead_seconds_sum{{{route}}}"));
    let within = |le: &str| {
        series(&format!(
            "lh_gateway_overhead_seconds_bucket{{le=\"{le}\",{route}}}"
        )) / count
    };
    println!(
        "\nlh_gateway_overhead_seconds, all {count:.0} requests: mean {:.0} us; {:.1}% within 100 us, \
         {:.1}% within 500 us, {:.2}% within 2 ms",
        sum / count * 1e6,
        within("0.0001") * 100.0,
        within("0.0005") * 100.0,
        within("0.002") * 100.0,
    );
}
