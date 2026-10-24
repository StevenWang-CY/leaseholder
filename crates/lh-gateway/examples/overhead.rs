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
//!
//! A second table prices each admission policy: background requests, which every policy holds, through one
//! gateway per policy with limits the runs never reach, so that what a policy adds is its bookkeeping and never
//! a wait. "Added" there is over a gateway without a policy, which runs the same requests just before; each
//! gateway's `lh_gateway_overhead_seconds` follows. Its realtime rows send voice requests, which flow control,
//! preempt-resubmit and policy leaseholder hold. Policy leaseholder runs twice: alone, where no realtime session
//! is in a turn, and with events, while a task posts a lifecycle event every millisecond (four voice sessions in
//! turn, through speech onset, end of turn, playback and its end), so that requests are decided beside sessions
//! that speak and are in their critical windows.
//!
//! A third table times `POST /v1/events`, batches of one and four events, through a gateway without a policy
//! and through policy leaseholder, which moves its sessions by each event under the admission lock.
//!
//! A fourth prices placement: the same requests through a gateway on two nodes, each a mock engine of its own,
//! and through a gateway on one with the same admission policy, the two started together so that they serve the
//! same requests: without an admission policy (placement least-loaded) and with policy leaseholder (placement
//! leaseholder). Every request names the same session, so all but the first are requests of a placed session,
//! which find their node in the placement table and keep it. The second engine and these gateways start once
//! the first three tables have run, so those run in the process they always had.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::response::IntoResponse;
use axum::routing::{get, post};
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
/// The key of the voice tenant, whose requests are realtime.
const VOICE_KEY: &str = "overhead-bench-voice-key";
const CHUNKS: usize = 8;
const CHUNK_GAP: Duration = Duration::from_millis(1);
const WORKERS: usize = 4;
/// The class of the main table's requests, which a trusted tenant chooses with `X-LH-Class`.
const INTERACTIVE: &str = "interactive";
/// The class of the policy table's requests: every policy holds background requests.
const BACKGROUND: &str = "background";
/// The class of the voice tenant's requests.
const REALTIME: &str = "realtime";
/// The time between two lifecycle events while events flow.
const EVENT_GAP: Duration = Duration::from_millis(1);
/// Each admission policy, with limits no run reaches.
const POLICIES: [(&str, &str); 5] = [
    (
        "static-cap",
        "policy = \"static-cap\"\nbackground_max_in_flight = 4096",
    ),
    (
        "token-reservation",
        "policy = \"token-reservation\"\npool_tokens = 1000000000\nreserved_tokens = 36400",
    ),
    (
        "flow-control",
        "policy = \"flow-control\"\nceilings = { realtime = 4096, interactive = 4096, background = 4096 }",
    ),
    ("preempt-resubmit", "policy = \"preempt-resubmit\""),
    (
        "leaseholder",
        "policy = \"leaseholder\"\npool_tokens = 1000000000\nheadroom_tokens = 1024\n\
         critical_prefill_budget_ms = 1000000000\ndegrade_max_tokens = 64",
    ),
];
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
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_writer(std::io::sink)
        .init();
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
    let gateway_on = |engines: &[SocketAddr], config: &str| Target {
        addr: gateway_rt.block_on(start_gateway(engines, config)),
        runtime: Some(gateway_rt.handle().clone()),
    };
    let gateway = |config: &str| gateway_on(&[engine], config);
    let targets = [direct, bare, gateway("")];
    // A gateway without a policy for the policy table, with its own overhead histogram, and policy leaseholder
    // once more for the run with events. These gateways have a voice tenant too.
    let voice = format!(
        "[tenants.voice]\nkeys_sha256 = [\"{}\"]\ntrusted = true\ndefault_class = \"realtime\"",
        digest(VOICE_KEY)
    );
    let mut policies: Vec<Priced> = [("none", "")]
        .into_iter()
        .chain(POLICIES)
        .map(|(name, admission)| Priced {
            name,
            target: gateway(&format!("{voice}\n[admission]\n{admission}")),
            events: false,
        })
        .collect();
    let leaseholder = POLICIES[POLICIES.len() - 1].1;
    policies.push(Priced {
        name: "leaseholder, with events",
        target: gateway(&format!("{voice}\n[admission]\n{leaseholder}")),
        events: true,
    });
    client_rt.block_on(async {
        measure(targets).await;
        price_policies(&policies).await;
        time_events(&[&policies[0], &policies[POLICIES.len()]]).await;
    });
    // The placement table's second engine and its gateways start only now, so that the tables above run in the
    // process they ran in before placement. Each gateway on two nodes is priced against one on a single node
    // with its admission policy.
    let engines = [engine, engine_rt.block_on(start_engine())];
    let priced = |name, engines: &[SocketAddr], config: String| Priced {
        name,
        target: gateway_on(engines, &format!("{voice}\n{config}")),
        events: false,
    };
    let pairs = [
        (
            priced("one node, none", &engines[..1], String::new()),
            priced(
                "two nodes, least-loaded",
                &engines,
                "[placement]\npolicy = \"least-loaded\"".into(),
            ),
        ),
        (
            priced(
                "one node, leaseholder",
                &engines[..1],
                format!("[admission]\n{leaseholder}"),
            ),
            priced(
                "two nodes, leaseholder",
                &engines,
                format!("[placement]\npolicy = \"leaseholder\"\n[admission]\n{leaseholder}"),
            ),
        ),
    ];
    client_rt.block_on(price_placement(&pairs));
    let _ = std::fs::remove_file(salt_secret());
}

/// The gateways' cache-salt secret, a file of mode 600 that `main` removes at the end.
fn salt_secret() -> &'static Path {
    static FILE: OnceLock<PathBuf> = OnceLock::new();
    FILE.get_or_init(|| {
        let name = format!("lh-gw-overhead-{}-cache-salt-secret", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, format!("{}\n", "ab".repeat(32))).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        path
    })
}

/// A gateway of the policy table: its policy, and whether lifecycle events flow while its requests run.
struct Priced {
    name: &'static str,
    target: Target,
    events: bool,
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

/// The mock engine: JSON at once, or `CHUNKS` stamped SSE chunks `CHUNK_GAP` apart (the first at once). Its
/// `/health` answers the probes of a gateway on nodes.
async fn start_engine() -> SocketAddr {
    let app = Router::new().route("/health", get(|| async {}));
    serve(app.route(
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

fn digest(key: &str) -> String {
    Sha256::digest(key)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A gateway in front of `engines`, its upstream if there is one and nodes `a` and `b` if there are two, with
/// `extra` (more tenants, `[admission]` and `[placement]` tables, or nothing) in its configuration.
async fn start_gateway(engines: &[SocketAddr], extra: &str) -> SocketAddr {
    let (upstream, nodes) = match engines {
        [engine] => (format!("upstream = \"http://{engine}\""), String::new()),
        _ => (
            String::new(),
            engines
                .iter()
                .zip(["a", "b"])
                .map(|(engine, name)| format!("[nodes.{name}]\nupstream = \"http://{engine}\"\n"))
                .collect(),
        ),
    };
    let config = Config::from_toml(&format!(
        "{upstream}\ncache_salt_secret_file = \"{}\"\n[[listen]]\ntcp = \"127.0.0.1:0\"\n\
         [tenants.agents]\nkeys_sha256 = [\"{}\"]\ntrusted = true\ndefault_class = \"background\"\n\
         max_class = \"interactive\"\n{nodes}{extra}",
        salt_secret().display(),
        digest(KEY)
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

fn http_client() -> HttpClient {
    let mut connector = HttpConnector::new();
    connector.set_nodelay(true);
    Client::builder(TokioExecutor::new()).build(connector)
}

async fn measure(targets: [Target; 3]) {
    let client = http_client();
    let small = chat_body(2 << 10, false);
    let large = chat_body(64 << 10, false);
    let streaming = chat_body(2 << 10, true);
    // Warm every path: connections, allocator, metric series.
    for target in &targets {
        run(&client, target, &small, Kind::Json, 64, 2_000, INTERACTIVE).await;
        run(
            &client,
            target,
            &streaming,
            Kind::Stream,
            16,
            200,
            INTERACTIVE,
        )
        .await;
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
                runs.push(run(&client, target, body, kind, concurrency, count, INTERACTIVE).await);
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
    println!();
    report_gateway_metric(&client, &targets[2], "lh-gateway").await;
}

/// The policy table: each scenario through a gateway without a policy and through one per policy.
async fn price_policies(gateways: &[Priced]) {
    let client = http_client();
    let small = chat_body(2 << 10, false);
    let streaming = chat_body(2 << 10, true);
    for gateway in gateways {
        let target = &gateway.target;
        run(&client, target, &small, Kind::Json, 64, 2_000, BACKGROUND).await;
        run(&client, target, &small, Kind::Json, 8, 2_000, REALTIME).await;
        run(
            &client,
            target,
            &streaming,
            Kind::Stream,
            16,
            200,
            BACKGROUND,
        )
        .await;
    }
    println!(
        "\nAdmission policies: background and realtime requests, limits never reached. Microseconds, p50 / p99.\n"
    );
    println!(
        "| Policy | Scenario | Concurrency | lh-gateway | Added over none | CPU per request |"
    );
    println!("|---|---|---:|---:|---:|---:|");
    let scenarios: [(&str, &Bytes, Kind, usize, usize, &str); 6] = [
        (
            "Non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            1,
            5_000,
            BACKGROUND,
        ),
        (
            "Non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            8,
            40_000,
            BACKGROUND,
        ),
        (
            "Non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            64,
            40_000,
            BACKGROUND,
        ),
        (
            "Streaming, first byte",
            &streaming,
            Kind::Stream,
            8,
            4_000,
            BACKGROUND,
        ),
        (
            "Realtime, non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            1,
            5_000,
            REALTIME,
        ),
        (
            "Realtime, non-streaming, 2 KiB body",
            &small,
            Kind::Json,
            8,
            40_000,
            REALTIME,
        ),
    ];
    for (scenario, body, kind, concurrency, count, class) in scenarios {
        let mut none = None;
        for gateway in gateways {
            let target = &gateway.target;
            let events = gateway.events.then(|| flow_events(&client, target));
            let run = run(&client, target, body, kind, concurrency, count, class).await;
            if let Some((stop, poster)) = events {
                stop.store(true, Ordering::Relaxed);
                assert!(poster.await.unwrap() > 0, "events flowed");
            }
            let samples = match kind {
                Kind::Json => &run.requests,
                Kind::Stream => &run.first_bytes,
            };
            let us = |q: f64| quantile(samples, q) as f64 / 1e3;
            let base = none.get_or_insert_with(|| (us(0.5), us(0.99)));
            let added = if gateway.name == "none" {
                "".to_owned()
            } else {
                format!(
                    "{:.0} / {:.0}",
                    (us(0.5) - base.0).max(0.0),
                    (us(0.99) - base.1).max(0.0)
                )
            };
            println!(
                "| {} | {scenario} | {concurrency} | {:.0} / {:.0} | {added} | {:.0} |",
                gateway.name,
                us(0.5),
                us(0.99),
                run.busy.as_secs_f64() * 1e6 / count as f64,
            );
        }
    }
    println!();
    for gateway in gateways {
        report_gateway_metric(&client, &gateway.target, gateway.name).await;
    }
}

/// The fourth table: each pair's gateway on two nodes against its gateway on one, in the policy table's request
/// scenarios. The two runs of a scenario alternate, the one-node run first, and the CPU columns are each run's.
/// Each gateway's `lh_gateway_overhead_seconds` follows.
async fn price_placement(pairs: &[(Priced, Priced)]) {
    let client = http_client();
    let small = chat_body(2 << 10, false);
    // Warm every path, and place the session.
    for (one, two) in pairs {
        for gateway in [one, two] {
            run(
                &client,
                &gateway.target,
                &small,
                Kind::Json,
                8,
                2_000,
                BACKGROUND,
            )
            .await;
            run(
                &client,
                &gateway.target,
                &small,
                Kind::Json,
                8,
                2_000,
                REALTIME,
            )
            .await;
        }
    }
    println!(
        "\nPlacement: the requests of one placed session, through two nodes and through one. Microseconds, p50 / p99.\n"
    );
    println!(
        "| Gateway | Scenario | Concurrency | One node | Two nodes | Added by placement | CPU per request, one / two |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|");
    let scenarios: [(&str, usize, usize, &str); 5] = [
        ("Non-streaming, 2 KiB body", 1, 5_000, BACKGROUND),
        ("Non-streaming, 2 KiB body", 8, 40_000, BACKGROUND),
        ("Non-streaming, 2 KiB body", 64, 40_000, BACKGROUND),
        ("Realtime, non-streaming, 2 KiB body", 1, 5_000, REALTIME),
        ("Realtime, non-streaming, 2 KiB body", 8, 40_000, REALTIME),
    ];
    for (scenario, concurrency, count, class) in scenarios {
        for (one, two) in pairs {
            let base = run(
                &client,
                &one.target,
                &small,
                Kind::Json,
                concurrency,
                count,
                class,
            )
            .await;
            let placed = run(
                &client,
                &two.target,
                &small,
                Kind::Json,
                concurrency,
                count,
                class,
            )
            .await;
            let us = |samples: &Samples, q: f64| quantile(&samples.requests, q) as f64 / 1e3;
            let cpu = |samples: &Samples| samples.busy.as_secs_f64() * 1e6 / count as f64;
            println!(
                "| {} | {scenario} | {concurrency} | {:.0} / {:.0} | {:.0} / {:.0} | {:.0} / {:.0} | {:.0} / {:.0} |",
                two.name,
                us(&base, 0.5),
                us(&base, 0.99),
                us(&placed, 0.5),
                us(&placed, 0.99),
                us(&placed, 0.5) - us(&base, 0.5),
                us(&placed, 0.99) - us(&base, 0.99),
                cpu(&base),
                cpu(&placed),
            );
        }
    }
    println!();
    for gateway in pairs.iter().flat_map(|(one, two)| [one, two]) {
        report_gateway_metric(&client, &gateway.target, gateway.name).await;
    }
}

/// Posts lifecycle events to `target`, one every [`EVENT_GAP`], until the flag is set; the task returns how many
/// it posted. Its posts are not counted as the gateway's requests, but its CPU is the gateway's.
fn flow_events(
    client: &HttpClient,
    target: &Target,
) -> (Arc<AtomicBool>, tokio::task::JoinHandle<u64>) {
    let stop = Arc::new(AtomicBool::new(false));
    let (client, addr, flag) = (client.clone(), target.addr, Arc::clone(&stop));
    let poster = tokio::spawn(async move {
        let mut tick = tokio::time::interval(EVENT_GAP);
        let mut posted = 0;
        while !flag.load(Ordering::Relaxed) {
            tick.tick().await;
            post_events(&client, addr, 1).await;
            posted += 1;
        }
        posted
    });
    (stop, poster)
}

/// The third table: `POST /v1/events` through a gateway without a policy and through policy leaseholder.
async fn time_events(gateways: &[&Priced]) {
    let client = http_client();
    for gateway in gateways {
        run_events(&client, &gateway.target, 8, 2_000, 1).await;
    }
    println!(
        "\n`POST /v1/events`: four voice sessions through their turns. Microseconds, p50 / p99.\n"
    );
    println!(
        "| Policy | Concurrency | Events per POST | POSTs | Latency | CPU per POST | Events per second |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (concurrency, per_post, posts) in [(1, 1, 5_000), (8, 1, 40_000), (8, 4, 20_000)] {
        for gateway in gateways {
            let started = Instant::now();
            let run = run_events(&client, &gateway.target, concurrency, posts, per_post).await;
            let elapsed = started.elapsed().as_secs_f64();
            let us = |q: f64| quantile(&run.requests, q) as f64 / 1e3;
            println!(
                "| {} | {concurrency} | {per_post} | {posts} | {:.0} / {:.0} | {:.0} | {:.0} |",
                gateway.name,
                us(0.5),
                us(0.99),
                run.busy.as_secs_f64() * 1e6 / posts as f64,
                (posts * per_post) as f64 / elapsed,
            );
        }
    }
}

/// Posts `total` batches of `per_post` events from `concurrency` tasks, timing each POST.
async fn run_events(
    client: &HttpClient,
    target: &Target,
    concurrency: usize,
    total: usize,
    per_post: usize,
) -> Samples {
    let busy_before = target.busy();
    let tasks: Vec<_> = (0..concurrency)
        .map(|task| {
            let (client, addr) = (client.clone(), target.addr);
            let count = total / concurrency + usize::from(task < total % concurrency);
            tokio::spawn(async move {
                let mut samples = Vec::with_capacity(count);
                for _ in 0..count {
                    let started = Instant::now();
                    post_events(&client, addr, per_post).await;
                    samples.push(started.elapsed().as_nanos() as u64);
                }
                samples
            })
        })
        .collect();
    let mut all = Samples::default();
    for task in tasks {
        all.requests.extend(task.await.unwrap());
    }
    all.busy = target.busy() - busy_before;
    all.requests.sort_unstable();
    all
}

/// Posts `count` lifecycle events in one batch: four voice sessions in turn, each through speech onset, end of
/// turn, playback and its end, with ids unique in the process.
async fn post_events(client: &HttpClient, addr: SocketAddr, count: usize) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    const TYPES: [&str; 4] = [
        "user_speech_started",
        "user_turn_ended",
        "playback_started",
        "playback_ended",
    ];
    let first = NEXT.fetch_add(count as u64, Ordering::Relaxed);
    let events: Vec<String> = (first..first + count as u64)
        .map(|n| {
            format!(
                r#"{{"event_id":"bench-{n}","session_id":"voice-bench-{}","at_ms":1792650010000,"source":"voice-worker","type":"{}","played_ms":0}}"#,
                n / 4 % 4,
                TYPES[n as usize % 4]
            )
        })
        .collect();
    let request = http::Request::post(format!("http://{addr}/v1/events"))
        .header(HOST, "gateway")
        .header(AUTHORIZATION, format!("Bearer {KEY}"))
        .header(CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(format!("[{}]", events.join(",")))))
        .unwrap();
    let response = client.request(request).await.unwrap();
    assert_eq!(response.status(), http::StatusCode::ACCEPTED);
    response.into_body().collect().await.unwrap();
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

/// Runs `total` requests of `class` from `concurrency` tasks. Streaming tasks start staggered over one
/// stream's length, so equal-length streams do not start in lockstep.
async fn run(
    client: &HttpClient,
    target: &Target,
    body: &Bytes,
    kind: Kind,
    concurrency: usize,
    total: usize,
    class: &'static str,
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
                    one(&client, addr, body.clone(), kind, class, &mut samples).await;
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
    class: &str,
    samples: &mut Samples,
) {
    let key = if class == REALTIME { VOICE_KEY } else { KEY };
    let request = http::Request::post(format!("http://{addr}/v1/chat/completions"))
        .header(HOST, "gateway")
        .header(AUTHORIZATION, format!("Bearer {key}"))
        .header(CONTENT_TYPE, "application/json")
        .header("x-session-id", "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41")
        .header("x-lh-class", class)
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

/// The gateway's own view of its overhead, from `lh_gateway_overhead_seconds`; `label` names the gateway.
async fn report_gateway_metric(client: &HttpClient, gateway: &Target, label: &str) {
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
        "lh_gateway_overhead_seconds, {label}, all {count:.0} requests: mean {:.0} us; {:.1}% within 100 us, \
         {:.1}% within 500 us, {:.2}% within 2 ms",
        sum / count * 1e6,
        within("0.0001") * 100.0,
        within("0.0005") * 100.0,
        within("0.002") * 100.0,
    );
}
