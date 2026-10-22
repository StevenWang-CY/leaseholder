//! Relaying: cancellation, backpressure, upstream failures, shutdown, and the Unix socket listener.

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;
use lh_gateway::{Config, LogSink, Server};
use tokio::net::{TcpListener, TcpStream};

use crate::common::*;

const OUTCOME: &str = "lh_gateway_request_outcomes_total{class=\"background\",outcome=";

#[tokio::test]
async fn a_client_that_leaves_mid_stream_closes_the_upstream_within_100_ms() {
    let chunk = Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    let mut mock = Mock::start(Script::Endless {
        chunk,
        gap: Duration::from_millis(20),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let (response, connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let mut body = response.into_body();
    for _ in 0..3 {
        body.frame().await.unwrap().unwrap();
    }
    gw.await_metric("lh_gateway_streams_active{class=\"background\"}", 1.0)
        .await;

    let left = Instant::now();
    connection.abort();
    drop(body);
    let dropped = mock.next_drop(Duration::from_secs(2)).await;
    let after = dropped.saturating_duration_since(left);
    assert!(
        after < Duration::from_millis(100),
        "the upstream body was dropped {after:?} after the client left"
    );
    gw.await_metric(&format!("{OUTCOME}\"cancelled\"}}"), 1.0)
        .await;
    gw.await_metric("lh_gateway_streams_active{class=\"background\"}", 0.0)
        .await;
}

#[tokio::test]
async fn a_client_that_leaves_before_the_response_cancels_the_upstream_request() {
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let gw = Gateway::start(mock.addr, "").await;
    let (mut sender, connection) = connect(&gw.tcp).await;
    let pending = tokio::spawn(sender.send_request(request(
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )));
    let deadline = Instant::now() + Duration::from_secs(2);
    while mock.requests().is_empty() {
        assert!(
            Instant::now() < deadline,
            "the request never reached the mock engine"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let left = Instant::now();
    connection.abort();
    pending.abort();
    let dropped = mock.next_drop(Duration::from_secs(2)).await;
    let after = dropped.saturating_duration_since(left);
    assert!(
        after < Duration::from_millis(100),
        "upstream cancelled {after:?} after the client left"
    );
    gw.await_metric(&format!("{OUTCOME}\"cancelled\"}}"), 1.0)
        .await;
    gw.await_metric(
        "lh_gateway_requests_total{tenant=\"agents\",route=\"/v1/chat/completions\",class=\"background\",status=\"499\"}",
        1.0,
    )
    .await;
}

#[tokio::test]
async fn a_stalled_reader_is_cut_off_after_the_stall_timeout() {
    let mut mock = Mock::start(Script::Endless {
        chunk: Bytes::from(vec![b'x'; 64 << 10]),
        gap: Duration::ZERO,
    })
    .await;
    let top = "stall_timeout_s = 0.3\nstream_buffer_frames = 4";
    let gw = Gateway::start(mock.addr, top).await;
    let started = Instant::now();
    // The client never reads: its socket fills, then the gateway's buffer, and the stream stalls.
    let (response, _connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let dropped = mock.next_drop(Duration::from_secs(10)).await;
    let after = dropped.saturating_duration_since(started);
    assert!(
        after >= Duration::from_millis(300) && after < Duration::from_secs(5),
        "the upstream was aborted {after:?} after the request"
    );
    gw.await_metric(&format!("{OUTCOME}\"stalled\"}}"), 1.0)
        .await;

    // What was buffered still arrives, and then an error: a cut-off stream must not look complete.
    let mut body = response.into_body();
    let end = loop {
        match body.frame().await {
            Some(Ok(_)) => continue,
            Some(Err(_)) => break "error",
            None => break "clean end",
        }
    };
    assert_eq!(end, "error");
}

#[tokio::test]
async fn an_unreachable_upstream_is_502() {
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = closed.local_addr().unwrap();
    drop(closed);
    let gw = Gateway::start(addr, "").await;
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 502);
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "server_error");
    assert_eq!(error["code"], "upstream_unavailable");
    assert!(reply.header("x-lh-session-id").is_some());
    gw.await_metric("lh_gateway_upstream_errors_total{kind=\"connect\"}", 1.0)
        .await;
    gw.await_metric(&format!("{OUTCOME}\"error\"}}"), 1.0).await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
}

#[tokio::test]
async fn no_response_headers_within_the_first_byte_timeout_is_504() {
    let mut mock = Mock::start(Script::Slow(Duration::from_secs(10))).await;
    let gw = Gateway::start(mock.addr, "first_byte_timeout_s = 0.2").await;
    let started = Instant::now();
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 504);
    assert_eq!(reply.json()["error"]["code"], "upstream_timeout");
    assert!(started.elapsed() >= Duration::from_millis(200));
    // The engine's request was abandoned too: the gateway closed the connection, which dropped the handler.
    let dropped = mock.next_drop(Duration::from_secs(2)).await;
    assert!(dropped.saturating_duration_since(started) < Duration::from_secs(2));
    gw.await_metric("lh_gateway_upstream_errors_total{kind=\"timeout\"}", 1.0)
        .await;
}

#[tokio::test]
async fn shutdown_stops_accepting_and_drains_in_flight_streams() {
    let chunks: Vec<Bytes> = (0..5)
        .map(|i| Bytes::from(format!("data: {i}\n\n")))
        .collect();
    let expected: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(100),
    })
    .await;
    let mut gw = Gateway::start(mock.addr, "drain_timeout_s = 5").await;
    let (response, _connection) = open(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    let mut body = response.into_body();
    let mut received = body
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap()
        .to_vec();

    gw.shut_down();
    let Target::Tcp(addr) = gw.tcp.clone() else {
        unreachable!()
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    while TcpStream::connect(addr).await.is_ok() {
        assert!(
            Instant::now() < deadline,
            "the gateway still accepts after shutdown"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The stream that was in flight completes.
    while let Some(frame) = body.frame().await {
        received.extend_from_slice(&frame.unwrap().into_data().unwrap());
    }
    assert_eq!(received, expected);
    tokio::time::timeout(Duration::from_secs(2), gw.stopped())
        .await
        .expect("the gateway exits once drained");
}

#[tokio::test]
async fn the_unix_listener_replaces_a_stale_socket_is_private_and_cleans_up() {
    let mock = Mock::start(Script::Json("{}")).await;
    let path = socket_path();
    // A socket file left behind by a process that died.
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    assert!(path.exists());
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    let mut gw = Gateway::from_config(config.clone(), Arc::new(LogSink)).await;
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);

    // A live socket is never taken over.
    let err = Server::bind(config, Arc::new(LogSink)).await.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::AddrInUse);

    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &[], CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(mock.last().header("x-vllm-priority"), Some("0"));
    gw.stopped().await;
    assert!(!path.exists(), "the socket file is removed at exit");

    // A path that holds something other than a socket is left alone.
    std::fs::write(&path, "not a socket").unwrap();
    let config = Config::from_toml(&gateway_toml(mock.addr, &path, "")).unwrap();
    assert!(Server::bind(config, Arc::new(LogSink)).await.is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "not a socket");
    std::fs::remove_file(&path).unwrap();
}
