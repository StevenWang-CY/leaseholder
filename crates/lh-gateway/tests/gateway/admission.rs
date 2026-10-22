//! The admission point: caps, queues, timeouts and token reservations.

use std::time::{Duration, Instant};

use crate::common::*;

const WAITING: &str = "lh_gateway_admission_waiting{class=\"background\"}";

/// A background request of the agents tenant, named `tag`, whose answer the mock engine holds.
fn background(tag: &'static str) -> Vec<(&'static str, &'static str)> {
    vec![AGENTS_AUTH, (TAG, tag), (GATE, "1")]
}

#[tokio::test]
async fn the_static_cap_holds_background_requests_and_queues_the_rest_in_order() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top = "[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 2\nqueue_limit = 2";
    let gw = Gateway::start(mock.addr, top).await;
    let mut replies = Vec::new();
    for (n, tag) in ["bg1", "bg2"].into_iter().enumerate() {
        replies.push(spawn_call(&gw.tcp, background(tag), CHAT));
        mock.await_requests(n + 1).await;
    }
    for (n, tag) in ["bg3", "bg4"].into_iter().enumerate() {
        replies.push(spawn_call(&gw.tcp, background(tag), CHAT));
        gw.await_metric(WAITING, n as f64 + 1.0).await;
    }
    // The queue is full.
    let full = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &background("bg5"),
        CHAT,
    )
    .await;
    assert_eq!(full.status, 429);
    let error = &full.json()["error"];
    assert_eq!(
        (&error["type"], &error["code"]),
        (&"rate_limit_error".into(), &"queue_full".into())
    );
    // Realtime requests are not held.
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &[(TAG, "rt")],
        CHAT,
    )
    .await;
    assert_eq!(voice.status, 200);
    assert_eq!(mock.tags(), ["bg1", "bg2", "rt"]);

    // Each answer lets the next waiting request in, in arrival order.
    mock.release(1);
    mock.await_requests(4).await;
    mock.release(1);
    mock.await_requests(5).await;
    assert_eq!(mock.tags(), ["bg1", "bg2", "rt", "bg3", "bg4"]);
    mock.release(2);
    for reply in replies {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    gw.await_metric(WAITING, 0.0).await;
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "lh_gateway_admission_admitted_total{class=\"background\"}",
            4.0,
        ),
        (
            "lh_gateway_admission_admitted_total{class=\"realtime\"}",
            1.0,
        ),
        (
            "lh_gateway_admission_wait_seconds_count{class=\"background\"}",
            4.0,
        ),
        (
            "lh_gateway_admission_refused_total{class=\"background\",reason=\"queue_full\"}",
            1.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn a_request_not_admitted_within_the_queue_timeout_gets_503() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top =
        "[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1\nqueue_timeout_s = 0.2";
    let gw = Gateway::start(mock.addr, top).await;
    let first = spawn_call(&gw.tcp, background("bg1"), CHAT);
    mock.await_requests(1).await;
    let started = Instant::now();
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &background("bg2"),
        CHAT,
    )
    .await;
    assert!(started.elapsed() >= Duration::from_millis(200));
    assert_eq!(reply.status, 503);
    let error = &reply.json()["error"];
    assert_eq!(
        (&error["type"], &error["code"]),
        (&"server_error".into(), &"queue_timeout".into())
    );
    assert!(reply.header("x-request-id").is_some() && reply.header("x-lh-session-id").is_some());
    assert_eq!(
        mock.tags(),
        ["bg1"],
        "a refused request never reaches the engine"
    );
    gw.await_metric(WAITING, 0.0).await;
    gw.await_metric(
        "lh_gateway_admission_refused_total{class=\"background\",reason=\"timeout\"}",
        1.0,
    )
    .await;
    mock.release(1);
    assert_eq!(first.await.unwrap().status, 200);
}

#[tokio::test]
async fn a_client_that_leaves_while_waiting_frees_its_place() {
    let mock = Mock::start(Script::Json("{}")).await;
    let top = "[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1\nqueue_limit = 1";
    let gw = Gateway::start(mock.addr, top).await;
    let first = spawn_call(&gw.tcp, background("bg1"), CHAT);
    mock.await_requests(1).await;
    let (mut sender, connection) = connect(&gw.tcp).await;
    let leaving = tokio::spawn(sender.send_request(request(
        "POST",
        "/v1/chat/completions",
        &background("bg2"),
        CHAT,
    )));
    gw.await_metric(WAITING, 1.0).await;
    connection.abort();
    leaving.abort();
    gw.await_metric(WAITING, 0.0).await;

    // The queue has room again, and the request that left is never sent.
    let third = spawn_call(&gw.tcp, background("bg3"), CHAT);
    gw.await_metric(WAITING, 1.0).await;
    mock.release(2);
    assert_eq!(first.await.unwrap().status, 200);
    assert_eq!(third.await.unwrap().status, 200);
    assert_eq!(mock.tags(), ["bg1", "bg3"]);
}

#[tokio::test]
async fn token_reservation_admits_background_while_its_estimate_fits() {
    // Every answer reports 50 prompt tokens.
    let usage = r#"{"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":1}}"#;
    let mock = Mock::start(Script::Json(usage)).await;
    let top =
        "[admission]\npolicy = \"token-reservation\"\npool_tokens = 1000\nreserved_tokens = 200";
    let gw = Gateway::start(mock.addr, top).await;
    // About 2,400 bytes, so about 600 tokens at 4 bytes a token: one fits beside the reservation, two do not.
    let body: &'static str =
        format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(2_350)).leak();
    let first = spawn_call(&gw.tcp, background("bg1"), body);
    mock.await_requests(1).await;
    let second = spawn_call(&gw.tcp, background("bg2"), body);
    gw.await_metric(WAITING, 1.0).await;
    mock.release(2);
    for reply in [first, second] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    gw.await_metric(WAITING, 0.0).await;

    // A session's last usage prices its next request: the same body, at 50 tokens, now fits beside another.
    let session = |tag| {
        vec![
            AGENTS_AUTH,
            ("x-session-id", "agent-7"),
            (TAG, tag),
            (GATE, "1"),
        ]
    };
    let learn = spawn_call(&gw.tcp, session("s1"), body);
    mock.await_requests(3).await;
    mock.release(1);
    assert_eq!(learn.await.unwrap().status, 200);
    let other = spawn_call(&gw.tcp, background("bg3"), body);
    mock.await_requests(4).await;
    let priced = spawn_call(&gw.tcp, session("s2"), body);
    mock.await_requests(5).await;
    mock.release(2);
    for reply in [other, priced] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    assert_eq!(mock.tags(), ["bg1", "bg2", "s1", "bg3", "s2"]);
}
