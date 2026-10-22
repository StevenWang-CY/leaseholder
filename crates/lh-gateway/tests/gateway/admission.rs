//! The admission point: caps, queues, timeouts, token reservations and flow control.

use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;

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

/// An interactive request of the agents tenant, named `tag`, whose answer the mock engine holds.
fn interactive(tag: &'static str) -> Vec<(&'static str, &'static str)> {
    vec![
        AGENTS_AUTH,
        ("x-lh-class", "interactive"),
        (TAG, tag),
        (GATE, "1"),
    ]
}

/// A realtime request on the voice socket, named `tag`; its answer is held only if `gated`.
fn realtime(tag: &'static str, gated: bool) -> Vec<(&'static str, &'static str)> {
    let mut headers = vec![(TAG, tag)];
    if gated {
        headers.push((GATE, "1"));
    }
    headers
}

fn flow_control(realtime: u32, interactive: u32, background: u32) -> String {
    format!(
        "[admission]\npolicy = \"flow-control\"\n\
         ceilings = {{ realtime = {realtime}, interactive = {interactive}, background = {background} }}"
    )
}

#[tokio::test]
async fn flow_control_dispatches_bands_strictly_and_each_in_arrival_order() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, &flow_control(1, 1, 1)).await;
    let mut replies = vec![spawn_call(&gw.tcp, interactive("int0"), CHAT)];
    mock.await_requests(1).await;
    // Arrivals in this order, all held behind int0: nothing in flight can be shed for the realtime request.
    let waiting = |class| format!("lh_gateway_admission_waiting{{class=\"{class}\"}}");
    let queued = [
        (&gw.tcp, background("bg1"), "background", 1.0),
        (&gw.tcp, interactive("int1"), "interactive", 1.0),
        (&gw.unix, realtime("rt1", true), "realtime", 1.0),
        (&gw.tcp, background("bg2"), "background", 2.0),
        (&gw.tcp, interactive("int2"), "interactive", 2.0),
    ];
    for (target, headers, class, count) in queued {
        replies.push(spawn_call(target, headers, CHAT));
        gw.await_metric(&waiting(class), count).await;
    }
    for n in 2..=6 {
        mock.release(1);
        mock.await_requests(n).await;
    }
    assert_eq!(mock.tags(), ["int0", "rt1", "int1", "int2", "bg1", "bg2"]);
    mock.release(1);
    for reply in replies {
        assert_eq!(reply.await.unwrap().status, 200);
    }
}

#[tokio::test]
async fn flow_control_sheds_the_newest_background_request_for_a_blocked_realtime_one() {
    let mut mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, &flow_control(2, 2, 2)).await;
    let older = spawn_call(&gw.tcp, background("bg1"), CHAT);
    mock.await_requests(1).await;
    let newer = spawn_call(&gw.tcp, background("bg2"), CHAT);
    mock.await_requests(2).await;

    // Two in flight: the realtime request takes the newest background request's place at once.
    let voice = spawn_call(&gw.unix, realtime("rt", false), CHAT);
    let shed = newer.await.unwrap();
    assert_eq!(shed.status, 429);
    assert_eq!(shed.json()["error"]["code"], "shed");
    mock.dropped("bg2", Duration::from_secs(2)).await;
    assert_eq!(voice.await.unwrap().status, 200);
    assert_eq!(mock.tags(), ["bg1", "bg2", "rt"]);

    mock.release(1);
    assert_eq!(older.await.unwrap().status, 200);
    let outcome = |outcome| {
        format!("lh_gateway_request_outcomes_total{{class=\"background\",outcome=\"{outcome}\"}}")
    };
    gw.await_metric(&outcome("shed"), 1.0).await;
    gw.await_metric(&outcome("completed"), 1.0).await;
}

#[tokio::test]
async fn a_shed_stream_is_cut_off_with_an_error() {
    let chunks: Vec<Bytes> = (0..5)
        .map(|i| Bytes::from(format!("data: {i}\n\n")))
        .collect();
    let mut mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(100),
    })
    .await;
    let gw = Gateway::start(mock.addr, &flow_control(1, 1, 1)).await;
    let headers = [AGENTS_AUTH, (TAG, "bg")];
    let (response, _connection) =
        open(&gw.tcp, "POST", "/v1/chat/completions", &headers, CHAT).await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    body.frame().await.unwrap().unwrap();

    let voice = spawn_call(&gw.unix, realtime("rt", false), CHAT);
    let end = loop {
        match body.frame().await {
            Some(Ok(_)) => continue,
            Some(Err(_)) => break "error",
            None => break "clean end",
        }
    };
    assert_eq!(end, "error", "a shed stream must not look complete");
    mock.dropped("bg", Duration::from_secs(2)).await;
    assert_eq!(voice.await.unwrap().status, 200);
    gw.await_metric(
        "lh_gateway_request_outcomes_total{class=\"background\",outcome=\"shed\"}",
        1.0,
    )
    .await;
}

const PREEMPT: &str = "[admission]\npolicy = \"preempt-resubmit\"";

fn event_stream_chunks() -> Vec<Bytes> {
    [
        "data: {\"n\":1}\n\n",
        "data: {\"n\":2}\n\n",
        "data: [DONE]\n\n",
    ]
    .into_iter()
    .map(Bytes::from)
    .collect()
}

#[tokio::test]
async fn preempt_resubmit_sends_background_work_again_after_the_realtime_first_byte() {
    let chunks = event_stream_chunks();
    let expected = chunks.concat();
    let mut mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start(mock.addr, PREEMPT).await;
    // In prefill at the engine: its response has headers and no first byte yet.
    let first = spawn_call(&gw.tcp, background("bg1"), CHAT);
    mock.await_requests(1).await;

    // A realtime arrival preempts it: the engine's request is dropped, and background waits for the realtime
    // request's first byte, which the engine holds too.
    let voice = spawn_call(&gw.unix, realtime("rt", true), CHAT);
    mock.dropped("bg1", Duration::from_secs(2)).await;
    let second = spawn_call(&gw.tcp, background("bg2"), CHAT);
    gw.await_metric(WAITING, 2.0).await;
    assert_eq!(mock.tags(), ["bg1", "rt"]);

    mock.release(1);
    mock.await_requests(4).await;
    let mut tags = mock.tags();
    tags[2..].sort();
    assert_eq!(tags, ["bg1", "rt", "bg1", "bg2"]);
    mock.release(2);
    for reply in [first, second, voice] {
        let reply = reply.await.unwrap();
        assert_eq!(reply.status, 200);
        assert_eq!(reply.header("content-type"), Some("text/event-stream"));
        assert_eq!(&reply.body[..], &expected[..], "one whole response");
    }
    let metrics = gw.metrics().await;
    let series = |name| format!("lh_gateway_admission_{name}_total{{class=\"background\"}}");
    assert_eq!(metric(&metrics, &series("resubmits")), Some(1.0));
    assert!(metric(&metrics, &series("discarded_tokens")).unwrap() > 0.0);
    assert_eq!(metric(&metrics, &series("admitted")), Some(2.0));
}

#[tokio::test]
async fn a_request_waiting_for_its_response_headers_is_preempted_and_answered_once() {
    let mut mock = Mock::start(Script::Json(r#"{"choices":[]}"#)).await;
    let gw = Gateway::start(mock.addr, PREEMPT).await;
    let background = spawn_call(&gw.tcp, background("bg"), CHAT);
    mock.await_requests(1).await;
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt", false),
        CHAT,
    )
    .await;
    assert_eq!(voice.status, 200);
    mock.dropped("bg", Duration::from_secs(2)).await;
    mock.await_requests(3).await;
    mock.release(1);
    let reply = background.await.unwrap();
    assert_eq!(
        (reply.status, &reply.body[..]),
        (http::StatusCode::OK, &br#"{"choices":[]}"#[..])
    );
    assert_eq!(mock.tags(), ["bg", "rt", "bg"]);
}

#[tokio::test]
async fn a_background_request_already_streaming_is_never_preempted() {
    let chunks = event_stream_chunks();
    let expected = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(100),
    })
    .await;
    let gw = Gateway::start(mock.addr, PREEMPT).await;
    let headers = [AGENTS_AUTH, (TAG, "bg")];
    let (response, _connection) =
        open(&gw.tcp, "POST", "/v1/chat/completions", &headers, CHAT).await;
    let mut body = response.into_body();
    let mut received = body
        .frame()
        .await
        .unwrap()
        .unwrap()
        .into_data()
        .unwrap()
        .to_vec();

    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt", false),
        CHAT,
    )
    .await;
    assert_eq!(voice.status, 200);
    while let Some(frame) = body.frame().await {
        received.extend_from_slice(&frame.unwrap().into_data().unwrap());
    }
    assert_eq!(received, expected);
    assert_eq!(mock.tags(), ["bg", "rt"]);
    let metrics = gw.metrics().await;
    let resubmits = "lh_gateway_admission_resubmits_total{class=\"background\"}";
    assert_eq!(metric(&metrics, resubmits), Some(0.0));
}
