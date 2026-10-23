//! Policy leaseholder: lifecycle events through `/v1/events`, then requests. The pool is 1,000 tokens and a turn's
//! headroom 200, so one armed session leaves background requests 800; a background request of about 600 tokens
//! fits alone, and two do not.

use std::time::{Duration, Instant};

use crate::common::*;

const WAITING: &str = "lh_gateway_admission_waiting{class=\"background\"}";
const SESSION: &str = "voice-RM_a1B2c3";
/// The headroom alone.
const HEADROOM: &str = "headroom_tokens = 200\nthrottle = false\ndegrade = false";

/// About 2,400 bytes once the gateway adds its cache salt: 600 tokens at 4 bytes a token, and a cold prefill of
/// 55 ms on the frozen box.
fn six_hundred() -> &'static str {
    format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(2_350)).leak()
}

/// About 900 tokens: more than an armed session leaves background requests, even alone.
fn nine_hundred() -> &'static str {
    format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(3_550)).leak()
}

/// Policy leaseholder on a 1,000-token pool, with its mechanisms' keys.
fn leaseholder(keys: &str) -> String {
    format!("[admission]\npolicy = \"leaseholder\"\npool_tokens = 1000\n{keys}")
}

/// A background request of the agents tenant, named `tag`, whose answer the mock engine holds.
fn background(tag: &'static str) -> Vec<(&'static str, &'static str)> {
    vec![AGENTS_AUTH, (TAG, tag), (GATE, "1")]
}

/// A realtime request of the voice session on the voice socket, named `tag`; held by the engine if `gated`.
fn realtime(tag: &'static str, gated: bool) -> Vec<(&'static str, &'static str)> {
    let mut headers = vec![(TAG, tag), ("x-session-id", SESSION)];
    if gated {
        headers.push((GATE, "1"));
    }
    headers
}

/// Posts lifecycle events of the voice session in one batch, as `(event_id, type)`, as the voice worker does.
async fn post(gw: &Gateway, events: &[(&str, &str)]) {
    let events: Vec<String> = events
        .iter()
        .map(|(id, kind)| {
            format!(
                r#"{{"event_id":"{id}","session_id":"{SESSION}","at_ms":1792650010000,"source":"voice-worker",
                    "type":"{kind}","played_ms":0}}"#
            )
        })
        .collect();
    let body = format!("[{}]", events.join(","));
    let reply = call(&gw.unix, "POST", "/v1/events", &[], &body).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
}

fn sessions(state: &str) -> String {
    format!("lh_gateway_realtime_sessions{{state=\"{state}\"}}")
}

fn deferred(reason: &str) -> String {
    format!("lh_gateway_admission_deferred_total{{reason=\"{reason}\"}}")
}

#[tokio::test]
async fn background_waits_while_a_turn_is_imminent_and_goes_after_it() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, &leaseholder(HEADROOM)).await;
    let running = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;

    // The user starts speaking: the next background turn waits, and the running one goes on.
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    gw.await_metric(&sessions("user_speaking"), 1.0).await;
    let deferred_turn = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(&deferred("headroom"), 1.0).await;
    post(&gw, &[("vw-2", "user_turn_ended")]).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    // The realtime request goes at once, past the waiting background request.
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt", false),
        CHAT,
    )
    .await;
    assert_eq!(voice.status, 200);
    assert_eq!(mock.tags(), ["bg1", "rt"]);

    // The reply's first audio ends the turn's claim: the deferred request goes.
    post(&gw, &[("vw-3", "playback_started")]).await;
    mock.await_requests(3).await;
    assert_eq!(mock.tags(), ["bg1", "rt", "bg2"]);
    mock.release(2);
    for reply in [running, deferred_turn] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    post(&gw, &[("vw-4", "playback_ended")]).await;
    gw.await_metric(&sessions("idle"), 1.0).await;
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "lh_gateway_admission_deferral_seconds_count{reason=\"headroom\"}",
            1.0,
        ),
        ("lh_gateway_admission_false_alarms_total", 0.0),
        ("lh_gateway_admission_degraded_total", 0.0),
        ("lh_gateway_admission_aged_total{oversize=\"false\"}", 0.0),
        (
            "lh_gateway_realtime_state_timeouts_total{state=\"critical\"}",
            0.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn the_critical_window_spends_a_budget_of_predicted_prefill_time() {
    let mock = Mock::start(Script::Json("{}")).await;
    // The throttle alone: 100 ms of prefill per window, and each cold 600-token prompt takes 55 ms.
    let keys = "headroom = false\ndegrade = false\ncritical_prefill_budget_ms = 100";
    let gw = Gateway::start(mock.addr, &leaseholder(keys)).await;
    post(&gw, &[("vw-1", "user_turn_ended")]).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let first = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;
    let second = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(&deferred("critical"), 1.0).await;
    // A lost playback_started: playback_ended still ends the window.
    post(&gw, &[("vw-2", "playback_ended")]).await;
    mock.await_requests(2).await;
    mock.release(2);
    for reply in [first, second] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    assert_eq!(mock.tags(), ["bg1", "bg2"]);
}

#[tokio::test]
async fn aging_admits_a_deferred_request_after_max_defer_s() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &leaseholder(&format!("{HEADROOM}\nmax_defer_s = 0.3")),
    )
    .await;
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    gw.await_metric(&sessions("user_speaking"), 1.0).await;
    // 900 tokens cannot go beside the turn's headroom even alone: only aging lets them in.
    let started = Instant::now();
    let oversize = spawn_call(&gw.tcp, background("big"), nine_hundred());
    mock.await_requests(1).await;
    assert!(started.elapsed() >= Duration::from_millis(300));
    gw.await_metric("lh_gateway_admission_aged_total{oversize=\"true\"}", 1.0)
        .await;
    // A request that then waits for the headroom ages too, without being oversize: alone it would fit.
    let behind = spawn_call(&gw.tcp, background("small"), six_hundred());
    mock.await_requests(2).await;
    gw.await_metric("lh_gateway_admission_aged_total{oversize=\"false\"}", 1.0)
        .await;
    mock.release(2);
    for reply in [oversize, behind] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    gw.await_metric(
        "lh_gateway_admission_deferral_seconds_count{reason=\"headroom\"}",
        2.0,
    )
    .await;
    let refused = "lh_gateway_admission_refused_total{class=\"background\",reason=\"timeout\"}";
    assert_eq!(metric(&gw.metrics().await, refused), Some(0.0));
}

#[tokio::test]
async fn degrade_caps_a_realtime_reply_when_the_headroom_was_not_kept() {
    let mock = Mock::start(Script::Json("{}")).await;
    // Degrade alone: nothing keeps the background request from taking the pool.
    let keys = "headroom = false\nthrottle = false\nheadroom_tokens = 200\ndegrade_max_tokens = 16";
    let gw = Gateway::start(mock.addr, &leaseholder(keys)).await;
    let large = spawn_call(&gw.tcp, background("bg"), nine_hundred());
    mock.await_requests(1).await;
    let voice = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt1", false),
        r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":200}"#,
    )
    .await;
    assert_eq!(voice.status, 200);
    let body = mock.last().json();
    assert_eq!(
        (&body["max_tokens"], &body["max_completion_tokens"]),
        (&16.into(), &serde_json::Value::Null),
        "{body}"
    );
    assert_eq!(body["messages"][0]["content"], "hi");
    gw.await_metric("lh_gateway_admission_degraded_total", 1.0)
        .await;

    // With the pool free again, a reply is not capped. The request leaves the admission point just after its
    // client has the whole response, and before its outcome is counted.
    mock.release(1);
    assert_eq!(large.await.unwrap().status, 200);
    gw.await_metric(
        "lh_gateway_request_outcomes_total{class=\"background\",outcome=\"completed\"}",
        1.0,
    )
    .await;
    call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &realtime("rt2", false),
        CHAT,
    )
    .await;
    let body = mock.last().json();
    assert!(body.get("max_completion_tokens").is_none(), "{body}");
}

#[tokio::test]
async fn a_state_no_event_ends_times_out_and_its_arming_is_a_false_alarm() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &leaseholder(&format!("{HEADROOM}\nstate_timeout_s = 0.3")),
    )
    .await;
    let running = spawn_call(&gw.tcp, background("bg1"), six_hundred());
    mock.await_requests(1).await;
    // A cough: speech onset, and no turn after it.
    post(&gw, &[("vw-1", "user_speech_started")]).await;
    let held = spawn_call(&gw.tcp, background("bg2"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    gw.await_metric(
        "lh_gateway_realtime_state_timeouts_total{state=\"user_speaking\"}",
        1.0,
    )
    .await;
    mock.await_requests(2).await;
    gw.await_metric(&sessions("user_speaking"), 0.0).await;
    let metrics = gw.metrics().await;
    assert_eq!(
        metric(&metrics, "lh_gateway_admission_false_alarms_total"),
        Some(1.0)
    );
    // About 600 tokens held back for most of the 0.3 s.
    let cost = metric(
        &metrics,
        "lh_gateway_admission_false_alarm_token_seconds_total",
    )
    .unwrap();
    assert!((60.0..=600.0).contains(&cost), "{cost}");
    mock.release(2);
    for reply in [running, held] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
}

#[tokio::test]
async fn without_events_a_realtime_request_is_its_own_critical_window() {
    let mock = Mock::start(Script::Json("{}")).await;
    // The throttle admits no background prefill in a window.
    let keys = "headroom = false\ndegrade = false\ncritical_prefill_budget_ms = 0";
    let gw = Gateway::start(mock.addr, &leaseholder(keys)).await;
    let voice = spawn_call(&gw.unix, realtime("rt", true), CHAT);
    mock.await_requests(1).await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let deferred_turn = spawn_call(&gw.tcp, background("bg"), six_hundred());
    gw.await_metric(WAITING, 1.0).await;
    // The reply ends, and with it the window: nothing else would end it.
    mock.release(1);
    assert_eq!(voice.await.unwrap().status, 200);
    mock.await_requests(2).await;
    gw.await_metric(&sessions("critical"), 0.0).await;
    gw.await_metric(&sessions("idle"), 0.0).await;
    mock.release(1);
    assert_eq!(deferred_turn.await.unwrap().status, 200);
}

#[tokio::test]
async fn realtime_is_never_queued_or_refused() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(
        mock.addr,
        &leaseholder(&format!("{HEADROOM}\nqueue_limit = 1")),
    )
    .await;
    post(
        &gw,
        &[("vw-1", "user_speech_started"), ("vw-2", "user_turn_ended")],
    )
    .await;
    gw.await_metric(&sessions("critical"), 1.0).await;
    let oversize = spawn_call(&gw.tcp, background("big"), nine_hundred());
    gw.await_metric(WAITING, 1.0).await;
    let full = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &background("bg2"),
        CHAT,
    )
    .await;
    assert_eq!(full.status, 429);
    for tag in ["rt1", "rt2"] {
        let voice = call(
            &gw.unix,
            "POST",
            "/v1/chat/completions",
            &realtime(tag, false),
            CHAT,
        )
        .await;
        assert_eq!(voice.status, 200);
    }
    assert_eq!(mock.tags(), ["rt1", "rt2"]);
    post(&gw, &[("vw-3", "session_closed")]).await;
    mock.await_requests(3).await;
    mock.release(1);
    assert_eq!(oversize.await.unwrap().status, 200);
    let refused = "lh_gateway_admission_refused_total{class=\"realtime\",reason=\"queue_full\"}";
    assert_eq!(metric(&gw.metrics().await, refused), Some(0.0));
}
