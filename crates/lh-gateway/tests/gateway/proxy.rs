//! Requests: authentication, sanitization, session and class resolution, bodies, passthrough, other routes.

use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use crate::common::*;

#[tokio::test]
async fn engine_inputs_from_clients_are_replaced_by_the_gateways() {
    let mock = Mock::start(Script::Json(r#"{"id":"c-1","choices":[]}"#)).await;
    let gw = Gateway::start(mock.addr, "").await;
    let body = r#"{"model":"qwen","messages":[],"priority":-100,"session_id":"s","kv_transfer_params":{"a":1},
        "cache_salt":"theirs","request_id":"r-1","vllm_xargs":{"session_id":"s"},"prompt_cache_key":"bench-7"}"#;
    let headers = [
        BENCH_AUTH,
        ("x-vllm-priority", "-100"),
        ("x-session-id", "someone-else"),
        ("x-data-parallel-rank", "3"),
        ("x-request-id", "theirs"),
        ("x-lh-class", "realtime"),
        ("x-lh-synthetic", "1"),
        ("connection", "x-hop"),
        ("x-hop", "1"),
        ("x-trace", "kept"),
    ];
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &headers, body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(&reply.body[..], br#"{"id":"c-1","choices":[]}"#);

    let seen = mock.last();
    let request_id = reply.header("x-request-id").unwrap();
    assert!(request_id.len() == 32 && request_id != "theirs");
    assert_eq!(seen.header("x-request-id"), Some(request_id));
    // An untrusted tenant's X-Session-ID and X-LH-Class are ignored: the session comes from prompt_cache_key
    // and the class is the tenant's default, interactive.
    assert_eq!(seen.header("x-session-id"), Some("bench-7"));
    assert_eq!(reply.header("x-lh-session-id"), Some("bench-7"));
    assert_eq!(seen.header("x-vllm-priority"), Some("1"));
    assert_eq!(seen.header("content-type"), Some("application/json"));
    assert_eq!(seen.header("x-trace"), Some("kept"));
    for removed in [
        "authorization",
        "x-data-parallel-rank",
        "x-lh-class",
        "x-lh-synthetic",
        "x-hop",
    ] {
        assert_eq!(seen.header(removed), None, "{removed}");
    }
    assert_eq!(
        seen.json(),
        json!({"model": "qwen", "messages": [], "prompt_cache_key": "bench-7", "cache_salt": BENCH_SALT})
    );

    let metrics = gw.metrics().await;
    for field in [
        "x-vllm-priority",
        "x-session-id",
        "x-data-parallel-rank",
        "x-request-id",
        "x-lh-class",
        "x-lh-synthetic",
        "priority",
        "session_id",
        "kv_transfer_params",
        "cache_salt",
        "request_id",
        "vllm_xargs",
    ] {
        let series = format!("lh_gateway_sanitized_total{{field=\"{field}\"}}");
        assert_eq!(metric(&metrics, &series), Some(1.0), "{series}");
    }
}

#[tokio::test]
async fn trusted_tenants_choose_session_and_class_within_their_limits() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let send = async |headers: &[(&str, &str)], body: &str| {
        let reply = call(&gw.tcp, "POST", "/v1/chat/completions", headers, body).await;
        assert_eq!(reply.status, 200, "{:?}", reply.body);
        let seen = mock.last();
        assert_eq!(seen.header("x-session-id"), reply.header("x-lh-session-id"));
        (
            seen.header("x-session-id").unwrap().to_owned(),
            seen.header("x-vllm-priority").unwrap().to_owned(),
        )
    };

    // The default class, background, and a minted session.
    let (session, priority) = send(&[AGENTS_AUTH], CHAT).await;
    assert!(
        session.starts_with("m-") && session.len() == 34,
        "{session}"
    );
    assert_eq!(priority, "2");
    assert_eq!(mock.last().json()["cache_salt"], AGENTS_SALT);
    // X-LH-Class is honored up to max_class (interactive); realtime is clamped and counted.
    assert_eq!(
        send(&[AGENTS_AUTH, ("x-lh-class", "interactive")], CHAT)
            .await
            .1,
        "1"
    );
    assert_eq!(
        send(&[AGENTS_AUTH, ("x-lh-class", "realtime")], CHAT)
            .await
            .1,
        "1"
    );
    gw.await_metric("lh_gateway_class_clamped_total", 1.0).await;

    // Session sources in order: X-Session-ID, X-Claude-Code-Session-ID, prompt_cache_key.
    let pck = r#"{"model":"qwen","prompt_cache_key":"pck-1"}"#;
    let claude = (
        "x-claude-code-session-id",
        "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41",
    );
    let all = [AGENTS_AUTH, ("x-session-id", "agent-42"), claude];
    assert_eq!(send(&all, pck).await.0, "agent-42");
    assert_eq!(send(&[AGENTS_AUTH, claude], pck).await.0, claude.1);
    assert_eq!(send(&[AGENTS_AUTH], pck).await.0, "pck-1");
    // A source that is not a session id is hashed with the tenant: sha256("agents", NUL, "has space").
    let hashed = send(&[AGENTS_AUTH, ("x-session-id", "has space")], CHAT)
        .await
        .0;
    assert_eq!(hashed, "h-e19e109c9994ca8a87bf597385621785");

    // The voice tenant needs no key on its Unix socket and may use realtime: priority 0.
    let reply = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &[("x-session-id", "voice-RM_a1B2c3")],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 200);
    let seen = mock.last();
    assert_eq!(seen.header("x-vllm-priority"), Some("0"));
    assert_eq!(seen.header("x-session-id"), Some("voice-RM_a1B2c3"));
    assert_eq!(seen.json()["cache_salt"], VOICE_SALT);
}

#[tokio::test]
async fn an_unknown_class_is_rejected() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let headers = [AGENTS_AUTH, ("x-lh-class", "urgent")];
    let reply = call(&gw.tcp, "POST", "/v1/completions", &headers, CHAT).await;
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(
        (&error["type"], &error["code"]),
        (&json!("invalid_request_error"), &json!("invalid_class"))
    );
    assert!(error["message"].as_str().unwrap().contains("urgent"));
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn requests_without_a_known_key_get_401_except_on_the_tenant_socket() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    for headers in [
        &[][..],
        &[("authorization", "Bearer agents-key2")],
        &[("authorization", "Basic YWdlbnRzLWtleQ==")],
    ] {
        let reply = call(&gw.tcp, "POST", "/v1/responses", headers, CHAT).await;
        assert_eq!(reply.status, 401, "{headers:?}");
        assert_eq!(
            reply.json(),
            json!({"error": {"message": "missing or unknown API key", "type": "authentication_error",
                             "code": "invalid_api_key"}})
        );
        assert!(reply.header("x-request-id").is_some());
    }
    assert!(mock.requests().is_empty());
    // The voice worker's OpenAI client sends a placeholder key; on the voice socket that is still voice.
    let reply = call(
        &gw.unix,
        "POST",
        "/v1/responses",
        &[("authorization", "Bearer local")],
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 200);
    assert_eq!(mock.last().path, "/v1/responses");
    assert_eq!(mock.last().header("x-vllm-priority"), Some("0"));
}

#[tokio::test]
async fn oversized_and_malformed_bodies_are_rejected() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "max_body_bytes = 1024").await;
    let large = format!(r#"{{"model":"{}"}}"#, "x".repeat(2000));
    for headers in [
        &[AGENTS_AUTH][..],
        &[AGENTS_AUTH, ("transfer-encoding", "chunked")],
    ] {
        let reply = call(&gw.tcp, "POST", "/v1/chat/completions", headers, &large).await;
        assert_eq!(reply.status, 413, "{headers:?}");
        assert_eq!(reply.json()["error"]["code"], "body_too_large");
    }
    for body in ["{not json", "[1, 2]", ""] {
        let reply = call(
            &gw.tcp,
            "POST",
            "/v1/chat/completions",
            &[AGENTS_AUTH],
            body,
        )
        .await;
        assert_eq!(reply.status, 400, "{body}");
        assert_eq!(reply.json()["error"]["code"], "invalid_json");
    }
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn event_streams_pass_through_byte_for_byte() {
    // A comment, CRLF line ends, a UTF-8 character split across chunks, and two events in one chunk.
    let chunks: Vec<Bytes> = [
        &b": keep-alive\r\n\r\n"[..],
        b"data: {\"choices\":[{\"delta\":{\"content\":\"h\xc3",
        b"\xa9llo\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\
          \"prompt_tokens_details\":{\"cached_tokens\":8}}}\n\n",
        b"data: [DONE]\n\n",
    ]
    .into_iter()
    .map(Bytes::from_static)
    .collect();
    let expected: Vec<u8> = chunks.concat();
    let mock = Mock::start(Script::Sse {
        chunks,
        gap: Duration::from_millis(5),
    })
    .await;
    let gw = Gateway::start(mock.addr, "").await;
    let headers = [AGENTS_AUTH, ("x-lh-class", "interactive")];
    let body = r#"{"model":"qwen","stream":true,"stream_options":{"include_usage":true}}"#;
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &headers, body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("content-type"), Some("text/event-stream"));
    assert_eq!(&reply.body[..], &expected[..]);

    let class = "class=\"interactive\"";
    gw.await_metric(
        &format!("lh_gateway_request_outcomes_total{{{class},outcome=\"completed\"}}"),
        1.0,
    )
    .await;
    let metrics = gw.metrics().await;
    for (kind, tokens) in [("prompt", 12.0), ("completion", 3.0), ("cached", 8.0)] {
        let series = format!("lh_gateway_usage_tokens_total{{{class},kind=\"{kind}\"}}");
        assert_eq!(metric(&metrics, &series), Some(tokens), "{series}");
    }
    let first_byte = format!("lh_gateway_first_byte_seconds_count{{{class}}}");
    assert_eq!(metric(&metrics, &first_byte), Some(1.0));
}

#[tokio::test]
async fn models_pass_through_and_other_routes_are_openai_errors() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let reply = call(&gw.tcp, "GET", "/v1/models", &[BENCH_AUTH], "").await;
    assert_eq!(reply.status, 200);
    assert_eq!(reply.json()["data"][0]["id"], "qwen");
    let seen = mock.last();
    assert_eq!(seen.path, "/v1/models");
    assert_eq!(seen.header("x-request-id"), reply.header("x-request-id"));
    assert_eq!(seen.header("authorization"), None);
    assert_eq!(
        call(&gw.tcp, "GET", "/v1/models", &[], "").await.status,
        401
    );

    let reply = call(&gw.tcp, "POST", "/v1/embeddings", &[BENCH_AUTH], CHAT).await;
    assert_eq!(reply.status, 404);
    assert_eq!(reply.json()["error"]["code"], "not_found");
    // The error names what was asked for: its route label is only "other".
    assert_eq!(
        reply.json()["error"]["message"],
        "no such route: POST /v1/embeddings"
    );
    let reply = call(&gw.tcp, "GET", "/v1/chat/completions", &[BENCH_AUTH], "").await;
    assert_eq!(reply.status, 405);
    assert_eq!(reply.json()["error"]["code"], "method_not_allowed");
    assert_eq!(
        reply.json()["error"]["message"],
        "the route does not accept this method: GET /v1/chat/completions"
    );
}

#[tokio::test]
async fn metrics_expose_every_series() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &[AGENTS_AUTH],
        CHAT,
    )
    .await;
    call(&gw.tcp, "POST", "/v1/chat/completions", &[], CHAT).await;
    let event =
        r#"{"event_id":"e-1","session_id":"s","at_ms":1,"source":"client","type":"turn_started"}"#;
    call(&gw.tcp, "POST", "/v1/events", &[AGENTS_AUTH], event).await;

    gw.await_metric(
        "lh_gateway_overhead_seconds_count{route=\"/v1/chat/completions\"}",
        1.0,
    )
    .await;
    let reply = call(&gw.tcp, "GET", "/metrics", &[], "").await;
    assert!(
        reply
            .header("content-type")
            .unwrap()
            .starts_with("application/openmetrics-text")
    );
    let text = String::from_utf8(reply.body.to_vec()).unwrap();
    for series in [
        "lh_gateway_requests_total{tenant=\"agents\",route=\"/v1/chat/completions\",class=\"background\",status=\"200\"} 1",
        "lh_gateway_requests_total{tenant=\"\",route=\"/v1/chat/completions\",class=\"\",status=\"401\"} 1",
        "lh_gateway_request_outcomes_total{class=\"background\",outcome=\"completed\"} 1",
        "lh_gateway_overhead_seconds_bucket{le=\"0.002\",route=\"/v1/chat/completions\"}",
        "lh_gateway_first_byte_seconds_count{class=\"background\"} 1",
        "lh_gateway_duration_seconds_count{class=\"background\"} 1",
        "lh_gateway_streams_active{class=\"background\"} 0",
        "lh_gateway_sanitized_total{field=\"priority\"} 0",
        "lh_gateway_class_clamped_total 0",
        "lh_gateway_upstream_errors_total{kind=\"timeout\"} 0",
        "lh_gateway_events_total{type=\"turn_started\",source=\"client\",duplicate=\"false\"} 1",
        "lh_gateway_usage_tokens_total{class=\"background\",kind=\"prompt\"} 0",
    ] {
        assert!(
            text.lines().any(|line| line.starts_with(series)),
            "missing {series}\n{text}"
        );
    }
}

#[tokio::test]
async fn health_answers_and_ready_follows_the_upstream() {
    let mock = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start(mock.addr, "").await;
    let health = call(&gw.tcp, "GET", "/health", &[], "").await;
    assert_eq!(
        (health.status, &health.body[..]),
        (http::StatusCode::OK, &b"ok\n"[..])
    );
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);

    let sick = Mock::start(Script::Json("{}")).await;
    sick.set_healthy(false);
    let gw = Gateway::start(sick.addr, "").await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
    assert_eq!(call(&gw.tcp, "GET", "/health", &[], "").await.status, 200);
}
