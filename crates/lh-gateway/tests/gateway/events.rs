//! `POST /v1/events`: validation, deduplication, trust, and the sink.

use std::sync::Arc;

use lh_proto::EventSource;
use serde_json::json;

use crate::common::*;

const SPEECH: &str = r#"{"event_id":"vw-1","session_id":"voice-RM_a1B2c3","at_ms":1792650010000,
    "source":"voice-worker","type":"user_speech_started","confidence":0.93}"#;

#[tokio::test]
async fn events_are_validated_deduplicated_and_taken_only_from_trusted_tenants() {
    let mock = Mock::start(Script::Json("{}")).await;
    let sink = Arc::new(RecordingSink::default());
    let gw = Gateway::start_with(mock.addr, "", sink.clone()).await;
    let post = async |target: &Target, headers: &[(&str, &str)], body: &str| {
        call(target, "POST", "/v1/events", headers, body).await
    };

    // The voice worker posts over its socket, with no key; unknown fields are ignored.
    let reply = post(&gw.unix, &[], SPEECH).await;
    assert_eq!(reply.status, 202);
    assert_eq!(reply.json(), json!({"accepted": 1, "duplicates": 0}));
    assert_eq!(
        post(&gw.unix, &[], SPEECH).await.json(),
        json!({"accepted": 0, "duplicates": 1})
    );

    // A batch: a repeat, and two new events. The key is (source, event_id), so the same id from another
    // source is a different event.
    let batch = format!(
        r#"[{SPEECH},
            {{"event_id":"aa-1","session_id":"s-1","at_ms":2,"source":"agentic-api","type":"tool_wait_started",
              "tool":"web_search","eta_ms":2400}},
            {{"event_id":"vw-1","session_id":"s-1","at_ms":3,"source":"client","type":"session_closed"}}]"#
    );
    let reply = post(&gw.tcp, &[AGENTS_AUTH], &batch).await;
    assert_eq!(reply.json(), json!({"accepted": 2, "duplicates": 1}));

    // A batch with an unknown type is refused whole, naming the problem.
    let bad = r#"[{"event_id":"x-1","session_id":"s","at_ms":1,"source":"client","type":"turn_started"},
                  {"event_id":"x-2","session_id":"s","at_ms":1,"source":"client","type":"user_laughed"}]"#;
    let reply = post(&gw.tcp, &[AGENTS_AUTH], bad).await;
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["code"], "invalid_event");
    let message = error["message"].as_str().unwrap();
    assert!(
        message.contains("event 1") && message.contains("user_laughed"),
        "{message}"
    );
    let x1 =
        r#"{"event_id":"x-1","session_id":"s","at_ms":1,"source":"client","type":"turn_started"}"#;
    assert_eq!(
        post(&gw.tcp, &[AGENTS_AUTH], x1).await.json(),
        json!({"accepted": 1, "duplicates": 0}),
        "nothing from the refused batch was recorded"
    );

    // Untrusted and unauthenticated callers.
    let reply = post(&gw.tcp, &[BENCH_AUTH], SPEECH).await;
    assert_eq!(reply.status, 403);
    assert_eq!(reply.json()["error"]["type"], "permission_error");
    assert_eq!(post(&gw.tcp, &[], SPEECH).await.status, 401);

    let accepted: Vec<_> = sink
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|a| {
            (
                a.tenant.to_string(),
                a.event.source,
                a.event.event_id.clone(),
            )
        })
        .collect();
    assert_eq!(
        accepted,
        [
            ("voice".into(), EventSource::VoiceWorker, "vw-1".into()),
            ("agents".into(), EventSource::AgenticApi, "aa-1".into()),
            ("agents".into(), EventSource::Client, "vw-1".into()),
            ("agents".into(), EventSource::Client, "x-1".into()),
        ]
    );
    let metrics = gw.metrics().await;
    let series = |duplicate: bool| {
        format!(
            "lh_gateway_events_total{{type=\"user_speech_started\",source=\"voice-worker\",duplicate=\"{duplicate}\"}}"
        )
    };
    assert_eq!(metric(&metrics, &series(false)), Some(1.0));
    assert_eq!(metric(&metrics, &series(true)), Some(2.0));
}
