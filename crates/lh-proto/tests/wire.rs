//! The wire format, pinned by examples. `testdata/events.jsonl` holds one example of every event type; the
//! voice worker's tests hold the events it emits to the JSON Schema generated from these types.

use lh_proto::{
    EventKind, EventSource, LifecycleEvent, SessionId, SessionIdError, SloClass, headers,
};
use serde_json::{Value, json};

fn sid(id: &str) -> SessionId {
    SessionId::new(id).unwrap()
}

#[test]
fn every_example_event_round_trips() {
    let examples = include_str!("../testdata/events.jsonl");
    let mut types = Vec::new();
    for line in examples.lines() {
        let event: LifecycleEvent = serde_json::from_str(line).unwrap();
        let wire: Value = serde_json::from_str(line).unwrap();
        assert_eq!(serde_json::to_value(&event).unwrap(), wire, "{line}");
        assert_eq!(event.kind.label(), wire["type"], "{line}");
        assert_eq!(event.source.as_str(), wire["source"], "{line}");
        types.push(wire["type"].as_str().unwrap().to_owned());
    }
    types.sort();
    types.dedup();
    assert_eq!(
        types.len(),
        10,
        "the examples cover every event type: {types:?}"
    );
}

#[test]
fn an_event_is_one_flat_object() {
    let event = LifecycleEvent {
        event_id: "vw-4".into(),
        session_id: sid("voice-RM_a1B2c3"),
        at_ms: 1_792_650_014_940,
        source: EventSource::VoiceWorker,
        kind: EventKind::Interrupted { played_ms: 1840 },
    };
    assert_eq!(
        serde_json::to_value(&event).unwrap(),
        json!({"event_id": "vw-4", "session_id": "voice-RM_a1B2c3", "at_ms": 1_792_650_014_940u64,
               "source": "voice-worker", "type": "interrupted", "played_ms": 1840})
    );
}

#[test]
fn unknown_fields_are_ignored_and_unknown_types_rejected() {
    let newer = json!({"event_id": "e", "session_id": "s", "at_ms": 1, "source": "client",
                       "type": "user_turn_ended", "confidence": 0.93});
    let event: LifecycleEvent = serde_json::from_value(newer).unwrap();
    assert_eq!(event.kind, EventKind::UserTurnEnded);

    let unknown = json!({"event_id": "e", "session_id": "s", "at_ms": 1, "source": "client",
                         "type": "user_laughed"});
    assert!(serde_json::from_value::<LifecycleEvent>(unknown).is_err());

    // The control plane infers no events, so no client may claim that it did.
    let inferred = json!({"event_id": "e", "session_id": "s", "at_ms": 1, "source": "inferred",
                          "type": "turn_started"});
    assert!(serde_json::from_value::<LifecycleEvent>(inferred).is_err());
}

#[test]
fn session_ids_are_header_safe() {
    assert!(SessionId::new("0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41").is_ok());
    assert!(SessionId::new("voice-RM_a1B2c3").is_ok());
    assert_eq!(SessionId::new(""), Err(SessionIdError::Length(0)));
    assert_eq!(
        SessionId::new("a".repeat(129)),
        Err(SessionIdError::Length(129))
    );
    assert_eq!(
        SessionId::new("a\r\nX-Vllm-Priority: -1"),
        Err(SessionIdError::Char('\r'))
    );
    assert!(serde_json::from_value::<SessionId>(json!("has space")).is_err());
}

#[test]
fn classes_map_to_engine_priorities_in_order() {
    let priorities: Vec<i32> = SloClass::ALL.iter().map(|c| c.engine_priority()).collect();
    assert_eq!(priorities, [0, 1, 2]);
    assert_eq!("realtime".parse::<SloClass>().unwrap(), SloClass::Realtime);
    assert!("batch".parse::<SloClass>().is_err());
    assert_eq!(SloClass::default(), SloClass::Interactive);
    assert_eq!(
        serde_json::to_value(SloClass::Background).unwrap(),
        json!("background")
    );
}

#[test]
fn client_controlled_engine_inputs_are_listed() {
    for header in ["x-vllm-priority", "x-smg-priority", headers::SESSION_ID] {
        assert!(
            headers::ENGINE_CONTROL_HEADERS.contains(&header),
            "{header}"
        );
    }
    // vLLM 0.32's, then SGLang 0.5.22's request id, cache namespace, session and data-parallel routing.
    for field in [
        "session_id",
        "priority",
        "kv_transfer_params",
        "ec_transfer_params",
        "cache_salt",
        "rid",
        "extra_key",
        "session_params",
        "routed_dp_rank",
    ] {
        assert!(
            headers::ENGINE_CONTROL_BODY_FIELDS.contains(&field),
            "{field}"
        );
    }
}
