//! The wire format, pinned by examples. `testdata/events.jsonl` holds one example of every event type;
//! the Python components are to test against it once they emit events (W3).

use lh_proto::{
    EngineLeaseHint, EventKind, EventSource, HintOp, LedgerError, LedgerRow, LifecycleEvent,
    Outcome, RecomputeCause, RecomputeSpan, SessionId, SessionIdError, SloClass, StopReason,
    headers,
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
fn stop_reasons_normalize_both_apis() {
    for (api, expected) in [
        ("stop", StopReason::Stop),
        ("end_turn", StopReason::Stop),
        ("length", StopReason::Length),
        ("max_tokens", StopReason::Length),
        ("tool_calls", StopReason::ToolCalls),
        ("tool_use", StopReason::ToolCalls),
        ("pause_turn", StopReason::Stop),
    ] {
        assert_eq!(StopReason::from_api(api), expected, "{api}");
    }
}

#[test]
fn the_engine_hint_matches_what_the_cache_policy_reads() {
    let hint = EngineLeaseHint {
        session_id: sid("s-1"),
        class: SloClass::Realtime,
        ttl_s: 300.0,
        expect_return_s: Some(2.5),
        op: None,
    };
    assert_eq!(
        hint.to_kv_transfer_params(),
        json!({"leaseholder": {"session_id": "s-1", "class": "realtime", "ttl_s": 300.0,
                               "expect_return_s": 2.5}})
    );
    let release: EngineLeaseHint = serde_json::from_value(
        json!({"session_id": "s-1", "class": "background", "op": "release"}),
    )
    .unwrap();
    assert_eq!((release.ttl_s, release.op), (0.0, Some(HintOp::Release)));
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

fn row() -> LedgerRow {
    LedgerRow {
        request_id: "req-1".into(),
        session_id: sid("s-1"),
        class: SloClass::Interactive,
        node: "phastform".into(),
        engine_epoch: "engine-3".into(),
        config: "phastform-frozen-v0".into(),
        synthetic: false,
        outcome: Outcome::Completed,
        prompt_tokens: 18_000,
        completion_tokens: 120,
        expected_reusable_tokens: 17_500,
        cached_device_tokens: 9_000,
        cached_host_tokens: 4_000,
        recomputed_tokens: 4_500,
        recompute: vec![
            RecomputeSpan {
                tokens: 4_000,
                cause: RecomputeCause::EvictedGpuAndCpu,
            },
            RecomputeSpan {
                tokens: 500,
                cause: RecomputeCause::PrefixBreak {
                    position: 13_000,
                    message: Some(14),
                    field: Some("tool_calls[0].arguments".into()),
                },
            },
        ],
        received_at_ms: 1_792_650_000_000,
        queue_ms: Some(3.1),
        prefill_ms: Some(212.0),
        restore_ms: Some(96.0),
        first_token_ms: Some(318.5),
        first_sentence_ms: None,
        total_ms: Some(2_410.0),
    }
}

#[test]
fn ledger_rows_keep_their_accounting_identities() {
    let row = row();
    assert_eq!(row.check(), Ok(()));
    assert_eq!(row.uncached_tokens(), 5_000);
    let wire = serde_json::to_value(&row).unwrap();
    assert_eq!(
        wire["recompute"][1],
        json!({"tokens": 500, "cause": "prefix_break", "position": 13_000,
                                             "message": 14, "field": "tool_calls[0].arguments"})
    );
    assert_eq!(serde_json::from_value::<LedgerRow>(wire).unwrap(), row);
}

#[test]
fn ledger_rows_with_unattributed_tokens_are_rejected() {
    let mut bad = row();
    bad.recompute.pop();
    assert_eq!(
        bad.check(),
        Err(LedgerError::UnattributedTokens {
            recomputed: 4_500,
            attributed: 4_000
        })
    );
    let mut bad = row();
    bad.cached_host_tokens = 10_000;
    assert_eq!(bad.check(), Err(LedgerError::CachedExceedsPrompt));
}

#[test]
fn out_of_range_counts_are_rejected_not_wrapped() {
    // The two cached counts sum past u32::MAX: wrapped, they would look smaller than the prompt.
    let mut bad = row();
    bad.cached_device_tokens = u32::MAX;
    bad.cached_host_tokens = 1;
    assert_eq!(bad.check(), Err(LedgerError::CachedExceedsPrompt));
    assert_eq!(bad.uncached_tokens(), 0);
    // Spans whose tokens sum past u32::MAX never match the recomputed count.
    let mut bad = row();
    bad.recompute[0].tokens = u32::MAX;
    assert_eq!(
        bad.check(),
        Err(LedgerError::UnattributedTokens {
            recomputed: 4_500,
            attributed: u64::from(u32::MAX) + 500,
        })
    );
}

#[test]
fn cause_labels_match_the_wire_names() {
    for cause in [
        RecomputeCause::ColdFirstTurn,
        RecomputeCause::PrefixBreak {
            position: 0,
            message: None,
            field: None,
        },
        RecomputeCause::EvictedGpuAndCpu,
        RecomputeCause::LeaseExpired,
        RecomputeCause::HolderMiss,
        RecomputeCause::Preempted,
        RecomputeCause::EpochChange,
        RecomputeCause::Unknown,
    ] {
        let wire = serde_json::to_value(RecomputeSpan {
            tokens: 1,
            cause: cause.clone(),
        })
        .unwrap();
        assert_eq!(wire["cause"], cause.label());
    }
}
