//! Two GPU nodes behind one gateway, each a mock engine of its own: placement by each policy, sticky sessions,
//! readiness and re-placement, the placement endpoint, and each node's own admission point.

use serde_json::json;
use tokio::net::TcpListener;

use crate::common::*;

/// The gateway on two nodes, `a` and `b`, with `tables`: `[placement]` and any other table.
async fn two_nodes(tables: &str) -> (Mock, Mock, Gateway) {
    let a = Mock::start(Script::Json("{}")).await;
    let b = Mock::start(Script::Json("{}")).await;
    let nodes = format!(
        "[nodes.a]\nupstream = \"http://{}\"\n[nodes.b]\nupstream = \"http://{}\"\n{tables}",
        a.addr, b.addr
    );
    let gw = Gateway::start_nodes(&nodes).await;
    (a, b, gw)
}

/// A background request of the agents tenant in `session`, named `tag`; the engine holds its answer if `gated`.
fn agent(
    session: &'static str,
    tag: &'static str,
    gated: bool,
) -> Vec<(&'static str, &'static str)> {
    let mut headers = vec![AGENTS_AUTH, ("x-session-id", session), (TAG, tag)];
    if gated {
        headers.push((GATE, "1"));
    }
    headers
}

/// Sends a background request of `session`, named `tag`, and returns its status.
async fn send(gw: &Gateway, session: &'static str, tag: &'static str) -> u16 {
    let headers = agent(session, tag, false);
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &headers, CHAT).await;
    reply.status.as_u16()
}

/// `POST /v1/sessions` of a realtime session on the voice socket.
async fn place(gw: &Gateway, session: &str) -> Reply {
    let body = format!(r#"{{"session_id":"{session}","class":"realtime"}}"#);
    call(&gw.unix, "POST", "/v1/sessions", &[], &body).await
}

/// Posts one lifecycle event of `session` on the voice socket.
async fn post_event(gw: &Gateway, session: &str, id: &str, kind: &str) {
    let event = format!(
        r#"{{"event_id":"{id}","session_id":"{session}","at_ms":1792650010000,"source":"voice-worker",
            "type":"{kind}","played_ms":0}}"#
    );
    let reply = call(&gw.unix, "POST", "/v1/events", &[], &event).await;
    assert_eq!(reply.status, 202, "{:?}", reply.body);
}

/// About 600 tokens at 4 bytes a token.
fn six_hundred() -> &'static str {
    format!(r#"{{"model":"qwen","prompt":"{}"}}"#, "x".repeat(2_350)).leak()
}

/// `count` requests of the bench tenant, which is untrusted, 64 at a time: each names a session of its own by its
/// `prompt_cache_key`, hashed with the tenant. They go over the Unix socket, where the tenant's key selects it as on
/// any listener, so that thousands of connections leave no TCP ports waiting to close.
async fn flood(gw: &Gateway, count: usize) {
    for chunk in (0..count).collect::<Vec<_>>().chunks(64) {
        let calls: Vec<_> = chunk
            .iter()
            .map(|n| {
                let body =
                    format!(r#"{{"model":"qwen","messages":[],"prompt_cache_key":"flood-{n}"}}"#);
                let target = gw.unix.clone();
                tokio::spawn(async move {
                    call(
                        &target,
                        "POST",
                        "/v1/chat/completions",
                        &[BENCH_AUTH],
                        &body,
                    )
                    .await
                    .status
                })
            })
            .collect();
        for call in calls {
            assert_eq!(call.await.unwrap(), 200);
        }
    }
}

/// The sessions placed on each node, by class.
async fn sessions(gw: &Gateway, class: &str) -> [f64; 2] {
    let metrics = gw.metrics().await;
    ["a", "b"].map(|node| {
        let series = format!("lh_gateway_sessions{{node=\"{node}\",class=\"{class}\"}}");
        metric(&metrics, &series).unwrap()
    })
}

#[tokio::test]
async fn least_loaded_places_a_new_session_on_the_quieter_node_and_keeps_it_there() {
    let (a, b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"").await;
    // Nothing is in flight: a tie, which goes to the first node by name.
    let held = spawn_call(&gw.tcp, agent("s-1", "s1-held", true), CHAT);
    a.await_requests(1).await;
    assert_eq!(send(&gw, "s-2", "s2").await, 200);
    assert_eq!(b.tags(), ["s2"], "a has a request in flight");
    // s-1 stays on its node, though that is now the busier one.
    assert_eq!(send(&gw, "s-1", "s1").await, 200);
    assert_eq!(a.tags(), ["s1-held", "s1"]);
    a.release(1);
    assert_eq!(held.await.unwrap().status, 200);
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "lh_gateway_placements_total{policy=\"least-loaded\",reason=\"new\"}",
            2.0,
        ),
        (
            "lh_gateway_placements_total{policy=\"least-loaded\",reason=\"node_down\"}",
            0.0,
        ),
        ("lh_gateway_sessions{node=\"a\",class=\"background\"}", 1.0),
        ("lh_gateway_sessions{node=\"b\",class=\"background\"}", 1.0),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
}

#[tokio::test]
async fn dedicated_places_each_class_on_its_own_node() {
    let voice = Mock::start(Script::Json("{}")).await;
    let agents = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start_nodes(&format!(
        "[nodes.voice]\nupstream = \"http://{}\"\nclasses = [\"realtime\"]\n\
         [nodes.agents]\nupstream = \"http://{}\"\nclasses = [\"interactive\", \"background\"]\n\
         [placement]\npolicy = \"dedicated\"",
        voice.addr, agents.addr
    ))
    .await;
    // The agents' node is busy and the voice node idle: the class decides, not the load.
    let held = spawn_call(&gw.tcp, agent("s-1", "bg-held", true), CHAT);
    agents.await_requests(1).await;
    assert_eq!(send(&gw, "s-2", "bg").await, 200);
    let interactive = [
        AGENTS_AUTH,
        ("x-session-id", "s-3"),
        ("x-lh-class", "interactive"),
        (TAG, "int"),
    ];
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &interactive, CHAT).await;
    assert_eq!(reply.status, 200);
    let call_headers = [("x-session-id", "voice-RM_1"), (TAG, "rt")];
    let reply = call(
        &gw.unix,
        "POST",
        "/v1/chat/completions",
        &call_headers,
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 200);
    assert_eq!(voice.tags(), ["rt"]);
    assert_eq!(agents.tags(), ["bg-held", "bg", "int"]);
    agents.release(1);
    assert_eq!(held.await.unwrap().status, 200);
}

#[tokio::test]
async fn an_untrusted_tenant_cannot_name_another_tenants_session() {
    let voice = Mock::start(Script::Json("{}")).await;
    let agents = Mock::start(Script::Json("{}")).await;
    let gw = Gateway::start_nodes(&format!(
        "[nodes.voice]\nupstream = \"http://{}\"\nclasses = [\"realtime\"]\n\
         [nodes.agents]\nupstream = \"http://{}\"\nclasses = [\"interactive\", \"background\"]\n\
         [placement]\npolicy = \"dedicated\"",
        voice.addr, agents.addr
    ))
    .await;
    assert_eq!(place(&gw, "voice-RM_victim").await.json()["node"], "voice");
    // The bench tenant (untrusted) names the call's session in both sources it may set. Each becomes a session of
    // its own, hashed with the tenant (`printf 'bench\0voice-RM_victim' | shasum -a 256 | cut -c1-32`), and is
    // placed by its own class: the call's node and the call's session at the engine stay out of its reach.
    let own = "h-468e550b540d0075ae9e649823e0977e";
    let claude = [
        BENCH_AUTH,
        ("x-claude-code-session-id", "voice-RM_victim"),
        (TAG, "claude"),
    ];
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &claude, CHAT).await;
    assert_eq!(
        (reply.status.as_u16(), reply.header("x-lh-session-id")),
        (200, Some(own))
    );
    let body = r#"{"model":"qwen","messages":[],"prompt_cache_key":"voice-RM_victim"}"#;
    let pck = [BENCH_AUTH, (TAG, "pck")];
    let reply = call(&gw.tcp, "POST", "/v1/chat/completions", &pck, body).await;
    assert_eq!(
        (reply.status.as_u16(), reply.header("x-lh-session-id")),
        (200, Some(own))
    );
    assert!(voice.tags().is_empty());
    assert_eq!(agents.tags(), ["claude", "pck"]);
    assert_eq!(agents.last().header("x-session-id"), Some(own));
}

#[tokio::test]
async fn a_node_that_fails_readiness_takes_no_new_sessions_and_its_sessions_move() {
    let tables = "[placement]\npolicy = \"least-loaded\"\nprobe_interval_s = 0.05";
    let (a, b, gw) = two_nodes(tables).await;
    assert_eq!(send(&gw, "s-1", "s1").await, 200);
    assert_eq!(a.tags(), ["s1"]);
    // a stops answering /health: after three failed probes it takes no new session, and s-1 moves to b.
    a.set_healthy(false);
    gw.await_metric("lh_gateway_node_ready{node=\"a\"}", 0.0)
        .await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 200);
    assert_eq!(send(&gw, "s-1", "s1-moved").await, 200);
    assert_eq!(send(&gw, "s-2", "s2").await, 200);
    assert_eq!(b.tags(), ["s1-moved", "s2"]);
    let moved = "lh_gateway_placements_total{policy=\"least-loaded\",reason=\"node_down\"}";
    assert_eq!(metric(&gw.metrics().await, moved), Some(1.0));
    // a answers again: s-1 stays where it went, and a new session takes the quieter node.
    a.set_healthy(true);
    gw.await_metric("lh_gateway_node_ready{node=\"a\"}", 1.0)
        .await;
    assert_eq!(send(&gw, "s-1", "s1-again").await, 200);
    assert_eq!(send(&gw, "s-3", "s3").await, 200);
    assert_eq!(b.tags(), ["s1-moved", "s2", "s1-again"]);
    assert_eq!(a.tags(), ["s1", "s3"]);
    // With no node ready, the gateway is not ready and refuses requests at once.
    a.set_healthy(false);
    b.set_healthy(false);
    gw.await_metric("lh_gateway_node_ready{node=\"a\"}", 0.0)
        .await;
    gw.await_metric("lh_gateway_node_ready{node=\"b\"}", 0.0)
        .await;
    assert_eq!(call(&gw.tcp, "GET", "/ready", &[], "").await.status, 503);
    let reply = call(
        &gw.tcp,
        "POST",
        "/v1/chat/completions",
        &agent("s-4", "s4", false),
        CHAT,
    )
    .await;
    assert_eq!(reply.status, 503);
    assert_eq!(reply.json()["error"]["code"], "no_ready_node");
    assert_eq!(a.requests().len() + b.requests().len(), 5);
}

#[tokio::test]
async fn an_idle_session_is_placed_again() {
    let (a, b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"\nidle_s = 0.2").await;
    assert_eq!(send(&gw, "s-1", "s1").await, 200);
    let held = spawn_call(&gw.tcp, agent("s-2", "s2", true), CHAT);
    a.await_requests(2).await;
    // s-1 idles out, and is placed again by its next request: on b, the quieter node.
    gw.await_metric("lh_gateway_sessions{node=\"a\",class=\"background\"}", 1.0)
        .await;
    assert_eq!(send(&gw, "s-1", "s1-again").await, 200);
    assert_eq!(b.tags(), ["s1-again"]);
    a.release(1);
    assert_eq!(held.await.unwrap().status, 200);
}

#[tokio::test]
async fn the_placement_endpoint_places_a_call_ahead_of_its_first_request() {
    let (a, b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"").await;
    let reply = place(&gw, "voice-RM_1").await;
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.json(),
        json!({"session_id": "voice-RM_1", "class": "realtime", "node": "a"})
    );
    // Asked again, the same node. The call holds a place there until its first request, so the next call that
    // starts goes to b.
    assert_eq!(place(&gw, "voice-RM_1").await.json()["node"], "a");
    assert_eq!(place(&gw, "voice-RM_2").await.json()["node"], "b");
    for (session, tag) in [("voice-RM_1", "rt1"), ("voice-RM_2", "rt2")] {
        let headers = [("x-session-id", session), (TAG, tag)];
        let reply = call(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await;
        assert_eq!(reply.status, 200);
    }
    assert_eq!(
        (a.tags(), b.tags()),
        (vec!["rt1".into()], vec!["rt2".into()])
    );
    let metrics = gw.metrics().await;
    for (series, value) in [
        ("lh_gateway_sessions{node=\"a\",class=\"realtime\"}", 1.0),
        ("lh_gateway_sessions{node=\"b\",class=\"realtime\"}", 1.0),
        (
            "lh_gateway_placements_total{policy=\"least-loaded\",reason=\"new\"}",
            2.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }

    // A class stricter than the tenant's is clamped; an untrusted tenant may not place, and bodies are checked.
    let body = r#"{"session_id":"agent-1","class":"realtime"}"#;
    let agents = call(&gw.tcp, "POST", "/v1/sessions", &[AGENTS_AUTH], body).await;
    assert_eq!(
        (agents.status.as_u16(), &agents.json()["class"]),
        (200, &json!("interactive"))
    );
    let untrusted = call(&gw.tcp, "POST", "/v1/sessions", &[BENCH_AUTH], body).await;
    assert_eq!(untrusted.status, 403);
    assert_eq!(
        call(&gw.tcp, "POST", "/v1/sessions", &[], body)
            .await
            .status,
        401
    );
    for body in [
        r#"{"class":"realtime"}"#,
        r#"{"session_id":"has space"}"#,
        r#"{"session_id":"voice-RM_3","class":"urgent"}"#,
        "[]",
    ] {
        let reply = call(&gw.unix, "POST", "/v1/sessions", &[], body).await;
        assert_eq!(reply.status, 400, "{body}");
        assert_eq!(reply.json()["error"]["code"], "invalid_session", "{body}");
    }

    // With one upstream there is nothing to place: the route does not exist.
    let one = Gateway::start(a.addr, "").await;
    assert_eq!(place(&one, "voice-RM_4").await.status, 404);
}

/// The review's probe P4: an untrusted tenant names as many sessions as the table holds (14 a second would do, at
/// `idle_s` 300), and a call placed after them keeps the node `POST /v1/sessions` answered.
#[tokio::test]
async fn a_call_keeps_its_node_after_an_untrusted_flood_of_sessions() {
    let (a, b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"").await;
    flood(&gw, 4096).await;
    // The call is placed ahead of its first request, on a: nothing is in flight, and ties go to the first node.
    let placed = place(&gw, "voice-RM_victim").await;
    assert_eq!(
        (placed.status.as_u16(), &placed.json()["node"]),
        (200, &json!("a"))
    );
    // Its place on a counts as a request in flight there, so an agent's new session goes to b.
    let held = spawn_call(&gw.tcp, agent("agent-held", "held", true), CHAT);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while a.tags().is_empty() && b.tags().is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "the agent's request went nowhere"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        b.tags(),
        ["held"],
        "the call's place on a was not remembered"
    );
    // The call's first request goes to its node, a, though the nodes tie now.
    let headers = [("x-session-id", "voice-RM_victim"), (TAG, "call-turn-1")];
    let reply = call(&gw.unix, "POST", "/v1/chat/completions", &headers, CHAT).await;
    assert_eq!(reply.status, 200);
    assert_eq!(a.tags(), ["call-turn-1"]);
    // The flood holds half the table.
    let flooded = sessions(&gw, "interactive").await;
    assert_eq!(flooded[0] + flooded[1], 2048.0);
    b.release(1);
    assert_eq!(held.await.unwrap().status, 200);
}

/// An untrusted tenant's sessions take at most half the table, so that its flood cannot make the table forget the
/// trusted tenants' idle sessions: an agent's session that went idle before the flood keeps its node.
#[tokio::test]
async fn untrusted_sessions_take_at_most_half_the_table() {
    let (_a, _b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"").await;
    assert_eq!(send(&gw, "agent-idle", "before").await, 200);
    flood(&gw, 4096).await;
    let flooded = sessions(&gw, "interactive").await;
    assert_eq!(flooded[0] + flooded[1], 2048.0);
    assert_eq!(sessions(&gw, "background").await, [1.0, 0.0]);
}

#[tokio::test]
async fn each_node_has_its_own_admission_point() {
    let tables = "[placement]\npolicy = \"least-loaded\"\n\
                  [admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1";
    let (a, b, gw) = two_nodes(tables).await;
    // A background request in flight on each node, under a cap of one: each node counts its own.
    let first = spawn_call(&gw.tcp, agent("s-1", "s1", true), CHAT);
    a.await_requests(1).await;
    let second = spawn_call(&gw.tcp, agent("s-2", "s2", true), CHAT);
    b.await_requests(1).await;
    // A third session ties, goes to a, and waits there behind the first.
    let third = spawn_call(&gw.tcp, agent("s-3", "s3", true), CHAT);
    gw.await_metric(
        "lh_gateway_admission_waiting{node=\"a\",class=\"background\"}",
        1.0,
    )
    .await;
    let metrics = gw.metrics().await;
    for (series, value) in [
        (
            "lh_gateway_admission_waiting{node=\"b\",class=\"background\"}",
            0.0,
        ),
        (
            "lh_gateway_admission_admitted_total{node=\"a\",class=\"background\"}",
            1.0,
        ),
        (
            "lh_gateway_admission_admitted_total{node=\"b\",class=\"background\"}",
            1.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}");
    }
    a.release(1);
    a.await_requests(2).await;
    a.release(1);
    b.release(1);
    for reply in [first, second, third] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
    assert_eq!(a.tags(), ["s1", "s3"]);
}

#[tokio::test]
async fn leaseholder_places_calls_where_their_turns_find_headroom_and_agents_by_their_kv() {
    let tables = "[placement]\npolicy = \"leaseholder\"\n\
                  [admission]\npolicy = \"leaseholder\"\npool_tokens = 1000\nheadroom_tokens = 200\n\
                  throttle = false\ndegrade = false";
    let (a, b, gw) = two_nodes(tables).await;
    // Two calls that start together: one on each node.
    assert_eq!(place(&gw, "voice-RM_1").await.json()["node"], "a");
    assert_eq!(place(&gw, "voice-RM_2").await.json()["node"], "b");
    // The first caller speaks: its events reach the admission point of a alone.
    post_event(&gw, "voice-RM_1", "vw-1", "user_speech_started").await;
    gw.await_metric(
        "lh_gateway_realtime_sessions{node=\"a\",state=\"user_speaking\"}",
        1.0,
    )
    .await;
    let speaking_on_b = "lh_gateway_realtime_sessions{node=\"b\",state=\"user_speaking\"}";
    assert_eq!(metric(&gw.metrics().await, speaking_on_b), Some(0.0));
    // Agents go where their KV leaves the most room: 600 tokens on a, the next 600 on b.
    let first = spawn_call(&gw.tcp, agent("agent-1", "bg1", true), six_hundred());
    a.await_requests(1).await;
    let second = spawn_call(&gw.tcp, agent("agent-2", "bg2", true), six_hundred());
    b.await_requests(1).await;
    // A third finds 600 on each and goes to a, where the speaking caller's headroom defers it.
    let third = spawn_call(&gw.tcp, agent("agent-3", "bg3", true), six_hundred());
    gw.await_metric(
        "lh_gateway_admission_deferred_total{node=\"a\",reason=\"headroom\"}",
        1.0,
    )
    .await;
    // The caller hangs up: its session closes on a, which frees the deferred agent and ends the placement.
    post_event(&gw, "voice-RM_1", "vw-2", "session_closed").await;
    a.await_requests(2).await;
    gw.await_metric("lh_gateway_sessions{node=\"a\",class=\"realtime\"}", 0.0)
        .await;
    assert_eq!(a.tags(), ["bg1", "bg3"]);
    assert_eq!(b.tags(), ["bg2"]);
    a.release(2);
    b.release(1);
    for reply in [first, second, third] {
        assert_eq!(reply.await.unwrap().status, 200);
    }
}

#[tokio::test]
async fn upstream_errors_count_on_their_node() {
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = closed.local_addr().unwrap();
    drop(closed);
    let b = Mock::start(Script::Json("{}")).await;
    // Node a has no engine, and its probes are too far apart to notice before the request.
    let gw = Gateway::start_nodes(&format!(
        "[nodes.a]\nupstream = \"http://{addr}\"\n[nodes.b]\nupstream = \"http://{}\"\n\
         [placement]\npolicy = \"least-loaded\"\nprobe_interval_s = 30",
        b.addr
    ))
    .await;
    assert_eq!(send(&gw, "s-1", "s1").await, 502);
    gw.await_metric(
        "lh_gateway_upstream_errors_total{node=\"a\",kind=\"connect\"}",
        1.0,
    )
    .await;
    let on_b = "lh_gateway_upstream_errors_total{node=\"b\",kind=\"connect\"}";
    assert_eq!(metric(&gw.metrics().await, on_b), Some(0.0));
}

#[tokio::test]
async fn the_series_each_node_keeps_carry_its_name() {
    let (_a, _b, gw) = two_nodes("[placement]\npolicy = \"least-loaded\"").await;
    let text = gw.metrics().await;
    for series in [
        "lh_gateway_admission_admitted_total{node=\"a\",class=\"background\"} 0",
        "lh_gateway_admission_waiting{node=\"b\",class=\"realtime\"} 0",
        "lh_gateway_admission_refused_total{node=\"a\",class=\"background\",reason=\"timeout\"} 0",
        "lh_gateway_admission_deferred_total{node=\"b\",reason=\"headroom\"} 0",
        "lh_gateway_admission_aged_total{node=\"a\",oversize=\"true\"} 0",
        "lh_gateway_admission_degraded_total{node=\"b\"} 0",
        "lh_gateway_admission_false_alarms_total{node=\"a\"} 0",
        "lh_gateway_admission_false_alarm_token_seconds_total{node=\"b\"} 0",
        "lh_gateway_realtime_sessions{node=\"a\",state=\"critical\"} 0",
        "lh_gateway_realtime_state_timeouts_total{node=\"b\",state=\"playback\"} 0",
        "lh_gateway_upstream_errors_total{node=\"a\",kind=\"connect\"} 0",
        "lh_gateway_sessions{node=\"b\",class=\"interactive\"} 0",
        "lh_gateway_placements_total{policy=\"least-loaded\",reason=\"node_down\"} 0",
        "lh_gateway_node_ready{node=\"a\"} 1",
        // The gateway's own series have no node.
        "lh_gateway_first_byte_seconds_count{class=\"background\"} 0",
        "lh_gateway_streams_active{class=\"realtime\"} 0",
    ] {
        assert!(
            text.lines().any(|line| line.starts_with(series)),
            "missing {series}\n{text}"
        );
    }
    for unlabeled in [
        "lh_gateway_admission_degraded_total ",
        "lh_gateway_admission_admitted_total{class=",
        "lh_gateway_upstream_errors_total{kind=",
    ] {
        assert!(
            !text.lines().any(|line| line.starts_with(unlabeled)),
            "{unlabeled}\n{text}"
        );
    }
}
