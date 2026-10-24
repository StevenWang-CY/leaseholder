//! What the gateway removes from a client request, what it resolves from it (session and class), and what it
//! sets for the engine. The names come from `lh_proto::headers`.

use std::fmt;
use std::io::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use http::header::{self, HeaderMap, HeaderName, HeaderValue};
use lh_proto::headers;
use lh_proto::{SessionId, SloClass, UnknownClass};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

use crate::auth::Tenant;
use crate::config::PrioritySignals;
use crate::metrics::{Metrics, Route};

pub(crate) static SESSION_ID: HeaderName = HeaderName::from_static(headers::SESSION_ID);
pub(crate) static VLLM_PRIORITY: HeaderName = HeaderName::from_static(headers::VLLM_PRIORITY);
pub(crate) static SMG_PRIORITY: HeaderName = HeaderName::from_static(headers::SMG_PRIORITY);
pub(crate) static REQUEST_ID: HeaderName = HeaderName::from_static(headers::REQUEST_ID);
pub(crate) static RESOLVED_SESSION_ID: HeaderName =
    HeaderName::from_static(headers::RESOLVED_SESSION_ID);

/// The `x-lh-` headers counted under their own name when removed.
const NAMED_LH_HEADERS: [&str; 3] = [
    headers::CLASS,
    headers::SYNTHETIC,
    headers::RESOLVED_SESSION_ID,
];

/// Metric label for removed `x-lh-` headers that `lh_proto` does not name, which keeps the label bounded.
const OTHER_LH_HEADER: &str = "x-lh-*";

/// Every `field` label of `lh_gateway_sanitized_total`.
pub(crate) fn sanitized_fields() -> impl Iterator<Item = &'static str> {
    headers::ENGINE_CONTROL_HEADERS
        .into_iter()
        .chain([headers::REQUEST_ID])
        .chain(NAMED_LH_HEADERS)
        .chain([OTHER_LH_HEADER])
        .chain(headers::ENGINE_CONTROL_BODY_FIELDS)
}

/// Hop-by-hop headers (RFC 9110 section 7.6.1, plus the legacy `Keep-Alive` and `Proxy-Connection`).
static HOP_BY_HOP: [HeaderName; 8] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    HeaderName::from_static("proxy-connection"),
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

/// Request headers that do not apply to the upstream request: the client's key, which only the gateway checks;
/// the gateway's host; the length and type of a body the gateway rewrites; and `Expect`, which the gateway
/// already answered by reading the body.
static NOT_FORWARDED: [HeaderName; 5] = [
    header::AUTHORIZATION,
    header::HOST,
    header::CONTENT_LENGTH,
    header::CONTENT_TYPE,
    header::EXPECT,
];

/// The values a request carries for the gateway itself.
#[derive(Debug, Default)]
pub(crate) struct ClientInputs {
    /// `X-Session-ID`, kept only from a trusted tenant.
    pub session_id: Option<HeaderValue>,
    pub claude_code_session_id: Option<HeaderValue>,
    /// `X-LH-Class`, kept only from a trusted tenant.
    pub class: Option<HeaderValue>,
}

/// Removes from `headers` everything the engine must not receive from a client, counting each input a client
/// set but may not control, and returns the inputs the gateway consumes.
pub(crate) fn sanitize_headers(
    headers: &mut HeaderMap,
    trusted: bool,
    metrics: &Metrics,
) -> ClientInputs {
    let listed = connection_listed(headers);
    let mut inputs = ClientInputs {
        claude_code_session_id: headers.remove(headers::CLAUDE_CODE_SESSION_ID),
        ..ClientInputs::default()
    };
    if trusted {
        inputs.session_id = headers.remove(&SESSION_ID);
        inputs.class = headers.remove(headers::CLASS);
    }
    for name in headers::ENGINE_CONTROL_HEADERS
        .into_iter()
        .chain([headers::REQUEST_ID])
    {
        if headers.remove(name).is_some() {
            metrics.sanitized(name);
        }
    }
    let lh_headers: Vec<HeaderName> = headers
        .keys()
        .filter(|name| name.as_str().starts_with(headers::LH_PREFIX))
        .cloned()
        .collect();
    for name in lh_headers {
        headers.remove(&name);
        metrics.sanitized(
            NAMED_LH_HEADERS
                .into_iter()
                .find(|known| *known == name.as_str())
                .unwrap_or(OTHER_LH_HEADER),
        );
    }
    for name in HOP_BY_HOP.iter().chain(&NOT_FORWARDED).chain(&listed) {
        headers.remove(name);
    }
    inputs
}

/// Removes hop-by-hop headers from an upstream response before it is relayed.
pub(crate) fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed = connection_listed(headers);
    for name in HOP_BY_HOP.iter().chain(&listed) {
        headers.remove(name);
    }
}

/// The header names a `Connection` header lists, which are hop-by-hop as well.
fn connection_listed(headers: &HeaderMap) -> Vec<HeaderName> {
    headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect()
}

/// The request's class: `X-LH-Class` from a trusted tenant, else the tenant's default, clamped to the tenant's
/// strictest allowed class. Returns whether it was clamped.
pub(crate) fn resolve_class(
    tenant: &Tenant,
    requested: Option<&HeaderValue>,
) -> Result<(SloClass, bool), UnknownClass> {
    let class = match requested {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| UnknownClass(String::from_utf8_lossy(value.as_bytes()).into_owned()))?,
        None => tenant.default_class,
    };
    // Classes order from strictest to most lenient, so "stricter than allowed" is "less than".
    Ok(if class < tenant.max_class {
        (tenant.max_class, true)
    } else {
        (class, false)
    })
}

/// The session of a request, from the first present source in PROTOCOL.md's order: a trusted `X-Session-ID`,
/// `X-Claude-Code-Session-ID`, the body's `prompt_cache_key`. A trusted tenant's valid session id is used as it
/// is. Any other source is hashed into one, scoped to the tenant: every source of an untrusted tenant, so that it
/// cannot name another tenant's session, and a source that is not a valid session id. With no source the gateway
/// mints one.
pub(crate) fn resolve_session(
    tenant: &Tenant,
    inputs: &ClientInputs,
    prompt_cache_key: Option<&str>,
) -> SessionId {
    match session_source(inputs, prompt_cache_key) {
        Some(raw) => std::str::from_utf8(raw)
            .ok()
            .filter(|_| tenant.trusted)
            .and_then(|id| SessionId::new(id).ok())
            .unwrap_or_else(|| hashed_session(&tenant.name, raw)),
        None => new_id("m-", &random_hex()),
    }
}

/// Whether a request names its session, so that its next requests can name the same one; a minted session is
/// the request's own.
pub(crate) fn names_session(inputs: &ClientInputs, prompt_cache_key: Option<&str>) -> bool {
    session_source(inputs, prompt_cache_key).is_some()
}

/// The first present source of a request's session, in PROTOCOL.md's order.
fn session_source<'a>(
    inputs: &'a ClientInputs,
    prompt_cache_key: Option<&'a str>,
) -> Option<&'a [u8]> {
    let sources = [
        inputs.session_id.as_ref().map(HeaderValue::as_bytes),
        inputs
            .claude_code_session_id
            .as_ref()
            .map(HeaderValue::as_bytes),
        prompt_cache_key.map(str::as_bytes),
    ];
    sources.into_iter().flatten().find(|raw| !raw.is_empty())
}

fn hashed_session(tenant: &str, raw: &[u8]) -> SessionId {
    let digest = Sha256::new()
        .chain_update(tenant)
        .chain_update([0])
        .chain_update(raw)
        .finalize();
    new_id("h-", &hex(&digest[..16]))
}

fn new_id(prefix: &str, hex: &str) -> SessionId {
    SessionId::new(format!("{prefix}{hex}"))
        .expect("a prefix and hex digits form a valid session id")
}

/// Sets the engine inputs the gateway owns: the session, the request id, and the class's priority in the
/// headers `signals` names.
pub(crate) fn set_engine_headers(
    headers: &mut HeaderMap,
    session: &SessionId,
    class: SloClass,
    request_id: &HeaderValue,
    signals: PrioritySignals,
) {
    headers.insert(SESSION_ID.clone(), session_header(session));
    if signals.vllm_header {
        headers.insert(
            VLLM_PRIORITY.clone(),
            HeaderValue::from(class.engine_priority()),
        );
    }
    if signals.smg_header {
        headers.insert(
            SMG_PRIORITY.clone(),
            HeaderValue::from_static(smg_class(class)),
        );
    }
    headers.insert(REQUEST_ID.clone(), request_id.clone());
}

/// The SMG class of an SLO class. SMG's fourth class, `default`, is what it gives a request without one.
fn smg_class(class: SloClass) -> &'static str {
    match class {
        SloClass::Realtime => "system",
        SloClass::Interactive => "interactive",
        SloClass::Background => "bulk",
    }
}

pub(crate) fn session_header(session: &SessionId) -> HeaderValue {
    HeaderValue::from_str(session.as_str()).expect("session ids are header-safe")
}

/// A request body after sanitization.
#[derive(Debug)]
pub(crate) struct Body {
    pub json: Bytes,
    pub prompt_cache_key: Option<String>,
}

/// Why a request body was rejected (400).
#[derive(Debug)]
pub(crate) struct InvalidBody(serde_json::Error);

impl fmt::Display for InvalidBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the request body must be a JSON object: {}", self.0)
    }
}

/// Removes the engine-control fields from a JSON object body and sets the tenant's `cache_salt`, and the
/// gateway's own `priority` if it is given one.
///
/// Only the top level is parsed: values stay raw slices of the input and are copied through unchanged, so the
/// cost is one validating scan and one copy whatever the body's size. Keys are compared after JSON unescaping,
/// so an escaped `"pri\u006frity"` is removed too.
pub(crate) fn rewrite_body(
    body: &[u8],
    cache_salt: &str,
    priority: Option<i32>,
    metrics: &Metrics,
) -> Result<Body, InvalidBody> {
    let TopLevel(fields) = serde_json::from_slice(body).map_err(InvalidBody)?;
    let mut json = Vec::with_capacity(body.len() + 64);
    let mut prompt_cache_key = None;
    json.push(b'{');
    for (key, value) in &fields {
        if let Some(field) = headers::ENGINE_CONTROL_BODY_FIELDS
            .into_iter()
            .find(|field| *field == key.as_str())
        {
            metrics.sanitized(field);
            continue;
        }
        if key == "prompt_cache_key" {
            prompt_cache_key = serde_json::from_str(value.get()).ok();
        }
        serde_json::to_writer(&mut json, key).expect("writing to a Vec cannot fail");
        json.push(b':');
        json.extend_from_slice(value.get().as_bytes());
        json.push(b',');
    }
    if let Some(priority) = priority {
        write!(json, "\"priority\":{priority},").expect("writing to a Vec cannot fail");
    }
    json.extend_from_slice(b"\"cache_salt\":");
    serde_json::to_writer(&mut json, cache_salt).expect("writing to a Vec cannot fail");
    json.push(b'}');
    Ok(Body {
        json: json.into(),
        prompt_cache_key,
    })
}

/// The body fields that cap a reply's tokens: Chat Completions' `max_completion_tokens` and its deprecated
/// `max_tokens`, which Completions reads, and the Responses API's `max_output_tokens`.
const REPLY_CAPS: [&str; 3] = ["max_completion_tokens", "max_tokens", "max_output_tokens"];

/// Caps the reply of a degraded request at `cap` tokens in a body [`rewrite_body`] wrote: each of the body's
/// reply caps is lowered to `cap` if it is above it or not a count, and a body with none gets the route's own.
pub(crate) fn cap_reply(body: &[u8], route: Route, cap: u32) -> Bytes {
    let TopLevel(fields) =
        serde_json::from_slice(body).expect("the gateway writes bodies as JSON objects");
    let mut json = Vec::with_capacity(body.len() + 32);
    json.push(b'{');
    if !fields
        .iter()
        .any(|(key, _)| REPLY_CAPS.contains(&key.as_str()))
    {
        let field = match route {
            Route::Completions => "max_tokens",
            Route::Responses => "max_output_tokens",
            _ => "max_completion_tokens",
        };
        write!(json, "\"{field}\":{cap},").expect("writing to a Vec cannot fail");
    }
    for (key, value) in &fields {
        serde_json::to_writer(&mut json, key).expect("writing to a Vec cannot fail");
        json.push(b':');
        if REPLY_CAPS.contains(&key.as_str()) {
            let asked = serde_json::from_str::<u32>(value.get()).ok();
            let capped = asked.filter(|&asked| asked <= cap).unwrap_or(cap);
            write!(json, "{capped}").expect("writing to a Vec cannot fail");
        } else {
            json.extend_from_slice(value.get().as_bytes());
        }
        json.push(b',');
    }
    // Every member, the one added among them, ends with a comma.
    *json.last_mut().expect("a member was written") = b'}';
    json.into()
}

/// The top-level members of a JSON object, in order, with raw values.
struct TopLevel<'a>(Vec<(String, &'a RawValue)>);

impl<'de> serde::Deserialize<'de> for TopLevel<'de> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Members;
        impl<'de> Visitor<'de> for Members {
            type Value = TopLevel<'de>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut members = Vec::with_capacity(map.size_hint().unwrap_or(16));
                while let Some(member) = map.next_entry()? {
                    members.push(member);
                }
                Ok(TopLevel(members))
            }
        }
        deserializer.deserialize_map(Members)
    }
}

/// Lowercase hexadecimal.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|b| [DIGITS[usize::from(b >> 4)], DIGITS[usize::from(b & 15)]])
        .map(char::from)
        .collect()
}

/// 128 random bits as 32 hex digits, from the operating system's generator.
pub(crate) fn random_hex() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the operating system's random number generator failed");
    hex(&bytes)
}

/// A request id: 32 hex digits, a random per-process prefix and a counter. Request ids must be unique, not
/// unpredictable (clients cannot set them), so this needs no system call per request.
pub(crate) fn request_id() -> HeaderValue {
    static PREFIX: LazyLock<u64> = LazyLock::new(|| {
        getrandom::u64().expect("the operating system's random number generator failed")
    });
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    HeaderValue::from_str(&format!("{:016x}{n:016x}", *PREFIX))
        .expect("hex digits are a valid header value")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn tenant(trusted: bool, default_class: SloClass, max_class: SloClass) -> Tenant {
        Tenant {
            name: "agents".into(),
            trusted,
            default_class,
            max_class,
            cache_salt: "salt".into(),
        }
    }

    fn named(name: &str, trusted: bool) -> Tenant {
        Tenant {
            name: name.into(),
            ..tenant(trusted, SloClass::Interactive, SloClass::Interactive)
        }
    }

    fn header(value: &str) -> Option<HeaderValue> {
        Some(HeaderValue::from_str(value).unwrap())
    }

    #[test]
    fn sessions_resolve_in_protocol_order() {
        let agents = named("agents", true);
        let all = ClientInputs {
            session_id: header("voice-RM_a1B2c3"),
            claude_code_session_id: header("0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41"),
            class: None,
        };
        assert_eq!(
            resolve_session(&agents, &all, Some("pck")).as_str(),
            "voice-RM_a1B2c3"
        );
        let claude = ClientInputs {
            session_id: None,
            ..all
        };
        assert_eq!(
            resolve_session(&agents, &claude, Some("pck")).as_str(),
            "0f3c2a9e-5b1d-4c8e-9a77-2d6f1e0b8c41"
        );
        assert_eq!(
            resolve_session(&agents, &ClientInputs::default(), Some("pck")).as_str(),
            "pck"
        );
        // An empty source is absent.
        let empty = ClientInputs {
            session_id: header(""),
            ..ClientInputs::default()
        };
        assert_eq!(
            resolve_session(&agents, &empty, Some("pck")).as_str(),
            "pck"
        );
    }

    #[test]
    fn invalid_sources_are_hashed_per_tenant_and_missing_ones_minted() {
        let (agents, inputs) = (named("agents", true), ClientInputs::default());
        let hashed = resolve_session(&agents, &inputs, Some("has spaces/and slashes"));
        // `printf 'agents\0has spaces/and slashes' | shasum -a 256 | cut -c1-32`
        assert_eq!(hashed.as_str(), "h-dc629ca91b9da396f87344e8a3e8ecbd");
        assert_ne!(
            resolve_session(
                &named("bench", true),
                &inputs,
                Some("has spaces/and slashes")
            ),
            hashed
        );
        let long = "a".repeat(SessionId::MAX_LEN + 1);
        assert!(
            resolve_session(&agents, &inputs, Some(&long))
                .as_str()
                .starts_with("h-")
        );

        let minted = resolve_session(&agents, &inputs, None);
        assert!(minted.as_str().starts_with("m-") && minted.as_str().len() == 34);
        assert!(minted.as_str()[2..].bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(minted, resolve_session(&agents, &inputs, None));
    }

    #[test]
    fn an_untrusted_tenant_names_only_sessions_of_its_own() {
        let bench = named("bench", false);
        let claude = ClientInputs {
            claude_code_session_id: header("voice-RM_victim"),
            ..ClientInputs::default()
        };
        // A valid session id from an untrusted tenant is hashed with the tenant all the same:
        // `printf 'bench\0voice-RM_victim' | shasum -a 256 | cut -c1-32`.
        assert_eq!(
            resolve_session(&bench, &claude, None).as_str(),
            "h-468e550b540d0075ae9e649823e0977e"
        );
        // `printf 'bench\0bench-7' | shasum -a 256 | cut -c1-32`
        assert_eq!(
            resolve_session(&bench, &ClientInputs::default(), Some("bench-7")).as_str(),
            "h-17636c9ec78323ab9bd295ae7d8a40c1"
        );
        // The same source names a trusted tenant's session as it is, and another untrusted tenant's own one.
        assert_eq!(
            resolve_session(&named("voice", true), &claude, None).as_str(),
            "voice-RM_victim"
        );
        assert_ne!(
            resolve_session(&named("other", false), &claude, None),
            resolve_session(&bench, &claude, None)
        );
    }

    #[test]
    fn classes_are_honored_defaulted_and_clamped() {
        use SloClass::*;
        let agents = tenant(true, Background, Interactive);
        assert_eq!(resolve_class(&agents, None).unwrap(), (Background, false));
        assert_eq!(
            resolve_class(&agents, header("interactive").as_ref()).unwrap(),
            (Interactive, false)
        );
        assert_eq!(
            resolve_class(&agents, header("realtime").as_ref()).unwrap(),
            (Interactive, true)
        );
        let err = resolve_class(&agents, header("urgent").as_ref()).unwrap_err();
        assert!(err.to_string().contains("\"urgent\""));
    }

    #[test]
    fn bodies_lose_engine_fields_and_gain_the_salt() {
        let metrics = Metrics::new();
        let body = br#"{"model":"m","pri\u006frity":-9,"messages":[{"role":"user","content":"hi \"priority\""}],
            "session_id":"x","kv_transfer_params":{"a":1},"cache_salt":"theirs","request_id":"r",
            "vllm_xargs":{"session_id":"y"},"rid":"theirs","extra_key":"other-tenant",
            "session_params":{"id":"s"},"custom_labels":{"x":"y"},"routed_dp_rank":1,"data_parallel_rank":1,
            "disagg_prefill_dp_rank":0,"bootstrap_host":"h","bootstrap_port":1,"bootstrap_room":7,
            "prompt_cache_key":"pck-1","stream":true}"#;
        let out = rewrite_body(body, "0123", None, &metrics).unwrap();
        let value: Value = serde_json::from_slice(&out.json).unwrap();
        assert_eq!(
            value,
            json!({"model": "m", "messages": [{"role": "user", "content": "hi \"priority\""}],
                   "prompt_cache_key": "pck-1", "stream": true, "cache_salt": "0123"})
        );
        assert_eq!(out.prompt_cache_key.as_deref(), Some("pck-1"));
        for field in headers::ENGINE_CONTROL_BODY_FIELDS {
            assert_eq!(metrics.sanitized_count(field), 1, "{field}");
        }
        // For an upstream that reads the priority from the body, the gateway's replaces the client's.
        let out = rewrite_body(body, "0123", Some(2), &metrics).unwrap();
        let value: Value = serde_json::from_slice(&out.json).unwrap();
        assert_eq!(value["priority"], 2);
        assert_eq!(metrics.sanitized_count("priority"), 2);
    }

    #[test]
    fn a_degraded_reply_is_capped_in_the_body() {
        let capped = |body: &str, route| -> Value {
            serde_json::from_slice(&cap_reply(body.as_bytes(), route, 24)).unwrap()
        };
        // A body without a cap gets the route's own; a smaller one stays, a larger one or a non-count is lowered.
        let chat = capped(r#"{"model":"m","cache_salt":"s"}"#, Route::ChatCompletions);
        assert_eq!(
            chat,
            json!({"max_completion_tokens": 24, "model": "m", "cache_salt": "s"})
        );
        let both = capped(
            r#"{"max_tokens":100,"max_completion_tokens":12,"cache_salt":"s"}"#,
            Route::ChatCompletions,
        );
        assert_eq!(
            both,
            json!({"max_tokens": 24, "max_completion_tokens": 12, "cache_salt": "s"})
        );
        let null = capped(
            r#"{"max_tokens":null,"cache_salt":"s"}"#,
            Route::Completions,
        );
        assert_eq!(null, json!({"max_tokens": 24, "cache_salt": "s"}));
        let completion = capped(r#"{"prompt":"p","cache_salt":"s"}"#, Route::Completions);
        assert_eq!(
            completion,
            json!({"max_tokens": 24, "prompt": "p", "cache_salt": "s"})
        );
        let response = capped(
            r#"{"input":"p","max_output_tokens":4000,"cache_salt":"s"}"#,
            Route::Responses,
        );
        assert_eq!(
            response,
            json!({"input": "p", "max_output_tokens": 24, "cache_salt": "s"})
        );
        // Every other value is copied as it is.
        let body =
            br#"{"messages":[{"role":"user","content":"\"max_tokens\":1"}],"cache_salt":"s"}"#;
        let out = cap_reply(body, Route::ChatCompletions, 24);
        assert_eq!(
            &out[..],
            br#"{"max_completion_tokens":24,"messages":[{"role":"user","content":"\"max_tokens\":1"}],"cache_salt":"s"}"#
        );
    }

    #[test]
    fn bodies_must_be_json_objects() {
        let metrics = Metrics::new();
        for body in [
            &b""[..],
            b"[1]",
            b"\"s\"",
            b"{\"a\":",
            b"{} {}",
            b"{\"a\":1,}",
        ] {
            assert!(rewrite_body(body, "s", None, &metrics).is_err(), "{body:?}");
        }
        let out = rewrite_body(b" { } ", "s", None, &metrics).unwrap();
        assert_eq!(&out.json[..], br#"{"cache_salt":"s"}"#);
        let out = rewrite_body(b"{}", "s", Some(0), &metrics).unwrap();
        assert_eq!(&out.json[..], br#"{"priority":0,"cache_salt":"s"}"#);
    }

    #[test]
    fn headers_lose_engine_inputs_and_hop_by_hop_fields() {
        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("x-vllm-priority", "-100"),
            ("x-smg-priority", "system"),
            ("x-data-parallel-rank", "1"),
            ("x-session-id", "s-1"),
            ("x-request-id", "theirs"),
            ("x-lh-class", "realtime"),
            ("x-lh-synthetic", "1"),
            ("x-lh-anything", "1"),
            ("x-claude-code-session-id", "cc-1"),
            ("authorization", "Bearer k"),
            ("connection", "keep-alive, x-hop"),
            ("x-hop", "1"),
            ("content-length", "10"),
            ("host", "gateway"),
            ("x-keep", "1"),
        ] {
            headers.append(
                HeaderName::from_static(name),
                HeaderValue::from_static(value),
            );
        }
        // A trusted tenant's session and class are consumed, not counted as removed.
        let trusted = Metrics::new();
        let inputs = sanitize_headers(&mut headers.clone(), true, &trusted);
        assert_eq!(inputs.session_id, header("s-1"));
        assert_eq!(inputs.class, header("realtime"));
        assert_eq!(trusted.sanitized_count("x-session-id"), 0);
        assert_eq!(trusted.sanitized_count("x-lh-class"), 0);
        assert_eq!(trusted.sanitized_count("x-vllm-priority"), 1);

        let untrusted = Metrics::new();
        let inputs = sanitize_headers(&mut headers, false, &untrusted);
        assert_eq!((inputs.session_id, inputs.class), (None, None));
        assert_eq!(inputs.claude_code_session_id, header("cc-1"));
        assert_eq!(headers.keys().collect::<Vec<_>>(), ["x-keep"]);
        for field in [
            "x-vllm-priority",
            "x-smg-priority",
            "x-data-parallel-rank",
            "x-session-id",
            "x-request-id",
            "x-lh-class",
            "x-lh-synthetic",
            OTHER_LH_HEADER,
        ] {
            assert_eq!(untrusted.sanitized_count(field), 1, "{field}");
        }
    }

    #[test]
    fn the_class_priority_goes_in_the_configured_headers() {
        let session = SessionId::new("s-1").unwrap();
        let id = HeaderValue::from_static("r-1");
        let priorities = |class, signals| {
            let mut headers = HeaderMap::new();
            set_engine_headers(&mut headers, &session, class, &id, signals);
            let value = |name| headers.get(name).map(|v| v.to_str().unwrap().to_owned());
            (value("x-vllm-priority"), value("x-smg-priority"))
        };
        let both = PrioritySignals {
            vllm_header: true,
            smg_header: true,
            ..PrioritySignals::default()
        };
        for (class, vllm, smg) in [
            (SloClass::Realtime, "0", "system"),
            (SloClass::Interactive, "1", "interactive"),
            (SloClass::Background, "2", "bulk"),
        ] {
            assert_eq!(
                priorities(class, both),
                (Some(vllm.into()), Some(smg.into()))
            );
            assert_eq!(priorities(class, PrioritySignals::default()), (None, None));
        }
    }

    #[test]
    fn hex_is_lowercase_and_ids_differ() {
        assert_eq!(hex(&[0x00, 0x9f, 0xff]), "009fff");
        assert_ne!(random_hex(), random_hex());
        let (a, b) = (request_id(), request_id());
        assert_eq!(a.len(), 32);
        assert_eq!(
            a.as_bytes()[..16],
            b.as_bytes()[..16],
            "one prefix per process"
        );
        assert_ne!(a, b);
    }
}
