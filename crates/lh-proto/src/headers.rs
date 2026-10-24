//! HTTP headers and body fields the control plane reads, sets and strips.
//!
//! Names are lowercase, the HTTP/2 form; HTTP/1.1 matches them case-insensitively.

/// Session id set by a trusted caller (agentic-api, the voice worker). vLLM reads the same header and records the
/// session id in its GPU-tier KV events.
pub const SESSION_ID: &str = "x-session-id";

/// Claude Code's own session header: the second source in session resolution.
pub const CLAUDE_CODE_SESSION_ID: &str = "x-claude-code-session-id";

/// Prefix of the headers between callers and the control plane. The control plane consumes them and never
/// forwards them to the engine.
pub const LH_PREFIX: &str = "x-lh-";

/// SLO class requested by a trusted caller: `realtime`, `interactive` or `background`.
pub const CLASS: &str = "x-lh-class";

/// Marks requests the control plane generates itself (prefetch warm-ups). They are excluded from SLIs, usage and
/// the ledger.
pub const SYNTHETIC: &str = "x-lh-synthetic";

/// Set on every response: the session id the control plane resolved or minted.
pub const RESOLVED_SESSION_ID: &str = "x-lh-session-id";

/// vLLM request priority. The control plane sets it from the class.
pub const VLLM_PRIORITY: &str = "x-vllm-priority";

/// The request class of SMG's priority scheduler: `system`, `interactive`, `default` or `bulk`. The control
/// plane sets it from the class when SMG is its upstream.
pub const SMG_PRIORITY: &str = "x-smg-priority";

/// vLLM data-parallel rank routing.
pub const DATA_PARALLEL_RANK: &str = "x-data-parallel-rank";

/// The request id. vLLM takes it as its own request id; the control plane generates one per request, sends it
/// to the engine and returns it to the caller.
pub const REQUEST_ID: &str = "x-request-id";

/// Headers the engine, or a router in front of it, trusts for scheduling and accounting. The control plane
/// removes them from every client request and sets its own.
pub const ENGINE_CONTROL_HEADERS: [&str; 4] =
    [SESSION_ID, VLLM_PRIORITY, SMG_PRIORITY, DATA_PARALLEL_RANK];

/// Body fields with the same role. vLLM 0.32 reads `session_id` before the header (and `vllm_xargs.session_id`
/// after it), takes `priority` for scheduling, `kv_transfer_params` for the offload policy, `cache_salt` for
/// prefix-cache isolation and `request_id` as its own request id. SGLang 0.5.22, which E2 runs behind the
/// control plane as well, takes `rid` as its request id, `extra_key` beside `cache_salt` as a prefix-cache
/// namespace, `session_params` to continue a session's KV, `custom_labels` as metric labels, and the routing
/// fields a router injects: the data-parallel rank (`routed_dp_rank`, and the deprecated `data_parallel_rank`)
/// and the prefill-decode disaggregation fields. The control plane removes them from client bodies and sets its
/// own; each tenant's `cache_salt` is derived from the tenant's name.
pub const ENGINE_CONTROL_BODY_FIELDS: [&str; 16] = [
    "session_id",
    "priority",
    "kv_transfer_params",
    "cache_salt",
    "request_id",
    "vllm_xargs",
    "rid",
    "extra_key",
    "session_params",
    "custom_labels",
    "routed_dp_rank",
    "data_parallel_rank",
    "disagg_prefill_dp_rank",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
];
