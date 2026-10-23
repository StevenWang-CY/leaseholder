//! The gateway configuration: one TOML file, validated in full before anything binds. The crate README lists
//! every key; `deploy/cyan/gateway.example.toml` is a complete file.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use http::uri::{Authority, Scheme, Uri};
use lh_core::FROZEN_BOX_POOL_TOKENS;
use lh_core::admission::{Ceilings, Leaseholder, Policy, Trigger};
use lh_core::placement;
use lh_proto::SloClass;
use serde::Deserialize;

/// A validated configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// The engine's address (`host:port`), spoken to over plain HTTP/1.1; with `[nodes]`, the first node's.
    pub upstream: Authority,
    /// `[nodes]` and `[placement]`: several GPU nodes and how sessions are placed on them; `None` for one
    /// `upstream`.
    pub placement: Option<Placement>,
    /// How long an idle upstream connection is kept for reuse. It must be shorter than the engine's own
    /// keep-alive, or a request can be sent on a connection the engine is closing.
    pub upstream_idle_timeout: Duration,
    /// How a request's class reaches the upstream as a priority.
    pub upstream_priority: PrioritySignals,
    /// The admission point; `None` (policy "none") sends every request upstream at once.
    pub admission: Option<Admission>,
    pub listeners: Vec<Listen>,
    pub tenants: Vec<Tenant>,
    /// Largest accepted request body; larger requests get 413.
    pub max_body_bytes: usize,
    /// Frames buffered between the upstream and a client before backpressure reaches the upstream.
    pub stream_buffer_frames: usize,
    /// How long a full stream buffer may stay full before the stream is aborted as stalled.
    pub stall_timeout: Duration,
    /// How long to wait for the upstream's response headers before answering 504.
    pub first_byte_timeout: Duration,
    /// How long a shutdown waits for in-flight requests.
    pub drain_timeout: Duration,
    /// Lifecycle-event ids remembered for deduplication, and for how long.
    pub event_dedup_capacity: usize,
    pub event_dedup_ttl: Duration,
    pub log_format: LogFormat,
    pub log_level: LogLevel,
}

/// One listener.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listen {
    pub addr: ListenAddr,
    /// The tenant that requests without a known API key belong to. Only Unix sockets may name one: their file
    /// mode keeps other users out, while any local user can reach a loopback TCP port.
    pub tenant: Option<String>,
}

/// Where a listener binds: a loopback TCP address or a Unix socket path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ListenAddr {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

/// A tenant: who a request belongs to, what it may ask for, and its cache namespace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tenant {
    pub name: String,
    /// SHA-256 digests of the tenant's API keys. The keys themselves never appear in the configuration.
    pub keys_sha256: Vec<[u8; 32]>,
    /// A trusted tenant may set `X-Session-ID`, choose its class with `X-LH-Class` and post lifecycle events.
    pub trusted: bool,
    pub default_class: SloClass,
    /// The strictest class the tenant may use; a stricter request is clamped to it.
    pub max_class: SloClass,
}

/// `upstream_priority`: where the class's priority goes in each upstream request. Engines and routers read it
/// in different places, and a stock deployment sends none, so each is a choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrioritySignals {
    /// `X-Vllm-Priority`, the class's engine priority (realtime 0, interactive 1, background 2), which vLLM's
    /// priority scheduling reads.
    pub vllm_header: bool,
    /// The same number as the body's `priority`, which SGLang's priority scheduling reads.
    pub body: bool,
    /// `X-SMG-Priority`: `system`, `interactive` or `bulk`, which SMG's priority scheduler reads.
    pub smg_header: bool,
}

/// `[admission]`: the policy that decides when a request goes upstream, and the bounds of its queues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub policy: Policy,
    /// The requests each class's queue may hold; a request that would wait beyond them gets 429.
    pub queue_limit: usize,
    /// How long a request may wait: `queue_timeout_s`, after which it gets 503, or under policy leaseholder,
    /// which refuses no request for its wait, `max_defer_s`, after which it is admitted regardless.
    pub max_wait: Duration,
    /// How long a realtime session stays in a state of its turn that no event moves it on from
    /// (`state_timeout_s`). Only policy leaseholder keeps the sessions' states.
    pub state_timeout: Option<Duration>,
    /// The engine's step budget (`max_num_batched_tokens`), in whose steps policy leaseholder's throttle prices a
    /// background request's prefill; `None` without the throttle.
    pub step_tokens: Option<u64>,
}

/// `[nodes]` and `[placement]`: the GPU nodes, and how sessions are placed on them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub policy: placement::Policy,
    /// The nodes, by name.
    pub nodes: Vec<Node>,
    /// How long a session keeps its node without a request (`idle_s`); its next request is placed again.
    pub idle: Duration,
    /// How often each node's `/health` is probed (`probe_interval_s`). A node takes no new sessions once three
    /// probes in a row fail.
    pub probe_interval: Duration,
}

/// One of `[nodes]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub upstream: Authority,
    /// Whether it serves each class, strictest first: those `classes` names under placement dedicated, and
    /// every class under the others.
    pub serves: [bool; 3],
}

/// One entry of `upstream_priority`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PrioritySignal {
    VllmHeader,
    Body,
    SmgHeader,
}

/// `log_format`: human-readable lines, or one JSON object per line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
}

/// `log_level`: the most verbose level logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Why a configuration was rejected.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Read(PathBuf, std::io::Error),
    /// The file is not TOML of the configuration's shape (an unknown key, a wrong type).
    Parse(PathBuf, toml::de::Error),
    /// A value the configuration's rules reject, such as a tenant no request can reach.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(path, err) => write!(f, "cannot read {}: {err}", path.display()),
            Self::Parse(path, err) => write!(f, "{}: {err}", path.display()),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(_, e) => Some(e),
            Self::Parse(_, e) => Some(e),
            Self::Invalid(_) => None,
        }
    }
}

/// The file as written; a missing key takes its value from [`RawConfig::default`].
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawConfig {
    upstream: String,
    upstream_idle_timeout_s: f64,
    upstream_priority: Vec<PrioritySignal>,
    nodes: BTreeMap<String, RawNode>,
    placement: Option<RawPlacement>,
    admission: RawAdmission,
    listen: Vec<RawListen>,
    tenants: BTreeMap<String, RawTenant>,
    max_body_bytes: usize,
    stream_buffer_frames: usize,
    stall_timeout_s: f64,
    first_byte_timeout_s: f64,
    drain_timeout_s: f64,
    event_dedup_capacity: usize,
    event_dedup_ttl_s: f64,
    log_format: LogFormat,
    log_level: LogLevel,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            upstream: String::new(),
            // vLLM closes idle connections after VLLM_HTTP_TIMEOUT_KEEP_ALIVE, 5 s by default.
            upstream_idle_timeout_s: 4.0,
            upstream_priority: vec![PrioritySignal::VllmHeader],
            nodes: BTreeMap::new(),
            placement: None,
            admission: RawAdmission::default(),
            listen: Vec::new(),
            tenants: BTreeMap::new(),
            max_body_bytes: 32 << 20,
            stream_buffer_frames: 64,
            stall_timeout_s: 30.0,
            first_byte_timeout_s: 600.0,
            drain_timeout_s: 30.0,
            event_dedup_capacity: 100_000,
            event_dedup_ttl_s: 600.0,
            log_format: LogFormat::Text,
            log_level: LogLevel::Info,
        }
    }
}

/// `[admission]` as written. Each key but `policy` belongs to some of the policies, and setting it for another
/// one is an error, so no setting silently does nothing; nor is a key of a mechanism switched off.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawAdmission {
    policy: PolicyName,
    background_max_in_flight: Option<u32>,
    pool_tokens: Option<u64>,
    reserved_tokens: Option<u64>,
    ceilings: Option<RawCeilings>,
    headroom: Option<bool>,
    headroom_tokens: Option<u64>,
    trigger: Option<TriggerName>,
    throttle: Option<bool>,
    critical_prefill_budget_ms: Option<f64>,
    max_num_batched_tokens: Option<u64>,
    degrade: Option<bool>,
    degrade_max_tokens: Option<u32>,
    max_defer_s: Option<f64>,
    state_timeout_s: Option<f64>,
    queue_limit: Option<usize>,
    queue_timeout_s: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCeilings {
    realtime: u32,
    interactive: u32,
    background: u32,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PolicyName {
    #[default]
    None,
    StaticCap,
    TokenReservation,
    FlowControl,
    PreemptResubmit,
    Leaseholder,
}

impl PolicyName {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::StaticCap => "static-cap",
            Self::TokenReservation => "token-reservation",
            Self::FlowControl => "flow-control",
            Self::PreemptResubmit => "preempt-resubmit",
            Self::Leaseholder => "leaseholder",
        }
    }
}

/// `trigger`: what arms policy leaseholder's headroom.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum TriggerName {
    Onset,
    Request,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawListen {
    tcp: Option<SocketAddr>,
    unix: Option<PathBuf>,
    tenant: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawTenant {
    keys_sha256: Vec<String>,
    trusted: bool,
    default_class: SloClass,
    /// Defaults to `default_class`: a tenant gets no stricter class unless the file says so.
    max_class: Option<SloClass>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNode {
    upstream: String,
    /// Placement dedicated only.
    classes: Option<Vec<SloClass>>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawPlacement {
    policy: PlacementName,
    idle_s: Option<f64>,
    probe_interval_s: Option<f64>,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum PlacementName {
    Dedicated,
    #[default]
    LeastLoaded,
    Leaseholder,
}

impl From<PlacementName> for placement::Policy {
    fn from(name: PlacementName) -> Self {
        match name {
            PlacementName::Dedicated => Self::Dedicated,
            PlacementName::LeastLoaded => Self::LeastLoaded,
            PlacementName::Leaseholder => Self::Leaseholder,
        }
    }
}

impl Config {
    /// Reads and validates a configuration file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Read(path.to_owned(), e))?;
        let raw: RawConfig =
            toml::from_str(&text).map_err(|e| ConfigError::Parse(path.to_owned(), e))?;
        raw.validate()
            .map_err(|message| ConfigError::Invalid(format!("{}: {message}", path.display())))
    }

    /// Parses and validates a configuration from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            toml::from_str(text).map_err(|e| ConfigError::Parse(PathBuf::from("<config>"), e))?;
        raw.validate().map_err(ConfigError::Invalid)
    }
}

impl RawConfig {
    fn validate(self) -> Result<Config, String> {
        // One engine is `upstream`; several are `[nodes]`, without it.
        let upstream = match self.nodes.iter().next() {
            None => parse_upstream(&self.upstream)?,
            Some(_) if !self.upstream.is_empty() => {
                return Err(
                    "set upstream for one engine or [nodes.<name>] for several, not both".into(),
                );
            }
            Some((name, node)) => node_upstream(name, &node.upstream)?,
        };
        let upstream_priority = priority_signals(&self.upstream_priority)?;
        let admission = self.admission.validate()?;
        let placement = placement(self.nodes, self.placement, admission.as_ref())?;

        let mut tenants = Vec::with_capacity(self.tenants.len());
        let mut digests = HashSet::new();
        for (name, raw) in self.tenants {
            let tenant = raw.validate(name)?;
            for digest in &tenant.keys_sha256 {
                if !digests.insert(*digest) {
                    return Err(format!(
                        "tenants.{}: an API key digest is listed twice",
                        tenant.name
                    ));
                }
            }
            tenants.push(tenant);
        }
        if tenants.is_empty() {
            return Err("no [tenants.<name>] table".into());
        }

        let mut listeners = Vec::with_capacity(self.listen.len());
        for (i, raw) in self.listen.into_iter().enumerate() {
            let addr = match (raw.tcp, raw.unix) {
                (Some(addr), None) => ListenAddr::Tcp(addr),
                (None, Some(path)) if path.is_absolute() => ListenAddr::Unix(path),
                (None, Some(path)) => {
                    return Err(format!(
                        "listen[{i}]: unix path {} is not absolute",
                        path.display()
                    ));
                }
                _ => return Err(format!("listen[{i}]: set exactly one of tcp or unix")),
            };
            if let Some(tenant) = &raw.tenant {
                if !tenants.iter().any(|t| &t.name == tenant) {
                    return Err(format!("listen[{i}]: unknown tenant {tenant:?}"));
                }
                if matches!(addr, ListenAddr::Tcp(_)) {
                    return Err(format!(
                        "listen[{i}]: a TCP listener cannot name a tenant, because other users of the host can \
                         reach it; give the tenant an API key instead"
                    ));
                }
            }
            listeners.push(Listen {
                addr,
                tenant: raw.tenant,
            });
        }
        if listeners.is_empty() {
            return Err("no [[listen]] entry".into());
        }
        for tenant in &tenants {
            let reachable = !tenant.keys_sha256.is_empty()
                || listeners
                    .iter()
                    .any(|l| l.tenant.as_ref() == Some(&tenant.name));
            if !reachable {
                return Err(format!(
                    "tenants.{}: no API key and no listener names it, so no request can reach it",
                    tenant.name
                ));
            }
        }

        if self.max_body_bytes == 0 {
            return Err("max_body_bytes must be positive".into());
        }
        if self.stream_buffer_frames == 0 {
            return Err("stream_buffer_frames must be positive".into());
        }
        if self.event_dedup_capacity == 0 {
            return Err("event_dedup_capacity must be positive".into());
        }
        Ok(Config {
            upstream,
            placement,
            upstream_idle_timeout: seconds(
                "upstream_idle_timeout_s",
                self.upstream_idle_timeout_s,
            )?,
            upstream_priority,
            admission,
            listeners,
            tenants,
            max_body_bytes: self.max_body_bytes,
            stream_buffer_frames: self.stream_buffer_frames,
            stall_timeout: seconds("stall_timeout_s", self.stall_timeout_s)?,
            first_byte_timeout: seconds("first_byte_timeout_s", self.first_byte_timeout_s)?,
            drain_timeout: seconds("drain_timeout_s", self.drain_timeout_s)?,
            event_dedup_capacity: self.event_dedup_capacity,
            event_dedup_ttl: seconds("event_dedup_ttl_s", self.event_dedup_ttl_s)?,
            log_format: self.log_format,
            log_level: self.log_level,
        })
    }
}

impl RawAdmission {
    fn validate(self) -> Result<Option<Admission>, String> {
        let name = self.policy.as_str();
        let reads: &[&str] = match self.policy {
            PolicyName::None => &[],
            PolicyName::StaticCap => {
                &["background_max_in_flight", "queue_limit", "queue_timeout_s"]
            }
            PolicyName::TokenReservation => &[
                "pool_tokens",
                "reserved_tokens",
                "queue_limit",
                "queue_timeout_s",
            ],
            PolicyName::FlowControl => &["ceilings", "queue_limit", "queue_timeout_s"],
            PolicyName::PreemptResubmit => &["queue_limit", "queue_timeout_s"],
            PolicyName::Leaseholder => &[
                "pool_tokens",
                "headroom",
                "headroom_tokens",
                "trigger",
                "throttle",
                "critical_prefill_budget_ms",
                "max_num_batched_tokens",
                "degrade",
                "degrade_max_tokens",
                "max_defer_s",
                "state_timeout_s",
                "queue_limit",
            ],
        };
        let set = [
            (
                "background_max_in_flight",
                self.background_max_in_flight.is_some(),
            ),
            ("pool_tokens", self.pool_tokens.is_some()),
            ("reserved_tokens", self.reserved_tokens.is_some()),
            ("ceilings", self.ceilings.is_some()),
            ("headroom", self.headroom.is_some()),
            ("headroom_tokens", self.headroom_tokens.is_some()),
            ("trigger", self.trigger.is_some()),
            ("throttle", self.throttle.is_some()),
            (
                "critical_prefill_budget_ms",
                self.critical_prefill_budget_ms.is_some(),
            ),
            (
                "max_num_batched_tokens",
                self.max_num_batched_tokens.is_some(),
            ),
            ("degrade", self.degrade.is_some()),
            ("degrade_max_tokens", self.degrade_max_tokens.is_some()),
            ("max_defer_s", self.max_defer_s.is_some()),
            ("state_timeout_s", self.state_timeout_s.is_some()),
            ("queue_limit", self.queue_limit.is_some()),
            ("queue_timeout_s", self.queue_timeout_s.is_some()),
        ];
        if let Some((key, _)) = set.iter().find(|(key, set)| *set && !reads.contains(key)) {
            return Err(format!("admission.{key} does not apply to policy {name}"));
        }
        // SMG's bulk class queues 1,024 requests.
        let queue_limit = self.queue_limit.unwrap_or(1024);
        if let PolicyName::Leaseholder = self.policy {
            return self.leaseholder(queue_limit).map(Some);
        }
        let needs = |key: &str| format!("admission: policy {name} needs {key}");
        let policy = match self.policy {
            PolicyName::None => return Ok(None),
            PolicyName::StaticCap => {
                let cap = self
                    .background_max_in_flight
                    .ok_or_else(|| needs("background_max_in_flight"))?;
                if cap == 0 {
                    return Err("admission.background_max_in_flight must be positive".into());
                }
                Policy::StaticCap {
                    background_max_in_flight: cap,
                }
            }
            PolicyName::TokenReservation => {
                let pool_tokens = self.pool_tokens.unwrap_or(FROZEN_BOX_POOL_TOKENS);
                let reserved_tokens = self
                    .reserved_tokens
                    .ok_or_else(|| needs("reserved_tokens"))?;
                if reserved_tokens >= pool_tokens {
                    return Err(format!(
                        "admission.reserved_tokens ({reserved_tokens}) must be less than pool_tokens \
                         ({pool_tokens})"
                    ));
                }
                Policy::TokenReservation {
                    pool_tokens,
                    reserved_tokens,
                }
            }
            PolicyName::FlowControl => {
                let RawCeilings {
                    realtime,
                    interactive,
                    background,
                } = self.ceilings.ok_or_else(|| needs("ceilings"))?;
                if background == 0 {
                    return Err("admission.ceilings must be positive".into());
                }
                if realtime < interactive || interactive < background {
                    return Err(
                        "admission.ceilings must not increase from realtime to background: the lower \
                         classes are held back first"
                            .into(),
                    );
                }
                Policy::FlowControl {
                    ceilings: Ceilings {
                        realtime,
                        interactive,
                        background,
                    },
                }
            }
            PolicyName::PreemptResubmit => Policy::PreemptResubmit,
            PolicyName::Leaseholder => unreachable!("validated above"),
        };
        Ok(Some(Admission {
            policy,
            queue_limit,
            // A minute ends a wait well before clients give up on their own (AIPerf's replay after 300 s,
            // OpenAI's SDK after 600 s).
            max_wait: seconds(
                "admission.queue_timeout_s",
                self.queue_timeout_s.unwrap_or(60.0),
            )?,
            state_timeout: None,
            step_tokens: None,
        }))
    }

    /// Policy leaseholder. Each mechanism is on unless switched off, needs its own keys while it is on, and
    /// takes none while it is off.
    fn leaseholder(self, queue_limit: usize) -> Result<Admission, String> {
        let headroom = self.headroom.unwrap_or(true);
        let throttle = self.throttle.unwrap_or(true);
        let degrade = self.degrade.unwrap_or(true);
        let sizing = headroom || degrade;
        for (key, set, on, off) in [
            (
                "trigger",
                self.trigger.is_some(),
                headroom,
                "headroom = false",
            ),
            (
                "critical_prefill_budget_ms",
                self.critical_prefill_budget_ms.is_some(),
                throttle,
                "throttle = false",
            ),
            (
                "max_num_batched_tokens",
                self.max_num_batched_tokens.is_some(),
                throttle,
                "throttle = false",
            ),
            (
                "degrade_max_tokens",
                self.degrade_max_tokens.is_some(),
                degrade,
                "degrade = false",
            ),
            (
                "headroom_tokens",
                self.headroom_tokens.is_some(),
                sizing,
                "headroom = false and degrade = false",
            ),
        ] {
            if set && !on {
                return Err(format!("admission.{key} does not apply with {off}"));
            }
        }
        let needs = |key: &str, off: &str| {
            format!("admission: policy leaseholder needs {key}, unless {off}")
        };
        let pool_tokens = self.pool_tokens.unwrap_or(FROZEN_BOX_POOL_TOKENS);
        let headroom_tokens = match self.headroom_tokens {
            Some(tokens) if (1..pool_tokens).contains(&tokens) => tokens,
            Some(tokens) => {
                return Err(format!(
                    "admission.headroom_tokens ({tokens}) must be positive and less than pool_tokens \
                     ({pool_tokens})"
                ));
            }
            None if sizing => {
                return Err(needs(
                    "headroom_tokens",
                    "headroom = false and degrade = false",
                ));
            }
            None => 0,
        };
        let critical_prefill_budget = match self.critical_prefill_budget_ms {
            Some(ms) => Some(Duration::try_from_secs_f64(ms / 1e3).map_err(|_| {
                "admission.critical_prefill_budget_ms must be a number of milliseconds, 0 or more"
            })?),
            None if throttle => {
                return Err(needs("critical_prefill_budget_ms", "throttle = false"));
            }
            None => None,
        };
        // The frozen box's `--max-num-batched-tokens`.
        let step_tokens = match self.max_num_batched_tokens {
            Some(0) => return Err("admission.max_num_batched_tokens must be positive".into()),
            tokens => throttle.then(|| tokens.unwrap_or(512)),
        };
        let degrade_max_tokens = match self.degrade_max_tokens {
            Some(0) => return Err("admission.degrade_max_tokens must be positive".into()),
            None if degrade => return Err(needs("degrade_max_tokens", "degrade = false")),
            cap => cap,
        };
        let trigger = match self.trigger {
            None | Some(TriggerName::Onset) => Trigger::Onset,
            Some(TriggerName::Request) => Trigger::Request,
        };
        let policy = Policy::Leaseholder(Leaseholder {
            pool_tokens,
            headroom_tokens,
            headroom: headroom.then_some(trigger),
            critical_prefill_budget,
            degrade_max_tokens,
        });
        // Longer than an E2 caller's turn is armed (about 4 s from speech onset to the reply's first audio), so
        // aging admits only what a stuck or unbroken arming holds back.
        let max_defer_s = self.max_defer_s.unwrap_or(10.0);
        // Longer than any state of an E2 caller's turn lasts (lines of 2 to 3 s, replies of a few seconds of
        // audio), so only a lost event ends a state by timeout.
        let state_timeout_s = self.state_timeout_s.unwrap_or(10.0);
        Ok(Admission {
            policy,
            queue_limit,
            max_wait: seconds("admission.max_defer_s", max_defer_s)?,
            state_timeout: Some(seconds("admission.state_timeout_s", state_timeout_s)?),
            step_tokens,
        })
    }
}

impl RawTenant {
    fn validate(self, name: String) -> Result<Tenant, String> {
        if !valid_name(&name) {
            return Err(format!(
                "tenant name {name:?}: use 1 to 64 characters from [A-Za-z0-9._-]"
            ));
        }
        let max_class = self.max_class.unwrap_or(self.default_class);
        // Classes order from strictest (realtime) to most lenient (background).
        if self.default_class < max_class {
            return Err(format!(
                "tenants.{name}: default_class {} is stricter than max_class {max_class}",
                self.default_class
            ));
        }
        let keys_sha256 = self
            .keys_sha256
            .iter()
            .map(|hex| {
                parse_digest(hex).ok_or_else(|| {
                    format!("tenants.{name}: keys_sha256 entries must be 64 hex characters")
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Tenant {
            name,
            keys_sha256,
            trusted: self.trusted,
            default_class: self.default_class,
            max_class,
        })
    }
}

/// `[nodes]` and `[placement]`. Without nodes there is nothing to place; with them, a node's `classes` belong to
/// placement dedicated alone, and setting them for another policy is an error, as under `[admission]`.
fn placement(
    nodes: BTreeMap<String, RawNode>,
    raw: Option<RawPlacement>,
    admission: Option<&Admission>,
) -> Result<Option<Placement>, String> {
    if nodes.is_empty() {
        return match raw {
            None => Ok(None),
            Some(_) => Err("[placement] needs [nodes.<name>]: one engine is upstream".into()),
        };
    }
    if nodes.len() == 1 {
        return Err("[nodes] needs two nodes or more: one engine is upstream".into());
    }
    let raw = raw.unwrap_or_default();
    let policy = placement::Policy::from(raw.policy);
    let dedicated = policy == placement::Policy::Dedicated;
    let leaseholder = matches!(
        admission,
        Some(Admission {
            policy: Policy::Leaseholder(_),
            ..
        })
    );
    if policy == placement::Policy::Leaseholder && !leaseholder {
        return Err(
            "placement leaseholder reads policy leaseholder's state: set [admission] policy = \
             \"leaseholder\""
                .into(),
        );
    }
    let mut validated: Vec<Node> = Vec::with_capacity(nodes.len());
    for (name, node) in nodes {
        if !valid_name(&name) {
            return Err(format!(
                "node name {name:?}: use 1 to 64 characters from [A-Za-z0-9._-]"
            ));
        }
        let upstream = node_upstream(&name, &node.upstream)?;
        if let Some(other) = validated.iter().find(|other| other.upstream == upstream) {
            return Err(format!(
                "nodes.{} and nodes.{name} have the same upstream",
                other.name
            ));
        }
        let serves = match (node.classes, dedicated) {
            (None, false) => [true; 3],
            (Some(_), false) => {
                return Err(format!(
                    "nodes.{name}.classes applies only to placement dedicated"
                ));
            }
            (None, true) => {
                return Err(format!(
                    "nodes.{name}: placement dedicated needs the classes it serves"
                ));
            }
            (Some(classes), true) => {
                let mut serves = [false; 3];
                for class in classes {
                    if std::mem::replace(&mut serves[class as usize], true) {
                        return Err(format!("nodes.{name}.classes lists {class} twice"));
                    }
                }
                if serves == [false; 3] {
                    return Err(format!("nodes.{name}.classes is empty"));
                }
                serves
            }
        };
        validated.push(Node {
            name,
            upstream,
            serves,
        });
    }
    let unserved = SloClass::ALL
        .into_iter()
        .find(|&class| !validated.iter().any(|node| node.serves[class as usize]));
    if let Some(class) = unserved {
        return Err(format!(
            "placement dedicated: no node serves {class} requests"
        ));
    }
    Ok(Some(Placement {
        policy,
        nodes: validated,
        // Anthropic's prompt cache keeps a prefix for 5 minutes after its last use; E2's agents think for at most
        // 2 minutes, and a call's turns are seconds apart.
        idle: seconds("placement.idle_s", raw.idle_s.unwrap_or(300.0))?,
        // The window of the gateway's own /ready.
        probe_interval: seconds(
            "placement.probe_interval_s",
            raw.probe_interval_s.unwrap_or(1.0),
        )?,
    }))
}

/// A tenant or node name: 1 to 64 characters from `[A-Za-z0-9._-]`, which metric labels, socket files and unit
/// names can carry.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn node_upstream(name: &str, raw: &str) -> Result<Authority, String> {
    parse_upstream(raw).map_err(|e| format!("nodes.{name}: {e}"))
}

fn parse_upstream(raw: &str) -> Result<Authority, String> {
    let uri: Uri = raw.parse().map_err(|e| format!("upstream {raw:?}: {e}"))?;
    let path_is_root = matches!(uri.path(), "" | "/") && uri.query().is_none();
    match (uri.scheme(), uri.authority()) {
        (Some(scheme), Some(authority)) if *scheme == Scheme::HTTP && path_is_root => {
            Ok(authority.clone())
        }
        _ => Err(format!(
            "upstream {raw:?}: expected http://host:port with no path"
        )),
    }
}

fn priority_signals(listed: &[PrioritySignal]) -> Result<PrioritySignals, String> {
    let mut signals = PrioritySignals::default();
    for signal in listed {
        let (set, name) = match signal {
            PrioritySignal::VllmHeader => (&mut signals.vllm_header, "vllm-header"),
            PrioritySignal::Body => (&mut signals.body, "body"),
            PrioritySignal::SmgHeader => (&mut signals.smg_header, "smg-header"),
        };
        if std::mem::replace(set, true) {
            return Err(format!("upstream_priority lists {name} twice"));
        }
    }
    Ok(signals)
}

fn parse_digest(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(digest)
}

fn seconds(key: &str, value: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(value)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| format!("{key} must be a positive number of seconds"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SHA-256 of "foo".
    const KEY: &str = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae";

    /// A configuration with one TCP listener, plus top-level keys and tables.
    fn config(top: &str, tables: &str) -> Result<Config, String> {
        Config::from_toml(&format!(
            "upstream = \"http://127.0.0.1:18000\"\n{top}\n[[listen]]\ntcp = \"127.0.0.1:0\"\n{tables}"
        ))
        .map_err(|e| e.to_string())
    }

    #[test]
    fn defaults_follow_the_spec() {
        let config = config("", &format!("[tenants.agents]\nkeys_sha256 = [\"{KEY}\"]")).unwrap();
        assert_eq!(config.upstream.as_str(), "127.0.0.1:18000");
        assert_eq!(config.upstream_idle_timeout, Duration::from_secs(4));
        let vllm = PrioritySignals {
            vllm_header: true,
            ..PrioritySignals::default()
        };
        assert_eq!(config.upstream_priority, vllm);
        assert_eq!(
            (config.max_body_bytes, config.stream_buffer_frames),
            (32 << 20, 64)
        );
        assert_eq!(
            [
                config.stall_timeout,
                config.first_byte_timeout,
                config.drain_timeout
            ],
            [30, 600, 30].map(Duration::from_secs)
        );
        assert_eq!(
            (config.event_dedup_capacity, config.event_dedup_ttl),
            (100_000, Duration::from_secs(600))
        );
        let tenant = &config.tenants[0];
        assert_eq!(
            (tenant.trusted, tenant.default_class, tenant.max_class),
            (false, SloClass::Interactive, SloClass::Interactive)
        );
        assert_eq!(tenant.keys_sha256[0][..2], [0x2c, 0x26]);
    }

    #[test]
    fn mistakes_are_rejected_with_their_cause() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        for (top, tables, cause) in [
            ("", "", "no [tenants"),
            ("", "[tenants.a]", "no API key and no listener"),
            ("", "[tenants.a]\nkeys_sha256 = [\"abc\"]", "64 hex"),
            (
                "",
                &format!("{a}\n[tenants.b]\nkeys_sha256 = [\"{KEY}\"]"),
                "listed twice",
            ),
            (
                "",
                &format!("{a}\ndefault_class = \"realtime\"\nmax_class = \"interactive\""),
                "stricter than",
            ),
            ("stall_timeout = 5", &a, "unknown field"),
            (
                "upstream_priority = [\"body\", \"body\"]",
                &a,
                "lists body twice",
            ),
            ("upstream_priority = [\"sglang\"]", &a, "unknown variant"),
            (
                "stall_timeout_s = 0",
                &a,
                "stall_timeout_s must be a positive",
            ),
            (
                "",
                "[[listen]]\ntcp = \"127.0.0.1:1\"\ntenant = \"a\"\n[tenants.a]",
                "TCP listener cannot",
            ),
            (
                "",
                "[[listen]]\nunix = \"relative.sock\"\ntenant = \"a\"\n[tenants.a]",
                "not absolute",
            ),
        ] {
            let err = config(top, tables).unwrap_err();
            assert!(err.contains(cause), "{err}");
        }
        let https = format!(
            "upstream = \"https://engine:8000/v1\"\n[[listen]]\ntcp = \"127.0.0.1:0\"\n{a}"
        );
        let err = Config::from_toml(&https).unwrap_err().to_string();
        assert!(err.contains("http://host:port"), "{err}");
    }

    #[test]
    fn upstream_priority_is_a_set_of_signals() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let none = config("upstream_priority = []", &a).unwrap();
        assert_eq!(none.upstream_priority, PrioritySignals::default());
        let all = config(
            "upstream_priority = [\"smg-header\", \"body\", \"vllm-header\"]",
            &a,
        )
        .unwrap();
        assert_eq!(
            all.upstream_priority,
            PrioritySignals {
                vllm_header: true,
                body: true,
                smg_header: true
            }
        );
    }

    #[test]
    fn admission_policies_take_their_own_keys_and_defaults() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let admission = |keys: &str| config(&format!("[admission]\n{keys}"), &a);
        assert_eq!(config("", &a).unwrap().admission, None);
        assert_eq!(admission("policy = \"none\"").unwrap().admission, None);
        let cap = admission("policy = \"static-cap\"\nbackground_max_in_flight = 4").unwrap();
        assert_eq!(
            cap.admission,
            Some(Admission {
                policy: Policy::StaticCap {
                    background_max_in_flight: 4
                },
                queue_limit: 1024,
                max_wait: Duration::from_secs(60),
                state_timeout: None,
                step_tokens: None,
            })
        );
        let reservation = admission(
            "policy = \"token-reservation\"\nreserved_tokens = 2048\nqueue_limit = 8\nqueue_timeout_s = 290",
        )
        .unwrap()
        .admission
        .unwrap();
        assert_eq!(
            reservation.policy,
            Policy::TokenReservation {
                pool_tokens: FROZEN_BOX_POOL_TOKENS,
                reserved_tokens: 2048
            }
        );
        assert_eq!(
            (reservation.queue_limit, reservation.max_wait),
            (8, Duration::from_secs(290))
        );
        let flow = admission(
            "policy = \"flow-control\"\nceilings = { realtime = 16, interactive = 12, background = 8 }",
        )
        .unwrap()
        .admission
        .unwrap();
        let ceilings = Ceilings {
            realtime: 16,
            interactive: 12,
            background: 8,
        };
        assert_eq!(flow.policy, Policy::FlowControl { ceilings });
        let preempt = admission("policy = \"preempt-resubmit\"")
            .unwrap()
            .admission;
        assert_eq!(preempt.map(|a| a.policy), Some(Policy::PreemptResubmit));

        for (keys, cause) in [
            ("policy = \"static_cap\"", "unknown variant"),
            (
                "policy = \"none\"\nqueue_limit = 4",
                "queue_limit does not apply to policy none",
            ),
            (
                "policy = \"static-cap\"",
                "static-cap needs background_max_in_flight",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 0",
                "must be positive",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 2\nreserved_tokens = 1",
                "reserved_tokens does not apply to policy static-cap",
            ),
            (
                "policy = \"static-cap\"\nbackground_max_in_flight = 2\nqueue_timeout_s = 0",
                "queue_timeout_s must be a positive",
            ),
            ("policy = \"token-reservation\"", "needs reserved_tokens"),
            (
                "policy = \"token-reservation\"\nreserved_tokens = 40000",
                "must be less than pool_tokens (36400)",
            ),
            ("polcy = \"none\"", "unknown field"),
            ("policy = \"flow-control\"", "flow-control needs ceilings"),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 8, background = 4 }",
                "missing field `interactive`",
            ),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 4, interactive = 8, background = 4 }",
                "must not increase",
            ),
            (
                "policy = \"flow-control\"\nceilings = { realtime = 4, interactive = 4, background = 0 }",
                "must be positive",
            ),
            (
                "policy = \"preempt-resubmit\"\nbackground_max_in_flight = 4",
                "does not apply to policy preempt-resubmit",
            ),
        ] {
            let err = admission(keys).unwrap_err();
            assert!(err.contains(cause), "{keys}: {err}");
        }
    }

    #[test]
    fn policy_leaseholder_switches_each_mechanism_with_its_own_keys() {
        let a = format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let admission = |keys: &str| {
            config(
                &format!("[admission]\npolicy = \"leaseholder\"\n{keys}"),
                &a,
            )
            .map(|config| config.admission.unwrap())
        };
        let all =
            "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 24";
        let arm9 = admission(all).unwrap();
        let full = Leaseholder {
            pool_tokens: FROZEN_BOX_POOL_TOKENS,
            headroom_tokens: 1024,
            headroom: Some(Trigger::Onset),
            critical_prefill_budget: Some(Duration::from_millis(250)),
            degrade_max_tokens: Some(24),
        };
        assert_eq!(
            arm9,
            Admission {
                policy: Policy::Leaseholder(full),
                queue_limit: 1024,
                max_wait: Duration::from_secs(10),
                state_timeout: Some(Duration::from_secs(10)),
                step_tokens: Some(512),
            }
        );
        let tuned = admission(&format!(
            "{all}\ntrigger = \"request\"\npool_tokens = 20000\nmax_defer_s = 4.5\nstate_timeout_s = 30\n\
             queue_limit = 16\nmax_num_batched_tokens = 1024"
        ))
        .unwrap();
        let request = Leaseholder {
            pool_tokens: 20_000,
            headroom: Some(Trigger::Request),
            ..full
        };
        assert_eq!(tuned.policy, Policy::Leaseholder(request));
        assert_eq!(
            (tuned.queue_limit, tuned.max_wait, tuned.state_timeout),
            (
                16,
                Duration::from_secs_f64(4.5),
                Some(Duration::from_secs(30))
            )
        );
        assert_eq!(tuned.step_tokens, Some(1024));

        // The ablations: each mechanism off, without its keys.
        let ablation = |keys: &str| match admission(keys).unwrap().policy {
            Policy::Leaseholder(lh) => lh,
            other => panic!("{other:?}"),
        };
        let no_throttle = "headroom_tokens = 1024\ndegrade_max_tokens = 24\nthrottle = false";
        assert_eq!(
            ablation(no_throttle),
            Leaseholder {
                critical_prefill_budget: None,
                ..full
            }
        );
        assert_eq!(admission(no_throttle).unwrap().step_tokens, None);
        let no_headroom = "headroom = false\nheadroom_tokens = 1024\ncritical_prefill_budget_ms = 250\n\
                           degrade_max_tokens = 24";
        assert_eq!(
            ablation(no_headroom),
            Leaseholder {
                headroom: None,
                ..full
            }
        );
        let no_degrade =
            "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade = false";
        assert_eq!(
            ablation(no_degrade),
            Leaseholder {
                degrade_max_tokens: None,
                ..full
            }
        );
        let throttle_alone = "headroom = false\ndegrade = false\ncritical_prefill_budget_ms = 0";
        assert_eq!(
            ablation(throttle_alone),
            Leaseholder {
                headroom_tokens: 0,
                headroom: None,
                critical_prefill_budget: Some(Duration::ZERO),
                degrade_max_tokens: None,
                ..full
            }
        );

        for (keys, cause) in [
            (
                "",
                "leaseholder needs headroom_tokens, unless headroom = false and degrade = false",
            ),
            (
                "headroom_tokens = 1024\ndegrade_max_tokens = 24",
                "needs critical_prefill_budget_ms, unless throttle = false",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250",
                "needs degrade_max_tokens, unless degrade = false",
            ),
            (
                &format!("{all}\nthrottle = false"),
                "critical_prefill_budget_ms does not apply with throttle = false",
            ),
            (
                &format!("{all}\ndegrade = false"),
                "degrade_max_tokens does not apply with degrade = false",
            ),
            (
                "headroom_tokens = 1024\ndegrade_max_tokens = 24\nthrottle = false\nmax_num_batched_tokens = 512",
                "max_num_batched_tokens does not apply with throttle = false",
            ),
            (
                &format!("{all}\nmax_num_batched_tokens = 0"),
                "max_num_batched_tokens must be positive",
            ),
            (
                &format!("{all}\nheadroom = false\ntrigger = \"onset\""),
                "trigger does not apply with headroom = false",
            ),
            (
                "headroom = false\ndegrade = false\nheadroom_tokens = 1\ncritical_prefill_budget_ms = 1",
                "headroom_tokens does not apply with headroom = false and degrade = false",
            ),
            (
                &format!("{all}\nqueue_timeout_s = 60"),
                "queue_timeout_s does not apply to policy leaseholder",
            ),
            (
                "headroom_tokens = 36400\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 24",
                "must be positive and less than pool_tokens (36400)",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = -1\ndegrade_max_tokens = 24",
                "a number of milliseconds, 0 or more",
            ),
            (
                "headroom_tokens = 1024\ncritical_prefill_budget_ms = 250\ndegrade_max_tokens = 0",
                "degrade_max_tokens must be positive",
            ),
            (
                &format!("{all}\nmax_defer_s = 0"),
                "max_defer_s must be a positive",
            ),
            (&format!("{all}\ntrigger = \"speech\""), "unknown variant"),
        ] {
            let err = admission(keys).unwrap_err();
            assert!(err.contains(cause), "{keys}: {err}");
        }
        let err = config(
            "[admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 1\nmax_defer_s = 1",
            &a,
        )
        .unwrap_err();
        assert!(
            err.contains("max_defer_s does not apply to policy static-cap"),
            "{err}"
        );
    }

    /// A configuration with two GPU nodes, one TCP listener, a tenant, and more tables.
    fn nodes(tables: &str) -> Result<Config, String> {
        Config::from_toml(&format!(
            "[nodes.phastform]\nupstream = \"http://127.0.0.1:18000\"\n\
             [nodes.gray]\nupstream = \"http://127.0.0.1:28000\"\n\
             [[listen]]\ntcp = \"127.0.0.1:0\"\n[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\n{tables}"
        ))
        .map_err(|e| e.to_string())
    }

    #[test]
    fn nodes_and_placement_take_their_own_keys_and_defaults() {
        assert_eq!(
            config("", &format!("[tenants.a]\nkeys_sha256 = [\"{KEY}\"]"))
                .unwrap()
                .placement,
            None
        );
        let pool = nodes("").unwrap();
        let placement = pool.placement.unwrap();
        assert_eq!(placement.policy, placement::Policy::LeastLoaded);
        assert_eq!(
            (placement.idle, placement.probe_interval),
            (Duration::from_secs(300), Duration::from_secs(1))
        );
        // By name, every class served; the configuration's upstream is the first node's.
        let names: Vec<_> = placement
            .nodes
            .iter()
            .map(|n| (n.name.as_str(), n.upstream.as_str(), n.serves))
            .collect();
        assert_eq!(
            names,
            [
                ("gray", "127.0.0.1:28000", [true; 3]),
                ("phastform", "127.0.0.1:18000", [true; 3])
            ]
        );
        assert_eq!(pool.upstream.as_str(), "127.0.0.1:28000");

        let dedicated = Config::from_toml(&format!(
            "[nodes.voice]\nupstream = \"http://127.0.0.1:18000\"\nclasses = [\"realtime\"]\n\
             [nodes.agents]\nupstream = \"http://127.0.0.1:28000\"\nclasses = [\"background\", \"interactive\"]\n\
             [placement]\npolicy = \"dedicated\"\nidle_s = 60\nprobe_interval_s = 0.25\n\
             [[listen]]\ntcp = \"127.0.0.1:0\"\n[tenants.a]\nkeys_sha256 = [\"{KEY}\"]\n"
        ))
        .unwrap()
        .placement
        .unwrap();
        assert_eq!(dedicated.policy, placement::Policy::Dedicated);
        let serves: Vec<_> = dedicated
            .nodes
            .iter()
            .map(|n| (n.name.as_str(), n.serves))
            .collect();
        assert_eq!(
            serves,
            [
                ("agents", [false, true, true]),
                ("voice", [true, false, false])
            ]
        );
        assert_eq!(
            (dedicated.idle, dedicated.probe_interval),
            (Duration::from_secs(60), Duration::from_millis(250))
        );

        let arm9 = "[admission]\npolicy = \"leaseholder\"\nheadroom_tokens = 1024\n\
                    critical_prefill_budget_ms = 150\ndegrade_max_tokens = 16";
        let leaseholder = nodes(&format!("[placement]\npolicy = \"leaseholder\"\n{arm9}")).unwrap();
        assert_eq!(
            leaseholder.placement.unwrap().policy,
            placement::Policy::Leaseholder
        );
    }

    #[test]
    fn placement_mistakes_are_rejected_with_their_cause() {
        let a =
            format!("[[listen]]\ntcp = \"127.0.0.1:0\"\n[tenants.a]\nkeys_sha256 = [\"{KEY}\"]");
        let one = "[nodes.phastform]\nupstream = \"http://127.0.0.1:18000\"";
        let dedicated = "[placement]\npolicy = \"dedicated\"";
        for (text, cause) in [
            (
                format!("upstream = \"http://127.0.0.1:18000\"\n{one}\n{a}"),
                "set upstream for one engine or [nodes.<name>] for several, not both",
            ),
            (format!("{one}\n{a}"), "[nodes] needs two nodes or more"),
            (
                format!(
                    "upstream = \"http://127.0.0.1:18000\"\n[placement]\npolicy = \"dedicated\"\n{a}"
                ),
                "[placement] needs [nodes.<name>]",
            ),
        ] {
            let err = Config::from_toml(&text).unwrap_err().to_string();
            assert!(err.contains(cause), "{text}: {err}");
        }
        for (tables, cause) in [
            ("[placement]\npolicy = \"round-robin\"", "unknown variant"),
            ("[placement]\nidle = 60", "unknown field"),
            (
                "[placement]\nidle_s = 0",
                "placement.idle_s must be a positive",
            ),
            (
                "[placement]\nprobe_interval_s = -1",
                "placement.probe_interval_s must be a positive",
            ),
            ("[nodes.gray.extra]\n", "unknown field"),
            (
                "[nodes.\"a b\"]\nupstream = \"http://127.0.0.1:38000\"",
                "node name \"a b\"",
            ),
            (
                "[nodes.three]\nupstream = \"http://127.0.0.1:18000\"",
                "nodes.phastform and nodes.three have the same upstream",
            ),
            (
                "[nodes.three]\nupstream = \"http://engine:8000/v1\"",
                "nodes.three: upstream \"http://engine:8000/v1\"",
            ),
            ("[nodes.three]", "missing field `upstream`"),
            (
                "[nodes.three]\nupstream = \"http://127.0.0.1:38000\"\nclasses = [\"realtime\"]",
                "nodes.three.classes applies only to placement dedicated",
            ),
            (
                dedicated,
                "nodes.gray: placement dedicated needs the classes it serves",
            ),
            (
                "[placement]\npolicy = \"leaseholder\"",
                "placement leaseholder reads policy leaseholder's state",
            ),
            (
                "[placement]\npolicy = \"leaseholder\"\n\
                 [admission]\npolicy = \"static-cap\"\nbackground_max_in_flight = 4",
                "placement leaseholder reads policy leaseholder's state",
            ),
        ] {
            let err = nodes(tables).unwrap_err();
            assert!(err.contains(cause), "{tables}: {err}");
        }
        let classes = |gray: &str, phastform: &str| {
            Config::from_toml(&format!(
                "[nodes.phastform]\nupstream = \"http://127.0.0.1:18000\"\nclasses = {phastform}\n\
                 [nodes.gray]\nupstream = \"http://127.0.0.1:28000\"\nclasses = {gray}\n{dedicated}\n{a}"
            ))
            .map_err(|e| e.to_string())
        };
        assert!(classes("[\"background\", \"interactive\"]", "[\"realtime\"]").is_ok());
        for (gray, phastform, cause) in [
            (
                "[\"background\"]",
                "[\"realtime\"]",
                "no node serves interactive requests",
            ),
            (
                "[\"background\", \"background\"]",
                "[\"realtime\"]",
                "nodes.gray.classes lists background twice",
            ),
            (
                "[]",
                "[\"realtime\", \"interactive\", \"background\"]",
                "nodes.gray.classes is empty",
            ),
            ("[\"bulk\"]", "[\"realtime\"]", "unknown variant"),
        ] {
            let err = classes(gray, phastform).unwrap_err();
            assert!(err.contains(cause), "{gray} {phastform}: {err}");
        }
    }

    #[test]
    fn the_deploy_example_is_valid_once_its_digests_are_filled_in() {
        let example = include_str!("../../../deploy/cyan/gateway.example.toml");
        let err = Config::from_toml(example).unwrap_err().to_string();
        assert!(err.contains("64 hex"), "placeholders are refused: {err}");

        // SHA-256 of "bar".
        let other = "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9";
        let filled = example
            .replace("<sha256 of the agents key>", KEY)
            .replace("<sha256 of the bench key>", other);
        let config = Config::from_toml(&filled).unwrap();
        let unix = ListenAddr::Unix("/run/user/1007/lh/gateway.sock".into());
        assert_eq!(config.listeners[1].addr, unix);
        assert_eq!(config.listeners[1].tenant.as_deref(), Some("voice"));
        let classes: Vec<_> = config
            .tenants
            .iter()
            .map(|t| (t.name.as_str(), t.trusted, t.default_class, t.max_class))
            .collect();
        use SloClass::*;
        assert_eq!(
            classes,
            [
                ("agents", true, Background, Interactive),
                ("bench", false, Interactive, Interactive),
                ("voice", true, Realtime, Realtime),
            ]
        );
    }
}
