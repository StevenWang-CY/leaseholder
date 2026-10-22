//! The gateway configuration: one TOML file, validated in full before anything binds. The crate README lists
//! every key; `deploy/cyan/gateway.example.toml` is a complete file.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use http::uri::{Authority, Scheme, Uri};
use lh_proto::SloClass;
use serde::Deserialize;

/// A validated configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// The engine's address (`host:port`), spoken to over plain HTTP/1.1.
    pub upstream: Authority,
    /// How long an idle upstream connection is kept for reuse. It must be shorter than the engine's own
    /// keep-alive, or a request can be sent on a connection the engine is closing.
    pub upstream_idle_timeout: Duration,
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
        let upstream = parse_upstream(&self.upstream)?;

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
            upstream_idle_timeout: seconds(
                "upstream_idle_timeout_s",
                self.upstream_idle_timeout_s,
            )?,
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

impl RawTenant {
    fn validate(self, name: String) -> Result<Tenant, String> {
        let valid_name = !name.is_empty()
            && name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !valid_name {
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
