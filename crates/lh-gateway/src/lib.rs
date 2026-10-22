//! `lh-gateway`: the Leaseholder harness in front of one vLLM engine.
//!
//! It carries Leaseholder's request contract and nothing a stock proxy would not need:
//! - **auth**: tenants from API-key digests (constant-time match), or from the listener for a private Unix
//!   socket;
//! - **sanitization**: the headers and body fields vLLM trusts for scheduling and accounting
//!   (`lh_proto::headers`) are removed from client requests, and the gateway sets its own;
//! - **session and class**: resolved per PROTOCOL.md and sent as `X-Session-ID` and `X-Vllm-Priority`, with a
//!   per-tenant `cache_salt`;
//! - **relay**: responses pass through byte for byte, with backpressure, client-disconnect cancellation and a
//!   stall timeout;
//! - **lifecycle events** at `POST /v1/events`, deduplicated, behind [`EventSink`];
//! - **metrics** at `GET /metrics` and one structured log line per request.
//!
//! Admission, placement and leases arrive with `lh-core`; the gateway is the place they will run.

mod auth;
pub mod config;
mod events;
mod exchange;
mod metrics;
mod proxy;
mod sanitize;
mod server;
mod usage;

pub use config::{Config, ConfigError, ListenAddr};
pub use events::{AcceptedEvent, EventSink, LogSink};
pub use server::Server;
