//! `lh-gateway`: the Leaseholder harness in front of one vLLM engine, or of a few GPU nodes.
//!
//! It carries Leaseholder's request contract and nothing a stock proxy would not need:
//! - **auth**: tenants from API-key digests (constant-time match), or from the listener for a private Unix
//!   socket;
//! - **sanitization**: the headers and body fields the engine, or a router in front of it, trusts for
//!   scheduling and accounting (`lh_proto::headers`) are removed from client requests, and the gateway sets its
//!   own;
//! - **session and class**: resolved per PROTOCOL.md and sent as `X-Session-ID` and a priority where the
//!   upstream reads one (`X-Vllm-Priority`, the body's `priority` or `X-SMG-Priority`), with a per-tenant
//!   `cache_salt`;
//! - **admission**: an `lh_core::admission` policy decides when a request goes upstream, and the gateway does
//!   the waiting and the cancellation; under policy leaseholder it also follows the realtime sessions' turns
//!   from lifecycle events and realtime requests;
//! - **relay**: responses pass through byte for byte, with backpressure, client-disconnect cancellation and a
//!   stall timeout;
//! - **lifecycle events** at `POST /v1/events`, deduplicated, then to the admission point and an [`EventSink`];
//! - **placement**, with `[nodes]`: each GPU node has its own admission point and readiness probe, and each
//!   session is placed on one node (by an `lh_core::placement` policy) and stays there, KV never moving between
//!   nodes; `POST /v1/sessions` places a session ahead of its first request;
//! - **metrics** at `GET /metrics` and one structured log line per request.
//!
//! The admission policies are E2's baselines and Leaseholder's own (E2 arm 9); the placement policies are those of
//! E2's two-node points, which have not run yet.

mod admission;
mod auth;
pub mod config;
mod events;
mod exchange;
mod metrics;
mod placement;
mod proxy;
mod realtime;
mod sanitize;
mod server;
mod usage;

pub use config::{Config, ConfigError, ListenAddr};
pub use events::{AcceptedEvent, EventSink, LogSink};
pub use server::Server;
