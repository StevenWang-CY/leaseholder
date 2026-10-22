//! Lifecycle events: what tells the control plane when a session will next need its KV.
//!
//! One schema serves every workload. A tool return and a user's speech onset are two triggers of the same
//! prefetch-and-headroom mechanism. Events arrive at `POST /v1/events`, idempotent by `event_id`, or the
//! control plane infers them from the traffic it proxies.
//!
//! On the wire an event is one flat JSON object; `type` selects the variant and its fields:
//!
//! ```json
//! {"event_id": "w1-0042", "session_id": "voice-RM_a1B2c3", "at_ms": 1792650000123,
//!  "source": "voice-worker", "type": "interrupted", "played_ms": 1840}
//! ```

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::SessionId;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LifecycleEvent {
    /// Idempotency key, 1 to 128 characters, unique per emitter. A repeated id is acknowledged and ignored.
    pub event_id: String,
    pub session_id: SessionId,
    /// When the emitter observed the event, in Unix epoch milliseconds. The control plane also stamps its own
    /// monotonic receipt time and orders events by that.
    pub at_ms: u64,
    pub source: EventSource,
    #[serde(flatten)]
    pub kind: EventKind,
}

impl LifecycleEvent {
    /// The longest accepted `event_id`, in bytes.
    pub const MAX_ID_LEN: usize = 128;
}

/// Who emitted an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum EventSource {
    /// Inferred by the control plane from proxied traffic (for example, a response ending in tool calls).
    Inferred,
    /// agentic-api's lifecycle sink: exact server-side tool timing.
    AgenticApi,
    /// The LiveKit voice worker: voice activity, end of turn, playback, barge-in.
    VoiceWorker,
    /// Any other trusted client.
    Client,
}

impl EventSource {
    /// The wire name, also used as a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inferred => "inferred",
            Self::AgenticApi => "agentic-api",
            Self::VoiceWorker => "voice-worker",
            Self::Client => "client",
        }
    }
}

/// What happened, with the fields of that event type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// A request for the session reached the control plane.
    TurnStarted,
    /// A response finished. A `tool_calls` stop moves the lease to tool wait, with an ETA from session history.
    TurnFinished {
        stop_reason: StopReason,
        #[serde(default)]
        prompt_tokens: u32,
        #[serde(default)]
        completion_tokens: u32,
    },
    /// A server-side tool started. `eta_ms`, when known, lets prefetch be timed exactly.
    ToolWaitStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        eta_ms: Option<u64>,
    },
    /// The tool returned; the next turn follows.
    ToolWaitEnded {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool: Option<String>,
    },
    /// Voice activity onset. Speech-onset prefetch and turn-imminent headroom start now.
    UserSpeechStarted,
    /// The end-of-turn detector decided the user has finished; the realtime request follows within
    /// milliseconds.
    UserTurnEnded,
    /// The agent started speaking.
    PlaybackStarted,
    /// The agent finished speaking.
    PlaybackEnded { played_ms: u64 },
    /// The user barged in. Cancel the in-flight stream and truncate the context to what the user heard.
    Interrupted { played_ms: u64 },
    /// The session ended; its KV becomes ordinary cache.
    SessionClosed,
}

impl EventKind {
    /// The wire `type`, also used as a metric label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::TurnStarted => "turn_started",
            Self::TurnFinished { .. } => "turn_finished",
            Self::ToolWaitStarted { .. } => "tool_wait_started",
            Self::ToolWaitEnded { .. } => "tool_wait_ended",
            Self::UserSpeechStarted => "user_speech_started",
            Self::UserTurnEnded => "user_turn_ended",
            Self::PlaybackStarted => "playback_started",
            Self::PlaybackEnded { .. } => "playback_ended",
            Self::Interrupted { .. } => "interrupted",
            Self::SessionClosed => "session_closed",
        }
    }
}

/// Why a response stopped, normalized across the OpenAI and Anthropic APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end: OpenAI `stop` or `content_filter`; Anthropic `end_turn`, `stop_sequence` or `refusal`.
    Stop,
    /// Token limit: OpenAI `length`; Anthropic `max_tokens`.
    Length,
    /// The model called tools: OpenAI `tool_calls` or `function_call`; Anthropic `tool_use`.
    ToolCalls,
    /// The client or the control plane aborted the stream.
    Cancelled,
    /// The upstream failed.
    Error,
}

impl StopReason {
    /// Maps an OpenAI `finish_reason` or an Anthropic `stop_reason`. Unknown values count as a natural stop.
    pub fn from_api(reason: &str) -> Self {
        match reason {
            "length" | "max_tokens" => Self::Length,
            "tool_calls" | "function_call" | "tool_use" => Self::ToolCalls,
            _ => Self::Stop,
        }
    }
}
