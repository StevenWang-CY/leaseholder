//! Lifecycle events: what tells the control plane when a session will next need its KV.
//!
//! One schema serves every workload (ADR-0002). Policy `leaseholder` follows the voice sessions' turns by their
//! events, which time its KV headroom and its critical window; the agents' events are recorded, and nothing acts
//! on them. Trusted clients post them to `POST /v1/events`, idempotent by `(tenant, source, event_id)`; the
//! control plane infers none from the traffic it proxies.
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

/// One lifecycle event of one session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LifecycleEvent {
    /// Idempotency key, 1 to 128 bytes, unique per emitter. A repeat of an id that the same tenant posted from the
    /// same source is acknowledged and ignored.
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
    /// A response finished: why it stopped, and its tokens.
    TurnFinished {
        stop_reason: StopReason,
        #[serde(default)]
        prompt_tokens: u32,
        #[serde(default)]
        completion_tokens: u32,
    },
    /// A server-side tool started; `eta_ms` is when it is expected to return, when known.
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
    /// Voice activity onset: a turn is coming, and policy `leaseholder` (`trigger = "onset"`) arms its KV headroom.
    UserSpeechStarted,
    /// The end-of-turn detector decided the user has finished; the realtime request follows within
    /// milliseconds.
    UserTurnEnded,
    /// The agent started speaking.
    PlaybackStarted,
    /// The agent finished speaking.
    PlaybackEnded { played_ms: u64 },
    /// The user barged in, `played_ms` into the reply: policy `leaseholder` takes it as the user speaking again.
    /// The voice agent ends the reply's stream itself.
    Interrupted { played_ms: u64 },
    /// The session ended: the gateway forgets its turn state and, with several GPU nodes, its placement.
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
