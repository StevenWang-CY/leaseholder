//! Session ids: the routing and accounting key of one conversation.

use std::borrow::Cow;
use std::fmt;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};

/// A session: one conversation, scoped to a tenant. It is the routing key, the accounting key and the OTel
/// `gen_ai.conversation.id`.
///
/// Session ids travel in HTTP headers and engine metadata, so they are 1 to 128 characters from
/// `[A-Za-z0-9._:-]`. Claude Code's UUIDs and the voice worker's `voice-<room SID>` ids fit as they are; the
/// control plane hashes any other source (an arbitrary `prompt_cache_key`, say) into this form when it resolves
/// the session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionId(String);

impl SessionId {
    /// The longest accepted session id, in bytes.
    pub const MAX_LEN: usize = 128;

    /// Accepts `id` if it is 1 to [`Self::MAX_LEN`] characters from `[A-Za-z0-9._:-]`.
    pub fn new(id: impl Into<String>) -> Result<Self, SessionIdError> {
        let id = id.into();
        if id.is_empty() || id.len() > Self::MAX_LEN {
            return Err(SessionIdError::Length(id.len()));
        }
        if let Some(c) = id.chars().find(|&c| !is_id_char(c)) {
            return Err(SessionIdError::Char(c));
        }
        Ok(Self(id))
    }

    /// The id as it travels in headers.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-')
}

impl TryFrom<String> for SessionId {
    type Error = SessionIdError;

    fn try_from(id: String) -> Result<Self, Self::Error> {
        Self::new(id)
    }
}

impl From<SessionId> for String {
    fn from(id: SessionId) -> Self {
        id.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl JsonSchema for SessionId {
    fn schema_name() -> Cow<'static, str> {
        "SessionId".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "One conversation, scoped to a tenant: 1 to 128 characters from [A-Za-z0-9._:-].",
            "type": "string",
            "minLength": 1,
            "maxLength": Self::MAX_LEN,
            "pattern": "^[A-Za-z0-9._:-]+$",
        })
    }
}

/// Why a string is not a session id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionIdError {
    /// The id is empty or longer than [`SessionId::MAX_LEN`] bytes.
    Length(usize),
    /// The id contains a character outside `[A-Za-z0-9._:-]`.
    Char(char),
}

impl fmt::Display for SessionIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(n) => write!(f, "session id must be 1 to 128 bytes, got {n}"),
            Self::Char(c) => write!(f, "session id contains {c:?}; allowed: [A-Za-z0-9._:-]"),
        }
    }
}

impl std::error::Error for SessionIdError {}
