//! SLO classes: which clock a request runs on, and the engine priority each maps to.

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The SLO class of a request: which clock it runs on. Classes order from strictest to most lenient.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SloClass {
    /// Voice turns, at engine priority 0. Policy `leaseholder` never holds or refuses them, and may cap a reply.
    Realtime,
    /// Agent loops with a person waiting, at engine priority 1. Policy `leaseholder` does not hold them.
    #[default]
    Interactive,
    /// Batch agents and trace replays, at engine priority 2. An admission policy may defer, shed or resubmit them
    /// while realtime needs room.
    Background,
}

impl SloClass {
    /// Every class, strictest first.
    pub const ALL: [Self; 3] = [Self::Realtime, Self::Interactive, Self::Background];

    /// The vLLM request priority for this class. vLLM schedules lower values first.
    pub const fn engine_priority(self) -> i32 {
        match self {
            Self::Realtime => 0,
            Self::Interactive => 1,
            Self::Background => 2,
        }
    }

    /// The wire name, also the `x-lh-class` header value and a metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Interactive => "interactive",
            Self::Background => "background",
        }
    }
}

impl fmt::Display for SloClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SloClass {
    type Err = UnknownClass;

    /// Parses an `x-lh-class` header value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|class| class.as_str() == s)
            .ok_or_else(|| UnknownClass(s.to_owned()))
    }
}

/// An `x-lh-class` value that names no class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownClass(pub String);

impl fmt::Display for UnknownClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown SLO class {:?}; expected realtime, interactive or background",
            self.0
        )
    }
}

impl std::error::Error for UnknownClass {}
