//! Token usage and finish reasons, read from responses as they pass through. The scanner only reads; the bytes
//! the client receives are the upstream's bytes.

use http::HeaderValue;
use memchr::{memchr, memmem};
use serde::Deserialize;

/// Token counts from a response's `usage` object (Chat Completions, Completions or Responses).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    pub prompt: u64,
    pub completion: u64,
    /// Prompt tokens served from the engine's prefix cache.
    pub cached: u64,
}

/// What a response said about itself.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Summary {
    pub usage: Option<Usage>,
    /// The last non-null `finish_reason` of a Chat Completions or Completions choice.
    pub finish_reason: Option<String>,
}

/// A non-streaming JSON body larger than this is relayed without reading its usage.
const MAX_JSON_BODY: usize = 4 << 20;
/// An SSE line longer than this is relayed without reading it; usage lines are a few hundred bytes.
const MAX_SSE_LINE: usize = 1 << 20;

/// Reads a response body chunk by chunk.
#[derive(Debug)]
pub(crate) enum Scanner {
    /// Server-sent events: each complete `data:` line that may carry usage or a finish reason is parsed.
    Sse {
        line: Vec<u8>,
        skipping: bool,
        summary: Summary,
    },
    /// A JSON body, parsed once complete.
    Json {
        body: Vec<u8>,
        too_large: bool,
    },
    Other,
}

impl Scanner {
    pub fn for_content_type(content_type: Option<&HeaderValue>) -> Self {
        let content_type = content_type.map_or(&b""[..], HeaderValue::as_bytes);
        if content_type.starts_with(b"text/event-stream") {
            Self::Sse {
                line: Vec::new(),
                skipping: false,
                summary: Summary::default(),
            }
        } else if content_type.starts_with(b"application/json") {
            Self::Json {
                body: Vec::new(),
                too_large: false,
            }
        } else {
            Self::Other
        }
    }

    pub fn feed(&mut self, mut chunk: &[u8]) {
        match self {
            Self::Sse {
                line,
                skipping,
                summary,
            } => {
                while let Some(end) = memchr(b'\n', chunk) {
                    if !*skipping {
                        if line.is_empty() {
                            summary.read_line(&chunk[..end]);
                        } else {
                            line.extend_from_slice(&chunk[..end]);
                            summary.read_line(line);
                        }
                    }
                    line.clear();
                    *skipping = false;
                    chunk = &chunk[end + 1..];
                }
                if line.len() + chunk.len() > MAX_SSE_LINE {
                    line.clear();
                    *skipping = true;
                } else if !*skipping {
                    line.extend_from_slice(chunk);
                }
            }
            Self::Json { body, too_large } => {
                if body.len() + chunk.len() > MAX_JSON_BODY {
                    *too_large = true;
                    *body = Vec::new();
                } else if !*too_large {
                    body.extend_from_slice(chunk);
                }
            }
            Self::Other => {}
        }
    }

    pub fn finish(self) -> Summary {
        match self {
            Self::Sse { summary, .. } => summary,
            Self::Json {
                body,
                too_large: false,
            } => {
                let mut summary = Summary::default();
                if let Ok(envelope) = serde_json::from_slice(&body) {
                    summary.absorb(envelope);
                }
                summary
            }
            Self::Json { .. } | Self::Other => Summary::default(),
        }
    }
}

impl Summary {
    fn read_line(&mut self, line: &[u8]) {
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        // Most chunks carry neither field, or `"finish_reason":null`: skip them without parsing.
        if (has_value(data, b"\"usage\"") || has_value(data, b"\"finish_reason\""))
            && let Ok(envelope) = serde_json::from_slice(data)
        {
            self.absorb(envelope);
        }
    }

    fn absorb(&mut self, envelope: Envelope) {
        if let Some(usage) = envelope
            .usage
            .or_else(|| envelope.response.and_then(|r| r.usage))
        {
            self.usage = Some(usage.into());
        }
        if let Some(reason) = envelope
            .choices
            .into_iter()
            .flatten()
            .find_map(|choice| choice.finish_reason)
        {
            self.finish_reason = Some(reason);
        }
    }
}

/// Whether `json` has `key` followed by a value other than `null`. Inside a JSON string a quote is escaped, so
/// `"usage"` with both quotes matches only a key.
fn has_value(json: &[u8], key: &[u8]) -> bool {
    memmem::find_iter(json, key).any(|at| {
        json[at + key.len()..]
            .trim_ascii_start()
            .strip_prefix(b":")
            .is_some_and(|value| !value.trim_ascii_start().starts_with(b"null"))
    })
}

/// The fields read from a response object, an SSE chunk, or a Responses API stream event (which wraps the
/// response object).
#[derive(Deserialize)]
struct Envelope {
    usage: Option<RawUsage>,
    choices: Option<Vec<Choice>>,
    response: Option<ResponseObject>,
}

#[derive(Deserialize)]
struct ResponseObject {
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct Choice {
    finish_reason: Option<String>,
}

/// Chat Completions and Completions name the fields `prompt_tokens` and `completion_tokens`; the Responses API
/// names them `input_tokens` and `output_tokens`.
#[derive(Deserialize)]
struct RawUsage {
    #[serde(alias = "input_tokens")]
    prompt_tokens: Option<u64>,
    #[serde(alias = "output_tokens")]
    completion_tokens: Option<u64>,
    #[serde(alias = "input_tokens_details")]
    prompt_tokens_details: Option<TokenDetails>,
}

#[derive(Deserialize)]
struct TokenDetails {
    cached_tokens: Option<u64>,
}

impl From<RawUsage> for Usage {
    fn from(raw: RawUsage) -> Self {
        Self {
            prompt: raw.prompt_tokens.unwrap_or(0),
            completion: raw.completion_tokens.unwrap_or(0),
            cached: raw
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(content_type: &'static str, chunks: &[&[u8]]) -> Summary {
        let mut scanner = Scanner::for_content_type(Some(&HeaderValue::from_static(content_type)));
        for chunk in chunks {
            scanner.feed(chunk);
        }
        scanner.finish()
    }

    const CHAT_STREAM: &[u8] = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\\\"usage\\\": 1\"},\"finish_reason\":null}]}\n\n\
        data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
        data: {\"choices\":[],\"usage\":{\"prompt_tokens\":18342,\"completion_tokens\":96,\"total_tokens\":18438,\
        \"prompt_tokens_details\":{\"cached_tokens\":16384}}}\n\ndata: [DONE]\n\n";

    #[test]
    fn sse_usage_and_finish_reason_survive_any_chunking() {
        let expected = Summary {
            usage: Some(Usage {
                prompt: 18342,
                completion: 96,
                cached: 16384,
            }),
            finish_reason: Some("tool_calls".into()),
        };
        for split in 0..CHAT_STREAM.len() {
            let (a, b) = CHAT_STREAM.split_at(split);
            assert_eq!(
                scan("text/event-stream", &[a, b]),
                expected,
                "split at {split}"
            );
        }
        let bytes: Vec<&[u8]> = CHAT_STREAM.chunks(1).collect();
        assert_eq!(scan("text/event-stream; charset=utf-8", &bytes), expected);
    }

    #[test]
    fn a_stream_without_usage_reports_none() {
        let summary = scan(
            "text/event-stream",
            &[b"data: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":null}],\"usage\":null}\r\n\r\n"],
        );
        assert_eq!(summary, Summary::default());
    }

    #[test]
    fn responses_api_events_and_bodies_use_input_and_output_names() {
        let event = b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\
            \"completed\",\"usage\":{\"input_tokens\":40,\"input_tokens_details\":{\"cached_tokens\":32},\
            \"output_tokens\":7,\"total_tokens\":47}}}\n\n";
        let usage = Usage {
            prompt: 40,
            completion: 7,
            cached: 32,
        };
        assert_eq!(scan("text/event-stream", &[event]).usage, Some(usage));
        let body = br#"{"id":"r","output":[],"usage":{"input_tokens":40,"output_tokens":7,
            "input_tokens_details":{"cached_tokens":32}}}"#;
        assert_eq!(scan("application/json", &[body]).usage, Some(usage));
    }

    #[test]
    fn json_bodies_are_read_once_complete() {
        let body: &[u8] = br#"{"choices":[{"message":{"content":"x"},"finish_reason":"length"}],
            "usage":{"prompt_tokens":5,"completion_tokens":null}}"#;
        let summary = scan("application/json", &[&body[..10], &body[10..]]);
        assert_eq!(
            summary,
            Summary {
                usage: Some(Usage {
                    prompt: 5,
                    ..Usage::default()
                }),
                finish_reason: Some("length".into()),
            }
        );
        assert_eq!(scan("application/json", &[&body[..10]]), Summary::default());
        assert_eq!(scan("text/plain", &[body]), Summary::default());
    }

    #[test]
    fn oversized_input_is_skipped_not_buffered() {
        let huge = vec![b'x'; MAX_SSE_LINE + 1];
        let tail = b"\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n";
        let summary = scan("text/event-stream", &[b"data: ", &huge, tail]);
        assert_eq!(summary.usage.map(|u| u.prompt), Some(1));

        let mut scanner =
            Scanner::for_content_type(Some(&HeaderValue::from_static("application/json")));
        scanner.feed(&vec![b' '; MAX_JSON_BODY + 1]);
        scanner.feed(br#"{"usage":{"prompt_tokens":1}}"#);
        assert_eq!(scanner.finish(), Summary::default());
    }
}
