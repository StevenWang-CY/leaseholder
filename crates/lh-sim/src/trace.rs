//! Weka-format agent traces: the SemiAnalysis Claude Code corpus, as `lh-bench scale-traces` writes it
//! (a directory of `<trace id>.json`) or as the original JSONL (one trace per line).
//!
//! Each trace records its requests as lists of prefix hash blocks of `block_size` tokens each (64
//! unscaled, 2 at 32x, 1 at 64x). A main-agent request has `t`, `model`, `in`, `out`, `hash_ids`,
//! `api_time` and `think_time`; an entry of type `subagent` holds its own requests. Hash ids are local
//! to one trace.

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};

/// One model request: a main-agent turn or a subagent's inner request.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// Start time in seconds from the trace start. Older captures record subagent requests relative
    /// to the subagent's spawn; [`crate::aiperf`] normalizes them as AIPerf does.
    pub t: f64,
    pub model: String,
    /// Recorded input tokens. Only AIPerf's naming of side chains reads it; the prompt length the
    /// simulator uses is `hash_ids.len() * block_size`, because the raw count overcounts.
    pub input: u64,
    pub out: u64,
    pub hash_ids: Vec<u32>,
    /// Recorded request duration in seconds.
    pub api_time: Option<f64>,
    /// Recorded idle time before this request, in seconds.
    pub think_time: Option<f64>,
}

/// A subagent: its own requests, run beside the parent.
#[derive(Clone, Debug, PartialEq)]
pub struct Subagent {
    /// Spawn time in seconds.
    pub t: f64,
    pub agent_id: String,
    pub duration_ms: Option<f64>,
    pub requests: Vec<Request>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Request(Request),
    Subagent(Subagent),
}

/// One recorded agent session.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub id: String,
    /// Tokens per hash block.
    pub block_size: u32,
    pub entries: Vec<Entry>,
}

impl Trace {
    /// Every model request: main-agent turns and subagent inner requests.
    pub fn requests(&self) -> impl Iterator<Item = &Request> {
        self.entries.iter().flat_map(|e| match e {
            Entry::Request(r) => std::slice::from_ref(r).iter(),
            Entry::Subagent(s) => s.requests.iter(),
        })
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let raw: RawTrace = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        raw.into_trace()
    }
}

/// A set of traces opened for lazy loading, in AIPerf's order: a directory's `*.json` files sorted by
/// name (as AIPerf's Weka loader enumerates them), or the lines of a JSONL file in file order.
#[derive(Debug)]
pub struct TraceSource {
    kind: SourceKind,
}

#[derive(Debug)]
enum SourceKind {
    Dir(Vec<PathBuf>),
    Jsonl {
        path: PathBuf,
        lines: Vec<(u64, usize)>,
    },
    Memory(Vec<Trace>),
}

impl TraceSource {
    pub fn open(path: &Path) -> Result<Self, TraceError> {
        let err = |source| TraceError {
            path: path.to_owned(),
            source,
        };
        if path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path).map_err(|e| err(ErrorKind::Io(e)))? {
                let p = entry.map_err(|e| err(ErrorKind::Io(e)))?.path();
                if p.extension().is_some_and(|x| x == "json") && p.is_file() {
                    files.push(p);
                }
            }
            files.sort();
            return Ok(Self {
                kind: SourceKind::Dir(files),
            });
        }
        let file = File::open(path).map_err(|e| err(ErrorKind::Io(e)))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut lines = Vec::new();
        let mut offset = 0u64;
        let mut buf = Vec::new();
        loop {
            buf.clear();
            let n = reader
                .read_until(b'\n', &mut buf)
                .map_err(|e| err(ErrorKind::Io(e)))?;
            if n == 0 {
                break;
            }
            if buf.iter().any(|b| !b.is_ascii_whitespace()) {
                lines.push((offset, n));
            }
            offset += n as u64;
        }
        Ok(Self {
            kind: SourceKind::Jsonl {
                path: path.to_owned(),
                lines,
            },
        })
    }

    /// Traces already in memory, in the given order.
    pub fn from_traces(traces: Vec<Trace>) -> Self {
        Self {
            kind: SourceKind::Memory(traces),
        }
    }

    /// The index of the file named `<id>.json`, for a directory source.
    pub fn file_index(&self, id: &str) -> Option<usize> {
        match &self.kind {
            SourceKind::Dir(files) => files
                .iter()
                .position(|p| p.file_stem().is_some_and(|s| s == id)),
            SourceKind::Jsonl { .. } | SourceKind::Memory(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        match &self.kind {
            SourceKind::Dir(files) => files.len(),
            SourceKind::Jsonl { lines, .. } => lines.len(),
            SourceKind::Memory(traces) => traces.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Parses trace `index`.
    pub fn load(&self, index: usize) -> Result<Trace, TraceError> {
        match &self.kind {
            SourceKind::Dir(files) => {
                let path = &files[index];
                let bytes = std::fs::read(path).map_err(|e| TraceError {
                    path: path.clone(),
                    source: ErrorKind::Io(e),
                })?;
                Trace::from_json(&bytes).map_err(|e| TraceError {
                    path: path.clone(),
                    source: ErrorKind::Parse(e),
                })
            }
            SourceKind::Jsonl { path, lines } => {
                let err = |source| TraceError {
                    path: path.clone(),
                    source,
                };
                let (offset, len) = lines[index];
                let mut file = File::open(path).map_err(|e| err(ErrorKind::Io(e)))?;
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| err(ErrorKind::Io(e)))?;
                let mut bytes = vec![0u8; len];
                file.read_exact(&mut bytes)
                    .map_err(|e| err(ErrorKind::Io(e)))?;
                Trace::from_json(&bytes)
                    .map_err(|e| err(ErrorKind::Parse(format!("line {}: {e}", index + 1))))
            }
            SourceKind::Memory(traces) => Ok(traces[index].clone()),
        }
    }
}

#[derive(Debug)]
pub struct TraceError {
    pub path: PathBuf,
    source: ErrorKind,
}

#[derive(Debug)]
enum ErrorKind {
    Io(io::Error),
    Parse(String),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            ErrorKind::Io(e) => write!(f, "{}: {e}", self.path.display()),
            ErrorKind::Parse(e) => write!(f, "{}: not a Weka trace: {e}", self.path.display()),
        }
    }
}

impl std::error::Error for TraceError {}

// The wire shape. One flat struct per entry, so serde never buffers the hash-id arrays.
#[derive(Deserialize)]
struct RawTrace {
    id: String,
    block_size: u32,
    #[serde(default, deserialize_with = "null_default")]
    requests: Vec<RawEntry>,
}

#[derive(Deserialize)]
struct RawEntry {
    #[serde(rename = "type")]
    kind: String,
    t: f64,
    #[serde(default, deserialize_with = "null_default")]
    model: String,
    #[serde(rename = "in", default, deserialize_with = "null_default")]
    input: u64,
    #[serde(default, deserialize_with = "null_default")]
    out: u64,
    #[serde(default, deserialize_with = "null_default")]
    hash_ids: Vec<u32>,
    #[serde(default)]
    api_time: Option<f64>,
    #[serde(default)]
    think_time: Option<f64>,
    #[serde(default, deserialize_with = "null_default")]
    agent_id: String,
    #[serde(default)]
    duration_ms: Option<f64>,
    #[serde(default, deserialize_with = "null_default")]
    requests: Vec<RawEntry>,
}

fn null_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

impl RawEntry {
    fn into_request(self) -> Result<Request, String> {
        if self.kind != "n" && self.kind != "s" {
            return Err(format!(
                "expected a request (type n or s), got type {:?}",
                self.kind
            ));
        }
        Ok(Request {
            t: self.t,
            model: self.model,
            input: self.input,
            out: self.out,
            hash_ids: self.hash_ids,
            api_time: self.api_time,
            think_time: self.think_time,
        })
    }
}

impl RawTrace {
    fn into_trace(self) -> Result<Trace, String> {
        if self.block_size == 0 {
            return Err(format!("trace {}: block_size must be positive", self.id));
        }
        let mut entries = Vec::with_capacity(self.requests.len());
        for raw in self.requests {
            entries.push(match raw.kind.as_str() {
                "n" | "s" => Entry::Request(raw.into_request()?),
                "subagent" => Entry::Subagent(Subagent {
                    t: raw.t,
                    agent_id: raw.agent_id,
                    duration_ms: raw.duration_ms,
                    requests: raw
                        .requests
                        .into_iter()
                        .map(RawEntry::into_request)
                        .collect::<Result<_, _>>()?,
                }),
                other => return Err(format!("trace {}: unknown request type {other:?}", self.id)),
            });
        }
        Ok(Trace {
            id: self.id,
            block_size: self.block_size,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRACE: &str = r#"{
        "id": "abc", "models": ["m"], "block_size": 2, "hash_id_scope": "local",
        "requests": [
            {"t": 0.0, "type": "n", "model": "m", "in": 4, "out": 3, "hash_ids": [0, 1],
             "think_time": null, "api_time": 1.5, "ttft": 0.2},
            {"t": 5.0, "type": "subagent", "agent_id": "a1", "subagent_type": "Explore",
             "duration_ms": 2000, "status": "completed", "models": ["h"],
             "requests": [{"t": 5.5, "type": "s", "model": "h", "in": 2, "out": 1, "hash_ids": [7]}]},
            {"t": 9.0, "type": "s", "model": "m", "in": 6, "out": 2, "hash_ids": [0, 1, 2],
             "think_time": 1.0}
        ]
    }"#;

    #[test]
    fn parses_requests_and_subagents() {
        let t = Trace::from_json(TRACE.as_bytes()).unwrap();
        assert_eq!(
            (t.id.as_str(), t.block_size, t.entries.len()),
            ("abc", 2, 3)
        );
        let Entry::Subagent(sa) = &t.entries[1] else {
            panic!("expected a subagent")
        };
        assert_eq!(
            (sa.agent_id.as_str(), sa.duration_ms, sa.requests.len()),
            ("a1", Some(2000.0), 1)
        );
        let ids: Vec<_> = t.requests().map(|r| r.hash_ids.clone()).collect();
        assert_eq!(ids, vec![vec![0, 1], vec![7], vec![0, 1, 2]]);
        let first = t.requests().next().unwrap();
        assert_eq!(
            (first.think_time, first.api_time, first.out),
            (None, Some(1.5), 3)
        );
    }

    #[test]
    fn rejects_unknown_types_and_zero_block_size() {
        let bad = TRACE.replace(
            r#""type": "s", "model": "m""#,
            r#""type": "x", "model": "m""#,
        );
        assert!(
            Trace::from_json(bad.as_bytes())
                .unwrap_err()
                .contains("unknown request type")
        );
        let zero = TRACE.replace(r#""block_size": 2"#, r#""block_size": 0"#);
        assert!(Trace::from_json(zero.as_bytes()).is_err());
    }

    #[test]
    fn reads_jsonl_lazily_and_directories_in_name_order() {
        let dir = std::env::temp_dir().join(format!("lh-sim-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let line = TRACE.replace('\n', " ");
        let other = line.replace(r#""id": "abc""#, r#""id": "abd""#);
        let jsonl = dir.join("traces.jsonl");
        std::fs::write(&jsonl, format!("{line}\n\n{other}\n")).unwrap();
        let src = TraceSource::open(&jsonl).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(src.load(1).unwrap().id, "abd");

        let traces = dir.join("traces");
        std::fs::create_dir_all(&traces).unwrap();
        std::fs::write(traces.join("b.json"), &other).unwrap();
        std::fs::write(traces.join("a.json"), &line).unwrap();
        std::fs::write(traces.join("notes.txt"), "not a trace").unwrap();
        let src = TraceSource::open(&traces).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(
            (src.load(0).unwrap().id, src.load(1).unwrap().id),
            ("abc".into(), "abd".into())
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
