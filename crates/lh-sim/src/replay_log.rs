//! A measured AIPerf run (its `profile_export.jsonl`) replayed through the engine model.
//!
//! Every request has the prompt its conversation turn has in the corpus, its measured output length,
//! and one KV namespace per session tree (AIPerf's cache-bust marker). Requests to different models
//! share KV, as they did against one served model.
//! - [`replay`]: every request enters the engine at its measured start and leaves at its measured
//!   end; one that finds no free blocks waits as in a simulated run. Comparing the hit pattern with
//!   the engine's counters isolates the cache model from the replay scheduler and the service model.
//! - [`replay_timed`]: every request is sent at its measured start and served by the step loop
//!   ([`crate::scheduler`]). Comparing its time to first token with the measured one isolates the
//!   service model from the replay's closed loop.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::BufRead;
use std::path::Path;

use lh_core::PromptSources;
use serde::Deserialize;

use crate::corpus::Corpus;
use crate::engine::{Engine, EngineRequest};
use crate::prefix::{BlockId, NodeId, Tokens};
use crate::scheduler::{Outcome, Scheduler};

/// One request of the log.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedRequest {
    pub conversation: String,
    pub turn: u32,
    pub warmup: bool,
    /// Prompt tokens as the engine counted them (chat template and cache-bust marker included).
    pub measured_prompt_tokens: u64,
    /// Prompt tokens the engine reported as cached, GPU and CPU together.
    pub measured_cached_tokens: u64,
    /// Prompt tokens of the turn in the corpus: hash blocks times the hash block size.
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Where the engine model found the prompt.
    pub sources: PromptSources,
    /// Time to first token as AIPerf measured it, in ms.
    pub measured_ttft_ms: f64,
    /// Time to first token in a timed replay, in ms.
    pub ttft_ms: Option<f64>,
}

/// The log replayed, with totals over its profiling requests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogReplay {
    pub requests: Vec<LoggedRequest>,
    pub profiling_requests: u64,
    pub measured_prompt_tokens: u64,
    pub measured_cached_tokens: u64,
    pub prompt_tokens: u64,
    pub sources: PromptSources,
    /// Requests that waited for free blocks, warmup included.
    pub stalls: u64,
    /// Mean over profiling requests of |simulated − measured| cached share of the prompt.
    pub mean_abs_cached_share_error: f64,
}

#[derive(Deserialize)]
struct Record {
    metadata: Metadata,
    metrics: HashMap<String, Metric>,
}

#[derive(Deserialize)]
struct Metadata {
    conversation_id: String,
    turn_index: u32,
    source_trace_id: String,
    root_correlation_id: String,
    benchmark_phase: String,
    request_start_ns: u64,
    request_end_ns: u64,
}

// Some metrics are per-token lists; only scalar ones are read.
#[derive(Deserialize)]
struct Metric {
    value: serde_json::Value,
}

impl Record {
    fn value(&self, name: &str) -> f64 {
        self.metrics
            .get(name)
            .and_then(|m| m.value.as_f64())
            .unwrap_or(0.0)
    }

    fn metric(&self, name: &str) -> u64 {
        self.value(name).max(0.0).round() as u64
    }
}

/// The requests of a log, with their engine prompts and measured start and end times.
struct Log {
    requests: Vec<LoggedRequest>,
    prompts: Vec<Vec<BlockId>>,
    spans: Vec<(u64, u64)>,
}

/// Reads the requests of `log`, their prompts cut in `block_tokens`-token blocks.
fn read(corpus: &mut Corpus, log: &Path, block_tokens: u32) -> Result<Log, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", log.display());
    let file = std::fs::File::open(log).map_err(|e| err(&e))?;
    let mut out = Log {
        requests: Vec::new(),
        prompts: Vec::new(),
        spans: Vec::new(),
    };
    let mut trees: HashMap<String, u64> = HashMap::new();
    let mut scratch: Vec<NodeId> = Vec::new();
    for (n, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| err(&e))?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Record =
            serde_json::from_str(&line).map_err(|e| err(&format!("line {}: {e}", n + 1)))?;
        let m = &r.metadata;
        let trace = corpus
            .find(&m.source_trace_id)?
            .ok_or_else(|| format!("trace {} is not in the corpus", m.source_trace_id))?;
        let plan = corpus.plan(trace)?;
        let conv = plan
            .conv_index(&m.conversation_id)
            .ok_or_else(|| format!("no conversation {}", m.conversation_id))?;
        let turn = plan.convs[conv as usize]
            .turns
            .get(m.turn_index as usize)
            .ok_or_else(|| format!("{} has no turn {}", m.conversation_id, m.turn_index))?;
        let next = trees.len() as u64;
        let play = *trees.entry(m.root_correlation_id.clone()).or_insert(next);
        let tokens = Tokens::new(plan.block_size, block_tokens);
        let mut prompt = Vec::new();
        plan.prefixes
            .engine_blocks(turn.prompt_end, tokens, play, &mut scratch, &mut prompt);
        out.spans
            .push((m.request_start_ns, m.request_end_ns.max(m.request_start_ns)));
        out.prompts.push(prompt);
        out.requests.push(LoggedRequest {
            conversation: m.conversation_id.clone(),
            turn: m.turn_index,
            warmup: m.benchmark_phase == "warmup",
            measured_prompt_tokens: r.metric("usage_prompt_tokens"),
            measured_cached_tokens: r.metric("usage_prompt_cache_read_tokens"),
            prompt_tokens: tokens.prompt_tokens(turn.hash_blocks),
            output_tokens: r.metric("output_sequence_length").max(1),
            sources: PromptSources::default(),
            measured_ttft_ms: r.value("time_to_first_token"),
            ttft_ms: None,
        });
    }
    Ok(out)
}

/// Replays the requests of `log` through `engine`, whose blocks hold `block_tokens` tokens, each
/// from its measured start to its measured end.
pub fn replay(
    corpus: &mut Corpus,
    log: &Path,
    mut engine: Engine,
    block_tokens: u32,
) -> Result<LogReplay, String> {
    let Log {
        requests,
        mut prompts,
        spans,
    } = read(corpus, log, block_tokens)?;
    let mut out = LogReplay {
        requests,
        ..LogReplay::default()
    };
    // (time, admission after finishes at the same instant, request)
    let mut events: Vec<(u64, bool, usize)> = Vec::with_capacity(2 * spans.len());
    for (i, &(start, end)) in spans.iter().enumerate() {
        events.push((start, true, i));
        events.push((end, false, i));
    }
    events.sort_unstable();
    let mut admitted = vec![false; out.requests.len()];
    let mut finish_pending = vec![false; out.requests.len()];
    for (t, admit, i) in events {
        if admit {
            let r = &out.requests[i];
            let req = EngineRequest {
                prompt: std::mem::take(&mut prompts[i]),
                prompt_tokens: r.prompt_tokens,
                output_tokens: r.output_tokens,
            };
            if let Some(a) = engine.submit(i as u64, req) {
                admitted[i] = true;
                out.requests[i].sources = a.sources;
            }
        } else if !admitted[i] {
            // Still waiting for blocks at its measured end: it finishes once it is admitted.
            finish_pending[i] = true;
        } else {
            let mut finishing = vec![i];
            while let Some(j) = finishing.pop() {
                for (q, a) in engine.finish(j as u64, t) {
                    let q = q as usize;
                    admitted[q] = true;
                    out.requests[q].sources = a.sources;
                    if finish_pending[q] {
                        finishing.push(q);
                    }
                }
            }
        }
    }
    totals(out, &engine)
}

/// Replays the requests of `log` through `engine` and `scheduler`: each is sent at its measured
/// start, reaches the scheduler after its host time and is served from there.
pub fn replay_timed(
    corpus: &mut Corpus,
    log: &Path,
    mut engine: Engine,
    block_tokens: u32,
    mut scheduler: Scheduler,
) -> Result<LogReplay, String> {
    let Log {
        requests,
        mut prompts,
        spans,
    } = read(corpus, log, block_tokens)?;
    let mut out = LogReplay {
        requests,
        ..LogReplay::default()
    };
    let origin = spans.iter().map(|s| s.0).min().unwrap_or(0);
    // (time, a send (0) or an arrival at the scheduler (1), request)
    let mut events: BinaryHeap<Reverse<(u64, u8, usize)>> = spans
        .iter()
        .enumerate()
        .map(|(i, s)| Reverse((s.0 - origin, 0, i)))
        .collect();
    let mut wake: Option<u64> = None;
    let mut outcomes = Vec::new();
    loop {
        let event = events.peek().map(|e| e.0);
        let now = match (event, wake) {
            (None, None) => break,
            (Some((t, ..)), Some(w)) => t.min(w),
            (Some((t, ..)), None) => t,
            (None, Some(w)) => w,
        };
        if let Some((t, stage, i)) = event
            && t == now
        {
            events.pop();
            let r = &out.requests[i];
            if stage == 0 {
                let req = EngineRequest {
                    prompt: Vec::new(),
                    prompt_tokens: r.prompt_tokens,
                    output_tokens: r.output_tokens,
                };
                if !engine.reject_oversized(&req) {
                    events.push(Reverse((
                        now + scheduler.host_ns(r.prompt_tokens, now),
                        1,
                        i,
                    )));
                }
                continue;
            }
            let req = EngineRequest {
                prompt: std::mem::take(&mut prompts[i]),
                prompt_tokens: r.prompt_tokens,
                output_tokens: r.output_tokens,
            };
            scheduler.arrive(i as u64, req);
            if scheduler.busy() {
                continue;
            }
        }
        wake = scheduler.run(now, &mut engine, &mut outcomes);
        for outcome in outcomes.drain(..) {
            match outcome {
                Outcome::Admitted { id, admission, .. } => {
                    out.requests[id as usize].sources = admission.sources;
                }
                Outcome::FirstToken(id) => {
                    let sent = spans[id as usize].0 - origin;
                    out.requests[id as usize].ttft_ms = Some((now - sent) as f64 / 1e6);
                }
                Outcome::Finished(_) => {}
            }
        }
    }
    totals(out, &engine)
}

/// Adds the totals over the profiling requests.
fn totals(mut out: LogReplay, engine: &Engine) -> Result<LogReplay, String> {
    if engine.stats.oversized > 0 {
        return Err(format!(
            "{} requests need more KV blocks than the GPU pool's {}",
            engine.stats.oversized,
            engine.gpu_capacity()
        ));
    }
    out.stalls = engine.stats.stalls;
    let mut error = 0.0;
    for r in out.requests.iter().filter(|r| !r.warmup) {
        out.profiling_requests += 1;
        out.measured_prompt_tokens += r.measured_prompt_tokens;
        out.measured_cached_tokens += r.measured_cached_tokens;
        out.prompt_tokens += r.prompt_tokens;
        out.sources.gpu += r.sources.gpu;
        out.sources.cpu += r.sources.cpu;
        out.sources.computed += r.sources.computed;
        let simulated = (r.sources.gpu + r.sources.cpu) as f64 / r.prompt_tokens.max(1) as f64;
        let measured = r.measured_cached_tokens as f64 / r.measured_prompt_tokens.max(1) as f64;
        error += (simulated - measured).abs();
    }
    out.mean_abs_cached_share_error = error / out.profiling_requests.max(1) as f64;
    Ok(out)
}
