//! The error of the token estimate (`lh_core::admission::estimate_tokens`) on an AIPerf replay, against the
//! prompt tokens the engine reported.
//!
//! `cargo run --release -p lh-core --example estimate_error -- RUN/aiperf/profile_export.jsonl...`
//!
//! Each record of AIPerf's per-request export has the bytes the request sent (`http_req_data_sent`: its
//! headers and its body), the prompt tokens the engine reported, and the root correlation id of its session
//! tree, which the `leaseholder_chat` endpoint sends as `X-Session-ID`. The records are replayed as the
//! gateway sees them: a request is estimated when it starts, from the last report of its session that had
//! ended by then. The error is given over all prompt tokens, and per request for requests of 1,000 tokens or
//! more, beside what bytes / 4 alone would give.

use std::collections::HashMap;

use lh_core::admission::{SessionUsage, estimate_tokens};
use serde_json::Value;

/// One request of the replay.
struct Record {
    start_ns: u64,
    end_ns: u64,
    session: String,
    bytes: u64,
    prompt_tokens: u64,
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: estimate_error PROFILE_EXPORT.jsonl...");
        std::process::exit(2);
    }
    for path in paths {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let records: Vec<Record> = text.lines().filter_map(record).collect();
        let (anchored, fallback, with_report) = estimates(&records);
        println!(
            "{path}: {} requests, {with_report} with an earlier report of their session",
            records.len()
        );
        println!("  the estimate:  {}", describe(&records, &anchored));
        println!("  bytes / 4:     {}", describe(&records, &fallback));
    }
}

fn record(line: &str) -> Option<Record> {
    let record: Value = serde_json::from_str(line).ok()?;
    let (meta, metrics) = (&record["metadata"], &record["metrics"]);
    let value = |name: &str| metrics[name]["value"].as_f64();
    let session = meta["root_correlation_id"]
        .as_str()
        .or_else(|| meta["x_correlation_id"].as_str())?;
    Some(Record {
        start_ns: meta["request_start_ns"].as_u64()?,
        end_ns: meta["request_end_ns"].as_u64()?,
        session: session.to_owned(),
        // AIPerf reports kilobytes of 1,024 bytes.
        bytes: (value("http_req_data_sent")? * 1024.0).round() as u64,
        prompt_tokens: value("usage_prompt_tokens")? as u64,
    })
}

/// Each request's estimate from its session's last report, and from its bytes alone, with the number of
/// requests whose session had reported before they started.
fn estimates(records: &[Record]) -> (Vec<u64>, Vec<u64>, usize) {
    // Starts and ends in time order, an end first at the same instant.
    let mut events: Vec<(u64, bool, usize)> = records
        .iter()
        .enumerate()
        .flat_map(|(i, r)| [(r.start_ns, true, i), (r.end_ns, false, i)])
        .collect();
    events.sort_unstable();
    let mut last: HashMap<&str, SessionUsage> = HashMap::new();
    let mut anchored = vec![0; records.len()];
    let mut with_report = 0;
    for (_, start, i) in events {
        let r = &records[i];
        if start {
            let usage = last.get(r.session.as_str()).copied();
            with_report += usize::from(usage.is_some());
            anchored[i] = estimate_tokens(r.bytes, usage);
        } else {
            let usage = SessionUsage {
                prompt_tokens: r.prompt_tokens,
                body_bytes: r.bytes,
            };
            last.insert(&r.session, usage);
        }
    }
    let fallback = records
        .iter()
        .map(|r| estimate_tokens(r.bytes, None))
        .collect();
    (anchored, fallback, with_report)
}

fn describe(records: &[Record], estimates: &[u64]) -> String {
    let reported: u64 = records.iter().map(|r| r.prompt_tokens).sum();
    let off: u64 = records
        .iter()
        .zip(estimates)
        .map(|(r, &e)| e.abs_diff(r.prompt_tokens))
        .sum();
    let mut errors: Vec<f64> = records
        .iter()
        .zip(estimates)
        .filter(|(r, _)| r.prompt_tokens >= 1_000)
        .map(|(r, &e)| e.abs_diff(r.prompt_tokens) as f64 / r.prompt_tokens as f64)
        .collect();
    errors.sort_by(f64::total_cmp);
    let within = |q: f64| errors[((errors.len() - 1) as f64 * q).round() as usize] * 100.0;
    format!(
        "off by {:.2}% of all prompt tokens; of {} requests of 1,000 tokens or more, 50% within {:.1}%, \
         90% within {:.1}%, 99% within {:.1}%, all within {:.1}%",
        off as f64 / reported as f64 * 100.0,
        errors.len(),
        within(0.5),
        within(0.9),
        within(0.99),
        within(1.0),
    )
}
