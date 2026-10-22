//! End to end on the 16-trace subset in `testdata/`: the AIPerf port against AIPerf's own replay
//! structure, and a run of every arm.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lh_core::TierCapacity;
use lh_sim::aiperf::{PlanOptions, reference};
use lh_sim::corpus::Corpus;
use lh_sim::replay::ReplayConfig;
use lh_sim::run::{self, Arm, ArmResult, RunConfig};
use serde_json::Value;

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn corpus() -> Corpus {
    Corpus::open(&testdata("weka-x32-sub16"), PlanOptions::default()).unwrap()
}

/// Collects the differences between two JSON values; numbers agree to 1e-9 relative.
fn diff(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                diff(
                    &format!("{path}.{k}"),
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                out.push(format!("{path}: {} vs {} items", x.len(), y.len()));
            }
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                diff(&format!("{path}[{i}]"), u, v, out);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > 1e-9 * x.abs().max(1.0) {
                out.push(format!("{path}: {x} vs {y}"));
            }
        }
        _ if a != b => out.push(format!("{path}: {a} vs {b}")),
        _ => {}
    }
}

#[test]
fn the_replay_structure_matches_aiperf() {
    // testdata/aiperf-0.13.0-sub16.json is AIPerf's own plan (scripts/aiperf_reference.py): 16 lanes,
    // seed 11, think times capped at 120 s, start ratios 0 to 1.
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(testdata("aiperf-0.13.0-sub16.json")).unwrap(),
    )
    .unwrap();
    let mut c = corpus();
    let plans: Vec<_> = (0..c.len()).map(|i| c.plan(i).unwrap()).collect();
    let refs: Vec<&_> = plans.iter().map(|p| p.as_ref()).collect();
    let ours = reference::dump(&refs, 16, 11, 0.0, 1.0);
    let by_id = |v: &Value| -> HashMap<String, Value> {
        v["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (c["id"].as_str().unwrap().to_owned(), c.clone()))
            .collect()
    };
    let (theirs, mine) = (by_id(&golden), by_id(&ours));
    assert_eq!(theirs.len(), 98);
    let mut problems = Vec::new();
    for (id, conv) in &theirs {
        match mine.get(id) {
            Some(m) => diff(id, conv, m, &mut problems),
            None => problems.push(format!("{id}: missing")),
        }
    }
    problems.extend(
        mine.keys()
            .filter(|id| !theirs.contains_key(*id))
            .map(|id| format!("{id}: not in AIPerf's plan")),
    );
    diff("lanes", &golden["lanes"], &ours["lanes"], &mut problems);
    diff(
        "recycle",
        &golden["recycle"],
        &ours["recycle"],
        &mut problems,
    );
    assert!(
        problems.is_empty(),
        "{} differences, first: {:?}",
        problems.len(),
        &problems[..problems.len().min(10)]
    );
}

fn run_all(seed: u64) -> Vec<ArmResult> {
    let cfg = RunConfig::frozen_box(ReplayConfig {
        seed,
        ..ReplayConfig::default()
    });
    let capacity = TierCapacity::frozen_box(&cfg.geometry);
    run::run(&mut corpus(), &cfg, &Arm::ALL, &[capacity], |_| {}).unwrap()
}

#[test]
fn every_arm_runs_consistently_on_the_subset() {
    let results = run_all(11);
    let names: Vec<_> = results.iter().map(|r| r.summary.arm.as_str()).collect();
    assert_eq!(names, ["gpu-lru", "lru-cpu", "s3fifo-cpu", "clairvoyant"]);
    let bytes_per_token = lh_core::KvGeometry::QWEN3_4B_BF16.bytes_per_token;
    for r in &results {
        let s = &r.summary;
        assert!(s.requests > 100, "{}: {} requests", s.arm, s.requests);
        assert_eq!(s.requests as usize, r.records.len());
        assert_eq!(s.mix.main + s.mix.subagent + s.mix.flattened, s.requests);
        assert_eq!(
            s.gpu_tokens + s.cpu_tokens + s.computed_tokens,
            s.prompt_tokens
        );
        assert_eq!(
            s.prompt_tokens,
            r.records.iter().map(|x| x.prompt_tokens).sum::<u64>()
        );
        let sh = &s.shares;
        assert!(
            (sh.gpu + sh.cpu + sh.computed - 1.0).abs() < 1e-9,
            "{}: shares {sh:?}",
            s.arm
        );
        assert!((sh.first_turn + sh.new_context + sh.recomputed - sh.computed).abs() < 1e-9);
        assert!(
            [sh.first_turn, sh.new_context, sh.recomputed]
                .iter()
                .all(|&x| x >= 0.0)
        );
        assert_eq!(s.pcie_restored_bytes, s.cpu_tokens * bytes_per_token);
        assert!(s.ttft_ms.p50 <= s.ttft_ms.p90 && s.ttft_ms.p90 <= s.ttft_ms.p99);
        assert!(s.regime.kv_pressure_peak >= s.regime.kv_pressure_mean && s.regime.rho > 0.0);
        assert_eq!(s.replay.lanes, 16);
    }
    let [gpu_lru, lru_cpu, s3fifo, clairvoyant] = &results[..] else {
        unreachable!()
    };
    assert_eq!(
        (gpu_lru.summary.cpu_blocks, gpu_lru.summary.cpu_tokens),
        (0, 0)
    );
    assert!(lru_cpu.summary.cpu_tokens > 0 && s3fifo.summary.cpu_tokens > 0);
    // The clairvoyant reference re-prices the requests of the lru-cpu run, on its timeline.
    let c = &clairvoyant.summary;
    assert_eq!(c.reference.as_deref(), Some("lru-cpu"));
    assert_eq!(
        (c.requests, c.prompt_tokens, c.stalls),
        (
            lru_cpu.summary.requests,
            lru_cpu.summary.prompt_tokens,
            lru_cpu.summary.stalls
        )
    );
    for (x, y) in clairvoyant.records.iter().zip(&lru_cpu.records) {
        assert_eq!(
            (x.conv, x.turn, x.issued_ns, x.admitted_ns),
            (y.conv, y.turn, y.issued_ns, y.admitted_ns)
        );
    }
}

#[test]
fn a_fixed_seed_gives_the_same_run() {
    let (a, b) = (run_all(5), run_all(5));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.summary, y.summary);
        assert_eq!(x.records, y.records);
    }
    // Another seed starts the lanes at other points of their traces.
    let other = run_all(6);
    assert_ne!(other[1].summary.prompt_tokens, a[1].summary.prompt_tokens);
}
