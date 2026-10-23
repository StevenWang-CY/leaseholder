//! vLLM 0.32's step loop for the contention model (`Scheduler.schedule()` in `v1/core/sched`): which
//! requests each engine step carries, and when their first and last tokens come out.
//!
//! - **Host.** A request reaches the scheduler after its host time ([`StepCost::host_ms`]). The API
//!   server renders and tokenizes one request at a time, in the order they are sent (its renderer
//!   pool has `--renderer-num-workers` threads, 1 by default), so a burst of sends queues there.
//! - **Budget.** A step carries at most `max_num_batched_tokens` tokens. Running requests come first,
//!   in the order they were admitted: a decode takes one token, a prefill the rest of its prompt, at
//!   most `long_prefill_token_threshold` per step when more than one request is running or waiting
//!   (alone, a request may take the whole budget). A request reached with the budget spent waits
//!   for the next step, decodes included.
//! - **Waiting requests** follow, first come first served (every request has the background class's
//!   priority), after those whose CPU restore has finished. A request is admitted into the cache
//!   model when it is first reached with budget left, as vLLM looks up its prefix cache then; one
//!   that does not fit stops the loop.
//! - **Restores.** An admitted request with blocks in the CPU tier waits for their load without
//!   taking budget. The load starts once the step that admitted it has run and the loads before it
//!   have finished (the connector issues them on one stream), and takes E0's restore term. The step
//!   loop stalls for `restore_stall` of it, charged to the first step that starts during the load.
//!   The engine learns that a load has finished from the output of the step in flight, and has
//!   already scheduled the step after that one: the request computes from the second step that
//!   starts after its load, or at once when nothing else runs. A prefetch ([`Scheduler::load`]) is a
//!   load that no request waits for.
//! - **Timing.** A step lasts what [`StepCost`] says for its load. At its end a request whose prompt
//!   is complete has its first token, a decode one more, and a request with all its output tokens
//!   finishes: the engine writes its blocks through and frees them. A request that arrives during a
//!   step waits for the next one.
//!
//! Steps are otherwise synchronous. vLLM schedules the next step while the current one runs; a GPU
//! hit that arrives beside one decode waits about half a step in the measured runs, as it does here,
//! not the step and a half that scheduling a step ahead would add. `max_num_seqs` (256) never binds
//! at these loads, and nothing is preempted, because a request reserves its whole footprint at
//! admission.

use std::collections::VecDeque;

use lh_core::{StepCost, StepLoad, TierCost};
use serde::Serialize;

use crate::engine::{Admission, Engine, EngineRequest};
use crate::prefix::IdMap;

/// The step budget: vLLM's `max_num_batched_tokens` and `long_prefill_token_threshold`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StepBudget {
    pub max_batched_tokens: u64,
    pub long_prefill_threshold: u64,
}

impl StepBudget {
    /// The frozen box: `--max-num-batched-tokens 512 --long-prefill-token-threshold 256`.
    pub const FROZEN_BOX: Self = Self {
        max_batched_tokens: 512,
        long_prefill_threshold: 256,
    };
}

/// What happened to a request at a step boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It was admitted into the cache model after waiting `blocked_ns` for free blocks.
    Admitted {
        id: u64,
        admission: Admission,
        blocked_ns: u64,
    },
    /// Its first token came out.
    FirstToken(u64),
    /// Its last token came out, and the engine has freed its blocks.
    Finished(u64),
}

#[derive(Debug)]
struct Request {
    /// The request until it is admitted.
    pending: Option<EngineRequest>,
    prompt_tokens: u64,
    /// Tokens whose KV exists: those found cached at admission and those computed since.
    computed: u64,
    /// Output tokens still to come.
    output_left: u64,
    blocked_since: Option<u64>,
}

impl Request {
    fn decoding(&self) -> bool {
        self.computed >= self.prompt_tokens
    }
}

/// A load of CPU-tier blocks: an admitted request's, or a prefetch.
#[derive(Debug)]
struct Load {
    /// The request waiting for it.
    id: Option<u64>,
    start: u64,
    end: u64,
    /// The stall it causes the step loop, in ms, until a step is charged with it.
    stall_ms: Option<f64>,
    /// Steps started since it ended.
    steps_after: u32,
}

/// The step in flight.
#[derive(Debug)]
struct Step {
    end: u64,
    /// Each scheduled request and the tokens it computes.
    batch: Vec<(u64, u64)>,
}

/// vLLM's scheduler driving the engine model.
#[derive(Debug)]
pub struct Scheduler {
    budget: StepBudget,
    cost: StepCost,
    tier: TierCost,
    requests: IdMap<Request>,
    /// Arrived and not yet admitted, in arrival order.
    waiting: VecDeque<u64>,
    /// How many requests at the front of `waiting` are waiting for free blocks: those queued when
    /// the head did not fit.
    blocked: usize,
    /// Loads not yet done with, in the order they run.
    loads: Vec<Load>,
    /// In the order they started computing.
    running: Vec<u64>,
    step: Option<Step>,
    /// When the API server is done with the requests sent so far.
    host_free: u64,
}

impl Scheduler {
    pub fn new(budget: StepBudget, cost: StepCost, tier: TierCost) -> Self {
        Self {
            budget,
            cost,
            tier,
            requests: IdMap::default(),
            waiting: VecDeque::new(),
            blocked: 0,
            loads: Vec::new(),
            running: Vec::new(),
            step: None,
            host_free: 0,
        }
    }

    /// Whether a step is in flight.
    pub fn busy(&self) -> bool {
        self.step.is_some()
    }

    /// How long after `now` a request with `prompt_tokens` tokens sent at `now` reaches the
    /// scheduler: the API server prepares one request at a time, in the order they are sent.
    pub fn host_ns(&mut self, prompt_tokens: u64, now: u64) -> u64 {
        let decoding = self
            .running
            .iter()
            .filter(|id| self.requests[id].decoding())
            .count();
        let start = self.host_free.max(now);
        self.host_free = start + ns(self.cost.host_ms(prompt_tokens, decoding as u64));
        self.host_free - now
    }

    /// Loads `tokens` tokens from the CPU tier for no request, issued at `now`: after the loads
    /// before it, stalling the steps beside it as a restore does. Returns when it ends.
    pub fn load(&mut self, now: u64, tokens: u64) -> u64 {
        let restore_ms = self.tier.restore_ms(tokens);
        let start = self.loads.last().map_or(now, |l| l.end.max(now));
        let end = start + ns(restore_ms);
        self.loads.push(Load {
            id: None,
            start,
            end,
            stall_ms: Some(self.cost.restore_stall * restore_ms),
            steps_after: 0,
        });
        end
    }

    /// Request `id` reaches the scheduler; it is considered from the next step on.
    pub fn arrive(&mut self, id: u64, req: EngineRequest) {
        self.requests.insert(
            id,
            Request {
                prompt_tokens: req.prompt_tokens,
                computed: 0,
                output_left: req.output_tokens.max(1),
                pending: Some(req),
                blocked_since: None,
            },
        );
        self.waiting.push_back(id);
    }

    /// Runs the engine at `now`: ends the step that ends now, then starts the next one when any
    /// request can compute. Appends what happened to `out` and returns when to run it next: the end
    /// of the new step, or the next load to end when nothing can compute.
    pub fn run(&mut self, now: u64, engine: &mut Engine, out: &mut Vec<Outcome>) -> Option<u64> {
        if let Some(step) = &self.step {
            if step.end > now {
                return Some(step.end);
            }
            self.end_step(engine, out);
        }
        self.start_step(now, engine, out)
    }

    fn end_step(&mut self, engine: &mut Engine, out: &mut Vec<Outcome>) {
        let step = self.step.take().expect("a step is in flight");
        for (id, n) in step.batch {
            let r = self
                .requests
                .get_mut(&id)
                .expect("scheduled requests exist");
            let prefilling = !r.decoding();
            r.computed += n;
            if !r.decoding() {
                continue;
            }
            if prefilling {
                out.push(Outcome::FirstToken(id));
            }
            r.output_left -= 1;
            if r.output_left == 0 {
                self.requests.remove(&id);
                self.running.retain(|&x| x != id);
                engine.finish(id, step.end);
                out.push(Outcome::Finished(id));
            }
        }
    }

    fn start_step(&mut self, now: u64, engine: &mut Engine, out: &mut Vec<Outcome>) -> Option<u64> {
        let mut budget = self.budget.max_batched_tokens;
        let restoring = self.loads.iter().filter(|l| l.id.is_some()).count();
        let cap = if self.running.len() + restoring + self.waiting.len() > 1 {
            self.budget.long_prefill_threshold
        } else {
            budget
        };
        let mut load = StepLoad::default();
        let mut batch = Vec::new();
        let mut schedule = |id: u64, r: &Request, budget: &mut u64| {
            let n = if r.decoding() {
                load.add_decode(r.computed);
                1
            } else {
                let n = (r.prompt_tokens - r.computed).min(cap).min(*budget);
                load.add_prefill(n, r.computed);
                n
            };
            batch.push((id, n));
            *budget -= n;
        };
        for &id in &self.running {
            if budget == 0 {
                break;
            }
            schedule(id, &self.requests[&id], &mut budget);
        }
        // This step carries the stall of the loads running now; a load that ended before any step
        // started stalled nothing.
        let mut stall_ms = 0.0;
        for l in &mut self.loads {
            if l.end <= now {
                l.steps_after += 1;
                l.stall_ms = None;
            } else if l.start <= now {
                stall_ms += l.stall_ms.take().unwrap_or(0.0);
            }
        }
        let alone = self.running.is_empty();
        let mut i = 0;
        while i < self.loads.len() {
            let l = &self.loads[i];
            let done = l.end <= now;
            match l.id {
                None if done => {
                    self.loads.remove(i);
                }
                Some(id) if done && (alone || l.steps_after >= 2) && budget > 0 => {
                    self.loads.remove(i);
                    schedule(id, &self.requests[&id], &mut budget);
                    self.running.push(id);
                }
                _ => i += 1,
            }
        }
        let mut restores = Vec::new();
        while budget > 0
            && let Some(&id) = self.waiting.front()
        {
            let r = self.requests.get_mut(&id).expect("waiting requests exist");
            let req = r.pending.take().expect("a waiting request is not admitted");
            match engine.admit(id, req) {
                Ok(admission) => {
                    self.waiting.pop_front();
                    self.blocked = self.blocked.saturating_sub(1);
                    let blocked_ns = r.blocked_since.map_or(0, |t| now - t);
                    out.push(Outcome::Admitted {
                        id,
                        admission,
                        blocked_ns,
                    });
                    // A prompt found whole recomputes its last token so the next one can be sampled.
                    let cached = admission.sources.gpu + admission.sources.cpu;
                    r.computed = cached.min(r.prompt_tokens - 1);
                    if admission.sources.cpu > 0 {
                        restores.push((id, self.tier.restore_ms(admission.sources.cpu)));
                        continue;
                    }
                    schedule(id, r, &mut budget);
                    self.running.push(id);
                }
                Err(req) => {
                    r.pending = Some(req);
                    // Every request queued now waits for free blocks: a stall, as in the engine's
                    // own queue.
                    for &queued in self.waiting.iter().skip(self.blocked) {
                        self.requests
                            .get_mut(&queued)
                            .expect("waiting requests exist")
                            .blocked_since = Some(now);
                        engine.stats.stalls += 1;
                    }
                    self.blocked = self.waiting.len();
                    break;
                }
            }
        }
        let end = if batch.is_empty() {
            now
        } else {
            now + ns(self.cost.step_ms(&load) + stall_ms).max(1)
        };
        for (id, restore_ms) in restores {
            let start = self.loads.last().map_or(end, |l| l.end.max(end));
            self.loads.push(Load {
                id: Some(id),
                start,
                end: start + ns(restore_ms),
                stall_ms: Some(self.cost.restore_stall * restore_ms),
                steps_after: 0,
            });
        }
        if batch.is_empty() {
            return self.loads.iter().map(|l| l.end).min();
        }
        engine.stats.steps += 1;
        engine.stats.step_tokens += load.tokens;
        engine.stats.step_ns += end - now;
        self.step = Some(Step { end, batch });
        Some(end)
    }
}

fn ns(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        (ms * 1e6).round() as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use lh_core::PromptSources;

    use super::*;
    use crate::engine::{CpuPolicyKind, EngineConfig, GpuPolicyKind};

    /// A step model with round numbers: 10 ms a step, 10 µs a token, nothing for attention.
    fn flat() -> StepCost {
        StepCost {
            step_ms: 10.0,
            us_per_token: 10.0,
            ns_per_pair: 0.0,
            ns_per_context: 0.0,
            mixed_ms: 0.0,
            host: lh_core::Linear {
                ms: 0.0,
                us_per_token: 0.0,
            },
            host_per_decode_ms: 0.0,
            restore_stall: 0.0,
        }
    }

    const MS: u64 = 1_000_000;

    fn engine(gpu_blocks: usize, cpu_blocks: usize) -> Engine {
        Engine::new(
            &EngineConfig {
                gpu_blocks,
                cpu_blocks,
                block_tokens: 16,
                gpu_policy: GpuPolicyKind::Lru,
                cpu_policy: CpuPolicyKind::Lru,
            },
            None,
        )
    }

    /// A request of `tokens` prompt tokens (block ids from `base`) and `out` output tokens.
    fn req(base: u64, tokens: u64, out: u64) -> EngineRequest {
        EngineRequest {
            prompt: (base..base + tokens / 16).collect(),
            prompt_tokens: tokens,
            output_tokens: out,
        }
    }

    /// Runs the scheduler until it is idle, returning each outcome with its time.
    fn drain(s: &mut Scheduler, e: &mut Engine, mut now: u64) -> Vec<(u64, Outcome)> {
        let mut log = Vec::new();
        let mut out = Vec::new();
        while let Some(next) = s.run(now, e, &mut out) {
            log.extend(out.drain(..).map(|o| (now, o)));
            now = next;
        }
        log.extend(out.drain(..).map(|o| (now, o)));
        log
    }

    fn at(log: &[(u64, Outcome)], what: Outcome) -> u64 {
        log.iter()
            .find(|(_, o)| *o == what)
            .map(|&(t, _)| t)
            .unwrap_or_else(|| panic!("{what:?} did not happen"))
    }

    #[test]
    fn a_lone_prefill_takes_the_whole_budget() {
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), TierCost::frozen_box());
        // 1,200 tokens: steps of 512, 512 and 176 tokens, then 2 decodes.
        s.arrive(1, req(100, 1_200, 3));
        let log = drain(&mut s, &mut e, 0);
        let steps = [512, 512, 176, 1, 1].map(|n| 10 * MS + n * 10_000);
        let first = steps[..3].iter().sum::<u64>();
        assert_eq!(at(&log, Outcome::FirstToken(1)), first);
        assert_eq!(at(&log, Outcome::Finished(1)), steps.iter().sum::<u64>());
    }

    #[test]
    fn two_prefills_split_the_budget_at_the_threshold() {
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), TierCost::frozen_box());
        s.arrive(1, req(100, 800, 1));
        s.arrive(2, req(200, 400, 1));
        let log = drain(&mut s, &mut e, 0);
        // Step 1: 256 + 256 (two eligible requests); step 2: 256 + 144; request 2 is done.
        // Step 3: request 1 alone takes the rest, 288 tokens.
        let s1 = 10 * MS + 512 * 10_000;
        let s2 = 10 * MS + 400 * 10_000;
        let s3 = 10 * MS + 288 * 10_000;
        assert_eq!(at(&log, Outcome::Finished(2)), s1 + s2);
        assert_eq!(at(&log, Outcome::Finished(1)), s1 + s2 + s3);
    }

    #[test]
    fn decodes_behind_two_prefills_wait_for_budget() {
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), TierCost::frozen_box());
        let mut out = Vec::new();
        // Two long prefills admitted first, then a short request that decodes 3 tokens.
        s.arrive(1, req(100, 2_048, 1));
        s.arrive(2, req(300, 2_048, 1));
        s.arrive(3, req(500, 16, 3));
        let end = s.run(0, &mut e, &mut out).unwrap();
        // The prefills take 256 each and spend the budget: request 3 is not even admitted.
        assert_eq!(end, 10 * MS + 512 * 10_000);
        assert_eq!(out.len(), 2);
        let log = drain(&mut s, &mut e, end);
        // It waits until both prefills finish (8 steps of 512), then prefills and decodes alone.
        let prefill_end = 8 * (10 * MS + 512 * 10_000);
        let alone = |n: u64| 10 * MS + n * 10_000;
        assert_eq!(at(&log, Outcome::FirstToken(3)), prefill_end + alone(16));
        assert_eq!(
            at(&log, Outcome::Finished(3)),
            prefill_end + alone(16) + 2 * alone(1)
        );
    }

    #[test]
    fn a_request_that_does_not_fit_holds_back_the_queue() {
        // 8 usable blocks: request 1 takes 6, request 2 needs 4 and waits, request 3 waits behind it.
        let mut e = engine(9, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), TierCost::frozen_box());
        s.arrive(1, req(100, 96, 1));
        s.arrive(2, req(200, 64, 1));
        s.arrive(3, req(300, 16, 1));
        let log = drain(&mut s, &mut e, 0);
        let first = 10 * MS + 96 * 10_000;
        assert_eq!(at(&log, Outcome::Finished(1)), first);
        let blocked = log
            .iter()
            .find_map(|(t, o)| match *o {
                Outcome::Admitted {
                    id: 2, blocked_ns, ..
                } => Some((*t, blocked_ns)),
                _ => None,
            })
            .unwrap();
        assert_eq!(blocked, (first, first));
        assert_eq!(e.stats.stalls, 2);
    }

    #[test]
    fn a_restore_waits_for_its_transfer_without_taking_budget() {
        let mut e = engine(9, 64);
        let tier = TierCost::frozen_box();
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), tier.clone());
        // Cache 6 blocks, push them out of the GPU with 8 others, then come back.
        s.arrive(1, req(100, 96, 1));
        drain(&mut s, &mut e, 0);
        s.arrive(2, req(200, 128, 1));
        drain(&mut s, &mut e, 0);
        s.arrive(3, req(100, 112, 1));
        let log = drain(&mut s, &mut e, 0);
        let restore = ns(tier.restore_ms(96));
        let adm = log
            .iter()
            .find_map(|(_, o)| match *o {
                Outcome::Admitted {
                    id: 3, admission, ..
                } => Some(admission),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            adm.sources,
            PromptSources {
                gpu: 0,
                cpu: 96,
                computed: 16
            }
        );
        assert_eq!(
            at(&log, Outcome::FirstToken(3)),
            restore + 10 * MS + 16 * 10_000
        );
    }

    #[test]
    fn a_load_starts_after_its_step_and_its_request_computes_two_steps_after_it() {
        let cost = StepCost {
            restore_stall: 0.5,
            ..flat()
        };
        let tier = TierCost::frozen_box();
        let mut e = engine(17, 64);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, cost, tier.clone());
        // Write two 4-block prompts through to the CPU tier, then push them out of the GPU.
        s.arrive(1, req(100, 64, 1));
        s.arrive(2, req(200, 64, 1));
        drain(&mut s, &mut e, 0);
        s.arrive(3, req(300, 256, 1));
        drain(&mut s, &mut e, 0);
        // A request decoding 3 tokens beside two restores of those prompts (one more block each).
        let t0 = 10_000 * MS;
        s.arrive(4, req(900, 16, 4));
        s.arrive(5, req(100, 80, 1));
        s.arrive(6, req(200, 80, 1));
        let mut out = Vec::new();
        // Step 1 prefills request 4 and admits both restores; it does not stall.
        let end1 = s.run(t0, &mut e, &mut out).unwrap();
        assert_eq!(end1, t0 + ns(10.16));
        let restore = tier.restore_ms(64);
        let stalled = ns(10.01 + 0.5 * restore);
        // Step 2 runs during request 5's load; step 3 during request 6's, which started after it.
        let end2 = s.run(end1, &mut e, &mut out).unwrap();
        assert_eq!(end2, end1 + stalled);
        let log = drain(&mut s, &mut e, end2);
        let end3 = end2 + stalled;
        // Request 5's load ended during step 2: step 3 was already scheduled, so it computes in step
        // 4, beside request 4's last decode. Request 6's ended during step 3: it computes in step 5.
        let end4 = end3 + ns(10.17);
        assert_eq!(at(&log, Outcome::FirstToken(5)), end4);
        assert_eq!(at(&log, Outcome::Finished(4)), end4);
        assert_eq!(at(&log, Outcome::FirstToken(6)), end4 + ns(10.16));
    }

    #[test]
    fn a_prompt_found_whole_recomputes_its_last_token() {
        let mut e = engine(9, 64);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, flat(), TierCost::frozen_box());
        s.arrive(1, req(100, 64, 1));
        drain(&mut s, &mut e, 0);
        s.arrive(2, req(200, 128, 1));
        drain(&mut s, &mut e, 0);
        // All 4 blocks are in the CPU tier: they are restored, and the last token is computed again.
        s.arrive(3, req(100, 64, 2));
        let log = drain(&mut s, &mut e, 0);
        let restore = ns(TierCost::frozen_box().restore_ms(64));
        assert_eq!(at(&log, Outcome::FirstToken(3)), restore + 10 * MS + 10_000);
        assert_eq!(
            at(&log, Outcome::Finished(3)),
            restore + 2 * (10 * MS + 10_000)
        );
    }

    #[test]
    fn decode_steps_read_every_context() {
        let cost = StepCost {
            ns_per_context: 1_000.0,
            ..flat()
        };
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, cost, TierCost::frozen_box());
        let mut out = Vec::new();
        s.arrive(1, req(100, 160, 2));
        s.arrive(2, req(200, 240, 2));
        // Prefill both (400 tokens, contexts 160 and 240 after the step), then one decode each.
        let t1 = s.run(0, &mut e, &mut out).unwrap();
        assert_eq!(t1, 10 * MS + 400 * 10_000 + 400 * 1_000);
        let t2 = s.run(t1, &mut e, &mut out).unwrap();
        assert_eq!(t2 - t1, 10 * MS + 2 * 10_000 + (161 + 241) * 1_000);
        assert_eq!(s.run(t2, &mut e, &mut out), None);
        let finished = out
            .iter()
            .filter(|o| matches!(o, Outcome::Finished(_)))
            .count();
        assert_eq!(finished, 2);
    }

    #[test]
    fn a_prefetch_runs_with_the_loads_and_stalls_the_steps_beside_it() {
        let cost = StepCost {
            restore_stall: 0.5,
            ..flat()
        };
        let tier = TierCost::frozen_box();
        let mut e = engine(64, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, cost, tier.clone());
        let mut out = Vec::new();
        s.arrive(1, req(100, 16, 3));
        let t1 = s.run(0, &mut e, &mut out).unwrap();
        // Two prefetches issued during the first step: the second starts when the first ends.
        let restore = ns(tier.restore_ms(64));
        assert_eq!(s.load(MS, 64), MS + restore);
        assert_eq!(s.load(MS, 64), MS + 2 * restore);
        // The next step starts during the first and carries its stall.
        let t2 = s.run(t1, &mut e, &mut out).unwrap();
        assert_eq!(t2 - t1, ns(10.01 + 0.5 * tier.restore_ms(64)));
    }

    #[test]
    fn a_prefill_beside_a_decode_pays_the_mixed_step_cost() {
        let cost = StepCost {
            mixed_ms: 5.0,
            ..flat()
        };
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, cost, TierCost::frozen_box());
        let mut out = Vec::new();
        s.arrive(1, req(100, 16, 3));
        let t1 = s.run(0, &mut e, &mut out).unwrap();
        assert_eq!(t1, 10 * MS + 16 * 10_000);
        s.arrive(2, req(200, 32, 1));
        let t2 = s.run(t1, &mut e, &mut out).unwrap();
        assert_eq!(t2 - t1, 15 * MS + 33 * 10_000);
        let t3 = s.run(t2, &mut e, &mut out).unwrap();
        assert_eq!(t3 - t2, 10 * MS + 10_000);
    }

    #[test]
    fn the_api_server_prepares_one_request_at_a_time() {
        let cost = StepCost {
            host: lh_core::Linear {
                ms: 20.0,
                us_per_token: 10.0,
            },
            host_per_decode_ms: 3.0,
            ..flat()
        };
        let mut e = engine(1_024, 0);
        let mut s = Scheduler::new(StepBudget::FROZEN_BOX, cost, TierCost::frozen_box());
        // One request at a time: the second waits for the first.
        assert_eq!(s.host_ns(1_000, 0), 30 * MS);
        assert_eq!(s.host_ns(1_000, 10 * MS), 50 * MS);
        s.arrive(1, req(100, 16, 3));
        let mut out = Vec::new();
        let t1 = s.run(100 * MS, &mut e, &mut out).unwrap();
        // Prefilling, it does not count; decoding, it does.
        assert_eq!(s.host_ns(1_000, t1), 30 * MS);
        let t2 = s.run(t1, &mut e, &mut out).unwrap();
        assert_eq!(s.host_ns(1_000, t2 + 100 * MS), 33 * MS);
    }
}
