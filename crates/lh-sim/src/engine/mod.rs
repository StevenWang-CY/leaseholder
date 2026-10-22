//! The engine cache model: vLLM 0.32's GPU prefix cache beside the OffloadingConnector CPU tier.
//!
//! - **GPU pool.** A fixed number of blocks, one of them vLLM's reserved null block. A request takes
//!   the longest prefix of its full prompt blocks that the pool caches, at most `⌊(L − 1) / B⌋`
//!   blocks: vLLM always computes the last prompt token. Blocks of running requests are pinned
//!   (reference counted, shared between requests with a common prefix). A finished request's blocks
//!   are freed in reverse order: cached blocks join the back of the free queue, so the end of a prefix
//!   is reused before its start, and the partial last block joins the front. A block recomputed while
//!   a copy is cached (the last block of a fully cached prompt, or one past a gap in a cached prefix)
//!   is cached a second time, as vLLM keeps every block per hash; a hit takes a copy that running
//!   requests already pin when there is one.
//! - **CPU tier.** After the GPU hit, the tier extends it with the following blocks it holds (up to
//!   every full prompt block); they are restored into newly allocated GPU blocks. Every full block a
//!   request holds at its end, prompt and output, is offered to the tier (write-through,
//!   `offload_prompt_only: false`), from the start of the prompt, or, when the request restored blocks
//!   from the tier, from after them (vLLM's chunk-level store index).
//! - **Admission.** A request needs blocks for `L + out − 1` tokens. When the pool cannot supply them
//!   without touching pinned blocks, the request waits, first come first served, until finishing
//!   requests free enough; vLLM's scheduler stops at the first waiting request that does not fit.
//!   Requests reserve their whole footprint at admission, so a running request never needs to be
//!   preempted.

pub mod cpu;
pub mod gpu;
pub mod oracle;

use std::collections::VecDeque;
use std::rc::Rc;

use lh_core::PromptSources;
use serde::Serialize;

use self::cpu::{CpuClairvoyant, CpuLru, CpuPolicy, CpuS3Fifo};
use self::gpu::{ClairvoyantFree, GpuPolicy, LruQueue};
use self::oracle::Oracle;
use crate::prefix::{BlockId, IdMap, UNIQUE_BIT};

const NO_BLOCK: BlockId = u64::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuPolicyKind {
    /// vLLM's free queue.
    Lru,
    /// Belady's MIN over the recorded request sequence.
    Clairvoyant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CpuPolicyKind {
    /// vLLM's `LRUCachePolicy`.
    Lru,
    S3Fifo,
    /// Belady's MIN over the recorded request sequence.
    Clairvoyant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EngineConfig {
    /// vLLM's `num_gpu_blocks`, including the null block.
    pub gpu_blocks: usize,
    /// Blocks in the CPU tier; 0 for none.
    pub cpu_blocks: usize,
    pub block_tokens: u64,
    pub gpu_policy: GpuPolicyKind,
    pub cpu_policy: CpuPolicyKind,
}

/// One request as the engine sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineRequest {
    /// Identities of the full prompt blocks, in order.
    pub prompt: Vec<BlockId>,
    pub prompt_tokens: u64,
    /// Decode length, at least 1.
    pub output_tokens: u64,
}

/// Where an admitted request's prompt came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Admission {
    pub sources: PromptSources,
    /// GPU-hit blocks.
    pub gpu_blocks: u32,
    /// Blocks restored from the CPU tier.
    pub cpu_blocks: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct EngineStats {
    pub admitted: u64,
    /// Requests that had to wait for free blocks.
    pub stalls: u64,
    /// Cached GPU blocks overwritten by a new allocation.
    pub gpu_evictions: u64,
    pub cpu_evictions: u64,
    /// Blocks written to the CPU tier.
    pub cpu_stored_blocks: u64,
    /// Offers the CPU tier could not make room for.
    pub cpu_store_failures: u64,
    pub restored_blocks: u64,
    pub peak_pinned_blocks: usize,
    /// Requests whose footprint exceeds the whole pool. No engine admits them (vLLM refuses to
    /// start when `max_model_len` exceeds its pool), so they are dropped.
    pub oversized: u64,
}

#[derive(Debug)]
struct Running {
    slots: Vec<u32>,
    prompt: Vec<BlockId>,
    gpu_blocks: usize,
    cpu_blocks: usize,
    full_blocks: usize,
}

/// The GPU pool, the optional CPU tier and the wait queue.
#[derive(Debug)]
pub struct Engine {
    block_tokens: u64,
    refs: Vec<u32>,
    ids: Vec<BlockId>,
    /// The cached copy of each block that hits take, and its other copies.
    cached: IdMap<u32>,
    copies: IdMap<Vec<u32>>,
    gpu: Box<dyn GpuPolicy>,
    cpu: Option<Box<dyn CpuPolicy>>,
    oracle: Option<Rc<Oracle>>,
    queue: VecDeque<(u64, EngineRequest)>,
    running: IdMap<Running>,
    next_unique: u64,
    pinned: usize,
    pub stats: EngineStats,
}

impl Engine {
    /// A cold engine. The clairvoyant policies need `oracle`.
    pub fn new(cfg: &EngineConfig, oracle: Option<Rc<Oracle>>) -> Self {
        let usable = cfg.gpu_blocks.saturating_sub(1).max(1) as u32;
        let need_oracle = || {
            oracle
                .clone()
                .expect("the clairvoyant policies need a next-use oracle")
        };
        let gpu: Box<dyn GpuPolicy> = match cfg.gpu_policy {
            GpuPolicyKind::Lru => Box::new(LruQueue::new(usable)),
            GpuPolicyKind::Clairvoyant => Box::new(ClairvoyantFree::new(usable, need_oracle())),
        };
        let cpu: Option<Box<dyn CpuPolicy>> = match (cfg.cpu_blocks, cfg.cpu_policy) {
            (0, _) => None,
            (n, CpuPolicyKind::Lru) => Some(Box::new(CpuLru::new(n))),
            (n, CpuPolicyKind::S3Fifo) => Some(Box::new(CpuS3Fifo::new(n))),
            (n, CpuPolicyKind::Clairvoyant) => {
                Some(Box::new(CpuClairvoyant::new(n, need_oracle())))
            }
        };
        Self {
            block_tokens: cfg.block_tokens,
            refs: vec![0; usable as usize],
            ids: vec![NO_BLOCK; usable as usize],
            cached: IdMap::default(),
            copies: IdMap::default(),
            gpu,
            cpu,
            oracle,
            queue: VecDeque::new(),
            running: IdMap::default(),
            next_unique: 0,
            pinned: 0,
            stats: EngineStats::default(),
        }
    }

    /// GPU blocks that can hold KV (vLLM's pool minus the null block).
    pub fn gpu_capacity(&self) -> usize {
        self.refs.len()
    }

    pub fn cpu_len(&self) -> usize {
        self.cpu.as_ref().map_or(0, |c| c.len())
    }

    pub fn waiting(&self) -> usize {
        self.queue.len()
    }

    /// Admits `req` now, or queues it behind earlier waiting requests. A request larger than the
    /// whole pool is dropped and counted in [`EngineStats::oversized`].
    pub fn submit(&mut self, id: u64, req: EngineRequest) -> Option<Admission> {
        let footprint =
            (req.prompt_tokens + req.output_tokens.max(1) - 1).div_ceil(self.block_tokens);
        if footprint > self.gpu_capacity() as u64 {
            self.stats.oversized += 1;
            return None;
        }
        if self.queue.is_empty() {
            match self.try_admit(id, req) {
                Ok(a) => return Some(a),
                Err(req) => self.queue.push_back((id, req)),
            }
        } else {
            self.queue.push_back((id, req));
        }
        self.stats.stalls += 1;
        None
    }

    /// Ends request `id`: writes its blocks through to the CPU tier, frees its GPU blocks, and
    /// returns the waiting requests admitted into the freed space, in order.
    pub fn finish(&mut self, id: u64) -> Vec<(u64, Admission)> {
        let run = self
            .running
            .remove(&id)
            .expect("finish of a request that is not running");
        let full_prompt = run.prompt.len();
        if let Some(cpu) = &mut self.cpu {
            let from = if run.cpu_blocks > 0 {
                run.gpu_blocks + run.cpu_blocks
            } else {
                0
            };
            let mut offer: Vec<(BlockId, u32)> = (from..full_prompt)
                .map(|j| (run.prompt[j], j as u32))
                .collect();
            offer.extend(
                (full_prompt..run.full_blocks).map(|j| (self.ids[run.slots[j] as usize], j as u32)),
            );
            match cpu.store(&offer) {
                Some(out) => {
                    self.stats.cpu_stored_blocks += out.stored as u64;
                    self.stats.cpu_evictions += out.evicted as u64;
                }
                None => self.stats.cpu_store_failures += 1,
            }
            let loaded = &run.prompt[run.gpu_blocks..run.gpu_blocks + run.cpu_blocks];
            let mut touched: Vec<BlockId> = loaded.to_vec();
            touched.extend(offer.iter().map(|&(b, _)| b));
            cpu.on_finish(&touched);
        }
        for (j, &slot) in run.slots.iter().enumerate().rev() {
            let s = slot as usize;
            self.refs[s] -= 1;
            if self.refs[s] == 0 {
                self.pinned -= 1;
                let id = self.ids[s];
                self.gpu
                    .free(slot, (id != NO_BLOCK).then_some(id), j as u32);
            }
        }
        let mut admitted = Vec::new();
        while let Some((qid, req)) = self.queue.pop_front() {
            match self.try_admit(qid, req) {
                Ok(a) => admitted.push((qid, a)),
                Err(req) => {
                    self.queue.push_front((qid, req));
                    break;
                }
            }
        }
        admitted
    }

    fn try_admit(&mut self, id: u64, req: EngineRequest) -> Result<Admission, EngineRequest> {
        let bt = self.block_tokens;
        let full = req.prompt.len();
        let max_hit = full.min((req.prompt_tokens.saturating_sub(1) / bt) as usize);
        let mut g = 0;
        while g < max_hit && self.cached.contains_key(&req.prompt[g]) {
            g += 1;
        }
        let mut c = g;
        if let Some(cpu) = &self.cpu {
            while c < full && cpu.contains(req.prompt[c]) {
                c += 1;
            }
        }
        let kv_tokens = req.prompt_tokens + req.output_tokens.max(1) - 1;
        let total_blocks = kv_tokens.div_ceil(bt) as usize;
        let full_blocks = (kv_tokens / bt) as usize;
        let free_hits = req.prompt[..g]
            .iter()
            .filter(|&&b| self.refs[self.hit_slot(b) as usize] == 0)
            .count();
        if total_blocks - g > self.gpu.free_blocks() - free_hits {
            return Err(req);
        }

        self.stats.admitted += 1;
        if let Some(o) = &self.oracle {
            o.set_position(self.stats.admitted as u32);
            // Cached blocks past the hit are used by this request without being hit; their next
            // use moves on.
            for (j, &b) in req.prompt.iter().enumerate().skip(g) {
                let Some(&primary) = self.cached.get(&b) else {
                    continue;
                };
                let others = self.copies.get(&b).map_or(&[][..], Vec::as_slice);
                for &slot in std::iter::once(&primary).chain(others) {
                    if self.refs[slot as usize] == 0 {
                        self.gpu.refresh(slot, b, j as u32);
                    }
                }
            }
        }
        if let Some(cpu) = &mut self.cpu {
            cpu.on_admit(&req.prompt);
        }
        let mut slots = Vec::with_capacity(total_blocks);
        for &b in &req.prompt[..g] {
            let slot = self.hit_slot(b);
            if self.refs[slot as usize] == 0 {
                self.gpu.reuse(slot);
                self.pinned += 1;
            }
            self.refs[slot as usize] += 1;
            slots.push(slot);
        }
        for j in g..total_blocks {
            let slot = self.gpu.take();
            let s = slot as usize;
            let old = self.ids[s];
            if old != NO_BLOCK {
                self.stats.gpu_evictions += 1;
                self.forget(slot, old);
            }
            let block = if j < full {
                let b = req.prompt[j];
                if let Some(&primary) = self.cached.get(&b) {
                    debug_assert_ne!(primary, slot);
                    self.copies.entry(b).or_default().push(slot);
                } else {
                    self.cached.insert(b, slot);
                }
                b
            } else if j < full_blocks {
                self.next_unique += 1;
                UNIQUE_BIT | self.next_unique
            } else {
                NO_BLOCK
            };
            self.ids[s] = block;
            self.refs[s] = 1;
            self.pinned += 1;
            slots.push(slot);
        }
        self.stats.peak_pinned_blocks = self.stats.peak_pinned_blocks.max(self.pinned);
        if c > g {
            if let Some(cpu) = &mut self.cpu {
                cpu.on_load(&req.prompt[g..c]);
            }
            self.stats.restored_blocks += (c - g) as u64;
        }
        let gpu_tokens = g as u64 * bt;
        let cpu_tokens = (c - g) as u64 * bt;
        let admission = Admission {
            sources: PromptSources {
                gpu: gpu_tokens,
                cpu: cpu_tokens,
                computed: req.prompt_tokens - gpu_tokens - cpu_tokens,
            },
            gpu_blocks: g as u32,
            cpu_blocks: (c - g) as u32,
        };
        self.running.insert(
            id,
            Running {
                slots,
                prompt: req.prompt,
                gpu_blocks: g,
                cpu_blocks: c - g,
                full_blocks,
            },
        );
        Ok(admission)
    }

    /// The copy of cached block `b` a hit takes: one that running requests pin if there is one, so
    /// a hit on a running request's prefix never takes a free block.
    fn hit_slot(&self, b: BlockId) -> u32 {
        let primary = self.cached[&b];
        if self.refs[primary as usize] > 0 {
            return primary;
        }
        self.copies
            .get(&b)
            .and_then(|others| others.iter().copied().find(|&s| self.refs[s as usize] > 0))
            .unwrap_or(primary)
    }

    /// `slot` no longer holds block `b`.
    fn forget(&mut self, slot: u32, b: BlockId) {
        let Some(others) = self.copies.get_mut(&b) else {
            if self.cached.get(&b) == Some(&slot) {
                self.cached.remove(&b);
            }
            return;
        };
        if self.cached.get(&b) == Some(&slot) {
            let next = others.remove(0);
            self.cached.insert(b, next);
        } else if let Some(k) = others.iter().position(|&s| s == slot) {
            others.remove(k);
        }
        if others.is_empty() {
            self.copies.remove(&b);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

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

    /// A prompt of `blocks` full blocks plus `extra` tokens, identities `base..base + blocks`.
    fn req(base: u64, blocks: u64, extra: u64, out: u64) -> EngineRequest {
        EngineRequest {
            prompt: (base..base + blocks).collect(),
            prompt_tokens: blocks * 16 + extra,
            output_tokens: out,
        }
    }

    #[test]
    fn a_shared_prefix_is_a_hit_and_the_last_token_is_computed() {
        let mut e = engine(64, 0);
        let a = e.submit(1, req(100, 4, 5, 1)).unwrap();
        assert_eq!(
            a.sources,
            PromptSources {
                gpu: 0,
                cpu: 0,
                computed: 69
            }
        );
        e.finish(1);
        // Same 4 blocks plus 3 new ones.
        let b = e.submit(2, req(100, 7, 0, 1)).unwrap();
        assert_eq!((b.gpu_blocks, b.sources.computed), (4, 48));
        e.finish(2);
        // A block-aligned prompt that is fully cached still computes its last block.
        let c = e.submit(3, req(100, 7, 0, 1)).unwrap();
        assert_eq!((c.gpu_blocks, c.sources.computed), (6, 16));
    }

    #[test]
    fn freeing_evicts_the_end_of_a_prefix_first() {
        // 8 usable blocks.
        let mut e = engine(9, 0);
        e.submit(1, req(100, 6, 0, 1)).unwrap();
        e.finish(1);
        // A 4-block request reuses the 2 never-used blocks, then the cached tail 105 and 104.
        e.submit(2, req(200, 4, 0, 1)).unwrap();
        e.finish(2);
        assert_eq!(e.stats.gpu_evictions, 2);
        let again = e.submit(3, req(100, 6, 0, 1)).unwrap();
        assert_eq!(
            again.gpu_blocks, 4,
            "blocks 100..103 survived, 104 and 105 were evicted"
        );
    }

    #[test]
    fn pinned_blocks_make_requests_wait() {
        let mut e = engine(9, 0);
        assert!(e.submit(1, req(100, 6, 0, 1)).is_some());
        // 4 more blocks do not fit beside the 6 pinned ones.
        assert!(e.submit(2, req(200, 4, 0, 1)).is_none());
        // A later small request waits behind it (vLLM stops at the first that does not fit).
        assert!(e.submit(3, req(300, 1, 0, 1)).is_none());
        assert_eq!((e.stats.stalls, e.waiting()), (2, 2));
        let admitted = e.finish(1);
        assert_eq!(
            admitted.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(e.stats.peak_pinned_blocks, 6);
    }

    #[test]
    fn concurrent_requests_share_pinned_prefix_blocks() {
        let mut e = engine(9, 0);
        e.submit(1, req(100, 6, 0, 1)).unwrap();
        // The same prefix while the first request runs: a hit on its pinned blocks.
        let b = e.submit(2, req(100, 7, 0, 1)).unwrap();
        assert_eq!(b.gpu_blocks, 6);
        assert_eq!(e.stats.peak_pinned_blocks, 7);
    }

    #[test]
    fn evicted_blocks_are_restored_from_the_cpu_tier() {
        let mut e = engine(9, 64);
        e.submit(1, req(100, 6, 0, 1)).unwrap();
        e.finish(1);
        assert_eq!(
            e.stats.cpu_stored_blocks, 6,
            "write-through of every full block"
        );
        e.submit(2, req(200, 8, 0, 1)).unwrap();
        e.finish(2);
        // The GPU lost 100..105, the CPU tier still has them.
        let back = e.submit(3, req(100, 7, 0, 1)).unwrap();
        assert_eq!((back.gpu_blocks, back.cpu_blocks), (0, 6));
        assert_eq!(
            back.sources,
            PromptSources {
                gpu: 0,
                cpu: 96,
                computed: 16
            }
        );
        assert_eq!(e.stats.restored_blocks, 6);
        e.finish(3);
        // After a restore only the blocks past the restored prefix are offered: one new block.
        assert_eq!(e.stats.cpu_stored_blocks, 6 + 8 + 1);
    }

    #[test]
    fn a_recomputed_block_stays_cached_after_its_first_copy_is_evicted() {
        // 4 usable blocks.
        let mut e = engine(5, 0);
        e.submit(1, req(100, 2, 0, 1)).unwrap();
        e.finish(1);
        // Fully cached and block aligned: 101 is recomputed into a second copy.
        let again = e.submit(2, req(100, 2, 0, 1)).unwrap();
        assert_eq!(again.gpu_blocks, 1);
        e.finish(2);
        // A new prompt evicts the never-used block and then the first copy of 101.
        e.submit(3, req(200, 2, 0, 1)).unwrap();
        e.finish(3);
        assert_eq!(e.stats.gpu_evictions, 1);
        let back = e.submit(4, req(100, 2, 1, 1)).unwrap();
        assert_eq!(back.gpu_blocks, 2, "the second copy of 101 still hits");
    }

    #[test]
    fn a_hit_takes_the_copy_running_requests_pin() {
        // 5 usable blocks.
        let mut e = engine(6, 0);
        e.submit(1, req(100, 2, 0, 1)).unwrap();
        e.finish(1);
        // Running: 100 and a pinned second copy of 101; the first copy of 101 is free.
        e.submit(2, req(100, 2, 0, 1)).unwrap();
        let b = e.submit(3, req(100, 2, 1, 1)).unwrap();
        assert_eq!(b.gpu_blocks, 2);
        // Request 3 pins only its new partial block beside the two shared ones.
        assert_eq!(e.stats.peak_pinned_blocks, 3);
    }

    #[test]
    fn output_blocks_take_space_and_are_written_through() {
        let mut e = engine(64, 64);
        // 2 prompt blocks and 40 output tokens: 32 + 39 KV tokens = 4 full blocks and a partial one.
        e.submit(1, req(100, 2, 0, 40)).unwrap();
        assert_eq!(e.stats.peak_pinned_blocks, 5);
        e.finish(1);
        assert_eq!(e.stats.cpu_stored_blocks, 4);
        // Output blocks are never shared.
        let again = e.submit(2, req(100, 2, 0, 1)).unwrap();
        assert_eq!(again.gpu_blocks, 1);
    }

    /// The most hits a demand-paged cache of `capacity` blocks gets, by exhaustive search over the
    /// held sets (blocks are 0..32, a set is a bit mask).
    fn best_demand_hits(seq: &[BlockId], capacity: usize) -> usize {
        fn go(
            seq: &[BlockId],
            i: usize,
            held: u32,
            capacity: usize,
            memo: &mut HashMap<(usize, u32), usize>,
        ) -> usize {
            let Some(&b) = seq.get(i) else { return 0 };
            if let Some(&v) = memo.get(&(i, held)) {
                return v;
            }
            let v = if held >> b & 1 == 1 {
                1 + go(seq, i + 1, held, capacity, memo)
            } else if (held.count_ones() as usize) < capacity {
                go(seq, i + 1, held | 1 << b, capacity, memo)
            } else {
                (0..32)
                    .filter(|k| held >> k & 1 == 1)
                    .map(|k| go(seq, i + 1, (held & !(1 << k)) | 1 << b, capacity, memo))
                    .max()
                    .expect("a full cache holds blocks")
            };
            memo.insert((i, held), v);
            v
        }
        go(seq, 0, 0, capacity, &mut HashMap::new())
    }

    #[test]
    fn the_gpu_clairvoyant_matches_the_exhaustive_optimum() {
        // One-block prompts, one request at a time: the pool caches `capacity` blocks beside the one
        // block each request's partial tail takes.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for case in 0..200 {
            let seq: Vec<BlockId> = (0..12)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    state >> 61
                })
                .collect();
            let capacity = 1 + case % 3;
            let reqs: Vec<Vec<BlockId>> = seq.iter().map(|&b| vec![b]).collect();
            let oracle = Rc::new(Oracle::new(reqs.iter().map(Vec::as_slice)));
            let cfg = EngineConfig {
                gpu_blocks: capacity + 2,
                cpu_blocks: 0,
                block_tokens: 16,
                gpu_policy: GpuPolicyKind::Clairvoyant,
                cpu_policy: CpuPolicyKind::Lru,
            };
            let mut e = Engine::new(&cfg, Some(oracle));
            let mut hits = 0;
            for (i, prompt) in reqs.into_iter().enumerate() {
                let a = e
                    .submit(
                        i as u64,
                        EngineRequest {
                            prompt,
                            prompt_tokens: 17,
                            output_tokens: 1,
                        },
                    )
                    .unwrap();
                hits += a.gpu_blocks as usize;
                e.finish(i as u64);
            }
            assert_eq!(
                hits,
                best_demand_hits(&seq, capacity),
                "{seq:?} at capacity {capacity}"
            );
        }
    }

    #[test]
    fn a_gpu_only_engine_recomputes_what_it_evicted() {
        let mut e = engine(9, 0);
        e.submit(1, req(100, 6, 0, 1)).unwrap();
        e.finish(1);
        e.submit(2, req(200, 8, 0, 1)).unwrap();
        e.finish(2);
        let back = e.submit(3, req(100, 6, 0, 1)).unwrap();
        assert_eq!(
            (back.gpu_blocks, back.cpu_blocks, back.sources.computed),
            (0, 0, 96)
        );
    }
}
