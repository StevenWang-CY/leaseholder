//! Miss-ratio curves of the GPU tier alone, over the access sequence of a recorded run.
//!
//! The sequence is the run's admissions in order. A request looks up its full prompt blocks, then
//! its blocks become the most recent in the order vLLM frees them: output blocks first, then the
//! prompt from its last block to its first, so the start of a prefix is the most recent. (vLLM
//! reorders when a request finishes; the curves reorder at admission, so they ignore pinning.) Hits
//! are token weighted and counted two ways: per block, and per prefix, where a block counts only when
//! every block before it hits too, as in the engine. vLLM always computes a prompt's last token, so
//! the last block of a block-aligned prompt never hits.
//!
//! - **LRU**: Mattson's stack algorithm gives the hits at every capacity in one pass. A block hits at
//!   capacity `C` iff its stack distance (distinct blocks since its last use, itself included) is at
//!   most `C`; with prefix semantics, iff the largest distance up to it is.
//! - **MIN**: Belady's farthest-next-use eviction with bypass, one pass per capacity: after each
//!   request the cache keeps the blocks with the nearest next uses, and blocks never used again
//!   (output blocks among them) are not kept. Its per-block hits are the most any policy gets at that
//!   capacity on this sequence, so they bound the per-block hit ratio. Its prefix hits are a
//!   reference, not a bound.

use std::collections::BinaryHeap;

use crate::corpus::Corpus;
use crate::prefix::{BlockId, IdMap, NodeId, Tokens};
use crate::replay::Timeline;

/// One request of an access sequence.
#[derive(Clone, Copy, Debug)]
pub struct Access<'a> {
    /// Full prompt blocks, in order.
    pub prompt: &'a [BlockId],
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Its hits count (it is a profiling request of the run); the others only change the cache.
    pub counted: bool,
}

/// The accesses of a recorded run, in admission order.
pub fn accesses(timeline: &Timeline) -> Vec<Access<'_>> {
    timeline
        .requests
        .iter()
        .map(|r| Access {
            prompt: &r.prompt,
            prompt_tokens: r.prompt_tokens,
            output_tokens: r.output_tokens,
            counted: r.record.is_some(),
        })
        .collect()
}

/// The prompts of a recorded run cut again with every hash block and every output `factor` times
/// larger: the same admission order at unscaled token counts.
pub fn rescaled(
    corpus: &mut Corpus,
    timeline: &Timeline,
    factor: u32,
    block_tokens: u32,
) -> Result<Vec<(Vec<BlockId>, u64, u64)>, String> {
    let mut scratch: Vec<NodeId> = Vec::new();
    timeline
        .requests
        .iter()
        .map(|r| {
            let plan = corpus.plan(r.trace as usize)?;
            let turn = &plan.convs[r.conv as usize].turns[r.turn as usize];
            let tokens = Tokens::new(plan.block_size * factor, block_tokens);
            let mut prompt = Vec::new();
            plan.prefixes
                .engine_blocks(turn.prompt_end, tokens, r.play, &mut scratch, &mut prompt);
            let output = if r.warmup {
                r.output_tokens
            } else {
                r.output_tokens * factor as u64
            };
            Ok((prompt, tokens.prompt_tokens(turn.hash_blocks), output))
        })
        .collect()
}

impl Access<'_> {
    /// Prompt blocks a lookup can hit: vLLM computes at least the last prompt token.
    fn hittable(&self, block_tokens: u64) -> usize {
        self.prompt
            .len()
            .min((self.prompt_tokens.saturating_sub(1) / block_tokens) as usize)
    }

    /// Full blocks of output KV: never reused, but they take cache space.
    fn output_blocks(&self, block_tokens: u64) -> usize {
        let kv = self.prompt_tokens + self.output_tokens.max(1) - 1;
        ((kv / block_tokens) as usize).saturating_sub(self.prompt.len())
    }
}

/// Hit tokens at one capacity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Point {
    /// Cache capacity in blocks.
    pub capacity: u64,
    pub lru_block_hits: u64,
    pub lru_prefix_hits: u64,
    /// MIN results exist at the capacities MIN was run at.
    pub min_block_hits: Option<u64>,
    pub min_prefix_hits: Option<u64>,
}

/// A miss-ratio curve: hit tokens at a set of capacities, out of `prompt_tokens`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub points: Vec<Point>,
}

impl Curve {
    /// LRU at every capacity in `lru` (one stack pass) and MIN at every capacity in `min`.
    pub fn compute(accesses: &[Access<'_>], block_tokens: u64, lru: &[u64], min: &[u64]) -> Self {
        let mut capacities: Vec<u64> = lru.iter().chain(min).copied().collect();
        capacities.sort_unstable();
        capacities.dedup();
        let max = capacities.last().copied().unwrap_or(0);
        let (block, prefix) = stack_histograms(accesses, block_tokens, max);
        let next = next_uses(accesses);
        let points = capacities
            .iter()
            .map(|&c| {
                let (min_block_hits, min_prefix_hits) = if min.contains(&c) {
                    let (b, p) = min_hits(accesses, &next, block_tokens, c);
                    (Some(b), Some(p))
                } else {
                    (None, None)
                };
                let upto = |h: &[u64]| {
                    h[..=(c as usize).min(h.len() - 1)].iter().sum::<u64>() * block_tokens
                };
                Point {
                    capacity: c,
                    lru_block_hits: upto(&block),
                    lru_prefix_hits: upto(&prefix),
                    min_block_hits,
                    min_prefix_hits,
                }
            })
            .collect();
        let counted = accesses.iter().filter(|a| a.counted);
        Self {
            requests: counted.clone().count() as u64,
            prompt_tokens: counted.map(|a| a.prompt_tokens).sum(),
            points,
        }
    }
}

/// The capacities, in blocks, at `multiples` of a pool of `pool` blocks: each rounded, at least one.
pub fn capacities(pool: u64, multiples: &[f64]) -> Vec<u64> {
    multiples
        .iter()
        .map(|m| (m * pool as f64).round().max(1.0) as u64)
        .collect()
}

/// The miss-ratio table of the lh-sim README (`lh-sim mrc --markdown`): the per-block miss ratios of
/// LRU and MIN, to three decimals, at each of `multiples` of the pool (where MIN was run), on the
/// replay (`scaled`, with its pool in blocks) and, where there is one, on the same admission order at
/// the original token counts (`same_order`, `factor` times the replay's tokens).
pub fn markdown(
    multiples: &[f64],
    factor: u32,
    scaled: (&Curve, u64),
    same_order: Option<(&Curve, u64)>,
) -> String {
    let curves: Vec<(&Curve, u64)> = std::iter::once(scaled).chain(same_order).collect();
    let mut text = if same_order.is_some() {
        format!(
            "| Pool multiple | LRU, {factor}× | LRU, original tokens, same order | MIN, {factor}× \
             | MIN, original tokens, same order |\n|---|---|---|---|---|\n"
        )
    } else {
        "| Pool multiple | LRU | MIN |\n|---|---|---|\n".to_string()
    };
    for (i, multiple) in multiples.iter().enumerate() {
        let at: Vec<&Point> = curves
            .iter()
            .map(|(curve, pool)| {
                let capacity = capacities(*pool, multiples)[i];
                curve
                    .points
                    .iter()
                    .find(|p| p.capacity == capacity)
                    .expect("the curve has a point at each capacity MIN ran at")
            })
            .collect();
        let miss = |hits: u64, curve: &Curve| 1.0 - hits as f64 / curve.prompt_tokens.max(1) as f64;
        let mut row = format!("| {multiple} |");
        for (point, (curve, _)) in at.iter().zip(&curves) {
            row += &format!(" {:.3} |", miss(point.lru_block_hits, curve));
        }
        for (point, (curve, _)) in at.iter().zip(&curves) {
            let hits = point.min_block_hits.expect("MIN ran at this capacity");
            row += &format!(" {:.3} |", miss(hits, curve));
        }
        text += &row;
        text.push('\n');
    }
    text
}

/// A Fenwick tree of counts over access times.
struct Fenwick(Vec<i32>);

impl Fenwick {
    fn add(&mut self, at: usize, delta: i32) {
        let mut i = at + 1;
        while i < self.0.len() {
            self.0[i] += delta;
            i += i.isolate_lowest_one();
        }
    }

    /// Sum over times `0..=at`.
    fn prefix(&self, at: usize) -> i64 {
        let mut i = at + 1;
        let mut s = 0i64;
        while i > 0 {
            s += self.0[i] as i64;
            i -= i.isolate_lowest_one();
        }
        s
    }
}

/// Histograms (in blocks) of the stack distance of every hittable prompt block, and of the largest
/// distance up to it within its prompt; distances above `max` are left out (they miss at every
/// capacity of interest).
fn stack_histograms(accesses: &[Access<'_>], block_tokens: u64, max: u64) -> (Vec<u64>, Vec<u64>) {
    let touches: usize = accesses
        .iter()
        .map(|a| a.prompt.len() + a.output_blocks(block_tokens))
        .sum();
    let mut tree = Fenwick(vec![0; touches + 1]);
    let mut last: IdMap<u32> = IdMap::default();
    let (mut block, mut prefix) = (vec![0u64; max as usize + 1], vec![0u64; max as usize + 1]);
    let mut now = 0usize;
    for a in accesses {
        let mut deepest = 0u64;
        for b in &a.prompt[..a.hittable(block_tokens)] {
            // Distinct blocks touched at or after b's last touch: b's depth in the LRU stack.
            let depth = last.get(b).map_or(u64::MAX, |&t| {
                let before = if t == 0 {
                    0
                } else {
                    tree.prefix(t as usize - 1)
                };
                (tree.prefix(now.saturating_sub(1)) - before) as u64
            });
            deepest = deepest.max(depth);
            if a.counted && depth <= max {
                block[depth as usize] += 1;
            }
            if a.counted && deepest <= max {
                prefix[deepest as usize] += 1;
            }
        }
        for _ in 0..a.output_blocks(block_tokens) {
            tree.add(now, 1);
            now += 1;
        }
        for &b in a.prompt.iter().rev() {
            if let Some(t) = last.insert(b, now as u32) {
                tree.add(t as usize, -1);
            }
            tree.add(now, 1);
            now += 1;
        }
    }
    (block, prefix)
}

const NEVER: u32 = u32::MAX;

/// For every prompt block of every access, the index of the next access whose prompt holds it.
fn next_uses(accesses: &[Access<'_>]) -> Vec<Vec<u32>> {
    let mut seen: IdMap<u32> = IdMap::default();
    let mut next: Vec<Vec<u32>> = vec![Vec::new(); accesses.len()];
    for (i, a) in accesses.iter().enumerate().rev() {
        next[i] = a
            .prompt
            .iter()
            .map(|b| seen.get(b).copied().unwrap_or(NEVER))
            .collect();
        for &b in a.prompt {
            seen.insert(b, i as u32);
        }
    }
    next
}

/// MIN with bypass at `capacity` blocks: (block hit tokens, prefix hit tokens).
fn min_hits(
    accesses: &[Access<'_>],
    next: &[Vec<u32>],
    block_tokens: u64,
    capacity: u64,
) -> (u64, u64) {
    let mut cached: IdMap<u32> = IdMap::default();
    let mut heap: BinaryHeap<(u32, BlockId)> = BinaryHeap::new();
    let (mut block_hits, mut prefix_hits) = (0u64, 0u64);
    for (a, next) in accesses.iter().zip(next) {
        let mut whole = true;
        for b in &a.prompt[..a.hittable(block_tokens)] {
            let hit = cached.contains_key(b);
            whole &= hit;
            if a.counted {
                block_hits += u64::from(hit);
                prefix_hits += u64::from(whole);
            }
        }
        // Cached blocks of this request move to their next use before new blocks compete with them.
        for (&b, &n) in a.prompt.iter().zip(next) {
            if let Some(key) = cached.get_mut(&b) {
                if n == NEVER {
                    cached.remove(&b);
                } else {
                    *key = n;
                    heap.push((n, b));
                }
            }
        }
        for (&b, &n) in a.prompt.iter().zip(next) {
            if n == NEVER || cached.contains_key(&b) {
                continue;
            }
            if cached.len() as u64 >= capacity {
                // The cached block with the farthest next use; stale heap entries are dropped.
                let farthest = loop {
                    match heap.peek() {
                        Some(&(k, x)) if cached.get(&x) == Some(&k) => break Some(k),
                        Some(_) => {
                            heap.pop();
                        }
                        None => break None,
                    }
                };
                match farthest {
                    Some(k) if k > n => {
                        let (_, victim) = heap.pop().expect("peeked");
                        cached.remove(&victim);
                    }
                    _ => continue,
                }
            }
            cached.insert(b, n);
            heap.push((n, b));
        }
        if heap.len() > 4 * cached.len() + 4096 {
            heap.retain(|&(k, x)| cached.get(&x) == Some(&k));
        }
    }
    (block_hits * block_tokens, prefix_hits * block_tokens)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};

    use super::*;

    const B: u64 = 16;

    /// Requests of `blocks` full blocks plus one token, so every block is hittable.
    fn accesses(reqs: &[Vec<BlockId>], out: u64) -> Vec<Access<'_>> {
        reqs.iter()
            .map(|p| Access {
                prompt: p,
                prompt_tokens: p.len() as u64 * B + 1,
                output_tokens: out,
                counted: true,
            })
            .collect()
    }

    /// LRU simulated directly with the same lookup and touch order.
    fn direct_lru(acc: &[Access<'_>], capacity: usize) -> (u64, u64) {
        let mut stack: VecDeque<BlockId> = VecDeque::new(); // front is most recent
        let (mut block, mut prefix) = (0, 0);
        let mut unique = 1u64 << 62;
        for a in acc {
            let mut whole = true;
            for b in &a.prompt[..a.hittable(B)] {
                let hit = stack.contains(b);
                whole &= hit;
                block += u64::from(hit) * B;
                prefix += u64::from(whole) * B;
            }
            let touch = |b: BlockId, stack: &mut VecDeque<BlockId>| {
                if let Some(k) = stack.iter().position(|&x| x == b) {
                    stack.remove(k);
                }
                stack.push_front(b);
                stack.truncate(capacity);
            };
            for _ in 0..a.output_blocks(B) {
                unique += 1;
                touch(unique, &mut stack);
            }
            for &b in a.prompt.iter().rev() {
                touch(b, &mut stack);
            }
        }
        (block, prefix)
    }

    fn random_requests(seed: u64, n: usize, alphabet: u64) -> Vec<Vec<BlockId>> {
        let mut state = seed;
        let mut draw = move |k: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % k
        };
        // Prompts that extend, fork or restart a few chains, as agent sessions do.
        let mut chains: Vec<Vec<BlockId>> = vec![Vec::new(); 3];
        (0..n)
            .map(|_| {
                let c = draw(3) as usize;
                let keep = draw(chains[c].len() as u64 + 1) as usize;
                chains[c].truncate(keep);
                for _ in 0..1 + draw(3) {
                    chains[c].push(draw(alphabet) + 100 * c as u64);
                }
                // Prefix identity: a block is named by its chain position and the ids before it.
                let mut id = 0u64;
                chains[c]
                    .iter()
                    .map(|&x| {
                        id = id.wrapping_mul(31).wrapping_add(x + 1);
                        id
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn mattson_matches_a_direct_lru_simulation() {
        for seed in 0..20 {
            let reqs = random_requests(seed, 60, 4);
            let acc = accesses(&reqs, 20);
            let caps: Vec<u64> = vec![1, 2, 3, 5, 8, 13, 21, 40];
            let curve = Curve::compute(&acc, B, &caps, &[]);
            for p in &curve.points {
                let (block, prefix) = direct_lru(&acc, p.capacity as usize);
                assert_eq!(
                    (p.lru_block_hits, p.lru_prefix_hits),
                    (block, prefix),
                    "seed {seed} capacity {}",
                    p.capacity
                );
            }
        }
    }

    /// The most block hits any policy gets, by exhaustive search over the cached sets.
    fn best_block_hits(reqs: &[Vec<BlockId>], capacity: usize) -> u64 {
        let mut ids: Vec<BlockId> = reqs.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert!(ids.len() <= 20);
        let bit = |b: &BlockId| 1u32 << ids.binary_search(b).expect("listed");
        fn go(
            sets: &[u32],
            i: usize,
            held: u32,
            capacity: usize,
            memo: &mut HashMap<(usize, u32), u64>,
        ) -> u64 {
            let Some(&req) = sets.get(i) else { return 0 };
            if let Some(&v) = memo.get(&(i, held)) {
                return v;
            }
            let hits = u64::from((held & req).count_ones());
            let pool = held | req;
            let mut best = 0;
            let mut keep = pool;
            loop {
                if keep.count_ones() as usize <= capacity {
                    best = best.max(go(sets, i + 1, keep, capacity, memo));
                }
                if keep == 0 {
                    break;
                }
                keep = (keep - 1) & pool;
            }
            memo.insert((i, held), hits + best);
            hits + best
        }
        let sets: Vec<u32> = reqs
            .iter()
            .map(|r| r.iter().map(bit).fold(0, |m, b| m | b))
            .collect();
        go(&sets, 0, 0, capacity, &mut HashMap::new()) * B
    }

    #[test]
    fn min_block_hits_match_the_exhaustive_optimum() {
        let mut state = 7u64;
        for case in 0..150 {
            let reqs: Vec<Vec<BlockId>> = (0..9)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let n = 1 + (state >> 62) as usize;
                    let mut r: Vec<BlockId> = (0..n).map(|k| (state >> (8 * k + 3)) % 7).collect();
                    r.sort_unstable();
                    r.dedup();
                    r
                })
                .collect();
            let capacity = 1 + case % 4;
            let acc = accesses(&reqs, 1);
            let curve = Curve::compute(&acc, B, &[], &[capacity as u64]);
            assert_eq!(
                curve.points[0].min_block_hits,
                Some(best_block_hits(&reqs, capacity)),
                "{reqs:?} at {capacity}"
            );
        }
    }

    #[test]
    fn min_beats_lru_and_both_grow_with_capacity() {
        let reqs = random_requests(3, 200, 6);
        let acc = accesses(&reqs, 40);
        let caps = [2, 4, 8, 16, 32, 64];
        let curve = Curve::compute(&acc, B, &caps, &caps);
        let mut last = (0, 0, 0, 0);
        for p in &curve.points {
            let (mb, mp) = (p.min_block_hits.unwrap(), p.min_prefix_hits.unwrap());
            assert!(
                mb >= p.lru_block_hits,
                "MIN bounds LRU per block at {}",
                p.capacity
            );
            assert!(p.lru_block_hits >= p.lru_prefix_hits && mb >= mp);
            assert!(p.lru_block_hits >= last.0 && p.lru_prefix_hits >= last.1 && mb >= last.2);
            last = (p.lru_block_hits, p.lru_prefix_hits, mb, mp);
        }
        assert!(curve.points.last().unwrap().lru_block_hits <= curve.prompt_tokens);
    }

    #[test]
    fn the_markdown_table_gives_each_curves_miss_ratios_at_the_pools_multiples() {
        let point = |capacity, lru, min| Point {
            capacity,
            lru_block_hits: lru,
            lru_prefix_hits: lru,
            min_block_hits: min,
            min_prefix_hits: min,
        };
        // A replay of 1,000 prompt tokens on a pool of 9 blocks, and its same order at 32x.
        let scaled = Curve {
            requests: 3,
            prompt_tokens: 1_000,
            points: vec![
                point(2, 100, Some(400)),
                point(5, 300, None),
                point(9, 500, Some(700)),
            ],
        };
        let same = Curve {
            requests: 3,
            prompt_tokens: 32_000,
            points: vec![
                point(72, 3_300, Some(12_810)),
                point(287, 16_010, Some(22_400)),
            ],
        };
        assert_eq!(capacities(9, &[0.25, 1.0]), vec![2, 9]);
        assert_eq!(capacities(287, &[0.25, 1.0]), vec![72, 287]);
        assert_eq!(
            markdown(&[0.25, 1.0], 32, (&scaled, 9), Some((&same, 287))),
            "| Pool multiple | LRU, 32× | LRU, original tokens, same order | MIN, 32× \
             | MIN, original tokens, same order |\n|---|---|---|---|---|\n\
             | 0.25 | 0.900 | 0.897 | 0.600 | 0.600 |\n| 1 | 0.500 | 0.500 | 0.300 | 0.300 |\n"
        );
        assert_eq!(
            markdown(&[1.0], 1, (&scaled, 9), None),
            "| Pool multiple | LRU | MIN |\n|---|---|---|\n| 1 | 0.500 | 0.300 |\n"
        );
    }

    #[test]
    fn the_last_block_of_an_aligned_prompt_never_hits() {
        let reqs: Vec<Vec<BlockId>> = vec![vec![1, 2], vec![1, 2]];
        let acc: Vec<Access<'_>> = reqs
            .iter()
            .map(|p| Access {
                prompt: p,
                prompt_tokens: 32,
                output_tokens: 1,
                counted: true,
            })
            .collect();
        let curve = Curve::compute(&acc, B, &[8], &[8]);
        let p = &curve.points[0];
        assert_eq!(
            (p.lru_block_hits, p.lru_prefix_hits, p.min_block_hits),
            (16, 16, Some(16))
        );
    }
}
