//! CPU-tier policies: what vLLM's OffloadingConnector keeps in host RAM.
//!
//! The connector writes every full block a request computes through to the tier (prompt and, with
//! `offload_prompt_only: false`, output), offering blocks the tier already holds as reuses. A store
//! may evict, but never one of the blocks it offers; when the tier cannot make room, vLLM stores
//! nothing. Restores are instantaneous here, so no block is pinned by a pending load.

use std::collections::BinaryHeap;
use std::fmt;
use std::rc::Rc;

use super::oracle::{NEVER, Oracle};
use crate::prefix::{BlockId, IdMap, IdSet};

/// What one store did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreOutcome {
    pub stored: usize,
    pub evicted: usize,
}

/// Decides what the CPU tier keeps.
pub trait CpuPolicy: fmt::Debug {
    fn contains(&self, id: BlockId) -> bool;
    /// A request was admitted with these prompt blocks (`prompt[i]` at position `i`).
    fn on_admit(&mut self, _prompt: &[BlockId]) {}
    /// These blocks were restored from the tier for the request being admitted.
    fn on_load(&mut self, _ids: &[BlockId]) {}
    /// Write-through of a finished request's blocks, `(identity, position)` in prefix order. Blocks
    /// already held are reuses; the rest are stored, evicting as needed but never an offered block.
    /// `None` when the tier cannot make room: nothing is stored.
    fn store(&mut self, offer: &[(BlockId, u32)]) -> Option<StoreOutcome>;
    /// The request finished; it loaded or offered `ids`, in prefix order.
    fn on_finish(&mut self, _ids: &[BlockId]) {}
}

/// Doubly linked lists of block ids over one node arena; each list has a sentinel node. The front is
/// the newest end.
#[derive(Debug, Default)]
struct Lists {
    prev: Vec<u32>,
    next: Vec<u32>,
    key: Vec<BlockId>,
    spare: Vec<u32>,
}

impl Lists {
    fn new_list(&mut self) -> u32 {
        let s = self.prev.len() as u32;
        self.prev.push(s);
        self.next.push(s);
        self.key.push(0);
        s
    }

    fn link_after(&mut self, at: u32, node: u32) {
        let after = self.next[at as usize];
        self.next[at as usize] = node;
        self.prev[node as usize] = at;
        self.next[node as usize] = after;
        self.prev[after as usize] = node;
    }

    fn unlink(&mut self, node: u32) {
        let (p, n) = (self.prev[node as usize], self.next[node as usize]);
        self.next[p as usize] = n;
        self.prev[n as usize] = p;
    }

    fn push_front(&mut self, list: u32, key: BlockId) -> u32 {
        let node = match self.spare.pop() {
            Some(n) => {
                self.key[n as usize] = key;
                n
            }
            None => {
                self.prev.push(0);
                self.next.push(0);
                self.key.push(key);
                self.prev.len() as u32 - 1
            }
        };
        self.link_after(list, node);
        node
    }

    fn move_front(&mut self, list: u32, node: u32) {
        self.unlink(node);
        self.link_after(list, node);
    }

    fn remove(&mut self, node: u32) -> BlockId {
        self.unlink(node);
        self.spare.push(node);
        self.key[node as usize]
    }

    /// The oldest node of `list`.
    fn back(&self, list: u32) -> Option<u32> {
        let b = self.prev[list as usize];
        (b != list).then_some(b)
    }

    fn older(&self, list: u32, node: u32) -> Option<u32> {
        let p = self.prev[node as usize];
        (p != list).then_some(p)
    }
}

fn missing(offer: &[(BlockId, u32)], contains: impl Fn(BlockId) -> bool) -> usize {
    offer.iter().filter(|&&(id, _)| !contains(id)).count()
}

/// vLLM 0.32's `LRUCachePolicy`. A block becomes most recent when it is stored and when a request
/// that loaded or offered it finishes; at a finish the request's blocks are refreshed in reverse, so
/// the start of its prefix is the most recent and the end is evicted first.
#[derive(Debug)]
pub struct CpuLru {
    capacity: usize,
    lists: Lists,
    list: u32,
    nodes: IdMap<u32>,
}

impl CpuLru {
    pub fn new(capacity: usize) -> Self {
        let mut lists = Lists::default();
        let list = lists.new_list();
        Self {
            capacity,
            lists,
            list,
            nodes: IdMap::default(),
        }
    }
}

impl CpuPolicy for CpuLru {
    fn contains(&self, id: BlockId) -> bool {
        self.nodes.contains_key(&id)
    }

    fn store(&mut self, offer: &[(BlockId, u32)]) -> Option<StoreOutcome> {
        let new = missing(offer, |id| self.nodes.contains_key(&id));
        if new == 0 {
            return Some(StoreOutcome::default());
        }
        let need = (self.nodes.len() + new).saturating_sub(self.capacity);
        let mut victims = Vec::with_capacity(need);
        if need > 0 {
            let protected: IdSet = offer.iter().map(|&(id, _)| id).collect();
            let mut node = self.lists.back(self.list);
            while let Some(n) = node {
                if victims.len() == need {
                    break;
                }
                if !protected.contains(&self.lists.key[n as usize]) {
                    victims.push(n);
                }
                node = self.lists.older(self.list, n);
            }
            if victims.len() < need {
                return None;
            }
        }
        for &n in &victims {
            let id = self.lists.remove(n);
            self.nodes.remove(&id);
        }
        for &(id, _) in offer {
            if !self.nodes.contains_key(&id) {
                let n = self.lists.push_front(self.list, id);
                self.nodes.insert(id, n);
            }
        }
        Some(StoreOutcome {
            stored: new,
            evicted: victims.len(),
        })
    }

    fn on_finish(&mut self, ids: &[BlockId]) {
        for id in ids.iter().rev() {
            if let Some(&n) = self.nodes.get(id) {
                self.lists.move_front(self.list, n);
            }
        }
    }
}

/// Belady's MIN on the tier, with bypass: the tier keeps the blocks with the nearest next uses. A
/// block's next use is the next request whose prompt contains it, whichever tier serves that request.
/// A store evicts the held block whose next use is farthest, or stores nothing when the offered block's
/// own next use is farther still (blocks never used again are not stored). Offered blocks get no
/// protection.
#[derive(Debug)]
pub struct CpuClairvoyant {
    capacity: usize,
    oracle: Rc<Oracle>,
    /// Held blocks and the sequence number of their current heap entry.
    entries: IdMap<u64>,
    /// (next use, position, sequence, id); the largest next use is evicted first. Entries whose
    /// sequence is not their block's current one are stale.
    heap: BinaryHeap<(u64, u32, u64, BlockId)>,
    seq: u64,
}

impl CpuClairvoyant {
    pub fn new(capacity: usize, oracle: Rc<Oracle>) -> Self {
        Self {
            capacity,
            oracle,
            entries: IdMap::default(),
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    fn push(&mut self, id: BlockId, next: u64, position: u32) {
        self.seq += 1;
        self.entries.insert(id, self.seq);
        self.heap.push((next, position, self.seq, id));
        if self.heap.len() > 4 * self.entries.len() + 4096 {
            let entries = &self.entries;
            self.heap.retain(|e| entries.get(&e.3) == Some(&e.2));
        }
    }

    fn refresh(&mut self, id: BlockId, position: u32) {
        if self.entries.contains_key(&id) {
            self.push(id, self.oracle.next_use(id), position);
        }
    }

    /// The held block with the farthest next use, as (next use, position); stale entries are dropped.
    fn farthest(&mut self) -> Option<(u64, u32)> {
        while let Some(&(next, position, seq, id)) = self.heap.peek() {
            if self.entries.get(&id) == Some(&seq) {
                return Some((next, position));
            }
            self.heap.pop();
        }
        None
    }
}

impl CpuPolicy for CpuClairvoyant {
    fn contains(&self, id: BlockId) -> bool {
        self.entries.contains_key(&id)
    }

    fn on_admit(&mut self, prompt: &[BlockId]) {
        for (pos, &id) in prompt.iter().enumerate() {
            self.refresh(id, pos as u32);
        }
    }

    fn store(&mut self, offer: &[(BlockId, u32)]) -> Option<StoreOutcome> {
        let mut out = StoreOutcome::default();
        for &(id, position) in offer {
            if self.entries.contains_key(&id) {
                continue;
            }
            let next = self.oracle.next_use(id);
            if next == NEVER {
                continue;
            }
            if self.entries.len() >= self.capacity {
                match self.farthest() {
                    Some(far) if far > (next, position) => {
                        let (_, _, _, victim) = self.heap.pop().expect("farthest() peeked it");
                        self.entries.remove(&victim);
                        out.evicted += 1;
                    }
                    _ => continue,
                }
            }
            self.push(id, next, position);
            out.stored += 1;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn offer(ids: &[BlockId]) -> Vec<(BlockId, u32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect()
    }

    /// The blocks of `list`, newest first.
    fn keys(lists: &Lists, list: u32) -> Vec<BlockId> {
        let mut out = Vec::new();
        let mut n = lists.next[list as usize];
        while n != list {
            out.push(lists.key[n as usize]);
            n = lists.next[n as usize];
        }
        out
    }

    #[test]
    fn lru_refreshes_a_finished_prefix_head_first() {
        let mut t = CpuLru::new(4);
        t.store(&offer(&[1, 2, 3])).unwrap();
        assert_eq!(keys(&t.lists, t.list), vec![3, 2, 1]);
        t.on_finish(&[1, 2, 3]);
        assert_eq!(
            keys(&t.lists, t.list),
            vec![1, 2, 3],
            "the prefix start is most recent"
        );
        // Storing two more evicts the least recent: the end of the old prefix.
        let out = t.store(&offer(&[1, 7, 8])).unwrap();
        assert_eq!(
            out,
            StoreOutcome {
                stored: 2,
                evicted: 1
            }
        );
        assert!(!t.contains(3) && t.contains(1) && t.contains(2));
        assert_eq!(t.nodes.len(), 4);
    }

    #[test]
    fn a_store_never_evicts_what_it_offers() {
        let mut t = CpuLru::new(2);
        t.store(&offer(&[1, 2])).unwrap();
        // Offering both held blocks and a new one leaves nothing evictable.
        assert_eq!(t.store(&offer(&[1, 2, 3])), None);
        assert!(t.contains(1) && t.contains(2) && !t.contains(3));
    }

    #[test]
    fn clairvoyant_evicts_the_farthest_next_use_or_bypasses() {
        let reqs: Vec<Vec<BlockId>> = vec![vec![1, 2, 9], vec![3], vec![3], vec![1], vec![2]];
        let oracle = Rc::new(Oracle::new(reqs.iter().map(Vec::as_slice)));
        let mut t = CpuClairvoyant::new(2, oracle.clone());
        oracle.set_position(1);
        // 9 is never used again, so it is not stored.
        assert_eq!(
            t.store(&offer(&[1, 2, 9])).unwrap(),
            StoreOutcome {
                stored: 2,
                evicted: 0
            }
        );
        oracle.set_position(2);
        t.on_admit(&[3]);
        // 3 (next use 2) displaces 2 (next use 4); 1 (next use 3) stays.
        assert_eq!(
            t.store(&offer(&[3])).unwrap(),
            StoreOutcome {
                stored: 1,
                evicted: 1
            }
        );
        assert!(t.contains(1) && t.contains(3) && !t.contains(2));
        oracle.set_position(3);
        t.on_admit(&[3]);
        oracle.set_position(4);
        t.on_admit(&[1]);
        // Both held blocks are now never used again; 2 (next use 4) replaces one of them.
        assert_eq!(
            t.store(&offer(&[2])).unwrap(),
            StoreOutcome {
                stored: 1,
                evicted: 1
            }
        );
        assert!(t.contains(2));
    }

    /// The most hits any policy that may decline to store gets, by exhaustive search over the held
    /// sets (blocks are 0..32, a set is a bit mask).
    fn best_hits_with_bypass(seq: &[BlockId], capacity: usize) -> usize {
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
            let hit = (held >> b & 1) as usize;
            let pool = held | 1 << b;
            let mut best = 0;
            let mut keep = pool;
            loop {
                if keep.count_ones() as usize <= capacity {
                    best = best.max(go(seq, i + 1, keep, capacity, memo));
                }
                if keep == 0 {
                    break;
                }
                keep = (keep - 1) & pool;
            }
            memo.insert((i, held), hit + best);
            hit + best
        }
        go(seq, 0, 0, capacity, &mut HashMap::new())
    }

    #[test]
    fn clairvoyant_matches_the_exhaustive_optimum() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..200 {
            let mut seq = Vec::new();
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                seq.push(state >> 61);
            }
            let capacity = 1 + case % 3;
            let reqs: Vec<Vec<BlockId>> = seq.iter().map(|&b| vec![b]).collect();
            let oracle = Rc::new(Oracle::new(reqs.iter().map(Vec::as_slice)));
            let mut t = CpuClairvoyant::new(capacity, oracle.clone());
            let mut hits = 0;
            for (i, &b) in seq.iter().enumerate() {
                oracle.set_position(i as u32 + 1);
                hits += usize::from(t.contains(b));
                t.on_admit(&[b]);
                t.store(&offer(&[b]));
                assert!(t.entries.len() <= capacity);
            }
            assert_eq!(
                hits,
                best_hits_with_bypass(&seq, capacity),
                "{seq:?} at capacity {capacity}"
            );
        }
    }
}
