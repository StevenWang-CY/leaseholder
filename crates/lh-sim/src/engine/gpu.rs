//! GPU pool eviction policies: which free block the pool reuses next.
//!
//! The pool ([`super::Engine`]) owns reference counts and the prefix-cache map; a policy only orders
//! the free blocks. A block whose last reference is dropped joins the free list but keeps its contents
//! (and stays a cache hit) until the policy hands it out again.

use std::collections::BinaryHeap;
use std::fmt;
use std::rc::Rc;

use super::oracle::{NEVER, Oracle};
use crate::prefix::BlockId;

/// Orders the free blocks of the GPU pool.
pub trait GpuPolicy: fmt::Debug {
    /// A block lost its last reference. `cached` is its identity when it holds a full block (it stays
    /// hittable); `position` is the block's index in its prompt.
    fn free(&mut self, slot: u32, cached: Option<BlockId>, position: u32);
    /// A free cached block was hit and is in use again.
    fn reuse(&mut self, slot: u32);
    /// A request needed the free cached block in `slot` (`id`, at `position` of its prompt) but could
    /// not hit it, because an earlier block of its prompt missed. Only policies that rank blocks by
    /// their next use care.
    fn refresh(&mut self, _slot: u32, _id: BlockId, _position: u32) {}
    /// The free block to allocate next; the pool evicts whatever it caches.
    fn take(&mut self) -> u32;
    fn free_blocks(&self) -> usize;
}

/// vLLM's `FreeKVCacheBlockQueue`: allocation pops the front; a freed cached block is appended to the
/// back (least recently used first), a freed uncached block is prepended (reused first). The engine
/// frees a request's blocks in reverse order, so the end of a prefix is evicted before its start.
#[derive(Debug)]
pub struct LruQueue {
    prev: Vec<u32>,
    next: Vec<u32>,
    queued: Vec<bool>,
    len: usize,
}

impl LruQueue {
    /// All `blocks` free, in index order.
    pub fn new(blocks: u32) -> Self {
        let n = blocks as usize;
        let (head, tail) = (n as u32, n as u32 + 1);
        let mut prev = vec![0; n + 2];
        let mut next = vec![0; n + 2];
        let mut last = head;
        for i in 0..n as u32 {
            next[last as usize] = i;
            prev[i as usize] = last;
            last = i;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        Self {
            prev,
            next,
            queued: vec![true; n],
            len: n,
        }
    }

    fn head(&self) -> u32 {
        self.queued.len() as u32
    }

    fn tail(&self) -> u32 {
        self.queued.len() as u32 + 1
    }

    fn link_after(&mut self, at: u32, slot: u32) {
        let after = self.next[at as usize];
        self.next[at as usize] = slot;
        self.prev[slot as usize] = at;
        self.next[slot as usize] = after;
        self.prev[after as usize] = slot;
        self.queued[slot as usize] = true;
        self.len += 1;
    }

    fn unlink(&mut self, slot: u32) {
        debug_assert!(self.queued[slot as usize]);
        let (p, n) = (self.prev[slot as usize], self.next[slot as usize]);
        self.next[p as usize] = n;
        self.prev[n as usize] = p;
        self.queued[slot as usize] = false;
        self.len -= 1;
    }
}

impl GpuPolicy for LruQueue {
    fn free(&mut self, slot: u32, cached: Option<BlockId>, _position: u32) {
        if cached.is_some() {
            let last = self.prev[self.tail() as usize];
            self.link_after(last, slot);
        } else {
            self.link_after(self.head(), slot);
        }
    }

    fn reuse(&mut self, slot: u32) {
        self.unlink(slot);
    }

    fn take(&mut self) -> u32 {
        let slot = self.next[self.head() as usize];
        assert!(slot != self.tail(), "no free GPU block");
        self.unlink(slot);
        slot
    }

    fn free_blocks(&self) -> usize {
        self.len
    }
}

/// Belady's MIN on the pool: reuse the free block whose next use is farthest (never-used blocks and
/// blocks without contents first; among equal next uses, the deepest block of its prompt first, which
/// keeps cached prefixes contiguous).
#[derive(Debug)]
pub struct ClairvoyantFree {
    oracle: Rc<Oracle>,
    /// (next use, position, slot, version): the largest next use is reused first.
    heap: BinaryHeap<(u64, u32, u32, u32)>,
    /// Free blocks no future request needs: (slot, version).
    empty: Vec<(u32, u32)>,
    /// Bumped on every state change; an entry is valid only at its slot's current version.
    version: Vec<u32>,
    len: usize,
}

impl ClairvoyantFree {
    pub fn new(blocks: u32, oracle: Rc<Oracle>) -> Self {
        Self {
            oracle,
            heap: BinaryHeap::new(),
            empty: (0..blocks).rev().map(|s| (s, 0)).collect(),
            version: vec![0; blocks as usize],
            len: blocks as usize,
        }
    }

    fn claim(&mut self, slot: u32) -> u32 {
        self.version[slot as usize] += 1;
        self.len -= 1;
        slot
    }

    /// Ranks the free block in `slot` by the next use of `cached`.
    fn rank(&mut self, slot: u32, cached: Option<BlockId>, position: u32) {
        self.version[slot as usize] += 1;
        let version = self.version[slot as usize];
        match cached.map(|id| self.oracle.next_use(id)) {
            Some(next) if next != NEVER => self.heap.push((next, position, slot, version)),
            _ => self.empty.push((slot, version)),
        }
        if self.heap.len() > 2 * self.version.len() + 1024 {
            let current = &self.version;
            self.heap.retain(|e| current[e.2 as usize] == e.3);
            self.empty.retain(|e| current[e.0 as usize] == e.1);
        }
    }
}

impl GpuPolicy for ClairvoyantFree {
    fn free(&mut self, slot: u32, cached: Option<BlockId>, position: u32) {
        self.len += 1;
        self.rank(slot, cached, position);
    }

    fn reuse(&mut self, slot: u32) {
        self.claim(slot);
    }

    // The block's ranked next use has passed without a hit; rank it by the following one.
    fn refresh(&mut self, slot: u32, id: BlockId, position: u32) {
        self.rank(slot, Some(id), position);
    }

    fn take(&mut self) -> u32 {
        while let Some((slot, version)) = self.empty.pop() {
            if self.version[slot as usize] == version {
                return self.claim(slot);
            }
        }
        while let Some((_, _, slot, version)) = self.heap.pop() {
            if self.version[slot as usize] == version {
                return self.claim(slot);
            }
        }
        panic!("no free GPU block");
    }

    fn free_blocks(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The free blocks from the front (next to be reused) to the back.
    fn order(q: &LruQueue) -> Vec<u32> {
        let mut out = Vec::new();
        let mut s = q.next[q.head() as usize];
        while s != q.tail() {
            out.push(s);
            s = q.next[s as usize];
        }
        out
    }

    #[test]
    fn lru_reuses_uncached_first_and_cached_oldest_first() {
        let mut q = LruQueue::new(4);
        assert_eq!(order(&q), vec![0, 1, 2, 3]);
        let taken: Vec<u32> = (0..4).map(|_| q.take()).collect();
        assert_eq!(taken, vec![0, 1, 2, 3]);
        // A request frees its blocks tail first: 2 (partial, uncached), then 1 and 0 (cached).
        q.free(2, None, 2);
        q.free(1, Some(11), 1);
        q.free(0, Some(10), 0);
        q.free(3, Some(30), 0);
        assert_eq!(order(&q), vec![2, 1, 0, 3]);
        q.reuse(0);
        assert_eq!(order(&q), vec![2, 1, 3]);
        assert_eq!((q.take(), q.take(), q.free_blocks()), (2, 1, 1));
    }

    #[test]
    fn clairvoyant_reuses_the_farthest_next_use() {
        let reqs: Vec<Vec<BlockId>> = vec![vec![10, 11], vec![10], vec![11]];
        let oracle = Rc::new(Oracle::new(reqs.iter().map(Vec::as_slice)));
        let mut p = ClairvoyantFree::new(3, oracle.clone());
        let slots: Vec<u32> = (0..3).map(|_| p.take()).collect();
        oracle.set_position(1);
        p.free(slots[0], Some(10), 0); // next use 1
        p.free(slots[1], Some(11), 1); // next use 2
        p.free(slots[2], Some(99), 0); // never
        assert_eq!(p.take(), slots[2]);
        assert_eq!(p.take(), slots[1]);
        p.free(slots[1], Some(11), 1);
        p.reuse(slots[1]);
        assert_eq!((p.take(), p.free_blocks()), (slots[0], 0));
    }
}
