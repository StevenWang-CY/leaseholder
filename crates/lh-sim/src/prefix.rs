//! Prompts as paths in a prefix tree of hash blocks, and the identities of their engine blocks.
//!
//! vLLM identifies a cached block by a hash chained over its parent block and its tokens, so two
//! prompts share a block iff they share every token up to its end. Within one trace, the token content
//! of a hash block is a function of its id (AIPerf synthesizes text per trace and hash id), so the
//! prefix of tokens up to a block end is determined by the hash ids up to the hash block that holds the
//! block's last token. A [`PrefixTree`] holds every prompt of a trace as a path of hash ids; an engine
//! block is identified exactly by that tree node, the block's offset inside the node's hash block, and
//! a namespace (the session play and, optionally, the model), with no hashing and no collisions.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A node of a [`PrefixTree`]: the prefix of hash blocks on the path from the root to it.
pub type NodeId = u32;

/// The identity of one engine block's KV. See [`block_id`].
pub type BlockId = u64;

/// Bit set on identities minted for blocks that can never be shared (blocks holding output tokens).
pub const UNIQUE_BIT: BlockId = 1 << 63;

const SUB_BITS: u32 = 4;
const NODE_BITS: u32 = 28;
const MODEL_BITS: u32 = 4;
const PLAY_BITS: u32 = 63 - SUB_BITS - NODE_BITS - MODEL_BITS;

/// Most models one namespace distinguishes.
pub const MAX_MODELS: usize = 1 << MODEL_BITS;
/// Most session plays one run can hold.
pub const MAX_PLAYS: u64 = 1 << PLAY_BITS;

/// The identity of the engine block that ends inside the hash block at `node`, `sub` engine blocks
/// into it, for a session play and model. Two prompts get the same identity iff they share the play,
/// the model and every hash block up to `node`.
pub fn block_id(play: u64, model: u8, node: NodeId, sub: u32) -> BlockId {
    debug_assert!(play < MAX_PLAYS && (model as usize) < MAX_MODELS);
    debug_assert!(node < 1 << NODE_BITS && sub < 1 << SUB_BITS);
    (((play << MODEL_BITS | model as u64) << NODE_BITS | node as u64) << SUB_BITS) | sub as u64
}

/// Token geometry of a replay: tokens per hash block and per engine block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tokens {
    pub hash_block: u32,
    pub engine_block: u32,
}

impl Tokens {
    pub fn new(hash_block: u32, engine_block: u32) -> Self {
        assert!(hash_block > 0 && engine_block > 0);
        assert!(
            hash_block.div_ceil(engine_block) <= 1 << SUB_BITS,
            "a hash block may span at most 16 engine blocks"
        );
        Self {
            hash_block,
            engine_block,
        }
    }

    /// Prompt tokens of `hash_blocks` hash blocks.
    pub fn prompt_tokens(&self, hash_blocks: u32) -> u64 {
        hash_blocks as u64 * self.hash_block as u64
    }
}

/// Every prompt of one trace as a trie over hash ids; node 0 is the empty prefix.
#[derive(Debug, Default, Clone)]
pub struct PrefixTree {
    parent: Vec<NodeId>,
    depth: Vec<u32>,
}

impl PrefixTree {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.len() <= 1
    }

    pub fn depth(&self, node: NodeId) -> u32 {
        self.depth[node as usize]
    }

    pub fn parent(&self, node: NodeId) -> NodeId {
        self.parent[node as usize]
    }

    /// Hash blocks shared by the prompts ending at `a` and `b`.
    pub fn common_prefix(&self, mut a: NodeId, mut b: NodeId) -> u32 {
        while self.depth(a) > self.depth(b) {
            a = self.parent(a);
        }
        while self.depth(b) > self.depth(a) {
            b = self.parent(b);
        }
        while a != b {
            a = self.parent(a);
            b = self.parent(b);
        }
        self.depth(a)
    }

    /// Writes the path from the root to `end` into `path` (`path[d]` is the node at depth `d + 1`).
    pub fn path(&self, end: NodeId, path: &mut Vec<NodeId>) {
        path.clear();
        let mut node = end;
        while node != 0 {
            path.push(node);
            node = self.parent(node);
        }
        path.reverse();
    }

    /// Appends the identities of the full engine blocks of the prompt ending at `end` to `out`.
    /// `scratch` is reused between calls.
    pub fn engine_blocks(
        &self,
        end: NodeId,
        tokens: Tokens,
        play: u64,
        model: u8,
        scratch: &mut Vec<NodeId>,
        out: &mut Vec<BlockId>,
    ) {
        self.path(end, scratch);
        let prompt = scratch.len() as u64 * tokens.hash_block as u64;
        let (hb, eb) = (tokens.hash_block as u64, tokens.engine_block as u64);
        for j in 0..prompt / eb {
            let last = (j + 1) * eb - 1;
            let node = scratch[(last / hb) as usize];
            out.push(block_id(play, model, node, ((last % hb) / eb) as u32));
        }
    }
}

/// Builds a [`PrefixTree`], inserting prompts one at a time.
#[derive(Debug)]
pub struct PrefixTreeBuilder {
    tree: PrefixTree,
    children: IdMap<NodeId>,
    prev_ids: Vec<u32>,
    prev_end: NodeId,
}

impl Default for PrefixTreeBuilder {
    fn default() -> Self {
        Self {
            tree: PrefixTree {
                parent: vec![0],
                depth: vec![0],
            },
            children: IdMap::default(),
            prev_ids: Vec::new(),
            prev_end: 0,
        }
    }
}

impl PrefixTreeBuilder {
    /// Inserts a prompt and returns its end node. The walk starts from the common prefix with the
    /// previous prompt, so a session that grows its context costs only its new blocks.
    pub fn insert(&mut self, ids: &[u32]) -> NodeId {
        let lcp = ids
            .iter()
            .zip(&self.prev_ids)
            .take_while(|(a, b)| a == b)
            .count();
        let mut node = self.prev_end;
        for _ in lcp..self.prev_ids.len() {
            node = self.tree.parent(node);
        }
        for &id in &ids[lcp..] {
            let key = (node as u64) << 32 | id as u64;
            node = match self.children.get(&key) {
                Some(&child) => child,
                None => {
                    let child = self.tree.parent.len() as NodeId;
                    assert!(child < 1 << NODE_BITS, "prefix tree exceeds 2^28 nodes");
                    let depth = self.tree.depth(node) + 1;
                    self.tree.parent.push(node);
                    self.tree.depth.push(depth);
                    self.children.insert(key, child);
                    child
                }
            };
        }
        self.prev_ids.clear();
        self.prev_ids.extend_from_slice(ids);
        self.prev_end = node;
        node
    }

    pub fn finish(self) -> PrefixTree {
        self.tree
    }
}

/// A hasher for keys that are already well spread integers: the splitmix64 finalizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }

    fn write_u64(&mut self, x: u64) {
        let mut z = (self.0 ^ x).wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        self.0 = z ^ (z >> 31);
    }

    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }

    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
}

pub type IdMap<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;
pub type IdSet = HashSet<u64, BuildHasherDefault<IdHasher>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(prompts: &[&[u32]]) -> (PrefixTree, Vec<NodeId>) {
        let mut b = PrefixTreeBuilder::default();
        let ends = prompts.iter().map(|p| b.insert(p)).collect();
        (b.finish(), ends)
    }

    fn blocks(t: &PrefixTree, end: NodeId, tokens: Tokens, play: u64, model: u8) -> Vec<BlockId> {
        let (mut scratch, mut out) = (Vec::new(), Vec::new());
        t.engine_blocks(end, tokens, play, model, &mut scratch, &mut out);
        out
    }

    #[test]
    fn shared_prefixes_share_nodes() {
        let (t, ends) = tree(&[&[1, 2, 3], &[1, 2, 3, 4], &[1, 2, 9], &[5]]);
        assert_eq!(t.len(), 1 + 4 + 1 + 1);
        assert_eq!(t.common_prefix(ends[0], ends[1]), 3);
        assert_eq!(t.common_prefix(ends[1], ends[2]), 2);
        assert_eq!(t.common_prefix(ends[2], ends[3]), 0);
        // Inserting an existing prompt adds nothing.
        let mut b = PrefixTreeBuilder::default();
        let first = b.insert(&[1, 2, 3]);
        b.insert(&[7]);
        assert_eq!(b.insert(&[1, 2, 3]), first);
    }

    #[test]
    fn a_divergent_block_breaks_the_chain() {
        // 32x geometry: 2 tokens per hash block, 16 per engine block, so 8 hash blocks per engine block.
        let tok = Tokens::new(2, 16);
        let base: Vec<u32> = (0..24).collect();
        let mut fork = base.clone();
        fork[10] = 99; // inside the second engine block
        let (t, ends) = tree(&[&base, &fork]);
        let a = blocks(&t, ends[0], tok, 0, 0);
        let b = blocks(&t, ends[1], tok, 0, 0);
        assert_eq!((a.len(), b.len()), (3, 3));
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        // Every later block differs too: identity covers the whole prefix.
        assert_ne!(a[2], b[2]);
    }

    #[test]
    fn only_full_engine_blocks_have_identities() {
        let tok = Tokens::new(2, 16);
        let ids: Vec<u32> = (0..15).collect(); // 30 tokens: one full block and a partial one
        let (t, ends) = tree(&[&ids]);
        assert_eq!(blocks(&t, ends[0], tok, 0, 0).len(), 1);
    }

    #[test]
    fn unscaled_hash_blocks_span_several_engine_blocks() {
        let tok = Tokens::new(64, 16);
        let (t, ends) = tree(&[&[1, 2], &[1, 3]]);
        let a = blocks(&t, ends[0], tok, 0, 0);
        let b = blocks(&t, ends[1], tok, 0, 0);
        assert_eq!(a.len(), 8);
        assert_eq!(a[..4], b[..4]);
        assert!(a[4..].iter().zip(&b[4..]).all(|(x, y)| x != y));
        // The four blocks inside one hash block are distinct.
        let distinct: IdSet = a.iter().copied().collect();
        assert_eq!(distinct.len(), 8);
    }

    #[test]
    fn namespaces_separate_plays_and_models() {
        let tok = Tokens::new(2, 16);
        let ids: Vec<u32> = (0..16).collect();
        let (t, ends) = tree(&[&ids]);
        let base = blocks(&t, ends[0], tok, 3, 0);
        assert_eq!(base, blocks(&t, ends[0], tok, 3, 0));
        assert_ne!(
            base,
            blocks(&t, ends[0], tok, 4, 0),
            "another play (cache bust) shares nothing"
        );
        assert_ne!(
            base,
            blocks(&t, ends[0], tok, 3, 1),
            "another model shares nothing"
        );
        assert!(base.iter().all(|id| id & UNIQUE_BIT == 0));
    }

    #[test]
    fn id_maps_spread_structured_keys() {
        let mut m: IdMap<u32> = IdMap::default();
        for play in 0..100u64 {
            for node in 0..100u32 {
                m.insert(block_id(play, 0, node, 0), node);
            }
        }
        assert_eq!(m.len(), 10_000);
        assert_eq!(m[&block_id(42, 0, 7, 0)], 7);
    }
}
