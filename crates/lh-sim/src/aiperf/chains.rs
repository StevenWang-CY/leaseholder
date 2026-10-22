//! Flattened-agent chain detection, ported from AIPerf 0.13.0 (`weka_agent_chains.py` and the
//! preamble rule in `weka_trace.py`).
//!
//! Agent fan-outs are often recorded as interleaved requests of one list. Detection partitions such a
//! list into per-agent chains by hash-id LCP evidence: a request extends the deepest chain whose tail
//! is a complete prefix of it (same model, tail finished), and otherwise forks a new chain from the
//! tail it shares the longest prefix with. A second pass splices a fork back onto a dead tail when it
//! is the same agent continuing after a context edit (a "seam"). AIPerf replays each chain as its own
//! conversation, concurrently with the others, so the chain structure decides the replay's
//! concurrency.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::Req;

const EPS: f64 = 1e-6;
/// AIPerf's `WEKA_SEAM_MAX_GAP_SECONDS` and `WEKA_SEAM_MIN_OVERLAP_RATIO` defaults.
pub(crate) const SEAM_MAX_GAP_S: f64 = 3600.0;
pub(crate) const SEAM_MIN_OVERLAP: f64 = 0.5;
/// A leading prefix-disjoint request with at most this output is a title-generation preamble.
const TITLE_GEN_MAX_OUTPUT: u64 = 64;

/// One request of a list under detection: its outer index (position in the trace, or in a subagent's
/// requests) and its index in the planner's request table.
pub(crate) type Item = (u32, usize);

/// Where a chain forked off: the parent chain and its tail request, the hash blocks they share, and
/// when the fork's first request started.
#[derive(Clone, Debug)]
pub(crate) struct Fork {
    pub parent_chain: Option<usize>,
    pub fork_outer: Option<u32>,
    pub depth: usize,
    pub fork_time: f64,
}

/// One detected agent chain (`AgentChain`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Chain {
    pub items: Vec<Item>,
    pub fork: Option<Fork>,
    /// The chain a seam continuation was spliced onto.
    pub spliced_into: Option<usize>,
    tail: Option<(u32, usize)>,
    tail_end: f64,
    tail_model: u8,
}

/// The chains of one request list (`ChainDetectionResult`).
#[derive(Debug)]
pub(crate) struct Detection {
    pub chains: Vec<Chain>,
    pub main: usize,
    /// Live chains other than the main one, by first request `(t, outer)`.
    pub workers: Vec<usize>,
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn by_time(reqs: &[Req], a: &Item, b: &Item) -> Ordering {
    reqs[a.1].t.total_cmp(&reqs[b.1].t).then(a.0.cmp(&b.0))
}

/// Splits a leading title-generation or one-shot request off the front (`_split_off_preamble`).
/// Returns `(preamble, rest)`, `rest` in outer-index order.
pub(crate) fn split_off_preamble(reqs: &[Req], items: &[Item]) -> (Vec<Item>, Vec<Item>) {
    if items.len() < 2 {
        return (Vec::new(), items.to_vec());
    }
    let mut ordered = items.to_vec();
    ordered.sort_by(|a, b| by_time(reqs, a, b));
    let first = &reqs[ordered[0].1];
    if first.hash.is_empty() {
        return (Vec::new(), items.to_vec());
    }
    let rest = &ordered[1..];
    if rest
        .iter()
        .any(|&(_, r)| !reqs[r].hash.is_empty() && lcp(&first.hash, &reqs[r].hash) > 0)
    {
        return (Vec::new(), items.to_vec());
    }
    if first.out > TITLE_GEN_MAX_OUTPUT {
        let others: HashSet<u32> = rest
            .iter()
            .flat_map(|&(_, r)| reqs[r].hash.iter().copied())
            .collect();
        if first.hash.iter().any(|h| others.contains(h)) {
            return (Vec::new(), items.to_vec());
        }
    }
    let mut rest = rest.to_vec();
    rest.sort_by_key(|item| item.0);
    (vec![ordered[0]], rest)
}

struct Phase1<'a> {
    reqs: &'a [Req],
    chains: Vec<Chain>,
    chain_of: HashMap<u32, usize>,
    forks_by_tail: HashMap<u32, Vec<usize>>,
    req_by_outer: HashMap<u32, usize>,
}

impl Phase1<'_> {
    fn append(&mut self, ci: usize, item: Item) {
        let req = &self.reqs[item.1];
        let c = &mut self.chains[ci];
        c.items.push(item);
        self.chain_of.insert(item.0, ci);
        if !req.hash.is_empty() {
            c.tail = Some(item);
            c.tail_end = req.end();
            c.tail_model = req.model;
        }
    }

    fn tail_hash(&self, c: &Chain) -> &[u32] {
        c.tail.map_or(&[][..], |(_, r)| &self.reqs[r].hash)
    }

    fn classify(&mut self, item: Item) {
        self.req_by_outer.insert(item.0, item.1);
        let req = &self.reqs[item.1];
        if req.hash.is_empty() {
            // No LCP evidence: kept on the main chain, invisible to tails and forks.
            if self.chains.is_empty() {
                self.chains.push(Chain::default());
            }
            self.chains[0].items.push(item);
            self.chain_of.insert(item.0, 0);
            return;
        }
        if self.chains.is_empty() {
            self.chains.push(Chain::default());
            self.append(0, item);
            return;
        }
        if let Some(target) = self.extension_target(&req.hash, req.t, req.model) {
            self.append(target, item);
            return;
        }
        let (parent, depth) = self.max_lcp_chain(&req.hash);
        if parent.is_none() && self.chains.iter().all(|c| c.tail.is_none()) {
            self.append(0, item);
            return;
        }
        let fork_outer = parent.and_then(|p| self.chains[p].tail.map(|(o, _)| o));
        let new = self.chains.len();
        self.chains.push(Chain {
            fork: Some(Fork {
                parent_chain: parent,
                fork_outer,
                depth,
                fork_time: req.t,
            }),
            ..Chain::default()
        });
        self.append(new, item);
        if let Some(outer) = fork_outer
            && depth > 0
        {
            self.forks_by_tail.entry(outer).or_default().push(new);
        }
    }

    /// The deepest chain whose tail is a complete prefix of `h`, has ended by `t` and ran on `model`;
    /// ties go to the lowest index (`_find_extension_target`).
    fn extension_target(&self, h: &[u32], t: f64, model: u8) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (idx, c) in self.chains.iter().enumerate() {
            let tail = self.tail_hash(c);
            let tl = tail.len();
            if tl == 0 || tl > h.len() || best.is_some_and(|(_, len)| tl <= len) {
                continue;
            }
            if c.tail_model != model || c.tail_end > t + EPS {
                continue;
            }
            if tail[tl - 1] == h[tl - 1] && tail == &h[..tl] {
                best = Some((idx, tl));
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// The chain tail with the deepest LCP against `h`; ties: deeper tail, then lower index.
    fn max_lcp_chain(&self, h: &[u32]) -> (Option<usize>, usize) {
        let mut best = None;
        let mut best_key = (0, 0);
        for (idx, c) in self.chains.iter().enumerate() {
            let tail = self.tail_hash(c);
            if tail.is_empty() {
                continue;
            }
            let d = lcp(tail, h);
            if d == 0 {
                continue;
            }
            let key = (d, tail.len());
            if key > best_key {
                best = Some(idx);
                best_key = key;
            }
        }
        (best, best_key.0)
    }
}

/// Partitions `items` into per-agent chains (`detect_agent_chains`).
pub(crate) fn detect(reqs: &[Req], items: &[Item]) -> Detection {
    if items.is_empty() {
        return Detection {
            chains: Vec::new(),
            main: 0,
            workers: Vec::new(),
        };
    }
    let mut ordered = items.to_vec();
    ordered.sort_by(|a, b| by_time(reqs, a, b));
    let mut p = Phase1 {
        reqs,
        chains: Vec::new(),
        chain_of: HashMap::new(),
        forks_by_tail: HashMap::new(),
        req_by_outer: HashMap::new(),
    };
    for &item in &ordered {
        p.classify(item);
    }
    resolve_seams(&mut p);
    let alias: HashMap<usize, usize> = p
        .chains
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.spliced_into.map(|to| (i, to)))
        .collect();
    let resolve = |mut i: usize| {
        while let Some(&to) = alias.get(&i) {
            i = to;
        }
        i
    };
    for c in &mut p.chains {
        if c.spliced_into.is_none()
            && let Some(fork) = &mut c.fork
            && let Some(parent) = fork.parent_chain
        {
            fork.parent_chain = Some(resolve(parent));
        }
    }
    let main = resolve(p.chain_of[&ordered[0].0]);
    let mut workers: Vec<usize> = (0..p.chains.len())
        .filter(|&i| p.chains[i].spliced_into.is_none() && i != main)
        .collect();
    workers.sort_by(|&a, &b| by_time(reqs, &p.chains[a].items[0], &p.chains[b].items[0]));
    Detection {
        chains: p.chains,
        main,
        workers,
    }
}

fn last_hash_outer(reqs: &[Req], chain: &Chain) -> Option<u32> {
    chain
        .items
        .iter()
        .rev()
        .find(|&&(_, r)| !reqs[r].hash.is_empty())
        .map(|&(o, _)| o)
}

/// Phase 2: splices join-seam continuations onto dead tails (`_resolve_seams`).
fn resolve_seams(p: &mut Phase1) {
    let reqs = p.reqs;
    let mut alias: HashMap<usize, usize> = HashMap::new();
    let mut keys: BinaryHeap<Reverse<u32>> = p.forks_by_tail.keys().map(|&k| Reverse(k)).collect();
    let mut processed: HashSet<u32> = HashSet::new();
    while let Some(Reverse(key)) = keys.pop() {
        if !processed.insert(key) {
            continue;
        }
        let mut owner = p.chain_of[&key];
        while let Some(&to) = alias.get(&owner) {
            owner = to;
        }
        if last_hash_outer(reqs, &p.chains[owner]) != Some(key) {
            continue; // the longer state was extended: every fork is a spawn
        }
        let registered: Vec<usize> = p.forks_by_tail[&key]
            .iter()
            .copied()
            .filter(|&ci| p.chains[ci].spliced_into.is_none())
            .collect();
        let tail_req = &reqs[p.req_by_outer[&key]];
        let Some(elected) = elect_continuation(reqs, &p.chains, &registered, tail_req) else {
            continue;
        };
        let moved = p.chains[elected].items.clone();
        for &(outer, _) in &moved {
            p.chain_of.insert(outer, owner);
        }
        p.chains[owner].items.extend(moved);
        p.chains[elected].spliced_into = Some(owner);
        alias.insert(elected, owner);
        let Some(new_tail) = last_hash_outer(reqs, &p.chains[owner]) else {
            continue;
        };
        // Re-key the forks that lost the election against the merged chain's new tail.
        let new_hash = &reqs[p.req_by_outer[&new_tail]].hash;
        let mut rekeyed = false;
        for &ci in &registered {
            let c = &p.chains[ci];
            if ci == elected || c.spliced_into.is_some() || c.fork.is_none() {
                continue;
            }
            let d = lcp(new_hash, &reqs[c.items[0].1].hash);
            if d == 0 {
                continue;
            }
            let fork = p.chains[ci].fork.as_mut().expect("checked above");
            fork.fork_outer = Some(new_tail);
            fork.depth = d;
            fork.parent_chain = Some(owner);
            p.forks_by_tail.entry(new_tail).or_default().push(ci);
            rekeyed = true;
        }
        if rekeyed {
            processed.remove(&new_tail);
            keys.push(Reverse(new_tail));
        }
    }
}

/// Picks the seam continuation among the forks of a dead tail: positive depth, starting after the
/// tail ended, same model, and not a far-off low-overlap restart. Deepest wins, then the earliest
/// fork, then the lowest index.
fn elect_continuation(
    reqs: &[Req],
    chains: &[Chain],
    registered: &[usize],
    tail: &Req,
) -> Option<usize> {
    let t_end = tail.end();
    let tail_blocks = tail.hash.len();
    let blocked = |ci: usize| {
        if tail_blocks == 0 {
            return false;
        }
        let first = &reqs[chains[ci].items[0].1];
        let depth = chains[ci].fork.as_ref().map_or(0, |f| f.depth);
        first.t - t_end > SEAM_MAX_GAP_S && (depth as f64 / tail_blocks as f64) < SEAM_MIN_OVERLAP
    };
    let mut best: Option<usize> = None;
    for &ci in registered {
        let Some(fork) = &chains[ci].fork else {
            continue;
        };
        let first = &reqs[chains[ci].items[0].1];
        if fork.depth == 0 || t_end > first.t + EPS || first.model != tail.model || blocked(ci) {
            continue;
        }
        best = match best {
            None => Some(ci),
            Some(b) => {
                let fb = chains[b].fork.as_ref().expect("candidates have forks");
                let better = fork.depth > fb.depth
                    || (fork.depth == fb.depth
                        && (fork.fork_time < fb.fork_time
                            || (fork.fork_time == fb.fork_time && ci < b)));
                Some(if better { ci } else { b })
            }
        };
    }
    best
}

/// AIPerf's sidecar classification defaults (`WEKA_AUX_*`).
const AUX_MAX_REQUESTS: usize = 1;
const AUX_ISL_RATIO: f64 = 0.10;
const AUX_ISL_FLOOR: u64 = 16_384;
const AUX_REDUCTION_OSL_MAX: u64 = 4_000;
const AUX_REDUCTION_RATIO: f64 = 20.0;
const WORKER_GROUP_MIN: usize = 3;

/// The session-id suffix of a detected worker chain: `aux:NNN` for a one-shot sidecar, `wg:GGG_MMM`
/// for a member of a parallel fan-out, `fa:NNN` otherwise. Only names change; the replay is the same.
pub(crate) fn worker_suffix(
    reqs: &[Req],
    chain_items: &[Item],
    n: usize,
    main_peak_isl: u64,
    main_model: Option<u8>,
    wg: Option<(usize, usize)>,
) -> String {
    let chain: Vec<&Req> = chain_items.iter().map(|&(_, r)| &reqs[r]).collect();
    let aux = is_aux_chain(&chain, main_peak_isl, main_model);
    let reduction = !aux && is_reduction_chain(&chain);
    match (aux || reduction, wg) {
        (true, _) => format!("aux:{n:03}"),
        (false, Some((group, member))) => format!("wg:{group:03}_{member:03}"),
        (false, None) => format!("fa:{n:03}"),
    }
}

fn is_aux_chain(chain: &[&Req], main_peak_isl: u64, main_model: Option<u8>) -> bool {
    if chain.is_empty() || chain.len() > AUX_MAX_REQUESTS {
        return false;
    }
    let first = chain[0];
    if main_model.is_some_and(|m| m != first.model) {
        return true;
    }
    let threshold = AUX_ISL_FLOOR.max((AUX_ISL_RATIO * main_peak_isl as f64) as u64);
    first.input < threshold
}

fn is_reduction_chain(chain: &[&Req]) -> bool {
    if chain.len() != 1 {
        return false;
    }
    let first = chain[0];
    if first.out == 0 || first.out >= AUX_REDUCTION_OSL_MAX || first.input < AUX_ISL_FLOOR {
        return false;
    }
    first.input as f64 > AUX_REDUCTION_RATIO * first.out as f64
}

/// A worker chain as a fan-out candidate: (start, end, first outer index, chain index).
type Candidate = (f64, f64, u32, usize);

/// Parallel fan-out membership (`worker_group_assignment`): workers forked from the same parent
/// request whose active intervals overlap, in groups of at least three.
pub(crate) fn worker_groups(reqs: &[Req], d: &Detection) -> HashMap<usize, (usize, usize)> {
    // Candidates bucketed by fork point (parent chain, fork outer index).
    let mut buckets: Vec<((usize, u32), Vec<Candidate>)> = Vec::new();
    for &ci in &d.workers {
        let c = &d.chains[ci];
        let Some(fork) = &c.fork else { continue };
        let (Some(parent), Some(outer)) = (fork.parent_chain, fork.fork_outer) else {
            continue;
        };
        if fork.depth == 0 || c.items.is_empty() {
            continue;
        }
        let t0 = reqs[c.items[0].1].t;
        let t1 = c
            .items
            .iter()
            .map(|&(_, r)| reqs[r].end())
            .fold(f64::NEG_INFINITY, f64::max);
        let cand = (t0, t1, c.items[0].0, ci);
        match buckets.iter_mut().find(|(key, _)| *key == (parent, outer)) {
            Some((_, list)) => list.push(cand),
            None => buckets.push(((parent, outer), vec![cand])),
        }
    }
    let mut groups: Vec<Vec<Candidate>> = Vec::new();
    for (_, cands) in buckets {
        for comp in overlap_components(&cands) {
            if comp.len() >= WORKER_GROUP_MIN {
                groups.push(comp);
            }
        }
    }
    let first_key = |comp: &Vec<Candidate>| {
        comp.iter()
            .map(|c| (c.0, c.2))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .expect("non-empty")
    };
    groups.sort_by(|a, b| {
        let (ka, kb) = (first_key(a), first_key(b));
        ka.0.total_cmp(&kb.0).then(ka.1.cmp(&kb.1))
    });
    let mut out = HashMap::new();
    for (group, mut comp) in groups.into_iter().enumerate() {
        comp.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
        for (member, c) in comp.iter().enumerate() {
            out.insert(c.3, (group, member));
        }
    }
    out
}

/// Connected components of `[t0, t1)` overlap, each in `(t0, outer)` order. In start order, an
/// interval joins the current component iff it starts before the latest end so far, because it then
/// overlaps the interval that ends there. AIPerf's sweep with union-find finds the same components;
/// touching intervals (`t1 == t0`) do not overlap.
fn overlap_components(cands: &[Candidate]) -> Vec<Vec<Candidate>> {
    let mut sorted = cands.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let mut comps: Vec<Vec<Candidate>> = Vec::new();
    let mut end = f64::NEG_INFINITY;
    for cand in sorted {
        match comps.last_mut() {
            Some(comp) if cand.0 < end => comp.push(cand),
            _ => comps.push(vec![cand]),
        }
        end = end.max(cand.1);
    }
    comps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(t: f64, api: f64, hash: &[u32]) -> Req {
        Req {
            t,
            api: Some(api),
            think: None,
            model: 0,
            input: hash.len() as u64 * 2,
            out: 10,
            hash: hash.to_vec(),
        }
    }

    fn items(reqs: &[Req]) -> Vec<Item> {
        (0..reqs.len()).map(|i| (i as u32, i)).collect()
    }

    fn chain_outers(d: &Detection, ci: usize) -> Vec<u32> {
        d.chains[ci].items.iter().map(|&(o, _)| o).collect()
    }

    #[test]
    fn a_growing_conversation_is_one_chain() {
        let reqs = vec![
            req(0.0, 1.0, &[1, 2]),
            req(2.0, 1.0, &[1, 2, 3]),
            req(4.0, 1.0, &[1, 2, 3, 4]),
        ];
        let d = detect(&reqs, &items(&reqs));
        assert_eq!((d.main, d.workers.len()), (0, 0));
        assert_eq!(chain_outers(&d, 0), vec![0, 1, 2]);
    }

    #[test]
    fn a_fork_whose_parent_state_continues_is_a_worker() {
        // Request 1 extends request 0. Request 2 shares only [1, 2] while request 1 is still running,
        // and request 3 returns to request 1's longer state, so request 2 is another agent.
        // (AIPerf 0.13.0's detect_agent_chains gives the same partition.)
        let reqs = vec![
            req(0.0, 1.0, &[1, 2]),
            req(2.0, 10.0, &[1, 2, 7, 8]),
            req(3.0, 1.0, &[1, 2, 3]),
            req(20.0, 1.0, &[1, 2, 7, 8, 9]),
        ];
        let d = detect(&reqs, &items(&reqs));
        assert_eq!(d.workers.len(), 1);
        assert_eq!(chain_outers(&d, d.main), vec![0, 1, 3]);
        assert_eq!(chain_outers(&d, d.workers[0]), vec![2]);
    }

    #[test]
    fn a_compaction_is_spliced_back_as_a_seam() {
        // Request 2 keeps only a prefix of request 1 (a context edit) and nothing ever returns to the
        // longer state, so it is the same agent continuing.
        let reqs = vec![
            req(0.0, 1.0, &[1, 2, 3]),
            req(2.0, 1.0, &[1, 2, 3, 4]),
            req(4.0, 1.0, &[1, 2, 9]),
        ];
        let d = detect(&reqs, &items(&reqs));
        assert!(d.workers.is_empty());
        assert_eq!(chain_outers(&d, d.main), vec![0, 1, 2]);
    }

    #[test]
    fn a_disjoint_leader_is_a_preamble() {
        let reqs = vec![
            req(0.0, 1.0, &[50]),
            req(1.0, 1.0, &[1, 2]),
            req(3.0, 1.0, &[1, 2, 3]),
        ];
        let (pre, rest) = split_off_preamble(&reqs, &items(&reqs));
        assert_eq!(pre, vec![(0, 0)]);
        assert_eq!(rest, vec![(1, 1), (2, 2)]);
        // A leader that shares a prefix is not one.
        let reqs = vec![req(0.0, 1.0, &[1]), req(1.0, 1.0, &[1, 2])];
        assert!(split_off_preamble(&reqs, &items(&reqs)).0.is_empty());
    }

    #[test]
    fn worker_suffixes_name_sidecars_and_agents() {
        let mut reqs = vec![req(0.0, 1.0, &[1, 2]), req(1.0, 1.0, &[1, 2, 5])];
        // One short request: a sidecar.
        assert_eq!(
            worker_suffix(&reqs, &[(1, 1)], 0, 100, Some(0), None),
            "aux:000"
        );
        // Two requests: an agent; inside a fan-out group: a worker-group member.
        reqs.push(req(5.0, 1.0, &[1, 2, 5, 6]));
        assert_eq!(
            worker_suffix(&reqs, &[(1, 1), (2, 2)], 3, 100, Some(0), None),
            "fa:003"
        );
        assert_eq!(
            worker_suffix(&reqs, &[(1, 1), (2, 2)], 3, 100, Some(0), Some((1, 2))),
            "wg:001_002"
        );
    }

    #[test]
    fn overlapping_intervals_form_components() {
        let cands = vec![
            (0.0, 5.0, 0, 0),
            (1.0, 2.0, 1, 1),
            (5.0, 6.0, 2, 2),
            (5.5, 7.0, 3, 3),
        ];
        let comps = overlap_components(&cands);
        let sizes: Vec<usize> = comps.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![2, 2], "touching intervals do not overlap");
    }
}
