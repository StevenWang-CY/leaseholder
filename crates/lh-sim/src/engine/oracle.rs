//! Next-use oracle for the clairvoyant reference: when each block is next needed.

use std::cell::Cell;

use crate::prefix::{BlockId, IdMap, UNIQUE_BIT};

/// "Never used again."
pub const NEVER: u64 = u64::MAX;

/// For every prompt block of a recorded request sequence, the positions (request indices in admission
/// order) that use it. The engine advances [`Oracle::set_position`] as it admits requests, and
/// [`Oracle::next_use`] answers the first use at or after the current position.
#[derive(Debug, Default)]
pub struct Oracle {
    spans: IdMap<(u32, u32, Cell<u32>)>,
    uses: Vec<u32>,
    position: Cell<u32>,
}

impl Oracle {
    /// `requests[i]` lists the prompt blocks of the request admitted `i`-th.
    pub fn new<'a>(requests: impl IntoIterator<Item = &'a [BlockId]>) -> Self {
        let mut pairs: Vec<(BlockId, u32)> = Vec::new();
        for (pos, blocks) in requests.into_iter().enumerate() {
            pairs.extend(blocks.iter().map(|&b| (b, pos as u32)));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut spans = IdMap::default();
        let mut uses = Vec::with_capacity(pairs.len());
        let mut i = 0;
        while i < pairs.len() {
            let id = pairs[i].0;
            let start = uses.len() as u32;
            while i < pairs.len() && pairs[i].0 == id {
                uses.push(pairs[i].1);
                i += 1;
            }
            spans.insert(id, (start, uses.len() as u32 - start, Cell::new(0)));
        }
        Self {
            spans,
            uses,
            position: Cell::new(0),
        }
    }

    /// The engine has admitted `admitted` requests; next uses are counted from there.
    pub fn set_position(&self, admitted: u32) {
        self.position.set(admitted);
    }

    /// The first position at or after the current one that uses `id`, or [`NEVER`].
    pub fn next_use(&self, id: BlockId) -> u64 {
        if id & UNIQUE_BIT != 0 {
            return NEVER;
        }
        let Some((start, len, cursor)) = self.spans.get(&id) else {
            return NEVER;
        };
        let now = self.position.get();
        let mut c = cursor.get();
        while c < *len && self.uses[(start + c) as usize] < now {
            c += 1;
        }
        cursor.set(c);
        if c < *len {
            self.uses[(start + c) as usize] as u64
        } else {
            NEVER
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_uses_follow_the_position() {
        let reqs: Vec<Vec<BlockId>> = vec![vec![1, 2], vec![2, 3], vec![1]];
        let o = Oracle::new(reqs.iter().map(Vec::as_slice));
        o.set_position(1);
        assert_eq!((o.next_use(1), o.next_use(2), o.next_use(3)), (2, 1, 1));
        o.set_position(2);
        assert_eq!(
            (o.next_use(1), o.next_use(2), o.next_use(3)),
            (2, NEVER, NEVER)
        );
        assert_eq!(o.next_use(99), NEVER);
        assert_eq!(o.next_use(UNIQUE_BIT | 5), NEVER);
    }
}
