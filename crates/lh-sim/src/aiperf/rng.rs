//! AIPerf's per-lane start point: a SHA-256 seed and numpy's `default_rng(seed).uniform(lo, hi)`.
//!
//! `TrajectorySource` seeds `numpy.random.default_rng` with the first 8 bytes (big-endian) of
//! `sha256(f"{seed}:{trace_id}:{lane}")` and draws the snapshot instant `t*` uniformly. Reproducing
//! numpy's SeedSequence and PCG64 here gives the simulator the same lanes as the measured run.

use sha2::{Digest, Sha256};

/// The numpy seed of `lane` for `trace_id`: `_seed_for_trace_lane(base_seed, trace_id, lane)`.
pub fn lane_seed(base_seed: u64, trace_id: &str, lane: usize) -> u64 {
    let digest = Sha256::digest(format!("{base_seed}:{trace_id}:{lane}").as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("a SHA-256 digest has 32 bytes"),
    )
}

/// `numpy.random.default_rng(seed).uniform(lo, hi)`: one draw from a fresh generator.
pub fn uniform(seed: u64, lo: f64, hi: f64) -> f64 {
    let mut rng = Pcg64::new(seed);
    lo + (hi - lo) * rng.next_f64()
}

/// numpy's PCG64 (XSL-RR 128/64), seeded through SeedSequence.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

impl Pcg64 {
    /// `numpy.random.PCG64(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        let s = seed_sequence_state(seed);
        let initstate = (s[0] as u128) << 64 | s[1] as u128;
        let initseq = (s[2] as u128) << 64 | s[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// A double in [0, 1) from the top 53 bits, as numpy's `next_double`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL_SIZE: usize = 4;

/// `numpy.random.SeedSequence(entropy).generate_state(4, np.uint64)` for an integer entropy.
fn seed_sequence_state(entropy: u64) -> [u64; 4] {
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut v = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    // numpy splits the entropy into little-endian 32-bit words and hashes one into each pool slot, a
    // zero where the words run out. A u64 is at most two words, so the zero-padded pool is exact and
    // numpy's pass over words beyond the pool never runs.
    let words = [entropy as u32, (entropy >> 32) as u32, 0, 0];
    let mut pool: [u32; POOL_SIZE] = words.map(&mut hashmix);
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    let mut hash_const = INIT_B;
    let mut out = [0u32; 8];
    for (i, word) in out.iter_mut().enumerate() {
        let mut v = pool[i % POOL_SIZE] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash_const);
        *word = v ^ (v >> 16);
    }
    std::array::from_fn(|i| out[2 * i] as u64 | (out[2 * i + 1] as u64) << 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Values from numpy 2.x and hashlib, as AIPerf 0.13.0 computes them.
    #[test]
    fn lane_seeds_match_aiperf() {
        assert_eq!(
            lane_seed(11, "002001296e8a8c38ad9d7cc436d691afc602", 0),
            10_459_471_519_670_851_094
        );
        assert_eq!(
            lane_seed(11, "085aacf44ac5428ad3c6b4f6cd431e083832", 15),
            1_545_323_306_067_940_826
        );
        assert_eq!(lane_seed(7, "abc", 3), 12_892_420_142_722_637_021);
    }

    #[test]
    fn seed_sequence_matches_numpy() {
        let cases: [(u64, [u64; 4]); 4] = [
            (
                0,
                [
                    15793235383387715774,
                    12390638538380655177,
                    2361836109651742017,
                    3188717715514472916,
                ],
            ),
            (
                11,
                [
                    3926704849073358691,
                    2926583794887213564,
                    215141457385765089,
                    15564452721439488421,
                ],
            ),
            (
                1 << 32,
                [
                    5836529245451711556,
                    8811601478698894690,
                    4323349454413245209,
                    13668837544996263139,
                ],
            ),
            (
                u64::MAX,
                [
                    12591116029944179981,
                    1268942265879420026,
                    1998214320288580263,
                    12063755697402456310,
                ],
            ),
        ];
        for (seed, state) in cases {
            assert_eq!(seed_sequence_state(seed), state, "seed {seed}");
        }
    }

    #[test]
    fn pcg64_matches_numpy() {
        let mut rng = Pcg64::new(0);
        let draws = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
        assert_eq!(
            draws,
            [
                11749869230777074271,
                4976686463289251617,
                755828109848996024
            ]
        );
        let mut rng = Pcg64::new(9_223_372_036_854_788_153);
        let draws = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
        assert_eq!(
            draws,
            [
                9116129000838757916,
                10625643075464141531,
                787597703341980114
            ]
        );
    }

    #[test]
    fn uniform_matches_numpy() {
        for (seed, expected) in [
            (0, 1274286.7314364307),
            (1, 1024131.6836866256),
            (11, 258011.89962073142),
            (4_294_967_295, 504635.3184937321),
            (u64::MAX, 1360373.671257764),
        ] {
            assert_eq!(uniform(seed, 1000.0, 2_000_000.5), expected, "seed {seed}");
        }
    }
}
