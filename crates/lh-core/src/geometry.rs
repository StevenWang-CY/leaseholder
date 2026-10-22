//! KV geometry: bytes per token, the engine block, and tier capacities in blocks.

use serde::{Deserialize, Serialize};

/// The GPU KV pool of the frozen box, in tokens: `--kv-cache-memory` 5 GiB of Qwen3-4B KV.
pub const FROZEN_BOX_POOL_TOKENS: u64 = 36_400;

/// The CPU tier of the frozen box, in bytes: the OffloadingConnector's `cpu_bytes_to_use`, 20 GiB.
pub const FROZEN_BOX_CPU_BYTES: u64 = 20 << 30;

/// The KV footprint of one model in one engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvGeometry {
    /// KV bytes per token: layers × KV heads × head dim × 2 (K and V) × bytes per element.
    pub bytes_per_token: u64,
    /// Tokens per engine block (vLLM's block size).
    pub block_tokens: u64,
}

impl KvGeometry {
    /// Qwen3-4B with a BF16 KV cache, in 16-token blocks: 36 layers × 8 KV heads × 128 × 2 × 2 bytes
    /// = 147,456 bytes (144 KiB) per token. The frozen box serves the FP8 checkpoint, whose KV cache
    /// stays BF16.
    pub const QWEN3_4B_BF16: Self = Self::new(36, 8, 128, 2, 16);

    /// The geometry of a model's KV cache, from its attention shape and the engine block size.
    pub const fn new(
        layers: u64,
        kv_heads: u64,
        head_dim: u64,
        bytes_per_element: u64,
        block_tokens: u64,
    ) -> Self {
        Self {
            bytes_per_token: layers * kv_heads * head_dim * 2 * bytes_per_element,
            block_tokens,
        }
    }

    /// KV bytes of one engine block.
    pub const fn block_bytes(&self) -> u64 {
        self.bytes_per_token * self.block_tokens
    }

    /// KV bytes of `tokens` tokens.
    pub const fn bytes_for_tokens(&self, tokens: u64) -> u64 {
        self.bytes_per_token * tokens
    }

    /// Whole blocks that fit in `bytes`. vLLM sizes its CPU tier as `cpu_bytes_to_use` divided by the
    /// bytes of one block, rounded down.
    pub const fn blocks_in_bytes(&self, bytes: u64) -> u64 {
        bytes / self.block_bytes()
    }
}

/// How many blocks each KV tier holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TierCapacity {
    /// Blocks in the GPU pool, as vLLM reports `num_gpu_blocks`.
    pub gpu_blocks: u64,
    /// Blocks in the CPU tier; 0 means no CPU tier.
    pub cpu_blocks: u64,
}

impl TierCapacity {
    /// The frozen box: 36,400 tokens in 2,275 GPU blocks, and a 20 GiB CPU tier of 9,102 blocks.
    pub const fn frozen_box(geometry: &KvGeometry) -> Self {
        Self {
            gpu_blocks: FROZEN_BOX_POOL_TOKENS / geometry.block_tokens,
            cpu_blocks: geometry.blocks_in_bytes(FROZEN_BOX_CPU_BYTES),
        }
    }

    /// A GPU pool of `pool_tokens` and a CPU tier `ratio` times its size (0 for none).
    pub fn with_cpu_ratio(geometry: &KvGeometry, pool_tokens: u64, ratio: f64) -> Self {
        let gpu_blocks = pool_tokens / geometry.block_tokens;
        Self {
            gpu_blocks,
            cpu_blocks: (gpu_blocks as f64 * ratio.max(0.0)).round() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_4b_kv_is_144_kib_per_token() {
        let g = KvGeometry::QWEN3_4B_BF16;
        assert_eq!(g.bytes_per_token, 147_456);
        assert_eq!(g.bytes_per_token, 144 << 10);
        assert_eq!(g.block_bytes(), 2_359_296);
        assert_eq!(g.bytes_for_tokens(16_384), 2_415_919_104);
    }

    #[test]
    fn the_frozen_box_has_2275_gpu_and_9102_cpu_blocks() {
        let g = KvGeometry::QWEN3_4B_BF16;
        let cap = TierCapacity::frozen_box(&g);
        assert_eq!(cap.gpu_blocks, 2_275);
        // 20 GiB / (16 × 144 KiB) = 9,102.2
        assert_eq!(cap.cpu_blocks, 9_102);
        // The pool is 5 GiB of KV: 36,400 tokens × 144 KiB = 4.999 GiB.
        assert!(g.bytes_for_tokens(FROZEN_BOX_POOL_TOKENS) <= 5 << 30);
    }

    #[test]
    fn cpu_ratios_scale_the_pool() {
        let g = KvGeometry::QWEN3_4B_BF16;
        for (ratio, blocks) in [(0.0, 0), (2.0, 4_550), (4.0, 9_100), (8.0, 18_200)] {
            let cap = TierCapacity::with_cpu_ratio(&g, FROZEN_BOX_POOL_TOKENS, ratio);
            assert_eq!((cap.gpu_blocks, cap.cpu_blocks), (2_275, blocks));
        }
    }
}
