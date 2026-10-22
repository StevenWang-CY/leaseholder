# Benchmarks

What has been measured so far, on one RTX 5060 Ti 16 GB (sm_120, PCIe Gen3 x8) in Phastform: two Xeon E5-2609 v4 at 1.7 GHz, 188 GiB RAM, vLLM 0.32.0. The cost-model and voice numbers come from the pre-W0 probes of 2026-10-21/22. E0 (W1) re-measures them on the frozen box with the scripts in this repository, and this page is updated from those runs.

## Tier cost model: resuming an agent turn

| 16k-token context, BF16 weights | Time to first token |
|---|---|
| KV resident on the GPU (prefix-cache hit) | **0.18 s** |
| KV restored from host RAM (native CPU offload) | **0.57 s** |
| Recomputed | **4.75 s** |

- Restoring beats recomputing from about 64–128 tokens.
- The admission model fits these points linearly:
  - GPU hit ≈ 60 ms + 7.2 µs per token;
  - a CPU restore adds ≈ 8 ms + 23.6 µs per token.
- With FP8 weights, the frozen model's 2k / 8k-token TTFT is 203 / 1,082 ms cold and 47 / 92 ms warm. It is unchanged with the audio servers co-located.

## The frozen box

| | Value |
|---|---|
| GPU KV pool | 36,400 tokens, fixed at 5 GiB (32k context) |
| CPU tier | 20 GiB in `/dev/shm`: 17.35 GiB on NUMA node 1 (GPU-local), 2.65 GiB spilled to node 0 |
| GPU memory, all three servers idle | 14,408 / 16,311 MiB |
| GPU memory, stress peak (LLM c=16, 30k prefill, STT c=4, TTS c=4) | 14,086 / 16,311 MiB in the probe; no OOM, no preemption |
| Cold start, first boot (includes a one-time FlashInfer JIT) | 7.5 min for all three servers |
| Engine restart with warm caches | about 2 min |

## Voice on the shared GPU

| Server-side voice turn (STT → LLM first sentence → TTS first audio) | Latency |
|---|---|
| Idle | **415 ms** (STT 152 ms, LLM first sentence 86 ms, TTS 176–179 ms) |
| While background agents run 8k-token prefills | **876 ms** median |

- Under that background load:
  - LLM time to first token rises from 39 to 371 ms;
  - the co-located STT and TTS slow down by 43–44% under time-slicing.
- Whisper large-v3-turbo on vLLM takes 144 / 166 / 221 ms for 2 / 5 / 10 s utterances at concurrency 1.
- Kokoro-82M reaches first audio for one sentence in 154–179 ms.
- **A barge-in closes the LLM stream.** This is LiveKit Agents 1.8.8 with mock STT, LLM and TTS, in the voice-plane namespace on cyan (`python/leaseholder_voice`, 10 runs). Measured from the caller's barge-in onset to the LLM server seeing the disconnect:
  - 2.19 s median with the stock handling, which pauses the reply audio and interrupts it on the barge-in's final transcript;
  - 0.58 s with `resume_false_interruption=False`, which interrupts after 0.5 s of speech.
  - In stock mode the reply keeps streaming on the GPU for the whole barge-in.

## Engine plugins

`LeasePolicy` at a full-size tier of 18,204 chunks (40 GiB of Qwen3-4B BF16 KV), against vLLM's real offload manager on CPU:

| Operation | LeasePolicy | Stock LRU |
|---|---|---|
| `prepare_store` plus evicting 128 chunks, mostly leased | 0.30 ms | 0.21 ms |
| Finishing a 1,011-chunk session: renewal | 0.39 ms | — |
| Finishing a 1,011-chunk session: class change | 0.28 ms | — |

**Live check on the frozen box** (`python/leaseholder_vllm/tests/live`). A 7,921-token prompt was written through to the CPU tier. After the GPU prefix cache was reset, the same prompt was restored with 7,920 tokens from the CPU tier, as reported by both the hit receipt and `usage.prompt_tokens_details.cached_tokens`.

## Agent replay on the frozen box: a first retention-headroom signal

This was W0 checklist item 5. The setup: AIPerf `agentic_replay` of the 32×-scaled Claude Code corpus, 16 session trees, think times capped at 120 s, `ignore_eos`, 10 minutes, against the frozen box through the tunnel. `lh-bench kv-metrics` compared vLLM's sampled KV block histograms over the run.

**Load.** 688 requests, 0 errors.

| | p50 | p90 | p99 |
|---|---|---|---|
| TTFT | 135 ms | 815 ms | 8,501 ms |
| Inter-token latency | 15.6 ms | 40.9 ms | 59.1 ms |

**Where prompt tokens came from.** 93.4% were served from cache, and only 6.6% were computed, mostly the cold first turns of new session plays.

| Source | Share of prompt tokens |
|---|---|
| GPU prefix cache | about 57% |
| Restored from the CPU tier | about 37% |
| Computed | 6.6% |

**KV block lifecycle.**

| Histogram | p50 | p90 |
|---|---|---|
| Reuse gap | 2.54 s | 6.33 s |
| Idle before eviction | 6.22 s | 27.3 s |

**Reading.**
- Half of the evicted blocks (50.4%) had been idle no longer than the 90th-percentile reuse gap, so LRU evicts sessions that are coming back.
- With the 4× write-through CPU tier, those misses cost a restore, not a recompute. The headroom on this configuration is GPU placement and prefetch timing, not CPU-tier retention.
- The reuse-gap histogram counts only blocks that survived until reuse, so it is biased short. This is a signal, and the simulator's OPT bound is the measurement.

## Load generation

- AIPerf 0.13.0 `agentic_replay` over the SemiAnalysis Claude Code traces: 99.4% of turns arrive within 50 ms of their recorded delay.
- Block scaling makes all 98,827 requests fit:
  - 32× fits a 32k context, with main-agent context p50 7.9k and p90 20.3k tokens;
  - 64× fits 16k.
- One host to avoid for clients: its p99 noise of 200–350 ms (green) rules it out. Load runs on cyan, with pinned cores.
