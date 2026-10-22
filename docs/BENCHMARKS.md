# Benchmarks

What has been measured so far, on one RTX 5060 Ti 16 GB (sm_120, PCIe Gen3 x8) in Phastform: two Xeon E5-2609 v4 at 1.7 GHz, 188 GiB RAM, vLLM 0.32.0. W1 numbers come from the frozen box ([`deploy/phastform`](../deploy/phastform)) and the `lh-bench` probes in this repository, run from cyan through the SSH tunnel as the control plane reaches the engine. Earlier probe numbers are kept only where W1 has not re-measured them, and are marked.

## The frozen box

| | Value |
|---|---|
| GPU KV pool | 36,400 tokens, fixed at 5 GiB (32k context) |
| CPU tier | 20 GiB in `/dev/shm`: 17.35 GiB on NUMA node 1 (GPU-local), 2.65 GiB spilled to node 0. In the earlier probe, a remote NUMA node cost at most 1% of pinned bandwidth |
| GPU memory, all three servers idle | 14,408 / 16,311 MiB |
| GPU memory, stress peak (LLM c=16, 30k prefill, STT c=4, TTS c=4) | 14,086 / 16,311 MiB in the earlier probe; no OOM, no preemption |
| Cold start, first boot (includes a one-time FlashInfer JIT) | 7.5 min for all three servers |
| Engine restart with warm caches | about 2 min |

## Tier cost model: resuming an agent turn (E0)

`lh-bench tiers` sends a unique prompt of L random token ids three times, each asking for one token: cold (recomputed), again at once (a GPU prefix-cache hit), and after other prompts pushed it out of the GPU pool (restored from the CPU tier). vLLM's prefix-cache counters, read around each request, verify where its tokens came from; no sample of these runs was invalid. Time to first token (TTFT), p50 of 5 samples, frozen box (Qwen3-4B-FP8, audio servers resident, 36,400-token pool, 20 GiB write-through CPU tier, step cap 512/256), 2026-10-22:

| Prompt tokens | Recomputed | GPU hit | Restored from host RAM |
|---|---|---|---|
| 1,024 | 135 ms | 45 ms | 64 ms |
| 2,048 | 198 ms | 40 ms | 95 ms |
| 4,096 | 437 ms | 48 ms | 157 ms |
| 8,192 | 1,114 ms | 64 ms | 281 ms |
| 16,384 | **3,264 ms** | **97 ms** | **509 ms** |
| 24,576 | 6,506 ms | 128 ms | 722 ms |
| 31,744 | 10,239 ms | 156 ms | 903 ms |

**Fitted model** (`lh-core`'s defaults):
- GPU hit ≈ 35 ms + 3.8 µs/token;
- a CPU restore adds ≈ 10 ms + 23.6 µs/token over a GPU hit (about 6.2 GB/s of KV at 144 KiB/token), the same slope as the earlier BF16 probe;
- recompute ≈ 48 ms + 63 µs/token + 8.1 ns/token², the quadratic term being attention. The 512-token step cap that protects decode latency makes cold prefill slower than an uncapped engine.

Restoring beats recomputing at every measured length (from 1,024 tokens). In the dimensionless terms E1 uses, `lh-sim` takes the restore time from these fits as what a restore adds over a GPU hit. At the 8k median prompt, ρ = restore / recompute ≈ 0.18. π = the median pause / restore depends on which pause:
- Copilot's median tool call (166 ms) gives about 0.8;
- the W0 replay window's median think time (1.6 s, at its 3.4k median prompt) gives about 18.

**Under contention** (p50; same probe while background load runs):

| Prompt tokens | Background | Recomputed | GPU hit | Restored |
|---|---|---|---|---|
| 1,024 | 8 decode streams (1k prompt, 1k output) | 200 ms | 131 ms | 173 ms |
| 16,384 | 8 decode streams | 4,528 ms | 186 ms | 592 ms |
| 1,024 | one stream of cold 8k prefills | 342 ms | 283 ms | 263 ms |
| 16,384 | one stream of cold 8k prefills | 5,230 ms | 387 ms | 728 ms |

- **Decode load** adds a roughly constant ~85 ms to a GPU hit and to a restore (the probe shares steps with the decode batch); the restore slope is unchanged, so PCIe is not the bottleneck.
- **Prefill load** is worse. Even a GPU-cached 1k prompt waits ~250 ms behind the running chunked prefill, and recompute slows 1.6–2.5×. This is the compute-interference path that a critical-window prefill throttle targets.

The earlier probe (2026-10-21, BF16 weights, no audio servers) measured 0.18 s / 0.57 s / 4.75 s at 16k for GPU hit / restore / recompute.

## Voice on the shared GPU

`lh-bench voice-turn` runs the cascade a voice agent runs, against the frozen box through the tunnel:
1. the caller's utterance goes to STT (LibriSpeech clips of 2.0 and 3.2 s);
2. the transcript is streamed through the LLM at the realtime priority;
3. the first sentence goes to TTS as soon as it is complete. A sentence counts once it reaches 20 characters, LiveKit's default.

The turn runs from sending the audio to the first byte of TTS audio. That is the server-side share of voice-to-voice latency: endpointing, the network and playout come on top. 30 turns per row, p50 / p90 in ms (W1, 2026-10-22):

| Background agent load | STT | LLM first token | LLM first sentence | TTS first audio | **Turn** | Background prefill |
|---|---|---|---|---|---|---|
| None | 160 / 168 | 49 / 51 | 251 / 319 | 176 / 187 | **589 / 665** | — |
| One stream of cold 8k-token prefills | 233 / 240 | 172 / 197 | 807 / 937 | 226 / 251 | **1,261 / 1,413** | 6,220 tokens/s |
| Two streams | 233 / 248 | 565 / 1,207 | 1,802 / 2,113 | 231 / 254 | **2,279 / 2,593** | 6,966 tokens/s |

- **The LLM's first sentence carries most of the damage.** Decode steps for the voice request share the GPU with 512-token prefill chunks, so the inter-token gap stretches. The co-located STT and TTS slow by up to 46% and 31% (time-slicing between processes).
- **MPS does not fix it.** With the three servers as clients of one MPS daemon (variant `mps`):
  - idle turns are unchanged (606 / 669 ms);
  - STT and TTS gain 7–8% under load;
  - beside one prefill stream, the turn gets worse (1,403 / 2,061 ms) and background prefill throughput drops 13%;
  - beside two streams it is mixed (2,042 / 2,870 ms).

  The interference that matters is inside the engine, so the frozen box stays on time-slicing, and an MPS split is kept as an equal-hardware arm of E2.

**End to end, from waveforms.** `lh-voice bots` places calls through LiveKit (2 callers, 25 turns each, a barge-in on 25% of turns) to the stock voice agent (`lh-voice-worker`). Voice-to-voice (V2V) runs from the end of the caller's speech to the agent's first audio, read at the caller on one clock (W1, 2026-10-22; the LLM path goes straight to the engine):

| Setup | V2V p50 | p95 | p99 | Barge-in to silence p50 / p95 |
|---|---|---|---|---|
| Voice alone, LiveKit's end-of-turn model | 1,646 ms | 3,226 ms | 3,444 ms | 637 / 686 ms |
| Voice alone, fixed endpointing (experiments) | **1,614 ms** | **1,760 ms** | **1,889 ms** | 634 / 696 ms |
| Beside the agent replay (16 session trees, 1.3 requests/s), fixed endpointing | 1,664 ms | **2,312 ms** | 3,013 ms | 563 / 687 ms |

- **Endpointing dominates voice alone.** About 1 s of each turn is endpointing (VAD end of speech plus LiveKit's 0.5 s minimum); the server-side turn is the rest.
- **The end-of-turn model adds a 3 s mode.** It judged about one line in six from the synthesized callers unfinished and waited for its 3 s maximum. Experiments therefore end turns a fixed delay after speech (the `vad-turns` variant), the same for every arm.
- **Agent load breaks the voice SLO.** Beside a realistic agent replay, V2V p95 rises by 552 ms and p99 by 1.1 s, against an SLO of at most +250 ms. Agent TTFT stayed 149 ms p50 and 11.2 s p99.
- **Nothing ran out of memory.** GPU memory held at 14.05 GiB, and neither the calls (50 turns) nor the replay (507 requests) had an error.

**A barge-in closes the LLM stream** (W0: LiveKit Agents 1.8.8 with mock STT, LLM and TTS, in the voice-plane namespace on cyan, [`python/leaseholder_voice`](../python/leaseholder_voice), 10 runs). Measured from the caller's barge-in onset to the LLM server seeing the disconnect:
- 2.19 s median with the stock handling, which pauses the reply audio and interrupts it on the barge-in's final transcript;
- 0.58 s with `resume_false_interruption=False`, which interrupts after 0.5 s of speech.

In stock mode the reply keeps streaming for the whole barge-in.

**Earlier probe** (2026-10-21/22, 8-character first sentence):
- the server-side turn took 415 ms idle and 876 ms median under 8k prefills;
- Whisper large-v3-turbo on vLLM took 144 / 166 / 221 ms for 2 / 5 / 10 s utterances at concurrency 1;
- Kokoro-82M reached first audio for one sentence in 154–179 ms.

## The control plane: lh-gateway in front of the engine

On cyan, the voice agent and the agent replay reach the engine through `lh-gateway`:
- the voice agent on a private Unix socket, as tenant `voice` (realtime class);
- the agent replay over loopback TCP with a key, as tenant `agents` (background class).

The gateway forwards over the SSH tunnel to the frozen box. Measured W1, 2026-10-22.

**What it adds to a voice request, paired.** The voice agent's own LLM request ran 40 rounds, alternating between the engine's tunnel socket and the gateway's voice socket, with greedy decoding so both paths generate the same text:

| Path | First token p50 | First sentence p50 | End p50 | Gap between chunks, p50 / p99 |
|---|---|---|---|---|
| Engine socket | 43.7 ms | 223.0 ms | 370.1 ms | 12.13 / 12.64 ms |
| Through lh-gateway | 44.1 ms | 223.3 ms | 370.3 ms | 12.14 / 12.69 ms |
| **Paired median difference** | **+0.38 ms** | **+0.26 ms** | **+0.25 ms** | — |

**Its own time, in place** (`lh_gateway_overhead_seconds`: from the end of a request's body to the upstream write). The voice agent's turns are seconds apart:

| Upstream keep-alive (engine / gateway) | Requests | Mean | p99 | Within 0.5 ms | First byte p99 |
|---|---|---|---|---|---|
| 5 s / 4 s (vLLM's default) | 60 | 0.24 ms | 0.50 ms | 100% | 205 ms |
| 60 s / 55 s (the frozen box from W1) | 60 | 0.13 ms | 0.44 ms | 100% | 99 ms |
| 60 s / 55 s, beside the 16-session agent replay (both classes) | 603 | 0.16 ms | 0.58 ms | 98.5% (all within 2 ms) | 700 ms (realtime) |

- **A client's upload is not overhead.** The first in-place reading was 1.8 ms on average, against about 0.03 ms in process. Of that, 1.57 ms was the voice worker's event loop sending the request body after its headers, which a direct call waits for too. The overhead metric now starts at the end of the body, and the upload is timed separately (`lh_gateway_request_body_seconds`: 1.35–1.57 ms mean for the voice agent).
- **Connections must outlive the turn gap.**
  - vLLM drops idle connections after 5 s, and voice turns are often further apart. So most voice requests opened a new connection through the tunnel.
  - A new connection costs about 1 ms, and now and then 40 ms more. One-token requests 6 s apart show it:

    | | Gateway's own time | First byte mean | First byte p99 |
    |---|---|---|---|
    | 5 s keep-alive | 0.26 ms | 41.8 ms | about 90 ms |
    | 60 s keep-alive | 0.08 ms | 38.5 ms | 49.8 ms |

  - The engine now keeps idle connections for 60 s (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`) and the gateway for 55 s.
- **End to end it is inside the run-to-run spread.**
  - Voice alone through the gateway: V2V p50 1,684, 1,600 and 1,633 ms over three runs.
  - Straight to the engine: 1,614 and 1,579 ms over two runs, every run with 0 errors.
  - Beside the agent replay: 1,727 / 2,361 / 3,141 and 1,710 / 2,361 / 2,872 ms (p50 / p95 / p99) over two runs through the gateway, against 1,664 / 2,312 / 3,013 ms direct.
- **In process** (`cargo run --release -p lh-gateway --example overhead` on cyan, [`lh-gateway` README](../crates/lh-gateway/README.md#overhead)), the gateway's own time is 32 µs on average. Against a direct call it adds 226 µs at p50 for one client, and at most 1.92 ms at p99, with 64 concurrent requests sharing the process with their client and engine.
- **Under load it carried both classes without an error.** Over the two runs:
  - the replay sent 578 and 543 background requests and the calls 60 realtime ones each;
  - every request completed except one realtime stream, which the voice agent closed early, as it does on a barge-in or when it replaces a preemptive reply;
  - there were no upstream errors;
  - agent TTFT was 141 and 146 ms p50, and 9.9 and 9.2 s p99.

## The API layer: agentic-api in front of the engine

Claude Code and Codex reach the engine through [agentic-api](https://github.com/vllm-project/agentic-api), so its cost is part of every agent turn. agentic-api's own `response_provider` benchmark compares its WebSocket, HTTP/SSE and JSON paths with direct vLLM, all five arms at once against the frozen box with tool calling on (agentic-api a7dc7b5, 2026-10-22). Per-turn latency for a 4-round tool loop, p50 in ms:

| Sessions per arm | agentic-api HTTP/SSE | vLLM HTTP/SSE | Paired median difference |
|---|---|---|---|
| 1 | 3,173 | 3,176 | 0 ms |
| 5 | 4,489 | 4,482 | +3 ms |
| 10 | 6,102 | 6,072 | +1 ms |

- **Latency.** agentic-api's latency cost is below the noise at up to 50 concurrent tool loops; the inference server dominates.
- **Resources.** It used about 5.5 ms of CPU per Responses round and 44–54 MiB of memory.
- **Request size.** Clients send 3.6–3.8 KB per turn through it against 8.4 KB direct, because it holds the conversation state.

The method, all five arms, correctness and the history-rehydration runs are in [`benchmarks/results/2026-10-22-agentic-api-http-baseline`](../benchmarks/results/2026-10-22-agentic-api-http-baseline/).

## Engine plugins

`LeasePolicy` at a full-size tier of 18,204 chunks (40 GiB of Qwen3-4B BF16 KV), against vLLM's real offload manager on CPU:

| Operation | LeasePolicy | Stock LRU |
|---|---|---|
| `prepare_store` plus evicting 128 chunks, mostly leased | 0.30 ms | 0.21 ms |
| Finishing a 1,011-chunk session: renewal | 0.39 ms | — |
| Finishing a 1,011-chunk session: class change | 0.28 ms | — |

**Live check on the frozen box** ([`python/leaseholder_vllm/tests/live`](../python/leaseholder_vllm/tests/live)):
- the lease hint reached both plugins;
- `X-Session-ID` appeared in GPU-tier KV events and in no CPU-tier one, and `LeaseConnector` mirrored it into the hint;
- a 7,921-token prompt was written through to the CPU tier. After the GPU prefix cache was reset, the same prompt was restored with 7,920 tokens from the CPU tier, as reported by both the hit receipt and `usage.prompt_tokens_details.cached_tokens`.

## Agent replay on the frozen box: a first retention-headroom signal

Measured in W0: AIPerf `agentic_replay` of the 32×-scaled Claude Code corpus, 16 session trees, think times capped at 120 s, `ignore_eos`, 10 minutes, against the frozen box through the tunnel. `lh-bench kv-metrics` compared vLLM's sampled KV block histograms over the run.

**Load.** 688 requests, 0 errors.

| | p50 | p90 | p99 |
|---|---|---|---|
| TTFT | 135 ms | 815 ms | 8,501 ms |
| Inter-token latency | 15.6 ms | 40.9 ms | 59.1 ms |

**Where prompt tokens came from.** 93.4% were served from cache:

| Source | Share of prompt tokens |
|---|---|
| GPU prefix cache | about 57% |
| Restored from the CPU tier | about 37% |
| Computed | 6.6% |

`lh-sim`'s replay of this run splits the computed share (6.1% there) into new context each turn adds (3.15 points), recompute after eviction from both tiers (2.05) and cold first turns (0.91).

**KV block lifecycle.**

| Histogram | p50 | p90 |
|---|---|---|
| Reuse gap | 2.54 s | 6.33 s |
| Idle before eviction | 6.22 s | 27.3 s |

**Reading.**
- Half of the evicted blocks (50.4%) had been idle no longer than the 90th-percentile reuse gap, so LRU evicts sessions that are coming back.
- With the 4× write-through CPU tier, those misses cost a restore, not a recompute. The headroom on this configuration is GPU placement and prefetch timing, not CPU-tier retention.
- The reuse-gap histogram counts only blocks that survived until reuse, so it is biased short. This is a signal, and the simulator's clairvoyant reference is the measurement.

## Load generation

- AIPerf 0.13.0 `agentic_replay` over the SemiAnalysis Claude Code traces: 99.4% of turns arrive within 50 ms of their recorded delay.
- Block scaling makes all 98,827 requests fit:
  - 32× fits a 32k context, with main-agent context p50 7.9k and p90 20.3k tokens;
  - 64× fits 16k.
- Clients do not run on green, whose p99 noise of 200–350 ms rules it out. Load runs on cyan, on pinned cores.
