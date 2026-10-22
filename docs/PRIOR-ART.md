# Prior art

*Checked against primary sources (issue and PR threads, release notes, docs, code) on 2026-10-22. Status is as of that date. Dates are UTC.*

This page is for engineers asking "how is this different from X?". For reference, Leaseholder is a control plane in front of vLLM, sized for one commodity GPU per node, with no Kubernetes. It does five things:
- keeps KV-cache leases for coding-agent and voice sessions, driven by lifecycle events: tool wait and return; speech onset, end of turn and barge-in;
- holds GPU headroom when a voice turn is imminent;
- prefetches from vLLM's native CPU offload tier;
- maps Anthropic `cache_control` TTLs to lease floors;
- attributes every recomputed prompt token to a cause.

The lease guarantees a host-RAM floor for its term. GPU residency is best-effort.

## At a glance

| Prior art | Status on 2026-10-22 | Relation to Leaseholder |
|---|---|---|
| llm-d-router #2273 (KV leases) | Umbrella open; lease part neither filed nor built | Closest lease design; different trigger and scope |
| llm-d v1.0 (turn priority, session affinity, flow control) | Released 2026-10-20; turn priority is Beta | Reproduced as baselines (E2 arm 7) |
| llm-d-router ThunderAgent plugin | Merged 2026-10-09 to 10-13 (Alpha) | Same family as E1 arm 8 (active-session admission) |
| vllm-project/router Progress-TTL (#285, #286) | PR merged 2026-09-24; RFC open | Closest prior art in the vLLM org; acts through admission, not engine actions |
| AIBrix #2814 | Open RFC, no code | Closest lifecycle-signal pinning proposal |
| Dynamo `nvext.cache_control` TTL pinning | Added 2026-02-27; removed 2026-04-02 (v1.1.0) | A cache contract that was tried and withdrawn |
| Dynamo agent hints | Shipping | Complementary input signals |
| SGLang sessions and retention | Shipping; HiCache TTL pin reverted 2026-04-01 | Same ideas on another engine |
| vLLM RFCs #37003, #45405, #57103 | Open; some plumbing merged | Engine mechanisms to adopt; venues for data |
| Mooncake lease and soft pin | Shipping | Same word, different meaning |
| Continuum, ThunderAgent | ICLR 2026, ICML 2026 | Each covers one Leaseholder mechanism |
| MORI, CacheWise, Ask the Tool, CacheScout, Tokencake | 2025–2026 papers | Overlap E1's parking and prefetch arms |
| Metronome, DuplexCadence, Endpoint Anticipation, HyGen | Papers | Adjacent to E2 |
| Anthropic and OpenAI cache TTLs | Live | The contracts Leaseholder honors (Anthropic in v1) |

## Routers and control planes

### llm-d-router #2273: "Cost-Aware P2P KV Pulls and KV Leases"
- **What.** An umbrella design (draft v5) by kaushikmitr for P2P KV pulls between pods ([#2273](https://github.com/llm-d/llm-d-router/issues/2273)). Change 4 is a lease, specified in §6 and Appendix B.4:
  - an advisory "evict last until time T" mark on a session's *offloaded* (CPU-tier) blocks;
  - renewed by every turn and released by silence, with lazy expiry at eviction;
  - an optional `x-llm-d-kv-lease-ttl` hint, a 300 s default and a 900 s cap;
  - at most half of the CPU tier leased, with the caps enforced in the engine;
  - shed and expiry metrics.

  The lease needs a vLLM `OffloadingConnector` change behind an engine RFC. The design is "reactive-first: … agents owe it nothing".
- **Status.** Open, assigned to kaushikmitr since 2026-08-04. Change 1 has shipped: [#2443](https://github.com/llm-d/llm-d-router/issues/2443) closed 2026-10-01, and the cost scorer [#3008](https://github.com/llm-d/llm-d-router/pull/3008) merged 2026-10-12. The lease child issue is still a placeholder ("#TBD-4"), sequenced last. A bot noted on 2026-10-13 that the umbrella has no open PR.
- **Relation.** This is the closest design to Leaseholder's agent-side lease: the same guarantee level (a CPU-tier preference, with GPU best-effort) and the same TTL-plus-cap shape. The differences:
  - the trigger: #2273 renews on turn arrival and has no tool-wait, speech or barge-in events;
  - prefetch: it has no prefetch before a request exists, and mentions a "deferred prefetch verb" for later;
  - it has no realtime class;
  - it targets multi-pod Kubernetes fleets.

  In E1's terms, it combines a fixed TTL (arm 4) with lease-aware CPU eviction (arm 7), without prefetch timed by lifecycle events.

### llm-d v1.0: session affinity, turn priority, flow control
- **What.**
  - [llm-d v1.0.0](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) (2026-10-20) ships flow control that is priority- and fairness-aware, with load shedding: a probabilistic admitter, program-aware fairness and turn-priority scoring. It also ships a session-state producer and session affinity.
  - The [turn-priority guide](https://github.com/llm-d/llm-d/blob/main/guides/turn-priority-fairness/README.md) (Beta) scores queued requests as `turnNumber + 0.05 × headWaitSeconds`. A session idle for more than 120 s scores as turn one, and under sustained contention new sessions are shed with a 429. At concurrency 128 it served 150% more requests than FCFS and cut p50 latency from 586.7 s to 37.8 s.
  - The [session-affinity filter](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity/README.md) pins a session to the pod that served its first request.
- **ThunderAgent plugin.** llm-d-router merged a [ThunderAgent plugin](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/thunderagent/README.md) (Alpha; [#2968](https://github.com/llm-d/llm-d-router/pull/2968), [#3052](https://github.com/llm-d/llm-d-router/pull/3052), [#3076](https://github.com/llm-d/llm-d-router/pull/3076); 2026-10-09 to 10-13). It is session-level admission over a per-pod KV ledger:
  - idle sessions keep their room for `idleLeaseSeconds` (default 30 s);
  - paused sessions wait for their own pod.

  Its README notes that during a tool call a session's blocks "sit on the free list and are reported as unused".
- **Relation.** Turn priority is reproduced as E2 arm 7, and session affinity as a baseline. The ThunderAgent gate is in the same family as E1 arm 8 (active-session admission). Leaseholder differs in deployment (no Kubernetes or EPP; sessions stay on the node that holds their KV) and in its realtime policy: it makes room and never sheds.

### vllm-project/router #285 and #286: Program-level Progress-TTL
- **What.** RFC [#285](https://github.com/vllm-project/router/issues/285) treats one `LLM → tool → LLM` chain as a Program. The implementation, [#286](https://github.com/vllm-project/router/pull/286) (+12,111 lines, opt-in), does three things:
  - it holds requests in a router RequestPool while a backend lacks logical KV capacity;
  - it protects tool gaps with a per-request "acting TTL", computed from observed continuity and a calibrated cold-prefill cost model. The TTL is capped at 10 s by default, and a `kv_retention_ttl_seconds` hint overrides it;
  - it pauses, resumes and migrates Programs, with starvation bounds ([docs](https://github.com/vllm-project/router/blob/main/docs/program_scheduling.md)).

  The docs state that it "protects KV indirectly by controlling request admission. Physical eviction, offload, and prefetch are not executed".
- **Status.** The PR merged 2026-09-24; the RFC is open, assigned to warriorsniu. On Qwen3-Coder-30B-A3B-FP8 (DP2/TP1, Claude Code replay), the PR reports against request-level cache-aware routing: prompt hit 74.72% against 63.74%, wall time −24.9%, P99 latency −40.4%. Follow-ups are open: [#383](https://github.com/vllm-project/router/pull/383), [#388](https://github.com/vllm-project/router/pull/388), [#393](https://github.com/vllm-project/router/issues/393).
- **Relation.** This is the closest prior art inside the vLLM org, in the same family as E1 arm 8. Leaseholder adds three things it lacks:
  - engine-side actions: a CPU-tier lease floor and prefetch;
  - a realtime class;
  - a per-cause recompute ledger, where Progress-TTL logs `unexpected_cache_discontinuity`.

### AIBrix #2814: agent-aware KV affinity and lifecycle management
- **What.** RFC [#2814](https://github.com/vllm-project/aibrix/issues/2814) proposes:
  - `agent_hints` (`program_id`, `priority`, `context_type`, `speculative_prefill`, `session_ttl_hint`);
  - weighted eviction;
  - speculative prefill under P/D;
  - session pinning: `open`/`heartbeat`/`close` signals pin a session's L1 blocks with a TTL, and heartbeats extend it during tool calls.
- **Status.** Open since 2026-09-25 and unassigned. It has two comments, one with scoping questions (2026-09-30), and no implementing PR. [AIBrix v0.8.0](https://github.com/vllm-project/aibrix/releases/tag/v0.8.0) (2026-10-13) shipped Redis-backed session-key pinning ([#2742](https://github.com/vllm-project/aibrix/pull/2742)), which pins routing, not KV.
- **Relation.** This is the proposal closest to a lease driven by explicit lifecycle signals. Leaseholder holds a host-RAM floor rather than pinning L1, adds voice events, and measures the result.

### NVIDIA Dynamo: `nvext.cache_control` TTL pinning (removed) and agent hints
- **What it was.** [#6213](https://github.com/ai-dynamo/dynamo/pull/6213) (merged 2026-02-27) read `nvext.cache_control: {"type": "ephemeral", "ttl": "5m"}` and, after each stream, called `pin_prefix(token_ids, ttl)` on the worker. The pin was backed by SGLang's TTL pinning in HiCache ([sglang#18941](https://github.com/sgl-project/sglang/pull/18941), merged 2026-03-02). The PR's demo showed 1,606 ms falling to 315 ms TTFT after a flood.
- **Removal and stated reason.**
  - SGLang reverted the hard pin on 2026-04-01 ([sglang#21884](https://github.com/sgl-project/sglang/pull/21884)), and Dynamo unwound `nvext.cache_control` the next day ([#7790](https://github.com/ai-dynamo/dynamo/pull/7790)).
  - The [v1.1.0 release note](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0) (2026-05-04) says: "SGLang upstream chose a different direction".
  - The superseded [sglang#21045](https://github.com/sgl-project/sglang/pull/21045) (closed unmerged) gives the rationale: "Hard pinning couples routing to a specific worker and wastes host memory when the router can't honor sticky affinity at scale."
  - The follow-up, sticky sessions over SGLang streaming sessions ([#7665](https://github.com/ai-dynamo/dynamo/pull/7665)), was itself removed in [v1.3.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.3.0) ([#10214](https://github.com/ai-dynamo/dynamo/pull/10214)) in favour of trajectory tags on SGLang's session radix cache.
- **Agent hints today.**
  - [`nvext.agent_hints`](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/agent-hints.md) are `priority`, `strict_priority`, `osl` and `speculative_prefill`. Priority-based eviction is marked "Yes" for SGLang and "Planned" for vLLM and TensorRT-LLM.
  - [v1.5.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0) classifies requests by `input_trigger` (`user_message`, `tool_result`).
  - Dynamo's [agentic blog](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/blog/2026/agentic-inference-optimizations.mdx) (updated 2026-10-02) says Dynamo does not treat `cache_control` as TTL pinning and calls TTL directives "future API work". It also says prefetch hooks for tool-call returns are being built.
- **Relation.** Leaseholder honors `cache_control` TTLs, the contract Dynamo tried and withdrew. It does so with a weaker guarantee (host RAM, with GPU best-effort) and a violation SLI. Sessions stay on the node that holds their KV ([ARCHITECTURE.md](ARCHITECTURE.md), tenet 4), which addresses the routing-affinity concern SGLang cited. Agent hints are complementary inputs. `speculative_prefill` and the planned hooks overlap tool-return prefetch.

## Engines

### SGLang: streaming sessions and session-aware retention
- **Streaming sessions.** With `--enable-streaming-session`, a session's KV is held outside radix eviction and freed on close or idle timeout ([Dynamo's SGLang page](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/agents-on-sglang.md)). Sessions are append-only ([`streaming_session.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/session/streaming_session.py)).
  - The 300 s default is Dynamo's `session_control.timeout`. SGLang's own `OpenSessionReqInput.timeout` defaults to no timeout ([io_struct.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py)).
  - [v0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22) (2026-10-19) restricts streaming sessions to `UnifiedRadixCache`.
- **Session radix cache.** `--enable-session-radix-cache` ([#29173](https://github.com/sgl-project/sglang/pull/29173), merged 2026-08-02) evicts unreferenced KV before session-referenced KV, on device and host. It is "soft protection, not memory pins" ([docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/session_radix_cache.mdx)).
- **Priority eviction.** With HiCache, `--radix-eviction-policy priority` orders both GPU demotion and host deletion ([Dynamo's SGLang page](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/agents-on-sglang.md)). A soft TTL decay (`retention_seconds`) is proposed in [#27970](https://github.com/sgl-project/sglang/pull/27970), open since 2026-06-11.
- **Relation.** These are the same ideas on another engine. Leaseholder targets vLLM and does not use them. The session radix cache is the closest engine-side analogue of a lease floor.

### vLLM: retention RFCs and the merged plumbing
- **[#37003](https://github.com/vllm-project/vllm/issues/37003), Context-Aware KV-Cache Retention API.** Opened by vMaroon on 2026-03-13 and still open; 11 comments, the last on 2026-10-09. It proposes token-range directives with priority 0–100 and an optional duration. Implementation [#38514](https://github.com/vllm-project/vllm/pull/38514) (`kv.retain` hints) is open, took community bug fixes on 2026-10-19 and 10-20, and has no maintainer approval. Field data in the thread:
  - [rich7420](https://github.com/vllm-project/vllm/issues/37003#issuecomment-5078100306): stock LRU already protects returning sessions at short gaps and modest contexts. Per-request `cached_tokens` reconciles exactly, while global counters over-count by 1.3–2.0× under queue pressure;
  - [k-cross](https://github.com/vllm-project/vllm/issues/37003#issuecomment-5087831234): in llm-d's simulator, directives beat LRU once the re-arrival gap reaches cache residency, and not before; "it must be a hold, not a hint";
  - [mgabor3141](https://github.com/vllm-project/vllm/issues/37003#issuecomment-6090931030): a single-box proxy (4× RTX 3090, 10–20 coding agents) queues outside the engine and charges each finished turn's prefix as a "hold" sized by its re-prefill cost. It served 91% of prompt tokens from cache. Turns that queued over a minute almost all missed.
- **[#45405](https://github.com/vllm-project/vllm/issues/45405), ReqContext for the CPU-offload `CachePolicy`.** Opened 2026-06-12; still open, marked stale on 2026-10-06. The minimal plumbing merged as [#47744](https://github.com/vllm-project/vllm/pull/47744) on 2026-07-08 and passes request context to `touch`. Adapting the policies is not done.
- **[#57103](https://github.com/vllm-project/vllm/issues/57103), Programmable KV Cache.** Opened 2026-09-16 and still open, with 9 comments. Routers supply lifecycle intent, and the default engine executes explicit directives. On 2026-09-24 the author dropped Pin in favour of retention priority with an optional TTL. Draft [#62382](https://github.com/vllm-project/vllm/pull/62382) needs a rebase and reports conditional results.
  - yinli-systems [measured](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478) request-free prefetch from native CPU offload on one L20: prefetching at pause kept 95–100% of the oracle's gain, and an unfinished prefetch was worse than none.
  - The primitive is draft [#58198](https://github.com/vllm-project/vllm/pull/58198).
- **Merged building blocks.**
  - `session_id` ([#48048](https://github.com/vllm-project/vllm/pull/48048), 2026-08-03);
  - `session_id` on GPU `BlockStored` events ([#51381](https://github.com/vllm-project/vllm/pull/51381), 2026-09-04);
  - `KvHintsEnvelope` ([#53423](https://github.com/vllm-project/vllm/pull/53423), 2026-09-21);
  - NIXL lease renewal for P/D ([#41383](https://github.com/vllm-project/vllm/pull/41383), 2026-05-11; [design](https://docs.vllm.ai/en/latest/design/nixl_kv_cache_lease/)). Its `decoder_kv_blocks_ttl` (480 s) already expires between-turn blocks on decode instances.

  Open issue [#61478](https://github.com/vllm-project/vllm/issues/61478) (2026-10-13) asks for TTL-based eviction.
- **Relation.** These are mechanisms Leaseholder would adopt rather than duplicate ([ARCHITECTURE.md](ARCHITECTURE.md), tenet 1). Until then it uses:
  - stock knobs;
  - per-request `kv_transfer_params`;
  - an optional `CachePolicy` plugin;
  - prefetch by a `max_tokens=1` warm-up request, which #58198 would replace.

  Gate G-sim answers #37003's hold-or-hint question with a bound.

### Mooncake: lease TTL and soft pin
- **What.** Mooncake Store grants a per-object lease on `ExistKey` or `GetReplicaList`, which protects the object from removal and eviction while it is read. The default is 10 s ([design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md), [`types.h`](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/types.h)). A separate soft pin (30 min by default, a fixed deadline that reads do not extend) makes an object evict last.
- **Relation.** Same word, different meaning: a lease for transfer safety, plus a fixed-deadline priority. Leaseholder does not use Mooncake, because KV never leaves the node.

## Research

- **Continuum** ([arXiv 2511.02230](https://arxiv.org/abs/2511.02230) v7, 2026-09-08; [ICLR 2026](https://iclr.cc/virtual/2026/10012473)).
  - After a tool-calling turn, it pins KV in GPU memory with a TTL set from reload cost and queueing delay, plus program-level FCFS. It reports over 8× lower average job-completion time.
  - The code is a preview fork ([vllm-continuum](https://github.com/Hanchenli/vllm-continuum)); no upstream PR was found.
  - Relation: E1's fixed-TTL hold (arm 4) is in this family. Leaseholder holds KV in host RAM and prefetches it back instead of pinning GPU memory.
- **ThunderAgent** ([arXiv 2602.13692](https://arxiv.org/abs/2602.13692) v3, 2026-06-30; [ICML 2026](https://icml.cc/virtual/2026/poster/62040)).
  - It models agent workflows as LLM Programs and pauses programs under memory pressure, resuming them where capacity exists. It reports 1.5–3.6× serving throughput.
  - It is open source ([MIT](https://github.com/ThunderAgent-org/ThunderAgent)). Dynamo ships it as an experimental router (v1.3.0), and llm-d-router as the plugin above.
  - Relation: E1 arm 8 (active-session admission) is this mechanism, measured on one GPU against stock limits.
- **MORI** ([arXiv 2606.00866](https://arxiv.org/abs/2606.00866), 2026-05-30).
  - It ranks programs by idleness, keeps the busiest in HBM and the most idle in DRAM, shifts the boundary to match capacity, and admits per tier. It reports 20–71% higher throughput on Claude Code traces.
  - Relation: this is the research work closest to E1's parking. Leaseholder times moves from lifecycle events rather than from an idleness ranking.
- **Prediction and progress signals.**
  - CacheWise ([2606.16824](https://arxiv.org/abs/2606.16824)) does reuse-aware eviction from tool-call metadata in vLLM.
  - Ask the Tool ([2609.18849](https://arxiv.org/abs/2609.18849)) has tools report progress and cuts p90 post-tool TTFT by 20.7% against LRU.
  - CacheScout ([2608.14624](https://arxiv.org/abs/2608.14624)) and Tokencake ([2510.18586](https://arxiv.org/abs/2510.18586), EuroSys '27) predict offload and upload around function calls.
  - Relation: these bracket E1 arms 5–6, which time prefetch from history or from an oracle.
- **A counter-result.** A simulator study ([2609.16215](https://arxiv.org/abs/2609.16215)) found that tiering gains came from capacity rather than placement policy, and that prefetch never beat no-prefetch on migration traffic. Gate G-sim exists to test this on Leaseholder's traces.
- **Voice and co-location.**
  - Metronome ([2607.02640](https://arxiv.org/abs/2607.02640)): on one GPU, full-duplex models collapse when session KV fills the pool. A bounded KV window plus online admission fixes it.
  - DuplexCadence ([2609.35115](https://arxiv.org/abs/2609.35115)) speeds up the runtime of full-duplex models.
  - Endpoint Anticipation ([2606.13450](https://arxiv.org/abs/2606.13450)) forecasts the end of a turn up to 2.56 s ahead.
  - HyGen ([2501.14808](https://arxiv.org/abs/2501.14808)) co-locates online and offline LLM requests.
  - Relation: this search found no system that schedules a cascaded voice turn beside tool-using agents inside one vLLM engine while managing KV residency for both. Endpoint forecasts could serve as an earlier trigger for turn-imminent headroom.

## Commercial cache contracts
- **Anthropic.** `cache_control` caches for 5 minutes by default, and a 1-hour TTL is available. Cache writes cost 1.25× the base input price for the 5-minute TTL and 2× for the 1-hour TTL ([docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
- **OpenAI.** On GPT-5.6 and later, `prompt_cache_options.ttl` is `"30m"` (the default) or `"12h"`, a minimum lifetime ([docs](https://developers.openai.com/api/docs/guides/prompt-caching)).
- **Relation.** Leaseholder maps Anthropic TTLs to lease floors ([ARCHITECTURE.md](ARCHITECTURE.md) §4.5). The OpenAI contract is deferred.

## What is distinctive

Each mechanism above has prior art:
- leases and TTL holds: #2273, AIBrix #2814, Continuum, SGLang, Mooncake;
- active-session admission: ThunderAgent, Progress-TTL, the llm-d gate;
- CPU-tier prefetch: Tokencake, MORI, vLLM #58198, Dynamo's planned hooks;
- turn priority and session affinity: llm-d.

What Leaseholder offers is a combination on one commodity GPU:
1. **One lifecycle-event schema for agents and voice.** A tool return and a speech onset drive the same lease, prefetch and headroom path. Each system cited here takes its signals from one workload: tool calls, turn arrivals or speech frames.
2. **A realtime class that is never shed.** Turn-imminent GPU headroom is measured against tuned stock vLLM and an emulation of llm-d turn priority (E2 arms 2–7). By contrast, llm-d sheds with a 429, and Progress-TTL holds requests back.
3. **A cache contract honored with a stated guarantee.** Anthropic `cache_control` gets a host-RAM floor (GPU best-effort) and a violation SLI, in the space Dynamo withdrew from.
4. **A recompute ledger validated by injected causes**, with at least 95% correct attribution required. The stacks cited here expose per-request cached tokens or a discontinuity log.
5. **Simulator bounds against Belady-OPT** on block-scaled traces, so a negative result can be reported as a measured regime rather than an opinion.

This is a positioning statement, not a priority claim. Each comparison rests on the sources cited above.
