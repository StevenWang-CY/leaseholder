# Prior art

*Checked against primary sources (papers, issue and PR threads, release notes, documentation, code) on 2026-10-22. Status is as of that date. Dates are UTC.*

This page is for engineers asking "how is this different from X?". Leaseholder serves a realtime tier that is never shed (voice turns) beside a best-effort tier that is harvested (background agents), on one vLLM replica. It times the protection from lifecycle events: speech onset, end of turn and barge-in for voice, and tool waits for agents.

## Positioning

> Leaseholder applies known co-serving and KV-tiering mechanisms to a workload pair published systems have not measured together, a cascaded voice pipeline beside long-context agents on one GPU, and measures when lifecycle timing beats static reservation and engine priority, and when it does not.

**Claimed.** Three parts, argued in [What is distinctive](#what-is-distinctive):
1. **Lifecycle-timed headroom, freed by parking.** GPU room for the next realtime turn is held before the request exists, from speech onset or end of turn. It is counted in KV tokens and made by parking background sessions in host RAM rather than killing their requests.
2. **The engine "make room" fix.** A waiting high-priority request may preempt running work when the KV cache is full, keeping or offloading the victim's KV. Neither vLLM nor SGLang does this today.
3. **The validated recompute ledger.** Every recomputed prompt token is attributed to a cause, and the attribution is checked by injecting causes.

Everything else is a baseline or a reference: [At a glance](#at-a-glance) lists each mechanism and where it exists, and [What we do not claim](#what-we-do-not-claim) gathers them.

In the tables, **Relation** says how an item bears on Leaseholder:
- **baseline**: E1 or E2 must be compared against it;
- **pre-empts**: it ships or publishes a mechanism, so Leaseholder does not claim that mechanism;
- **complement**: an input, a venue for upstream work, or evidence for the problem.

## At a glance

| Mechanism | Where it exists on 2026-10-22 | Leaseholder |
|---|---|---|
| A latency class co-served with best-effort work | ConServe, HyGen, BROS, Niyama; Valve in production on 8,054 GPUs | Not claimed |
| Priority classes, budgets, aging, queue TTLs, in-flight eviction | llm-d v1.0 flow control, SMG, Dynamo, AIBrix | Not claimed; llm-d flow control (emulated) and SMG are E2 baselines |
| Headroom held back for priority work | Static: SMG reserved slots, llm-d holdback ceilings, vLLM `watermark` (class-blind). Tied to a running request: Llumnix. Sized from load or the agent graph: Valve, Tokencake | An equal-size static reservation is a baseline; timing it from voice events is claimed |
| Engine priority preemption | vLLM among running requests; SGLang at the request cap | Baseline; the KV-full case is the claimed fix |
| Parking KV during a tool call, prefetching before the return | Tokencake, KVFlow, CacheScout, MORI; proposed for vLLM (#57103, #58198) | Not claimed; E1 measures what lifecycle information adds |
| TTL holds and leases | Continuum, TensorRT-LLM retention, vLLM #37003, llm-d-router #2273, AIBrix #2814 | Not claimed; Continuum's TTL hold is an E1 baseline |
| Program- or session-level admission | Autellix, Parrot, ThunderAgent, vllm-project/router Progress-TTL | Not claimed |
| Starting the LLM before the turn ends | LiveKit preemptive generation (on by default), Endpoint Anticipation, PredGen | Triggers to compare, not claimed |
| Gap between LRU and Belady on prefix caches | "When Fancy Eviction Fails", LPC | Not claimed; E1 reports the gap closed per level of information |
| Recomputed tokens attributed to causes | Not found | Claimed |
| Cascaded voice beside long-context agents on one GPU, measured | Not found; NVIDIA's single-GPU voice-agent profile documents the regression | The measured workload |

## Co-serving latency-critical and best-effort work

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [ConServe](https://arxiv.org/abs/2410.01228) | arXiv 2024-10-02 (rev. 2025-09-03) | Co-serves online requests with offline tasks: token-level scheduling sizes offline batches to the online latency target, layer-wise preemption yields to online spikes, and incremental KV management preempts and resumes offline requests at near-zero cost. 2.2× throughput and 2.9× lower online tail latency on average | Pre-empts harvest-and-protect, including throttling offline work. E2's abort-and-resubmit baseline approximates its preempt-and-resume from outside the engine |
| [HyGen](https://arxiv.org/abs/2501.14808) | NeurIPS 2025 | A latency predictor and SLO-aware offline scheduling co-locate online and offline requests; up to 3.9–5.8× throughput while meeting latency SLOs | Pre-empts the co-location goal |
| [BROS](https://arxiv.org/abs/2504.09590) | arXiv 2025-04-13 | Real-time and best-effort requests in one engine, with dynamic priorities and bidirectional KV sharing between the classes; up to 74.2% lower real-time latency than vLLM and TGI | Pre-empts two classes in one engine sharing KV |
| [Niyama](https://arxiv.org/abs/2503.22562) | arXiv 2025-03-28 | Fine-grained QoS classes on shared infrastructure: dynamic chunking, hybrid prioritization, selective relegation under overload; 32% more capacity than siloed deployments | Pre-empts class-aware admission with graceful degradation |
| [Valve](https://arxiv.org/abs/2604.07874) | arXiv 2026-04-09 | Production online–offline colocation on 8,054 GPUs: sub-millisecond compute preemption at most once per online request, rate-limited memory reclamation, dynamic memory reservation. +34.6% cluster utilization, under 5% TTFT increase | Pre-empts co-location at fleet scale; the closest production analogue of E2 |
| [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) | OSDI 2024 | Live migration of requests between instances. A high-priority request adds a memory headroom to its "virtual usage", and normal requests migrate away when it runs short | Pre-empts class KV headroom. Its headroom belongs to a running request; Leaseholder's is held before the request arrives |
| [SLOs-Serve](https://arxiv.org/abs/2504.08784) | arXiv 2025-04-05 | Allocates tokens per SLO by dynamic programming over chunked prefill and speculative decoding | Pre-empts per-class token budgets |
| [QLM](https://arxiv.org/abs/2407.00047) | SoCC 2024 | Holds batch and interactive requests in one queue, ordered by estimated waiting time, with eviction and model swapping | Pre-empts SLO-ordered queueing |
| [Aladdin](https://arxiv.org/abs/2405.06856) | arXiv 2024-05-11 | SLO-aware query placement and worker scaling | Complement: the cluster-scale question |
| [Andes](https://arxiv.org/abs/2404.16283) | arXiv 2024-04-25 | Preemptive token-level scheduling for quality of experience in text streaming | Complement: the text analogue of pacing decode to audio playback |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | OSDI 2024 | Chunked prefill and stall-free batching | Baseline: vLLM's chunked prefill, swept in the tuned-engine arms |

## Routers and gateways

These ship classes, admission and eviction today.

| Stack (release) | Classes and admission | Making room | Capacity held for a class | Relation |
|---|---|---|---|---|
| [llm-d](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) v1.0.0 (2026-10-20); [llm-d-router](https://github.com/llm-d/llm-d-router/releases/tag/v1.0.0) v1.0.0 (2026-10-16) | [Flow control](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/flowcontrol/README.md), off by default behind a feature gate: priority bands, fairness policies, queue TTLs, [SLO-deadline ordering](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/flowcontrol/ordering/slodeadline/README.md) from an `x-llm-d-slo-ttft-ms` header, and a [latency-SLO admitter](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/requestcontrol/admitter/latencyslo/README.md) that rejects sheddable requests no endpoint can serve in time | [In-flight eviction](https://github.com/llm-d/llm-d-router/pull/2093) (merged 2026-08-11; `flowControl.enableEviction`, off by default): when higher-priority requests are blocked by saturation, sheddable (priority < 0) streams end with HTTP 429. Partial eviction, which would offload the KV and resume the request on retry ([#1119](https://github.com/llm-d/llm-d-router/issues/1119)), is deferred | [`priority-holdback-policy`](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/flowcontrol/usagelimits/priorityholdback/README.md) (beta in v1.0) lowers the admission ceiling of low priorities as saturation rises | Baseline: flow control with holdback and eviction, emulated in `lh-core` and labeled as such |
| [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v1.6.0) v1.6.0 (2026-08-17) | Moved its Endpoint Picker and the `InferenceObjective` priority API to llm-d, where `InferenceObjective` is now `llm-d.ai/v1`; GAIE keeps the specification, CRDs and conformance | — | — | Pre-empts the priority API; its policy lives in llm-d |
| [SMG](https://github.com/smg-project/smg) v1.13.0 (2026-10-15), Rust | Four classes from an `x-smg-priority` header, capped per tenant; per-class queues (429 when full, 408 on timeout); strict priority with starvation promotion ([docs](https://github.com/smg-project/smg/pull/1583)) | Preempts a lower-class request only before its first response byte; the victim gets a 503 | Reserved slots per class, as floors, counted in concurrency rather than tokens | Baseline: run as shipped (`smg launch --worker-urls`) in front of vLLM |
| [Dynamo](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1) v1.5.1 (2026-10-07) | Router queue keyed by `(strict_priority, due_at, policy)`; a soft `priority` is forwarded to vLLM and SGLang ([docs](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/agents/priority-scheduling.md)) | Strict priority "does not preempt admitted work" | Priority "does not reserve capacity for high-priority requests" | Pre-empts priority queueing at the router |
| [AIBrix](https://github.com/vllm-project/aibrix/releases/tag/v0.8.0) v0.8.0 (2026-10-13) | Queues by SLO tier at the gateway; [#2787](https://github.com/vllm-project/aibrix/pull/2787) (merged 2026-09-24) passes the tier to vLLM as the request priority | Through engine priority only | None found | Pre-empts SLO tiers mapped onto engine priority |
| [production-stack](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/routers/routing_logic.py) | `PriorityRouter` sends high-priority requests to the least-loaded engine, so "no engine is ever reserved and left idle" | No | No | Pre-empts priority-aware routing |
| [vllm-project/router](https://github.com/vllm-project/router) v0.1.16 (2026-10-09), Rust | [Program scheduling with Progress-TTL](https://github.com/vllm-project/router/pull/286) (merged 2026-09-24) admits agent Programs against logical KV capacity and covers tool gaps with an acting TTL (capped at 10 s by default). Per its [docs](https://github.com/vllm-project/router/blob/main/docs/program_scheduling.md), it "protects KV indirectly by controlling request admission. Physical eviction, offload, and prefetch are not executed" | No | Program capacity | Pre-empts active-session admission; complement as the home for Leaseholder's agent-side policy |

- **llm-d turn priority is not a voice baseline.** Its [turn-priority guide](https://github.com/llm-d/llm-d/blob/main/guides/turn-priority-fairness/README.md) (Beta) scores queued requests as `turnNumber + 0.05 × headWaitSeconds` and sheds new sessions with a 429 under sustained contention. It ranks agent turns for fairness; the voice-relevant llm-d policy is flow control.
- **vllm-project/router RFCs, all opened 2026-10-22.**
  - [#427](https://github.com/vllm-project/router/issues/427), engine-confirmed KV retention: "A Router-side TTL or reservation is only a logical hint". It asks to keep logical session TTL, requested retention, acknowledged retention and observed residency apart.
  - [#428](https://github.com/vllm-project/router/issues/428): workflow output-length hints that produce a bounded prefetch or offload hint before a tool returns, acknowledged by the backend and evaluated in shadow mode first.
  - [#435](https://github.com/vllm-project/router/issues/435): worker overload admission, bounded backpressure and load shedding.

  Relation: complement. These threads are where Leaseholder's agent-side lease and prefetch policy is to be proposed; `LeaseConnector`'s per-request hit receipts are one form of the acknowledgement #427 asks for.

## Engines

### vLLM 0.32.0 (2026-10-18)

Checked in the v0.32.0 source:
- **Priority preempts among running requests, not for waiting ones.** Under `--scheduling-policy priority`, a running request that cannot grow preempts the lowest-priority running request. The waiting loop stops when `allocate_slots` fails or the running set is full, so a waiting request never preempts running work ([`scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py)).
- **The open fixes cover only the request cap.**
  - [#40004](https://github.com/vllm-project/vllm/issues/40004) asks for waiting-request preemption; it has been open since 2026-04-16, with 22 comments.
  - [#54530](https://github.com/vllm-project/vllm/pull/54530) (opened 2026-08-31, needs a rebase), and before it [#45558](https://github.com/vllm-project/vllm/pull/45558), [#45561](https://github.com/vllm-project/vllm/pull/45561) and [#40087](https://github.com/vllm-project/vllm/pull/40087), add it at `max_num_seqs`. #54530 states that it leaves the KV-pressure path out.
  - [#61976](https://github.com/vllm-project/vllm/pull/61976) (opened 2026-10-16) fixes a priority self-preemption bug.
- **Scheduling policies under review do not preempt.** The SLO-aware policy [#53571](https://github.com/vllm-project/vllm/pull/53571) "only reorders waiting requests, does not preempt running requests". [#58420](https://github.com/vllm-project/vllm/pull/58420) adds shortest-remaining-prefill-first with aging.
- **Class-blind knobs** ([`config/scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/scheduler.py)):
  - `watermark`: a fraction of KV blocks kept free when admitting waiting or preempted requests (default 0, off);
  - `max_num_queued_tokens`: a "TTFT QoS mechanism" that rejects new requests with HTTP 503 once the prefill backlog reaches the limit;
  - `long_prefill_token_threshold_adaptive`: floors the long-prefill threshold at a fair share of the token budget;
  - `prefill_schedule_interval`: admits new prefills only every N engine steps. In v0.32.0 only the data-parallel MoE engine applies it; [#54627](https://github.com/vllm-project/vllm/pull/54627) (merged 2026-10-15, after the v0.32.0 tag) applies it to a single engine.
- **Priority does not reach retention.** [#47802](https://github.com/vllm-project/vllm/issues/47802) (open) reports that priority is not reflected in KV retention under cache pressure.

**Relation.** Baselines: priority scheduling with the step cap, `watermark`, `max_num_queued_tokens`, `prefill_schedule_interval` on a build that includes #54627, and vLLM with #54530 applied. The KV-full case is the claimed fix, planned as a vLLM PR under #40004 that keeps or offloads the victim's KV.

### SGLang 0.5.22 (2026-10-19)

- **Priority preemption ships.** It came in [#8746](https://github.com/sgl-project/sglang/pull/8746) (merged 2025-09-17) and [#17026](https://github.com/sgl-project/sglang/pull/17026) (merged 2026-03-04) ([flags](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/fields/schedule.py)):
  - with `--enable-priority-scheduling`, an incoming request may preempt running ones whose priority is lower by at least `--priority-scheduling-preemption-threshold` (default 10);
  - `--disable-priority-preemption` turns this off;
  - `--retraction-policy priority` retracts lower-priority decodes first when the KV cache is full.
- **But not when KV memory runs out.** Draft [#45686](https://github.com/sgl-project/sglang/pull/45686) (opened 2026-10-21) states that the admission loop tries preemption only when the running batch reaches its request cap; when KV memory runs out, "the loop stops". It adds preemption on KV exhaustion, and `--preempt-keep-kv`, which inserts the victim's KV, decoded tokens included, into the radix tree as a best-effort keep.
- **Relation.** Baseline: SGLang priority preemption, on the same model. #45686 is the SGLang counterpart of Leaseholder's vLLM fix.

### TensorRT-LLM

- [`KvCacheRetentionConfig`](https://nvidia.github.io/TensorRT-LLM/1.2.1/features/kvcache.html) gives token ranges a priority from 0 to 100 (default 35) with an optional duration, after which the priority reverts. Lower priorities are evicted first, and blocks below a threshold are not offloaded to host memory.
- Through Dynamo, per-request engine priority for TensorRT-LLM is "not currently exposed".
- **Relation.** Pre-empts retention priority with expiry inside an engine. Not a baseline, since Leaseholder targets vLLM.

## KV across tool calls and turns

### Research

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [InferCept](https://proceedings.mlr.press/v235/abhyankar24a.html) | ICML 2024 | At each tool interception, preserves, discards or swaps the KV to minimize wasted GPU resources. Recompute had been 37–40% of model forward time; 1.6–2× throughput | Pre-empts per-pause retention choices; its min-waste policy is an E1 baseline |
| [CachedAttention](https://arxiv.org/abs/2403.19708) | USENIX ATC 2024 | Multi-turn KV in host memory and disk, layer-wise preloading, and fetching and eviction guided by hints from the job scheduler; TTFT up to 87% lower | Pre-empts prefetch from a known future |
| [Pensieve](https://arxiv.org/abs/2312.05516) | EuroSys 2025 | Conversation state cached across GPU and CPU memory; 1.14–3.0× the throughput of vLLM and TensorRT-LLM | Pre-empts multi-turn parking |
| [HCache](https://arxiv.org/abs/2410.05004) | EuroSys 2025 | Restores state from cached intermediate activations instead of KV or tokens; up to 1.93× lower TTFT than KV offload | Complement: a third restore path |
| [KVFlow](https://arxiv.org/abs/2507.07400) | NeurIPS 2025 | Steps-to-execution eviction and overlapped CPU→GPU prefetch for the agents scheduled next; up to 1.83× (single workflows) and 2.19× (many concurrent workflows) over SGLang's hierarchical radix cache | Pre-empts prefetch timed by predicted activation |
| [Tokencake](https://arxiv.org/abs/2510.18586) | EuroSys '27 (arXiv v5, 2026-09-28) | Offloads idle KV when a function call starts, uploads it predictively before the call is expected to end, and partitions GPU memory to keep room for critical-path agents; up to 47% lower end-to-end latency than vLLM | Pre-empts "park in host RAM, return before the tool does"; E1 baseline. Its first author, Zhuohang Bian, also wrote vLLM RFC #57103 |
| [Continuum](https://arxiv.org/abs/2511.02230) | ICLR 2026 workshop poster ([Lifelong Agents](https://iclr.cc/virtual/2026/10012473)); arXiv v7, 2026-09-08 | Pins KV in GPU memory after a tool-calling turn, with a TTL set from reload cost and queueing delay, plus program-level FCFS; over 8× lower average job completion time | Pre-empts TTL holds; E1 baseline |
| [Autellix](https://arxiv.org/abs/2502.13965) | arXiv 2025-02-19 | Program-level preemptive scheduling of LLM calls; 4–15× program throughput at equal latency | Pre-empts program-level scheduling |
| [Parrot](https://arxiv.org/abs/2405.19888) | OSDI 2024 | Semantic Variables expose application structure to the serving system | Pre-empts application hints to the server |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | ICML 2026 ([poster](https://icml.cc/virtual/2026/poster/62040)) | Pauses LLM Programs under memory pressure and resumes them where capacity exists; 1.5–3.6× serving throughput. Ships as an llm-d-router plugin (v1.0) and an experimental Dynamo router (v1.3.0) | Pre-empts active-session admission |
| [MORI](https://arxiv.org/abs/2606.00866) | arXiv 2026-05-30 | Ranks programs by idleness, keeps the busiest in HBM and the most idle in DRAM, and moves the boundary to match capacity; 20–71% more throughput and 18–43% lower TTFT than the best offloading baseline, on Claude Code workloads | Pre-empts idleness-ranked parking; E1 baseline and the closest work to E1 |
| [CacheWise](https://arxiv.org/abs/2606.16824) | arXiv 2026-06-15 | Reuse-aware eviction from tool-call metadata, in vLLM; up to 2–2.6× fewer evictions | Pre-empts eviction from history |
| [CacheScout](https://arxiv.org/abs/2608.14624) | arXiv 2026-07-16 | Learns agent execution transitions online to drive eviction and prefetch, in vLLM; 18–45% lower mean TTFT | Pre-empts prefetch from history |
| [Ask the Tool](https://arxiv.org/abs/2609.18849) | arXiv 2026-09-16 | Tools report their progress while running; p90 TTFT after a tool call 20.7% (HBM) and 20.8% (HBM + DRAM) lower than LRU, "close to an oracle" | Complement: defines E1's oracle level, and the number E1 is compared with |
| [Adaptive KV retention](https://arxiv.org/abs/2608.30830) | arXiv 2026-08-31 | Prices retention in GPU opportunity cost for waits of minutes to hours (human approval); 23–51% more goodput than vLLM baselines, ahead of MORI and Continuum | Pre-empts retention for the think-time regime |
| [Leyline](https://arxiv.org/abs/2606.01065) | arXiv 2026-05-31 | KV directives for agent context edits (retries, dropped tool output) without re-prefilling the rest; +11.2 pp replay cache hits | Complement: context edits are a prefix-break cause for the ledger |
| [Strata](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang) | OSDI 2026 | Hierarchical context caching across HBM, host memory and SSD, with GPU-assisted I/O and a cache-aware scheduler, in SGLang | Complement: restore engineering |

**Verdict.** Parking KV during a tool call and prefetching it before the return is published. E1 does not claim the mechanism. It measures how much of the gap to a clairvoyant reference each level of lifecycle information closes (none, session history, a progress oracle), per regime.

### Engine and router threads

| Thread | Status on 2026-10-22 | Relation |
|---|---|---|
| vLLM [#57103](https://github.com/vllm-project/vllm/issues/57103), Programmable KV Cache RFC, and draft [#62382](https://github.com/vllm-project/vllm/pull/62382) | RFC open since 2026-09-16; the draft (2026-10-19) needs a rebase. Composable Retain, Pin, Prefetch and Demote directives, under which "TokenCake-like algorithms can integrate" | Complement: the upstream path of Tokencake's mechanism, and where E1 data goes |
| vLLM [#58198](https://github.com/vllm-project/vllm/pull/58198), request-free CPU→GPU prefetch | Draft since 2026-09-22, no reviews. Its author's [measurement](https://github.com/vllm-project/vllm/issues/57103#issuecomment-5733975478) on one L20 with Qwen3-4B: prefetch during the tool wait recovered 52–57% of resume TTFT, prefetching at pause kept 95–100% of the oracle's gain, and an unfinished prefetch was worse than none | Complement: the primitive that would replace Leaseholder's warm-up request |
| vLLM [#37003](https://github.com/vllm-project/vllm/issues/37003), retention API RFC, and [#38514](https://github.com/vllm-project/vllm/pull/38514) | RFC open since 2026-03-13; the implementation (token-range priority with a duration) open since 2026-03-30, with no approving review | Complement: engine retention that Leaseholder would adopt if it lands |
| vLLM merged plumbing | `session_id` ([#48048](https://github.com/vllm-project/vllm/pull/48048), 2026-08-03); session id on KV events ([#51381](https://github.com/vllm-project/vllm/pull/51381), 2026-09-04); `KvHintsEnvelope` ([#53423](https://github.com/vllm-project/vllm/pull/53423), 2026-09-21); request context for the offload `CachePolicy` ([#47744](https://github.com/vllm-project/vllm/pull/47744), 2026-07-08); NIXL lease renewal for P/D ([#41383](https://github.com/vllm-project/vllm/pull/41383), 2026-05-11) | Complement: hooks Leaseholder uses or would adopt |
| vLLM [#61478](https://github.com/vllm-project/vllm/issues/61478) and [#62855](https://github.com/vllm-project/vllm/issues/62855) | Open: TTL-based eviction requested (2026-10-13); the Responses API ignores `prompt_cache_retention` and `prompt_cache_options` (2026-10-21) | Complement: vLLM has no TTL contract today |
| SGLang [session radix cache](https://github.com/sgl-project/sglang/pull/29173) and retention | Merged 2026-08-02, "soft protection, not memory pins". The HiCache TTL pin was reverted ([#21884](https://github.com/sgl-project/sglang/pull/21884), 2026-04-01); soft retention by priority decay is open ([#27970](https://github.com/sgl-project/sglang/pull/27970)) | Pre-empts session-aware retention in an engine |
| Dynamo | `nvext.cache_control` TTL pinning was added ([#6213](https://github.com/ai-dynamo/dynamo/pull/6213), 2026-02-27) and removed in [v1.1.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.1.0) because "SGLang upstream chose a different direction". `agent_hints.speculative_prefill` warms the predicted next-turn prefix. [v1.5.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0) deprecated KVBM in favour of "the engine's native KV offloading". Prefetch hooks for tool returns are being built ([blog](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/blog/2026/agentic-inference-optimizations.mdx), 2026-10-02) | Pre-empts next-turn prefill; supports building on vLLM's native CPU tier |
| llm-d-router [#2273](https://github.com/llm-d/llm-d-router/issues/2273), KV leases | Umbrella open since 2026-08-04; the lease child issue is still a placeholder. An advisory "evict last until time T" on a session's offloaded blocks, renewed by every turn (300 s default, 900 s cap), and a deferred prefetch verb that would act before a request exists | Pre-empts the agent-side lease design (unbuilt) |
| AIBrix [#2814](https://github.com/vllm-project/aibrix/issues/2814) | RFC open since 2026-09-25, no code: `agent_hints`, and session pinning in L1 with a TTL, driven by `open`, `heartbeat` and `close` signals | Pre-empts pinning driven by lifecycle signals |

Mooncake Store uses "lease" for something else: a short lease that protects an object while it is read, plus a soft pin ([design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md)); standalone soft-pin controls were proposed in [#4858](https://github.com/kvcache-ai/Mooncake/pull/4858) on 2026-10-22. LMCache has an open L2 prefetch deadline with recompute fallback ([#5255](https://github.com/LMCache/LMCache/pull/5255)), the same restore-or-recompute choice E1 measures.

### Tool timing data

[TraceLab](https://github.com/uw-syfi/TraceLab) ([paper](https://arxiv.org/abs/2606.30560), 2026-06-29) is a corpus of Claude Code and Codex sessions; v0.0.2 (2026-07-24) holds 665,453 agent steps. Each tool call carries `tool_name`, `emitted_at`, `result_at` and `tool_wall_latency_ms`, and the dataset is CC BY 4.0. It supplies the pause distributions for E1's history level.

## Voice under contention

| Work | Venue, date | What it does | Relation |
|---|---|---|---|
| [LiveKit preemptive generation](https://docs.livekit.io/agents/multimodality/audio.md) | Shipping; on by default in LiveKit Agents 1.8.8 (2026-10-15) | Starts the LLM request when a final transcript arrives, before the end of turn is confirmed, and regenerates if the context changes. TTS waits unless `preemptive_tts` is set | Baseline: an E2 factor, on and off; its transcript is one of the headroom triggers compared |
| [Endpoint Anticipation](https://arxiv.org/abs/2606.13450) | Interspeech 2026 | Forecasts the end of a turn up to 2.56 s ahead; in Unmute, 505 ms lower average latency for 28.4% more speculative compute | Complement: an earlier trigger |
| [PredGen](https://arxiv.org/abs/2506.15556) | arXiv 2025-06-18 | Generates candidate responses while the user is still speaking; about 2× lower latency in simulation | Pre-empts input-time speculation |
| [Unmute](https://github.com/kyutai-labs/unmute) (Kyutai) | Open source, 2025 | STT, an LLM on vLLM and TTS. TTS latency is about 750 ms with everything on one L40S and about 450 ms on separate GPUs | Complement: practitioner evidence of the co-location cost; the shipped fix is separate GPUs |
| [Voice-Light](https://arxiv.org/abs/2609.20995) | arXiv 2026-09-17 | A full-duplex cascaded agent; 758 ms median from the final VAD endpoint to first server audio, over 36 turns | Complement: small-sample practice data |
| [Metronome](https://arxiv.org/abs/2607.02640) | arXiv 2026-07-02 | Full-duplex models on one GPU collapse metastably when session KV fills the pool: 14 of 20 runs without its KV window, 0 of 20 with it | Complement: E2 reports outcomes per run, not only means |
| [NVIDIA Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) blueprint | Active on 2026-10-22 | Single-GPU profiles (workstation, DGX Spark, Jetson Thor) run ASR, the LLM and TTS on one GPU, with streaming-input LLM prefill. Its [frontend/backend agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/frontend_backend_agent/README.md) runs a Talker and a Thinker on the same weights, and lists a known limitation: the single-GPU profile "shows a conversational-experience regression ... including longer backend-planning pauses" | Complement: the edge pattern E2 reproduces |
| [LiveKit Inference](https://livekit.com/blog/latency-optimized-inference-gemma-4-on-livekit) | 2026-07-02 | Serves Gemma behind SGLang "with more headroom than a throughput-maximized deployment would", because "queueing delay is very noticeable" to people speaking to agents. Its client marks evaluation judges `low` so they do not "land in the same lane as live voice traffic" ([livekit/agents#6794](https://github.com/livekit/agents/pull/6794), 2026-08-11) | Complement: the headroom that E2's frontier prices |
| vLLM realtime | Shipping | A WebSocket `/v1/realtime` endpoint for streaming ASR ([example](https://docs.vllm.ai/en/latest/examples/speech_to_text/realtime/)), on session-based streaming input ([#28973](https://github.com/vllm-project/vllm/pull/28973), merged 2026-01-24). Open bug [#60137](https://github.com/vllm-project/vllm/issues/60137) (2026-10-05): paused streaming sessions keep their KV and deadlock the scheduler when the KV cache is full; the fix attempt [#62439](https://github.com/vllm-project/vllm/pull/62439) closed unmerged on 2026-10-22 | Complement: a KV-holding realtime session inside vLLM, and a candidate fix on the realtime path |

**Verdict.** We found no system that schedules cascaded voice turns beside a background agent class in one engine while managing KV for both. Starting the LLM before the turn ends is published and shipped at the application layer; Leaseholder compares those signals as triggers.

## Simulators and eviction bounds

| Work | Venue, date | What it shows | Relation |
|---|---|---|---|
| ["When Fancy Eviction Fails"](https://arxiv.org/abs/2609.28870) (Liu, Yu, Yang) | arXiv 2026-09-24 | On production prefix-cache traces from two companies, 14 eviction policies give "little benefit over LRU" despite "a large gap to Belady", because recency is unusually predictive | The main risk to E1: gains must come from information, not from a cleverer policy. E1's question is set against it |
| [LPC](https://neurips.cc/virtual/2025/poster/117662) (Learned Prefix Caching) | NeurIPS 2025 | LRU "has a large gap to the optimal algorithm"; a learned policy needs 18–47% less cache for the same hit ratio | The opposite reading of the same gap |
| [KVSET](https://arxiv.org/abs/2609.27746) | arXiv 2026-09-23 | Computes the KV working set from LRU stack distances (the Mattson algorithm) | Complement: miss-ratio curves for block-scaled traces |
| [Vidur](https://arxiv.org/abs/2405.05465) | MLSys 2024 | A serving simulator built from profiled operators; under 9% latency error | Complement |
| [LLMServingSim 2.0](https://arxiv.org/abs/2602.23036) | arXiv 2026-02-26 | Batching, routing, offloading and memory in one runtime loop; 0.95% average error against real deployments | Complement |
| ["Where Should the KV Cache Live?"](https://arxiv.org/abs/2609.16215) | arXiv 2026-09-14 | Simulated GPU, CPU and SSD placement. Gains come from tier capacity, not placement policy, and "even an oracle with knowledge of future requests never beats no prefetch on migration traffic" | Not the E1 risk. Its prefetch verdict is stated on migration traffic, which prefetching cannot reduce by construction, in a setup where decode is compute-bound at batch size one. E1's objective is resume latency |

**Theory relevant to E1:**
- integrated prefetching and caching: Cao et al. (SIGMETRICS 1995); Albers, Garg and Leonardi (JACM 2000), where the optimal stall time is computable offline;
- bounds for variable-size objects: Berger et al. (SIGMETRICS 2018);
- scaled-down miss-ratio curves: SHARDS (FAST 2015) and miniature simulations (USENIX ATC 2017);
- TTL-cache models: Jung et al. (INFOCOM 2003).

Belady's MIN over fixed-size blocks is exact only for single-tier hit counts. With a CPU tier, prefetch and a PCIe limit, E1 calls it a clairvoyant reference rather than an optimum.

## Commercial cache contracts and latency tiers

| Provider | Latency tier | Best-effort tier | Cache contract |
|---|---|---|---|
| Anthropic | [Priority Tier](https://platform.claude.com/docs/en/api/service-tiers): "capacity commitments are no longer available for purchase"; existing commitments target 99.5% uptime and overflow to Standard. [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode) (`speed: "fast"`): up to 2.5× output tokens per second at premium prices, generally available on Claude Opus 5.5 | Batch | [`cache_control`](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): 5 minutes by default, "refreshed for no additional cost each time the cached content is used", or `ttl: "1h"`. Writes cost 1.25× (5 min) or 2× (1 h) the base input price; reads 0.1× (less on some models). The page states no residency guarantee |
| OpenAI | [Fast mode](https://developers.openai.com/api/docs/guides/fast-mode) (`service_tier: "fast"`, or `"priority"`; Priority processing was renamed on 2026-07-30): up to 2.5× faster on `gpt-5.6-sol`; a fast ramp can be served at standard speed | [Flex](https://developers.openai.com/api/docs/guides/flex-processing) (beta): Batch API rates, "slower response times and occasional resource unavailability" (429). Batch API | [`prompt_cache_options.ttl`](https://developers.openai.com/api/docs/guides/prompt-caching) of `"30m"` (default) or `"12h"` on GPT-5.6 and later: it "governs eligibility for cache-read pricing" and "does not guarantee that every request reuses stored KV tensors". `prompt_cache_key` influences routing but does not pin a machine |
| Google Gemini | [Priority](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-flex-and-priority-inference/) (2026-04-02), at a premium; overflow is "automatically served at the Standard tier instead of failing" | [Flex](https://ai.google.dev/gemini-api/docs/flex-inference): 50% off, on "sheddable" capacity; requests "may be preempted or evicted to ensure capacity for high-priority users"; a 1–15 minute target; clients retry with backoff | [Implicit caching](https://ai.google.dev/gemini-api/docs/caching) is on by default for Gemini 2.5 and later, with no hit guarantee |
| AWS Bedrock | [Priority](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-bedrock-priority-flex-inference-service-tiers) (2025-11-18): up to 25% better output tokens per second, at a premium | Flex, at a discount, for workloads that tolerate longer latency | — |

**Relation.** Complement. Providers sell the two tiers Leaseholder runs on one replica, and they harvest the cheap tier by shedding it and having clients retry. Their cache TTLs are billing terms, not residency promises. Leaseholder maps Anthropic `cache_control` TTLs to lease floors, reports Anthropic's usage fields (a cache read only for a real hit), and treats the honor rate inside the TTL as an SLI. The OpenAI contract is deferred.

## What is distinctive

Each point below rests on the sources above, as of 2026-10-22.
1. **Headroom timed by voice lifecycle events and freed by parking.**
   - Capacity held back for priority work exists. It is static (SMG reserved slots, llm-d holdback ceilings, vLLM `watermark`), tied to a running request (Llumnix), or sized from load or the agent graph (Valve's dynamic memory reservation; Tokencake, which keeps GPU memory for critical-path agents).
   - llm-d's in-flight eviction reacts to demand that is already queued, and Dynamo's priority "does not reserve capacity".
   - In shipping systems, making room means ending background work (llm-d's 429 eviction, SMG's cancel before the first byte, Gemini Flex's preemption with client retries) or engine preemption that frees the victim's blocks. Keeping the victim's KV is published (ConServe) and proposed for SGLang (`--preempt-keep-kv`, draft #45686) and llm-d (deferred partial eviction).
   - We found no system that holds KV headroom for a realtime turn from a voice event that precedes the request, such as speech onset or end of turn. The claim is about timing: E2 compares it with an equal-size static reservation and with engine priority, and reports where it does not help.
2. **The engine "make room" fix.**
   - vLLM preempts among running requests but never for a waiting one (#40004, open since 2026-04-16); its open fixes cover only the `max_num_seqs` cap.
   - SGLang's priority preemption does not fire when KV memory runs out; the fix is a draft opened on 2026-10-21 (#45686).
   - Leaseholder's version is planned as a vLLM PR under #40004 that keeps or offloads the victim's KV. Preempting work and resuming it from kept KV is published (ConServe); the contribution is the change to vLLM's scheduler, not the idea.
3. **A recompute ledger validated by injected causes.** The stacks above expose per-request cached-token counts (vLLM) or a cache-discontinuity log (vllm-project/router Progress-TTL). We found none that attributes recomputed tokens to causes. This is observability, not a research result.

Two measurements carry the rest:
- **The workload pair.** We found no published system that measures a cascaded voice pipeline beside long-context agent traffic on one GPU, with voice-to-voice latency taken from waveforms. ConServe, HyGen, BROS, Niyama and Valve co-serve online and offline LLM requests. Metronome, Voice-Light and Unmute measure voice workloads alone. NVIDIA's single-GPU profile documents the regression.
- **The parking information ladder.** E1 measures how much of the gap to a clairvoyant reference each level of lifecycle information closes, per regime. The mechanisms are Tokencake's, KVFlow's, MORI's and Ask the Tool's.

One lifecycle-event schema for both workloads is an engineering choice with little standalone value. Adopting it means mapping it onto existing envelopes: vLLM's `KvHintsEnvelope`, Dynamo's `agent_hints` and llm-d's headers.

## What we do not claim

- **That any of these mechanisms is new:**
  - co-serving a latency class with best-effort work (ConServe, HyGen, BROS, Niyama, Valve);
  - class headroom (Llumnix), reserved capacity per class (SMG, llm-d) or dynamic memory reservation (Valve, Tokencake);
  - preempting work and resuming it from kept KV (ConServe; proposed in SGLang #45686);
  - priority classes, budgets, aging, queue TTLs and in-flight eviction (llm-d, SMG, Dynamo, AIBrix);
  - a priority tier that degrades instead of failing (Gemini Priority overflow, OpenAI Fast mode, Niyama's relegation);
  - parking KV in host RAM and prefetching it before a tool returns (Pensieve, CachedAttention, KVFlow, Tokencake, CacheScout, MORI, vLLM #58198);
  - TTL holds and leases (Continuum, vLLM #37003, llm-d-router #2273, AIBrix #2814, TensorRT-LLM);
  - program- or session-level admission (Autellix, ThunderAgent, Progress-TTL);
  - starting the LLM before the turn ends (LiveKit preemptive generation, Endpoint Anticipation, PredGen);
  - a gap between LRU and Belady on prefix caches ("When Fancy Eviction Fails", LPC).
- **A cache residency guarantee.** Providers promise billing eligibility, not residency, and Leaseholder reports an honor rate instead.
- **A better router.** `lh-gateway` is a small harness for experiments on one GPU without Kubernetes, and the `lh-core` policy is to run inside it from W3. The policy is to be proposed upstream: a vLLM scheduler PR, the agent-side lease and prefetch policy on vllm-project/router RFCs #427 and #428, and the lifecycle-event sink in agentic-api.
- **Measured results on datacenter GPUs.** Measurements on one 16 GB consumer GPU with a 4B model and traces block-scaled 32× are stated in dimensionless terms (KV pressure, pause over restore time, restore over recompute time), and `lh-sim` places datacenter configurations on that map from modeled cost fits, labeled as modeled.
- **Precedence.** Nothing here claims to be first or the only one. This page is a positioning statement, and each comparison rests on the sources cited.
