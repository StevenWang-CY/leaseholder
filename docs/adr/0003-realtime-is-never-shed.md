# ADR-0003: Realtime is never shed

- **Status:** accepted, 2026-10-22
- **Deciders:** Chuyue Wang

## Context

A voice caller expects an answer about a second after they stop talking. Rejecting a voice turn with a 429 drops the call mid-conversation, which is the failure users notice most. Load-shedding routers (llm-d's flow control, for example) treat every class the same way at the limit.

On this hardware, the threat to voice is not request volume but background work holding the GPU:
- one server-side voice turn takes 415 ms idle and 876 ms median while background agents run 8k-token prefills, and LLM time to first token rises from 39 to 371 ms (the 2026-10-22 probe; [BENCHMARKS.md](../BENCHMARKS.md#voice-on-the-shared-gpu) has the frozen box's W1 values);
- vLLM's priority cannot preempt running requests to admit a waiting one (ADR-0001).

## Decision

Admission for the `realtime` class never rejects. Under pressure it does two things, in order:
1. **Make room.**
   - Cap background prefill admission at a small token budget.
   - Keep free KV at least equal to the realtime working set, by deferring new background turns and letting finished ones park in the CPU tier.
   - Prefetch the speaking session's KV while the user is still talking.
2. **Degrade the reply.** Use a shorter `max_tokens` or a filler phrase, so the reply still arrives.

Background traffic absorbs the overload. It keeps a minimum share and ages, so it is never starved.

## Consequences

- Capacity for realtime must be planned: a bounded number of concurrent calls per GPU, set from the voice SLO calibration.
- Background goodput becomes the measure of how well the GPU is used. The E2 frontier plots it against voice V2V p95.
- Rejections exist only for `interactive` (a deadline can be missed) and for `background` (deferral beyond its age bound). Both are reported as their own metrics.
- The decision binds Leaseholder's own admission policy. E2's baseline policies emulate what others ship (W2): llm-d's flow control, for one, can make a realtime request wait or refuse it when realtime and interactive requests fill its ceiling, and those outcomes are measured, not prevented.

## Update (2026-10-23, W4)

The decision stands: under the `leaseholder` policy a realtime request is never queued or refused. What was built of the two steps, and what E2 measured in its discriminating scenario ([ARCHITECTURE.md §4.3](../ARCHITECTURE.md#43-turn-imminent-headroom)):
- **Make room.** The gateway budgets background prefill in the critical window by predicted prefill time, not by tokens, and keeps KV headroom by deferring new background turns. Neither moved the voice tail measurably. The room is made in the engine, by the `make-room` change: lower-priority prefill sits out a realtime request's first tokens, with KV preemption or a class-aware reserve ([§4.6](../ARCHITECTURE.md#46-engine-side-policy)). Prefetching the speaking session's KV is not built.
- **Degrade the reply.** A shorter `max_tokens` is built (16 tokens in E2); a filler phrase is not. In that scenario degrade capped every reply, and the tail depends on it: the engine protects a reply's first 16 tokens, which undegraded replies outrun. A 48-token window without degrade is being measured.
