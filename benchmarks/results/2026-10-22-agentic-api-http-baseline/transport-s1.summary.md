# Responses provider benchmark: transport

Streaming TTFT is measured at the first `response.output_text.delta`; JSON runs report it as `n/a`. Request/response bytes are the client-visible wire payload per turn: flat across turn depth for a provider that rehydrates history server-side, growing for a provider the client must resend full history to.

| Provider | Transport | Success | Task correctness | Tool-call accuracy | p50 latency (ms) | p95 latency (ms) | p50 continuation round (ms) | p50 first output (ms) | p50 TTFT (ms) | Turns/s | p50 request bytes | p50 response bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agentic-api | responses_websocket | 100.0% | 100.0% | 100.0% | 3173 | 4194 | 683 | 175 | 2874 | 0.30 | 3773 | 44824 |
| agentic-api-http | responses_http_sse | 100.0% | 100.0% | 100.0% | 3173 | 4191 | 684 | 173 | 2872 | 0.30 | 3648 | 53728 |
| agentic-api-json | responses_http_json | 100.0% | 100.0% | 100.0% | 3175 | 4194 | 684 | n/a | n/a | 0.30 | 3653 | 7418 |
| vllm | responses_http_sse | 100.0% | 100.0% | 100.0% | 3176 | 4183 | 684 | 179 | 2874 | 0.30 | 8435 | 53770 |
| vllm-json | responses_http_json | 90.0% | 90.0% | 90.0% | 3187 | 4272 | 687 | n/a | n/a | 0.27 | 8440 | 8417 |

Paired successful turns: 10. Median Agentic API minus vLLM latency: 0 ms. Median Agentic API / vLLM latency ratio: 1.000.

> **Correctness, not competition.** Task correctness and tool-call accuracy are correctness/regression checks against the same underlying model, not a performance comparison: a gateway that owns conversation state is expected to present the model with equivalent context, so its accuracy should match direct vLLM. A gap that widens with turn depth indicates a rehydration bug, not a capability advantage.
