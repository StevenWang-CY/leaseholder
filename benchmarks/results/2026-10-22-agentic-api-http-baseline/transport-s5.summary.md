# Responses provider benchmark: transport

Streaming TTFT is measured at the first `response.output_text.delta`; JSON runs report it as `n/a`. Request/response bytes are the client-visible wire payload per turn: flat across turn depth for a provider that rehydrates history server-side, growing for a provider the client must resend full history to.

| Provider | Transport | Success | Task correctness | Tool-call accuracy | p50 latency (ms) | p95 latency (ms) | p50 continuation round (ms) | p50 first output (ms) | p50 TTFT (ms) | Turns/s | p50 request bytes | p50 response bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agentic-api | responses_websocket | 98.0% | 98.0% | 98.0% | 4488 | 7925 | 945 | 262 | 4045 | 0.96 | 3773 | 44899 |
| agentic-api-http | responses_http_sse | 98.0% | 98.0% | 98.0% | 4489 | 7686 | 969 | 266 | 4050 | 0.96 | 3648 | 53824 |
| agentic-api-json | responses_http_json | 98.0% | 98.0% | 98.0% | 4490 | 7837 | 970 | n/a | n/a | 0.96 | 3653 | 7418 |
| vllm | responses_http_sse | 98.0% | 98.0% | 98.0% | 4482 | 7890 | 949 | 255 | 4041 | 0.97 | 8435 | 53871 |
| vllm-json | responses_http_json | 98.0% | 98.0% | 98.0% | 4500 | 7772 | 974 | n/a | n/a | 0.96 | 8440 | 8412 |

Paired successful turns: 49. Median Agentic API minus vLLM latency: 3 ms. Median Agentic API / vLLM latency ratio: 1.000.

> **Correctness, not competition.** Task correctness and tool-call accuracy are correctness/regression checks against the same underlying model, not a performance comparison: a gateway that owns conversation state is expected to present the model with equivalent context, so its accuracy should match direct vLLM. A gap that widens with turn depth indicates a rehydration bug, not a capability advantage.
