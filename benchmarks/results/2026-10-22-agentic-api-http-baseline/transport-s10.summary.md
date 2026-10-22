# Responses provider benchmark: transport

Streaming TTFT is measured at the first `response.output_text.delta`; JSON runs report it as `n/a`. Request/response bytes are the client-visible wire payload per turn: flat across turn depth for a provider that rehydrates history server-side, growing for a provider the client must resend full history to.

| Provider | Transport | Success | Task correctness | Tool-call accuracy | p50 latency (ms) | p95 latency (ms) | p50 continuation round (ms) | p50 first output (ms) | p50 TTFT (ms) | Turns/s | p50 request bytes | p50 response bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agentic-api | responses_websocket | 99.0% | 99.0% | 99.0% | 6099 | 10949 | 1260 | 404 | 5478 | 1.40 | 3773 | 44941 |
| agentic-api-http | responses_http_sse | 99.0% | 99.0% | 99.0% | 6102 | 10953 | 1260 | 398 | 5478 | 1.40 | 3648 | 53842 |
| agentic-api-json | responses_http_json | 99.0% | 99.0% | 99.0% | 6094 | 10977 | 1248 | n/a | n/a | 1.40 | 3653 | 7418 |
| vllm | responses_http_sse | 99.0% | 99.0% | 99.0% | 6072 | 10945 | 1251 | 413 | 5488 | 1.40 | 8435 | 53884 |
| vllm-json | responses_http_json | 99.0% | 99.0% | 99.0% | 6097 | 10975 | 1255 | n/a | n/a | 1.40 | 8440 | 8412 |

Paired successful turns: 99. Median Agentic API minus vLLM latency: 1 ms. Median Agentic API / vLLM latency ratio: 1.000.

> **Correctness, not competition.** Task correctness and tool-call accuracy are correctness/regression checks against the same underlying model, not a performance comparison: a gateway that owns conversation state is expected to present the model with equivalent context, so its accuracy should match direct vLLM. A gap that widens with turn depth indicates a rehydration bug, not a capability advantage.
