# State-scaling: request bytes and accuracy versus turn depth

Each depth below ran as its own independent batch of sessions (not a bucket sliced out of one long run), so sample counts are even across depths instead of thinning out at the tail. Accuracy is a correctness/regression check, not a competition: it should not diverge between providers, and a widening gap at deeper turns points to a rehydration bug.

| Provider | Depth | Sessions | Success | Task correctness | p50 latency (ms) | p50 request bytes | p50 response bytes | total request bytes | total response bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agentic-api | 1 | 5 | 100.0% | 100.0% | 517 | 360 | 5966 | 1800 | 29983 |
| agentic-api | 5 | 5 | 40.0% | 40.0% | 396 | 424 | 6856 | 4235 | 68265 |
| agentic-api | 10 | 5 | 20.0% | 20.0% | 398 | 424 | 6856 | 4235 | 68265 |
| agentic-api | 25 | 5 | 8.0% | 8.0% | 409 | 424 | 6856 | 4235 | 68265 |
| agentic-api-http | 1 | 5 | 100.0% | 100.0% | 513 | 335 | 6823 | 1675 | 34503 |
| agentic-api-http | 5 | 5 | 40.0% | 40.0% | 409 | 398 | 7944 | 3985 | 78869 |
| agentic-api-http | 10 | 5 | 20.0% | 20.0% | 410 | 398 | 7944 | 3985 | 79061 |
| agentic-api-http | 25 | 5 | 8.0% | 8.0% | 409 | 398 | 7944 | 3985 | 79061 |
| agentic-api-json | 1 | 5 | 100.0% | 100.0% | 522 | 336 | 1061 | 1680 | 5307 |
| agentic-api-json | 5 | 5 | 40.0% | 40.0% | 424 | 400 | 1082 | 3995 | 10823 |
| agentic-api-json | 10 | 5 | 20.0% | 20.0% | 418 | 400 | 1082 | 3995 | 10822 |
| agentic-api-json | 25 | 5 | 8.0% | 8.0% | 418 | 400 | 1082 | 3995 | 10822 |
| vllm | 1 | 5 | 100.0% | 100.0% | 508 | 336 | 6851 | 1680 | 34064 |
| vllm | 5 | 5 | 40.0% | 40.0% | 408 | 598 | 7721 | 5980 | 76634 |
| vllm | 10 | 5 | 20.0% | 20.0% | 409 | 598 | 7721 | 5980 | 76826 |
| vllm | 25 | 5 | 8.0% | 8.0% | 414 | 598 | 7721 | 5980 | 76826 |
| vllm-json | 1 | 5 | 100.0% | 100.0% | 519 | 337 | 1187 | 1685 | 5936 |
| vllm-json | 5 | 5 | 40.0% | 40.0% | 423 | 599 | 1188 | 5990 | 11878 |
| vllm-json | 10 | 5 | 20.0% | 20.0% | 418 | 599 | 1188 | 5990 | 11877 |
| vllm-json | 25 | 5 | 8.0% | 8.0% | 428 | 599 | 1188 | 5990 | 11877 |
