# ADR-0005: Loopback-only services, SSH tunnels, a private network for voice, and node-sticky KV

- **Status:** accepted, 2026-10-22
- **Deciders:** Chuyue Wang

## Context

The hosts are shared university machines with public addresses and no firewall we control. Two probes on 2026-10-22 showed how easily media ports leak:
- `livekit-server --bind 127.0.0.1` still exposed its RTC ports on the public addresses;
- libwebrtc clients (the voice worker, the measurement bots) bind ICE sockets on every non-loopback interface, whatever the server's configuration.

vLLM also trusts several client-supplied inputs for scheduling and accounting: `X-Vllm-Priority`, `X-Session-ID`, `X-data-parallel-rank`, and body fields such as `priority`, `session_id` and `kv_transfer_params`.

The GPU nodes are joined by 1 GbE, slower than recomputing a prompt on the card.

## Decision

- **Loopback only.**
  - Every service binds 127.0.0.1.
  - Hosts reach each other only through SSH local forwards, run as `systemd --user` units that restart on failure.
  - The tunnel key is authorized on each GPU node with `restrict,port-forwarding`, `permitopen` for the listed loopback ports, and no shell.
- **A private network for voice.** On the control host, the LiveKit server, the voice worker and the bots run inside an unprivileged network namespace (`lh-voicenet.service`) with no route out:
  - **Interfaces.** It holds loopback plus one dummy interface (`lh0`, 10.255.0.1/32, no IPv6 link-local). libwebrtc ignores loopback adapters, so with loopback alone a client gathers no ICE candidate and media never connects; the dummy gives clients an address to bind. The server still advertises only 127.0.0.1.
  - **Reaching services.** The namespace reaches the engine, the audio servers and the control plane through Unix sockets that the tunnel and the control plane expose in `$XDG_RUNTIME_DIR/lh/`.
- **Sanitized inputs.** The control plane removes the engine-control headers and body fields from every client request and sets its own (`lh_proto::headers`). Each tenant's `cache_salt` is derived from the tenant's name, so tenants never share prefix-cache entries.
- **Node-sticky KV.** KV never crosses the network. Sessions stick to the node that holds their KV, and failover re-prefills from the transcript.

## Consequences

- No public ports, by construction. Every smoke test and deploy check audits the listening sockets.
- The voice plane cannot reach the internet or the host network directly. Joining a call from a browser would need an explicit bridge; the demo video needs none, since it is rendered from calls the measurement callers record (`lh-voice bots --save-audio`).
- A tunnel adds a loopback hop on each side, sub-millisecond on this network. Every measured path to the engine crosses it, direct or through the control plane; the control-plane overhead in the SLO ends at the upstream write and leaves it out.
- Losing a node costs a re-prefill of its sessions. The node-loss drill measures it.

## Co-tenants (2026-10-24)

The hosts' other users share their loopback, which this decision does not cover. On cyan, the tunnels' TCP ports (Phastform's 18000, 18100, 18880 and 15558, and Gray's 28000-series) are reachable by any local user, and the frozen box's engine runs with `VLLM_SERVER_DEV_MODE=1`, which serves development routes such as `/collective_rpc` and `/sleep`; the harness needs `/server_info` and the cache-reset routes. The GPU nodes' own users reach the engine, STT and TTS on their loopback with no tunnel at all, and on cyan SMG's ports and the gateway's `/metrics` (no key) are reachable too. The engine also publishes every cached block's token ids and session id to 127.0.0.1:5557, which nothing of ours binds, and serves a replay of them on 5558, forwarded to cyan's 15558; W5 removes that endpoint, which nothing reads. Closing the rest needs more than socket forwards on cyan, which `permitopen` cannot name and which the harness's TCP clients (the gateway, `lh-bench`, SMG) cannot use: the engine, STT and TTS on Unix sockets in a private directory on the GPU nodes, socket-to-socket forwards under a key restriction that names them, a `unix:` upstream in the gateway and Unix-socket transport in `lh-bench`, and development mode only in a drop-in for measurements. That is W6's production hardening. Until then every co-tenant of either host can drive the engine. The voice plane's namespace isolates its network, not its processes: they keep the deploying user's uid and reach that user's systemd bus.
