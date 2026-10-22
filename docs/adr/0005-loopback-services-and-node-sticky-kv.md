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
- **A private network for voice.** On the control host, the LiveKit server, the voice worker and the bots run inside an unprivileged network namespace whose only interface is loopback (`lh-voicenet.service`). They reach the engine, the audio servers and the control plane through Unix sockets that the tunnel and the control plane expose in `$XDG_RUNTIME_DIR/lh/`.
- **Sanitized inputs.** The control plane removes the engine-control headers and body fields from every client request and sets its own (`lh_proto::headers`). Each tenant's `cache_salt` is derived from its API key.
- **Node-sticky KV.** KV never crosses the network. Sessions stick to the node that holds their KV, and failover re-prefills from the transcript.

## Consequences

- No public ports, by construction. Every smoke test and deploy check audits the listening sockets.
- The voice plane cannot reach the internet or the host network directly. Joining a call from a browser, for the demo video, needs an explicit bridge, built when the demo is.
- A tunnel adds a loopback hop on each side, sub-millisecond on this network, and is measured as part of control-plane overhead.
- Losing a node costs a re-prefill of its sessions. The node-loss drill measures it.
