# Runbook

How to deploy, check, operate and recover the Leaseholder hosts. Every service runs as a `systemd --user` unit. Lingering is enabled on each host, so units survive logout and start at boot.

| Host | Units | Ports (all loopback) |
|---|---|---|
| Phastform (GPU) | `leaseholder.target` → `lh-stt`, `lh-tts`, `lh-engine` | engine 8000, STT 8100, TTS 8880, KV-event subscriber 5557 (bound by `lh-node` from W3; the engine connects to it), KV-event replay 5558 |
| cyan (control) | `lh-tunnel-phastform`, `lh-gateway`, `lh-voicenet`, `lh-livekit`, `lh-voice-worker` | tunnel 18000 → 8000, 18100 → 8100, 18880 → 8880, 15558 → 5558; gateway 18080; Unix sockets in `/run/user/$UID/lh/` (the gateway's is `gateway.sock`). LiveKit 17880 (TCP) and 17882 (UDP), inside the voice-plane namespace only |

**Cores on cyan** (48 threads):

| Cores | Used by |
|---|---|
| 0–7 | The control plane (`lh-gateway`) |
| 8–15 | agentic-api and tools |
| 16–31 | Load generators: AIPerf on 16–23, `lh-bench` on 16–31, callers on 24–31 |
| 32–47 | The voice plane: LiveKit and the worker |

The `lh-gateway`, `lh-livekit` and `lh-voice-worker` units set their own affinity, and load generators run under `taskset`.

## Deploy

From a workstation with SSH access to the hosts:

```sh
export LH_SSH="ssh -F $HOME/.ssh/penn_config"   # your SSH command for the hosts
deploy/deploy.sh phastform.cis.upenn.edu       # sync the repo to ~/leaseholder/src and install the units
deploy/deploy.sh cyan.cis.upenn.edu
```

**First time on a host:**
1. Enable lingering: `loginctl enable-linger $USER`.
2. Run its bootstrap:
   - Phastform: `deploy/phastform/bootstrap.sh` builds `~/leaseholder/venvs/{engine,tts}` from the lock files, plus the Kokoro checkout, the FlashInfer JIT workspace and the curand header links.
   - cyan: `deploy/cyan/bootstrap.sh` installs the LiveKit server, builds `~/leaseholder/venvs/{voice,bench}` from the lock files with uv (set `UV` when uv is neither in `~/.local/bin` nor on `PATH`), checks that the worker's models load offline, and generates the LiveKit keys in `~/.config/leaseholder/livekit.env` (mode 600) and the tunnel key. Stop `lh-voice-worker` before re-running it, because the sync may change the voice venv. Write the tunnel's SSH target to `~/.config/leaseholder/tunnel-phastform.env` as `LH_TUNNEL_TARGET=<user>@<gpu node>`.
3. Authorize the tunnel key on each GPU node: `deploy/authorize-tunnel.sh cyan.cis.upenn.edu phastform.cis.upenn.edu`. The entry may forward only the listed loopback ports, with no shell. Re-running the script replaces the entry.
4. On cyan, install the gateway:
   - Build it on a Linux host (`cargo build --release -p lh-gateway`) and copy `target/release/lh-gateway` to `~/leaseholder/bin/`.
   - Write `~/.config/leaseholder/gateway.toml` (mode 600) from `deploy/cyan/gateway.example.toml`. For each keyed tenant, make a key, give it to the client, and put only its digest in the file: `printf '%s' "$key" | sha256sum`.
   - `systemctl --user enable --now lh-gateway`.

## Start, stop, status

```sh
# Phastform
systemctl --user start leaseholder.target        # STT and TTS first, then the LLM; ~2 min warm, ~7.5 min on first boot
systemctl --user stop leaseholder.target
systemctl --user restart lh-engine               # one server; returns once /health answers
systemctl --user list-units 'lh-*'
journalctl --user -u lh-engine -f                # logs
bash ~/leaseholder/src/deploy/phastform/smoke.sh # end-to-end voice turn, GPU memory, port audit

# cyan
systemctl --user start lh-tunnel-phastform lh-voicenet lh-livekit lh-gateway
bash ~/leaseholder/src/deploy/phastform/smoke.sh http://127.0.0.1:18000 http://127.0.0.1:18100 http://127.0.0.1:18880
~/leaseholder/src/deploy/bin/in-voicenet ss -lntup   # what listens inside the voice-plane namespace
```

A unit counts as started only once its server answers its health endpoint (`wait-healthy` in `ExecStartPost`). Units that depend on it start only then.

## The gateway

`lh-gateway` serves the OpenAI routes and `/v1/events` on 127.0.0.1:18080, with an API key, and on `/run/user/$UID/lh/gateway.sock`, the voice plane's way in, without one. Its own `/health`, `/ready` and `/metrics` need no key. Its configuration reference and metrics list are in the [`lh-gateway` README](../crates/lh-gateway/README.md).

```sh
curl -s 127.0.0.1:18080/health                    # the process is up
curl -s 127.0.0.1:18080/ready                     # 200 while the engine's /health answers (through the tunnel)
curl -s 127.0.0.1:18080/metrics | grep -E 'lh_gateway_(request_outcomes|upstream_errors)_total'
journalctl --user -u lh-gateway -f                # one JSON line per request and per lifecycle event
systemctl --user restart lh-gateway               # rereads gateway.toml; drains in-flight requests for up to 30 s
```

## The voice plane

| Unit | What it runs | Notes |
|---|---|---|
| `lh-voicenet` | The network namespace holder | Loopback plus the dummy `lh0` (10.255.0.1/32), no route out |
| `lh-livekit` | LiveKit server in the namespace | 127.0.0.1:17880 (signalling), UDP 17882 (media) |
| `lh-voice-worker` | `lh-voice-worker start` (LiveKit Agents) in the namespace | Agent `leaseholder-voice`. STT and TTS through `%t/lh/phastform-{stt,tts}.sock`, the LLM through `%t/lh/gateway.sock` |

```sh
systemctl --user start lh-tunnel-phastform lh-gateway lh-voicenet lh-livekit lh-voice-worker
journalctl --user -u lh-voice-worker -f          # JSON logs; "registered worker" means it is ready
```

**Worker variants** are drop-ins linked into `~/.config/systemd/user/lh-voice-worker.service.d/`, followed by `systemctl --user daemon-reload` and a restart:
- `deploy/cyan/variants/vad-turns/lh-voice-worker.conf` sets fixed endpointing (`LH_TURN_DETECTION=vad`). Every experiment runs with it, because LiveKit's end-of-turn model waits its 3 s maximum on some synthesized turns ([BENCHMARKS.md](../docs/BENCHMARKS.md#voice-on-the-shared-gpu)).
- An `Environment=LH_LLM_UDS=%t/lh/phastform-engine.sock` drop-in sends the LLM straight to the engine, bypassing the gateway.

**Measurement callers.** Render the corpus once with the TTS server. The callers run inside the namespace:

```sh
~/leaseholder/venvs/voice/bin/lh-voice render-callers ~/leaseholder/data/callers --tts-url http://127.0.0.1:18880/v1
set -a; . ~/.config/leaseholder/livekit.env; set +a
taskset -c 24-31 ~/leaseholder/src/deploy/bin/in-voicenet ~/leaseholder/venvs/voice/bin/lh-voice bots \
  --corpus ~/leaseholder/data/callers --callers 2 --turns 25 --barge-in 0.25 --out ~/leaseholder/runs/<run>/bots
```

## Engine variants

A variant is a systemd drop-in in `~/.config/systemd/user/lh-engine.service.d/` that overrides `ExecStart` with extra flags. Flags on the command line take precedence over `deploy/phastform/engine.yaml`.
- **Apply:** link the drop-in, then `systemctl --user daemon-reload && systemctl --user restart lh-engine`.
- **Remove:** delete the link and repeat.
- **Example:** `python/leaseholder_vllm/tests/live/run.sh` runs the engine plugins check this way and always restores the frozen configuration.

A variant that spans several units lives in `deploy/phastform/variants/<name>/` and is switched as a whole: `deploy/phastform/variant.sh on mps` links its drop-ins and restarts the box, and `variant.sh off mps` restores the frozen configuration. `mps` runs all three servers as clients of one CUDA MPS daemon (`lh-mps.service`); the frozen box uses time-slicing ([BENCHMARKS.md](../docs/BENCHMARKS.md#voice-on-the-shared-gpu)).

## Checks before heavy work on a shared host

- `nvidia-smi` and `who`. `gpu-guard` refuses to start a GPU unit while another user's process holds the GPU.
- `free -g`, and `df -h /dev/shm`: the CPU tier occupies 20 GiB there.
- Port audit:
  - on Phastform: `ss -Hlntu | grep -E ':(8000|8100|8880|5557|5558)\b'` shows only `127.0.0.1`;
  - on cyan: `ss -Hlntup` shows no LiveKit socket on the host network.

## Failures

| Symptom | Likely cause | Action |
|---|---|---|
| `lh-engine` fails at start with "Free memory on device … less than desired GPU memory utilization" | Another process holds GPU memory, or the audio servers grew | `nvidia-smi`; stop the foreign job or wait. The engine needs 10.83 GiB free at start |
| A unit restarts in a loop, then stops (`start-limit-hit`) | Three failed starts within 30 min | `journalctl --user -u <unit> -n 200`, fix, `systemctl --user reset-failed <unit>`, start |
| The first start after an upgrade takes many minutes | FlashInfer JIT and torch.compile caches are cold | Expected once; caches persist in `~/leaseholder/cache/flashinfer` and `~/.cache/vllm` |
| No KV events reach a subscriber | vLLM connects (does not bind) to a specific TCP endpoint | The subscriber must bind `tcp://127.0.0.1:5557`; the engine reconnects to it within seconds |
| The tunnel is down (`curl 127.0.0.1:18000/health` fails on cyan) | Network blip or a restarted GPU node | `lh-tunnel-phastform` restarts itself every 5 s; check `journalctl --user -u lh-tunnel-phastform` |
| LiveKit is unreachable from the worker | The voice-plane namespace was restarted | `systemctl --user restart lh-voicenet lh-livekit`. Clients must run inside the namespace (`in-voicenet`) |
| The gateway answers 502 `upstream_unavailable` or `/ready` is 503 | The tunnel or the engine is down | `curl 127.0.0.1:18000/health`; see the tunnel row above |
| The gateway answers 401 `invalid_api_key` | The client's key has no digest in `gateway.toml` | `printf '%s' "$key" \| sha256sum` must match an entry; restart the gateway after editing |
| `lh-gateway` exits at start: "another process is accepting on this socket" | A second gateway serves `gateway.sock` | Stop the other process; a socket file left by a crash is replaced automatically |
| Some voice turns wait about 3 s before the agent answers | LiveKit's end-of-turn model is waiting its maximum endpointing delay | Link the `vad-turns` worker variant |
| A caller run fails with "the agent never listened" | The worker is not registered ("registered worker" missing from its log), or LiveKit was restarted under it | `journalctl --user -u lh-voice-worker -n 50`; `systemctl --user restart lh-voice-worker` |

## Pausing

Nothing should hold the shared GPU while idle. To pause the GPU node, run `systemctl --user stop leaseholder.target`; restart it with `systemctl --user start leaseholder.target`. The target stays enabled, so a reboot brings the frozen box back.
