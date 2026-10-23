# Runbook

How to deploy, check, operate and recover the Leaseholder hosts. Every service runs as a `systemd --user` unit. Lingering is enabled on each host, so units survive logout and start at boot.

| Host | Units | Ports (all loopback) |
|---|---|---|
| Phastform (GPU) | `leaseholder.target` → `lh-stt`, `lh-tts`, `lh-engine` | engine 8000, STT 8100, TTS 8880, KV-event subscriber 5557 (to be bound by `lh-node`, from W5; the engine connects to it), KV-event replay 5558 |
| Gray (second GPU node, booked windows) | The same units as Phastform, with Gray's drop-ins (`deploy/gray/systemd/`: the CUDA compiler from the engine venv, the engine on cores 0–7 of NUMA node 0, the GPU's, and the audio servers on cores 8–15 of node 1). `leaseholder.target` is not enabled at boot | As on Phastform |
| cyan (control) | `lh-tunnel-phastform`, `lh-gateway`, `lh-voicenet`, `lh-livekit`, `lh-voice-worker`; `lh-smg` for E2 arm 8b only; `lh-tunnel-gray` for two-node points only | tunnel 18000 → 8000, 18100 → 8100, 18880 → 8880, 15558 → 5558, and to Gray 28000, 28100, 28880, 25558; gateway 18080; Unix sockets in `/run/user/$UID/lh/` (the gateway's is `gateway.sock`). LiveKit 17880 (TCP) and 17882 (UDP), inside the voice-plane namespace only; SMG 18090, its metrics 18091 |

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
deploy/deploy.sh gray.cis.upenn.edu            # the second GPU node
```

`deploy.sh` writes the deployed commit to `~/leaseholder/src/DEPLOYED` on the host. Do not deploy to cyan while an E2 campaign runs: the points read their arm files and the harness from that tree.

**First time on a host:**
1. Enable lingering: `loginctl enable-linger $USER`.
2. Run its bootstrap:
   - Phastform: `deploy/phastform/bootstrap.sh` builds `~/leaseholder/venvs/{engine,tts}` from the lock files, plus the Kokoro checkout, the FlashInfer JIT workspace and the curand header links.
   - Gray: `deploy/gray/bootstrap.sh` runs Phastform's. It needs uv 0.11.27 or newer (set `UV`), and the models it downloads are the revisions `engine.yaml` and `stt.yaml` pin.
   - cyan: `deploy/cyan/bootstrap.sh` installs the LiveKit server, builds `~/leaseholder/venvs/{voice,bench}` from the lock files with uv (set `UV` when uv is neither in `~/.local/bin` nor on `PATH`), checks that the worker's models load offline, and generates the LiveKit keys in `~/.config/leaseholder/livekit.env` (mode 600) and the tunnel key. Stop `lh-voice-worker` before re-running it, because the sync may change the voice venv. Write the tunnel's SSH target to `~/.config/leaseholder/tunnel-phastform.env` as `LH_TUNNEL_TARGET=<user>@<gpu node>`, and Gray's to `tunnel-gray.env` for the two-node points.
3. Authorize the tunnel key on each GPU node: `deploy/authorize-tunnel.sh cyan.cis.upenn.edu phastform.cis.upenn.edu` (and `gray.cis.upenn.edu`). The entry may forward only the listed loopback ports, with no shell. Re-running the script replaces the entry.
4. On cyan, install the gateway:
   - Build it on a Linux host (`cargo build --release -p lh-gateway`) and copy `target/release/lh-gateway` to `~/leaseholder/bin/`.
   - Write `~/.config/leaseholder/gateway.toml` (mode 600) from `deploy/cyan/gateway.example.toml`. For each keyed tenant, make a key, give it to the client, and put only its digest in the file: `printf '%s' "$key" | sha256sum`.
   - `systemctl --user enable --now lh-gateway`.

## Start, stop, status

```sh
# Phastform
systemctl --user start leaseholder.target        # STT and TTS first, then the LLM; ~2 min warm, ~7.5 min on first boot (Gray: ~12)
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

`lh-gateway` serves the OpenAI routes and `/v1/events` on 127.0.0.1:18080, with an API key, and on `/run/user/$UID/lh/gateway.sock`, the voice plane's way in, without one. Its own `/health`, `/ready` and `/metrics` need no key. Its configuration reference and metrics list are in the [`lh-gateway` README](../crates/lh-gateway/README.md), including the E2 settings: `upstream_priority` (where each class's priority goes upstream: `X-Vllm-Priority`, the body's `priority`, or `X-SMG-Priority`) and the `[admission]` policies. Configuration changes take a restart.

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
- `deploy/cyan/variants/no-preemptive-generation/lh-voice-worker.conf` turns LiveKit's preemptive generation off (`LH_PREEMPTIVE_GENERATION=false`), for E2's factor.

**Measurement callers.** Render the corpus once with the TTS server. The callers run inside the namespace:

```sh
~/leaseholder/venvs/voice/bin/lh-voice render-callers ~/leaseholder/data/callers --tts-url http://127.0.0.1:18880/v1
set -a; . ~/.config/leaseholder/livekit.env; set +a
taskset -c 24-31 ~/leaseholder/src/deploy/bin/in-voicenet ~/leaseholder/venvs/voice/bin/lh-voice bots \
  --corpus ~/leaseholder/data/callers --callers 2 --turns 25 --barge-in 0.25 --out ~/leaseholder/runs/<run>/bots
```

## Engine variants

The frozen box is `deploy/phastform/engine.yaml` and the units in `deploy/phastform/systemd/`. `deploy/phastform/variant.sh use NAME|frozen [FLAG...]` makes the box run exactly one variant plus engine flags, restarting only what changes; `variant.sh use frozen` restores the frozen box. The variants, the flags the E2 arms use and how each was checked are in [`deploy/phastform/variants/README.md`](../deploy/phastform/variants/README.md):
- **Flags** go on the engine's command line through `LH_ENGINE_FLAGS`, set by a generated drop-in (`variant-flags.conf`); vLLM takes the command line over `engine.yaml`.
- **`pr54530`**: vLLM #54530 ported to 0.32.0, generated at each start by `build.py` and loaded with `--scheduler-cls`.
- **`sglang`**: SGLang 0.5.22 on the engine's port. Build its environment once with `bash ~/leaseholder/src/deploy/phastform/variants/sglang/install.sh`. Its unit counts as started only after one sampled request, because the first one compiles FlashInfer's kernels for minutes.
- **`mps`**: the three servers as clients of one CUDA MPS daemon (`lh-mps.service`); the frozen box uses time-slicing ([BENCHMARKS.md](../docs/BENCHMARKS.md#voice-on-the-shared-gpu)).

`systemctl --user cat lh-engine` shows what the box runs. `variant.sh on|off NAME` adds or removes a single variant, for work outside E2. `python/leaseholder_vllm/tests/live/run.sh` runs the engine plugins check with its own drop-in and always restores the frozen configuration.

## E2 points

An E2 point is one arm file (`benchmarks/e2/arms/`); `lh-bench e2 run ARM --out DIR` measures it on cyan, and `benchmarks/e2/campaign.sh` runs a campaign file from a workstation, switching the engine before each point ([`benchmarks/e2/README.md`](../benchmarks/e2/README.md)). What a point changes on cyan, and how to undo it:
- **The gateway.** An arm with a `[gateway]` table runs the gateway from `~/.config/leaseholder/gateway-e2.toml` (`gateway.toml` with the table merged in) through the `e2-gateway` drop-in.
- **The worker.** `preemptive_generation = false` links the `no-preemptive-generation` worker drop-in.
- **SMG.** An arm with `router = "smg"` starts `lh-smg`, and every other arm stops it.
- **Undo.** `~/leaseholder/venvs/bench/bin/lh-bench e2 reset` puts the gateway back on `gateway.toml`, preemptive generation on and SMG stopped. A campaign runs it at its end.

One point runs at a time: the pid of the running one is in `~/leaseholder/runs/.lh-bench-e2.lock` on cyan, and SIGINT to it stops AIPerf and the callers and marks the point failed.

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
| `lh-bench e2 run` fails at its engine check | The GPU node runs another variant or other flags than the arm names | `variant.sh use` with what `lh-bench e2 engine ARM` prints; a campaign does this itself |
| `lh-bench e2 run`: "another lh-bench e2 run or reset holds" the lock | A point is still running, or one was interrupted | The pid is in `~/leaseholder/runs/.lh-bench-e2.lock`; stop it with SIGINT, or wait |
| The gateway does not start for a point, and `run.json` quotes its log | The arm's `[gateway]` table sets a key this gateway build does not know | Deploy a gateway build that has it, or fix the arm; `lh-bench e2 reset` restores `gateway.toml` |
| The gateway answers 429 `queue_full` or 503 `queue_timeout` | An `[admission]` policy's queue is full or its wait ran out | Expected under the E2 policies, and counted in `lh_gateway_admission_refused_total` |
| The engine's `/server_info` answers 500 | vLLM collects its environment with `uv pip list` and cannot find uv | The unit's `PATH` must include `~/.local/bin` and `VIRTUAL_ENV` must name the engine venv (as `lh-engine.service` does) |

## Pausing

Nothing should hold the shared GPU while idle. To pause the GPU node, run `systemctl --user stop leaseholder.target`; restart it with `systemctl --user start leaseholder.target`. The target stays enabled, so a reboot brings the frozen box back. Gray is shared with other work: start it only for a booked window (`gpu-guard` refuses while another process holds the GPU) and stop it after; the engine takes about 20 s to stop, releasing its pinned CPU tier.
