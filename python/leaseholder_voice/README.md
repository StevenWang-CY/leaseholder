# leaseholder-voice

Leaseholder's realtime voice plane on LiveKit Agents 1.8.8 ([ARCHITECTURE §7](../../docs/ARCHITECTURE.md#7-realtime-voice-pipeline)). The package holds:
- the production voice agent;
- the lifecycle events it posts to the control plane, and its per-turn stage times;
- the measurement callers that read voice-to-voice latency from waveforms;
- the headers the agent sends to the control plane;
- a factory for the stock OpenAI STT, LLM and TTS plugins, pointed at OpenAI-compatible servers over TCP or a Unix socket.

| Module | Contents |
|---|---|
| `leaseholder_voice.worker` | The voice agent (`lh-voice-worker start`): Silero VAD, LiveKit's local end-of-turn model, STT/LLM/TTS over the voice plane's sockets, LiveKit's default turn handling |
| `leaseholder_voice.events` | `Lifecycle`: a session's LiveKit signals as lifecycle events, and a log line per turn with its stage times; `EventSender`: their delivery to `/v1/events` |
| `leaseholder_voice.bots` | Measurement callers (`lh-voice bots`): voice-to-voice latency, barge-in to silence, and with `--save-audio` each call's audio |
| `leaseholder_voice.callers` | The caller corpus (`lh-voice render-callers`): scripted lines in six stock Kokoro voices, 16 kHz WAVs |
| `leaseholder_voice.audio` | A real-time microphone track, the agent's speech spans and a recording of both, on one clock |
| `leaseholder_voice.protocol` | The agent name `leaseholder-voice`, the `X-Session-ID` and `X-LH-Class` names, `room_session_id(room_sid)`, `realtime_headers(session_id)` |
| `leaseholder_voice.plugins` | `make_stt`, `make_llm`, `make_tts`, `openai_client` |

## The voice agent

`lh-voice-worker start` registers the agent `leaseholder-voice` with the LiveKit server and serves each dispatched room with one `AgentSession`. It runs in the voice-plane namespace on cyan as `lh-voice-worker.service`, from `~/leaseholder/venvs/voice`, which `deploy/cyan/bootstrap.sh` builds from `uv.lock`. The environment configures it:

| Variable | Meaning |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | The LiveKit server (from `~/.config/leaseholder/livekit.env`) |
| `LH_LLM_UDS` | The control plane's socket (`gateway.sock`), or the engine's tunnel (`phastform-engine.sock`) |
| `LH_STT_UDS`, `LH_TTS_UDS` | The tunnels to the STT and TTS servers |
| `LH_IDLE_PROCESSES` | Pre-started job processes (default 2) |
| `LH_LLM_TIMEOUT` | Seconds before an LLM request fails (default 30) |
| `LH_TURN_DETECTION` | `model` (default) or `vad` |
| `LH_PREEMPTIVE_GENERATION` | `true` (default) or `false` |
| `LH_EVENTS` | `on` (default): lifecycle events to the control plane through `LH_LLM_UDS`; `off` |

Experiments change two settings. LiveKit's end-of-turn model waits up to the endpointing maximum (3 s) when it judges a line unfinished: on the synthesized callers that happened on about one turn in six, and put a 3 s mode in voice-to-voice latency unrelated to serving. Experiments therefore run with `LH_TURN_DETECTION=vad`. That variant is `deploy/cyan/variants/vad-turns`: a turn ends a fixed delay after the end of speech (VAD plus LiveKit's 0.5 s minimum), the same for every arm. LiveKit's preemptive generation, on by default, starts the reply from the final transcript before the end of the turn is confirmed; `LH_PREEMPTIVE_GENERATION=false` (the variant `deploy/cyan/variants/no-preemptive-generation`, which `lh-bench e2 run` links for an arm that asks for it) turns it off, one of E2's factors.

Everything else is LiveKit's default except two settings for this deployment, whose only callers are the measurement bots in the namespace: interruptions are detected by VAD (what production mode picks anyway; adaptive detection needs LiveKit Inference), and there is no AEC warm-up, which would discard a barge-in on the first reply of each call although the callers have no echo path.

## Lifecycle events and turn lines

Each session tells the control plane when it will next need the GPU ([PROTOCOL.md](../../docs/PROTOCOL.md#lifecycle-events)), with events of the session `voice-<room SID>` from source `voice-worker`. Each comes from one LiveKit hook:

| Event | LiveKit 1.8.8 hook | When it fires |
|---|---|---|
| `user_speech_started` | `user_state_changed` to `speaking` | At Silero VAD's start of speech, once it has heard 0.05 s: 0.10–0.16 s after the onset on cyan |
| `user_turn_ended` | `conversation_item_added` with the caller's message | Once the end of the turn is decided and its reply scheduled, as the agent turns `thinking`. The reply's LLM request starts then, or at the final transcript under preemptive generation |
| `playback_started` | the audio output's `playback_started` | As the reply's first audio frame goes to the published track |
| `playback_ended` | the audio output's `playback_finished`, not interrupted | Once the reply has played out; `played_ms` is all of it |
| `interrupted` | the audio output's `playback_finished`, interrupted | When a barge-in interrupts the reply: on its final transcript with LiveKit's default handling, which pauses the audio after 0.5 s of speech first, or after those 0.5 s with `resume_false_interruption=False`. `played_ms` is what the caller heard |
| `session_closed` | `close` | When the session closes, after the caller has left |

`metrics_collected` would give the end of the turn as well, but 1.8.8 deprecates it, and warns on every subscription, in favour of each message's `metrics`, which also hold the stage times below. A reply interrupted before any of its audio was made has no playback to report; the control plane sees its stream close.

**Delivery.** The events go to `POST /v1/events` over `LH_LLM_UDS`, the control plane's socket, where the voice tenant needs no key. A hook only queues its event, and one background task per session posts what is queued as a JSON array, with a 1 s timeout: events queued while a POST is under way go out together in the next. Nothing reaches the session from there. A full queue (64 events) drops its oldest event, and a POST that fails, times out or is not answered 202 loses its batch; a lost batch is not retried, because a late timing signal would mislead. The first failure of an outage is logged as a warning, and recovery as information. `event_id` is the session and a sequence number (`voice-RM_a1B2c3-7`), `at_ms` the wall clock when the hook ran. When the job shuts down it sends what is left, waiting at most 2 s.

**The switch.** `LH_EVENTS=off` stops the events (the turn lines stay), so that E2 can measure Leaseholder with and without them: `deploy/cyan/variants/no-events`, which `lh-bench e2 run` links for an arm with `[voice] events = false`. They stop as well when `LH_LLM_UDS` is an engine's tunnel (`*-engine.sock`, the bypass), which takes none.

**Turn lines.** Each reply logs one JSON line, logger `leaseholder_voice.turns`, message `turn`, with `session_id`, `turn` (from 0 in each session), `interrupted` and, in ms, the stages LiveKit measured (a stage it did not measure is left out):

| Field | From | To |
|---|---|---|
| `transcript_ms` | the end of speech (VAD) | the final transcript |
| `end_of_turn_ms` | the end of speech | the end-of-turn decision |
| `llm_ttft_ms` | the LLM request | its first token |
| `llm_first_sentence_ms` | the LLM request | the first sentence handed to TTS |
| `turn_to_first_sentence_ms` | the end-of-turn decision | that sentence: `e2e_ms` less the playback latency, `tts_first_audio_ms` and `end_of_turn_ms`, because TTS starts only once the reply is scheduled |
| `tts_first_audio_ms` | that sentence | its first audio |
| `e2e_ms` | the end of speech | the reply's first audio frame on the track: the agent's side of voice-to-voice |

When a session ends, one line (logger `leaseholder_voice.events`, message `session ended`) gives its `turns` and, with events on, `events_sent`, `events_dropped` (pushed out of the queue, or unsent at the end) and `events_failed` (sent but not accepted). `lh-bench e2 run` reads both kinds of line from the worker's journal.

## Measurement callers

`lh-voice render-callers` renders the corpus once through a TTS server:
- 40 caller lines and 8 interruptions for a restaurant's phone line;
- six stock Kokoro voices, none of them the agent's;
- 16 kHz mono WAVs with 0.2 s of silence at each end, plus `manifest.json`.

`lh-voice bots` runs callers inside the namespace, each in its own room. The join token dispatches the agent. Each turn:
1. The caller plays a line.
2. It waits for the agent's reply: voice-to-voice runs from the end of its speech to the agent's first audio above the speech threshold.
3. On a share of turns it interrupts after a random delay: barge-in to silence runs from the interruption's speech onset until the agent's audio stops, and the reply to the interruption is a second voice-to-voice sample.

Every time comes from `time.monotonic()` in one process, so no clock offset enters. The run directory gets `run.json`, `turns.jsonl` and `summary.json`, which includes the microphone loop's worst lateness as a check that the caller kept up.

```bash
lh-voice render-callers ~/leaseholder/data/callers --tts-url http://127.0.0.1:18880/v1
set -a; . ~/.config/leaseholder/livekit.env; set +a
~/leaseholder/src/deploy/bin/in-voicenet lh-voice bots --corpus ~/leaseholder/data/callers \
  --callers 2 --turns 25 --barge-in 0.25 --out ~/leaseholder/runs/<run>/bots
```

**Saving the calls' audio.** With `--save-audio` each caller also keeps the frames it already handles, what it sends and the agent's audio as received, each with the time it handled it, and once every call has ended it writes two files into the run directory:
- `call-<i>.wav`: stereo 16-bit at 24 kHz, the agent's rate. On the left is what the caller sent, upsampled from 16 kHz; on the right the agent's audio as received. Both are laid out on the caller's monotonic clock from the first frame of either: frames go one after another, as their sender made them, each as early as the frames after it allow, so the jitter of their arrival leaves no gap and a real pause stays a pause.
- `call-<i>.json`: when the WAV starts, on the monotonic and the wall clock (`start`), its `rate`, `channels` and length (`seconds`), and each of the caller's turns as `turns.jsonl` has it, with `at`: when its events happened, in seconds into the WAV. They are the line's `start` and `speech_end`, the `reply` (the agent's first audio), for a barge-in the interruption's onset (`barge_in`) and `barge_in_speech_end`, the agent's `silence` and its `barge_in_reply`, and the turn's `end`, when the agent's audio stopped for good. A turn that failed keeps the times it got to.

Recording does not change what is measured: each frame is appended where a caller already handles it (a copy of 480 bytes for each of the agent's), and the files are written once the calls have ended. A 6-minute call keeps about 32 MiB (90 KiB a second of call), and writing it takes about 115 MiB more for a moment, one call at a time. [`benchmarks/e2/demo.py`](../../benchmarks/e2/README.md#the-demo-video) renders the demo video from two saved calls.

The measured numbers are in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md#voice-on-the-shared-gpu).

## Using the plugin factory

```python
from leaseholder_voice import plugins
from leaseholder_voice.protocol import room_session_id

lh = f"{os.environ['XDG_RUNTIME_DIR']}/lh"
session_id = room_session_id(ctx.job.room.sid)  # "voice-RM_..."
llm = plugins.make_llm("http://lh/v1", session_id, uds=f"{lh}/gateway.sock")  # the control plane
stt = plugins.make_stt("http://lh/v1", uds=f"{lh}/phastform-stt.sock")  # the tunnel to the GPU node
tts = plugins.make_tts("http://lh/v1", uds=f"{lh}/phastform-tts.sock")  # raw 24 kHz PCM
```

The `phastform-*.sock` files are the tunnel's Unix-socket forwards (`deploy/cyan`). Create one LLM per session: the headers belong to the instance.

## Tests

Unit tests need no network. CI runs them with `make test-python`:

```bash
uv run --extra test pytest -m "not integration"      # in python/leaseholder_voice
```

`tests/test_events.py` drives a stand-in session's hooks with LiveKit's own event types and checks each event against lh-proto's schema, the turn lines and that a failing hook never reaches LiveKit; it delivers to a fake control plane on a Unix socket: order, batching, the ids, a timeout, a refusal and recovery, the full-queue drop, a missing socket, and the end of a session. `tests/test_callers_and_bots.py` covers the corpus, the summary and the saved calls: the layout of a recorded track, the caller's track upsampled stretch by stretch exactly as a whole, and two turns, one with a barge-in, through a fake microphone and agent track, whose WAV shows each line's end and each reply's onset where `call-<i>.json` puts them and V2V as the callers measured it.

The integration test (the W0 barge-in and header findings) runs on cyan. It needs Linux with unprivileged user namespaces,
`unshare`, `nsenter`, `setpriv`, `ip` and `ss`, and `livekit-server` 1.13.10:

```bash
cd ~/leaseholder/src/python/leaseholder_voice
export UV_PROJECT_ENVIRONMENT=~/leaseholder/venvs/voice
LIVEKIT_SERVER_BIN=~/leaseholder/bin/livekit-server uv run --locked --extra test pytest -m integration -v
```

A run takes about 35 s and prints the measured latencies at the end. Logs and a JSON
record per room are kept under pytest's base temp directory (`--basetemp` to choose it).
The test skips when it is not on Linux or cannot find `livekit-server`.

What the test does:

- It creates a private user and network namespace that holds `lo` and a dummy interface.
  The namespace has no route out.
- Inside the namespace it starts a LiveKit server with a loopback-only config (the
  `LIVEKIT_CONFIG` template in `tests/integration/plane.py`, ports 27880/tcp and
  27882/udp), an agent worker, and one bot per case.
- The agent worker is `tests/integration/agent_app.py`. It uses this package's factory
  and Silero VAD.
- The mock STT, LLM and TTS run in the test process, outside the namespace. The agent
  reaches them only through their Unix sockets, as the deployed worker reaches the
  control plane and the tunnels.
- The bot plays a LibriSpeech utterance. Once the agent's reply audio starts, the bot
  waits 1 s, then barges in with a second utterance. Meanwhile the mock LLM is still
  streaming: one token per 100 ms, for 20 s.
- The test asserts five things:
  - every LLM request in the room carries `X-Session-ID: voice-<room SID>` and
    `X-LH-Class: realtime`;
  - the stream that was open at the barge-in is closed by the client, within the bound;
  - the lifecycle events reach the mock LLM's socket, which takes them as the control
    plane's does, all of them and in order: the turn, the reply, the barge-in that
    interrupts it, within 1 s of the stream closing and with `played_ms` within 250 ms of
    what the bot heard, its turn, the reply to it, the session's end; and the worker logs
    turn lines;
  - nothing in the namespace listens outside loopback, except the WebRTC clients' ICE
    sockets on the dummy address;
  - none of the test's processes owns a TCP or UDP socket on the host.

Namespace commands, as the test runs them:

```bash
# holder: prints "ready", then keeps the namespace alive
unshare --user --map-root-user --net -- sh -c 'ip link set lo up && ip link add lh0 type dummy && ip link set lh0 addrgenmode none && ip addr add 10.255.0.1/32 dev lh0 && ip link set lh0 up && echo ready && exec sleep infinity'
# every other process
nsenter --target=<holder pid> --user --net --preserve-credentials -- <command>
```

## W0 findings (cyan, 2026-10-22)

**Per-session headers.**
- The mechanism is the 1.8.8 constructor option `livekit.plugins.openai.LLM(extra_headers=...)`.
  `LLM.chat()` passes it to `chat.completions.create` on every request of that instance.
- The plugin writes its own headers into the same dict:
  - `User-Agent: LiveKit Agents/1.8.8 ...`;
  - `X-LiveKit-Room-ID` (the room SID), `X-LiveKit-Job-ID` and `X-LiveKit-Agent-ID`;
  - a new `X-LiveKit-Session-ID: inference_...` on every request. Despite its name, that
    one is a request id, not the conversation.
- `api_key` goes out as `Authorization: Bearer <key>`. The worker sends the factory's
  placeholder key: the gateway's voice socket names its tenant, so it needs none.
- All three plugins accept `client=`. The factory passes an `openai.AsyncClient` whose
  `httpx2` transport is `AsyncHTTPTransport(uds=...)`.
- Request timeouts: the LLM plugin sets each request's timeout from the session's
  `llm_conn_options.timeout`, 10 s by default, for connect, read, write and pool. The mock
  saw `x-stainless-read-timeout: 10.0`. The client's own timeout covers only the prewarm
  `GET /models`. Cold prefills longer than 10 s need
  `AgentSession(conn_options=SessionConnectOptions(llm_conn_options=APIConnectOptions(timeout=...)))`.

**A barge-in closes the LLM stream.** Measured from the bot's barge-in speech onset to the
mock seeing the client disconnect, over 10 runs:

| Interruption handling | Median | Range | Bound in the test |
|---|---|---|---|
| Stock (`resume_false_interruption=True`) | 2.19 s | 2.17–2.20 s | 3.0 s |
| `resume_false_interruption=False` | 0.58 s | 0.57–0.62 s | 1.0 s |

In both modes, the agent's VAD saw the barge-in 0.10–0.16 s after onset. The agent's
audio stopped at the bot 0.61–0.68 s after onset.

`AgentActivity` decides when to interrupt, from Silero VAD:

- **Stock.** After 0.5 s of barge-in speech (`interruption.min_duration`), it only pauses
  the audio output. The agent state turns `listening`, but the LLM keeps streaming.
- The reply is interrupted only when the barge-in turn's final transcript arrives. That
  takes VAD end of speech (0.55 s of silence) plus the STT call. The call chain is
  `on_final_transcript` → `_cancel_speech_pause()` → `SpeechHandle.interrupt()`.
- So the stream stays open for the whole barge-in utterance plus about 0.9 s. The test's
  barge-in has 1.31 s of speech.
- **`resume_false_interruption=False`.** After 0.5 s of barge-in speech,
  `_interrupt_by_audio_activity()` calls `SpeechHandle.interrupt()` right away.
- A false barge-in (a cough) then kills the reply. In stock mode the reply would resume
  after `false_interruption_timeout` (2 s).

From `SpeechHandle.interrupt()`, both modes follow the same chain:

1. The reply task (`_pipeline_reply_task`) cancels its LLM task with `cancel_and_wait`.
2. The `CancelledError` unwinds the default `llm_node` out of `async with llm.chat(...)`.
3. `LLMStream.aclose()` cancels the plugin's `_run`.
4. In `_run`, `async with stream:` calls `openai.AsyncStream.close()`, and that calls
   `httpx2.Response.aclose()`.
5. The response is unfinished, so the client closes the HTTP/1.1 connection instead of
   pooling it. That close is the disconnect the control plane sees.

The agent's LLM node reports the cancellation within 10 ms of the mock seeing the
disconnect.

**Network.**
- libwebrtc (livekit-rtc 1.1.20) gathers ICE host candidates on every non-loopback
  interface, whatever the server's rtc block says. On cyan's host network, one process
  with two connected rooms held 9 UDP sockets and 7 passive ICE-TCP `LISTEN` sockets on
  the host's public address.
- With only `lo` in a namespace, libwebrtc gathers no candidate at all: it ignores
  loopback adapters. `Room.connect` then fails with `wait_pc_connection timed out`.
- A dummy interface with a /32 address fixes it. Media flowed within 0.75 s, and the
  server still offers only 127.0.0.1 candidates.
- On the server, `--bind 127.0.0.1` covers signalling only. `rtc.tcp_port` must be 0,
  because the ICE-TCP listener ignores the IP filters. The single `rtc.udp_port` binds
  only the filtered IPs.
- `AgentServer` defaults matter:
  - `host=""` puts its health server on all interfaces;
  - in production mode, `num_idle_processes` defaults to the CPU count (48 on cyan);
  - in production mode, the worker refuses jobs above 70 % host CPU load.

**Other 1.8.8 defaults, and what the worker and the test set.**
- `aec_warmup_duration` defaults to 3.0 s, which ignores barge-ins during the first 3 s of
  the session's first agent turn. The callers have no echo path, so the worker and the test
  set `None`.
- `turn_detection` defaults to LiveKit Inference's `TurnDetector()`, and an unset
  `interruption.mode` means adaptive (LiveKit Inference) in dev mode and VAD in production
  mode. The worker runs the local `v1-mini` model (or `"vad"` with `LH_TURN_DETECTION=vad`)
  and pins `"vad"` interruptions; the test pins `"vad"` for both. Nothing is downloaded:
  the `v1-mini` weights ship in the `livekit-local-inference` wheel and Silero's in its
  plugin's, so the worker starts with `HF_HUB_OFFLINE=1` and an empty Hugging Face cache
  (`deploy/cyan/bootstrap.sh` checks that both load offline). `download-files` serves the
  older text turn-detector plugin, which the package does not install.
- `record` follows the server's recording setting. The worker and the test pass
  `record=False`.
- When the user leaves, the job stays in the room until LiveKit's departure timeout,
  unless `RoomOptions(delete_room_on_close=True)` is set. The worker and the test set it.

Test audio: `tests/data/user_turn.wav` and `tests/data/barge_in.wav` are LibriSpeech
dev-clean utterances 1272-135031-0011 and 1272-135031-0012 (CC BY 4.0).
