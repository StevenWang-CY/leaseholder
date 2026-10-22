"""Live check of the engine plugins against a running vLLM 0.32 on the GPU node (started by run.sh).

The engine runs the frozen configuration with LeaseConnector and ProbeLeasePolicy (engine-check.conf). The checks
show that:
1. per-request ``kv_transfer_params`` reach the connector and the CPU-tier cache policy;
2. ``X-Session-ID`` reaches the engine: GPU-tier KV events carry it, and the connector mirrors it into the hint
   the policy reads;
3. generated tokens are written through to the CPU tier, and the connector's hit receipt reports a CPU restore
   after the GPU prefix cache is reset.
"""

import json
import os
import threading
import time
import uuid

import pytest

ENGINE = os.environ.get("LH_LIVE_ENGINE")
pytestmark = pytest.mark.skipif(not ENGINE, reason="needs a running engine; see run.sh")

httpx = pytest.importorskip("httpx")
msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")
kv_events = pytest.importorskip("vllm.distributed.kv_events")

MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
EVENTS = os.environ.get("LH_LIVE_EVENTS", "tcp://127.0.0.1:5557")


class EventTap:
    """Collects the engine's KV cache events.

    vLLM 0.32's publisher binds only wildcard and ipc endpoints; given a specific TCP address it connects to it.
    The frozen box gives it tcp://127.0.0.1:5557, so the subscriber binds there (as lh-node does) and the engine
    connects to it.
    """

    def __init__(self, endpoint):
        self.events = []
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.bind(endpoint)
        self._decoder = msgspec.msgpack.Decoder(type=kv_events.KVEventBatch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                _topic, _seq, payload = self._sock.recv_multipart()
            except zmq.Again:
                continue
            self.events.extend(self._decoder.decode(payload).events)

    def close(self):
        self._stop.set()
        self._thread.join()
        self._sock.close()


@pytest.fixture(scope="module")
def tap():
    tap = EventTap(EVENTS)
    time.sleep(2.0)  # the engine reconnects to the subscriber within its reconnect interval
    yield tap
    tap.close()


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=ENGINE, timeout=120) as client:
        yield client


def chat(client, session, hint, prompt, max_tokens=16):
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "kv_transfer_params": {"leaseholder": hint},
        },
    )
    response.raise_for_status()
    return response.json()


def long_prompt(tag, words=1500):
    return f"[{tag}] Summarize this inventory in one line: " + " ".join(
        f"item-{i}" for i in range(words)
    )


def probe_records():
    with open(os.environ["LH_PROBE_LOG"]) as log:
        return [json.loads(line) for line in log]


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.2)
    return predicate()


def test_the_hint_reaches_the_connector_and_the_policy(client):
    session = f"w0-hint-{uuid.uuid4().hex[:8]}"
    hint = {"session_id": session, "class": "background", "ttl_s": 300}
    reply = chat(client, session, hint, long_prompt(session))
    assert set(reply["kv_transfer_params"]["leaseholder"]) == {"gpu_hit_tokens", "cpu_hit_tokens"}

    seen = wait_for(lambda: [r for r in probe_records() if r["hint"]["session"] == session])
    assert seen, "the CPU-tier policy never saw the request's hint"
    assert {r["hint"]["rank"] for r in seen} == {1}  # background
    assert all(r["kv_transfer_params"]["leaseholder"]["ttl_s"] == 300 for r in seen)


def test_the_session_header_reaches_kv_events_and_the_hint(client, tap):
    session = f"w0-header-{uuid.uuid4().hex[:8]}"
    chat(client, session, {"class": "realtime", "ttl_s": 120}, long_prompt(session))

    def stored(medium):
        return [
            e for e in tap.events if isinstance(e, kv_events.BlockStored) and e.medium == medium
        ]

    gpu = wait_for(lambda: [e for e in stored("GPU") if e.session_id == session])
    assert gpu, "no GPU-tier BlockStored event carries the X-Session-ID"
    seen = wait_for(lambda: [r for r in probe_records() if r["hint"]["session"] == session])
    assert seen, "the connector did not mirror X-Session-ID into the hint"
    assert {r["hint"]["rank"] for r in seen} == {3}  # realtime
    cpu = stored("CPU")
    print(
        f"GPU-tier stores with the session id: {len(gpu)}; CPU-tier stores: {len(cpu)}, "
        f"with a session id: {sum(e.session_id is not None for e in cpu)}"
    )


def test_a_cpu_restore_shows_up_in_the_hit_receipt(client):
    session = f"w0-restore-{uuid.uuid4().hex[:8]}"
    hint = {"session_id": session, "class": "interactive", "ttl_s": 300}
    prompt = long_prompt(session)
    first = chat(client, session, hint, prompt)
    assert first["kv_transfer_params"]["leaseholder"]["cpu_hit_tokens"] == 0

    # Stores to the CPU tier finish asynchronously after the request; then drop only the GPU prefix cache.
    def reset_gpu_cache():
        return client.post("/reset_prefix_cache").status_code == 200

    time.sleep(1.0)
    assert wait_for(reset_gpu_cache), "could not reset the GPU prefix cache"

    second = chat(client, session, hint, prompt)
    receipt = second["kv_transfer_params"]["leaseholder"]
    cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
    prompt_tokens = second["usage"]["prompt_tokens"]
    print(f"prompt {prompt_tokens} tokens: receipt {receipt}, usage cached_tokens {cached}")
    assert receipt["cpu_hit_tokens"] > 0.9 * prompt_tokens - 64
    assert cached >= receipt["cpu_hit_tokens"]
