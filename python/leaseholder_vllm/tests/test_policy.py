"""LeasePolicy against vLLM 0.32's real CPUOffloadingManager."""

import random
import time

import pytest

manager_mod = pytest.importorskip(
    "vllm.v1.kv_offload.cpu.manager", reason="needs VLLM_SRC or an installed vLLM 0.32"
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key  # noqa: E402

import leaseholder_vllm.policy as policy_mod  # noqa: E402

CPUOffloadingManager = manager_mod.CPUOffloadingManager


def K(name: str):
    return make_offload_key(name.encode().ljust(16, b"\0"), 0)


def mgr(n, policy="LeasePolicy"):
    path = "leaseholder_vllm.policy" if policy == "LeasePolicy" else None
    return CPUOffloadingManager(
        num_chunks=n, cache_policy=policy, cache_policy_module_path=path, enable_events=True
    )


_rid = [0]


def ctx(session=None, cls="realtime", ttl=600.0, expect=None, op=None):
    _rid[0] += 1
    params = None
    if session is not None:
        hint = {"session_id": session, "class": cls, "ttl_s": ttl}
        if expect is not None:
            hint["expect_return_s"] = expect
        if op:
            hint["op"] = op
        params = {"leaseholder": hint}
    return ReqContext(req_id=f"r{_rid[0]}", kv_transfer_params=params)


def request(m, keys, **lease):
    """One request: register, store every key, complete the store, finish."""
    c = ctx(**lease)
    m.on_new_request(c)
    out = m.prepare_store([K(k) for k in keys], c)
    if out is not None:
        m.complete_store(out.keys_to_store, c)
    m.on_request_finished(c)
    return out


def present(m, names):
    return [n for n in names if m.lookup(K(n), None).name == "HIT"]


def check(m):
    """The policy's bookkeeping agrees with the manager's."""
    p = m._policy
    seen = set()
    for i, b in enumerate(p._buckets):
        for k in b.chunks:
            assert p._bucket_of.get(k) == i, "bucket_of mismatch"
            assert k not in seen, "key in two buckets"
            seen.add(k)
        for k in b._evictable:
            assert b.chunks[k].ref_cnt == 0, "evictable chunk with ref_cnt != 0"
    assert seen == set(p._bucket_of), "bucket_of has stale keys"
    total = sum(len(b._evictable) for b in p._buckets)
    assert total == m._num_evictable_cache_chunks
    for s, ks in p._keys_of.items():  # tags only for live leases or cached keys
        assert s in p._leases or all(k in p._bucket_of for k in ks), s


def test_unleased_chunks_go_first_tail_first():
    m = mgr(10)
    request(m, [f"A{i}" for i in range(5)], session="A", cls="realtime")
    request(m, [f"B{i}" for i in range(5)])  # no lease
    out = request(m, ["C0", "C1", "C2"])
    check(m)
    assert present(m, [f"A{i}" for i in range(5)]) == [f"A{i}" for i in range(5)]
    assert present(m, [f"B{i}" for i in range(5)]) == ["B0", "B1"]
    assert len(out.evicted_keys) == 3


def test_leased_chunks_go_lowest_class_first():
    m = mgr(10)
    request(m, ["A0", "A1", "A2", "A3"], session="A", cls="realtime")
    request(m, ["D0", "D1", "D2", "D3"], session="D", cls="background")
    request(m, ["U0", "U1"])
    out = request(m, ["N0", "N1", "N2", "N3", "N4"])
    check(m)
    assert present(m, ["U0", "U1"]) == []
    assert present(m, ["D0", "D1", "D2", "D3"]) == ["D0"]
    assert present(m, ["A0", "A1", "A2", "A3"]) == ["A0", "A1", "A2", "A3"]
    assert len(out.evicted_keys) == 5


def test_expired_and_released_leases_fall_back_to_lru():
    m = mgr(8)
    request(m, ["A0", "A1", "A2", "A3"], session="A", cls="realtime", ttl=0.05)
    request(m, ["D0", "D1", "D2", "D3"], session="D", cls="background", ttl=600)
    time.sleep(0.1)
    request(m, ["N0", "N1"])  # A expired: back in LRU, its tail goes first
    check(m)
    assert present(m, ["A0", "A1", "A2", "A3"]) == ["A0", "A1"]
    assert present(m, ["D0", "D1", "D2", "D3"]) == ["D0", "D1", "D2", "D3"]
    request(m, [], session="D", op="release")  # a control request: no tokens, only the op
    check(m)
    assert m._policy._bucket_of[K("D0")] == 0 and "D" not in m._policy._leases
    request(m, [f"N{i}" for i in range(2, 8)])  # evicts A1 A0 N1 N0 D3 D2
    check(m)
    assert present(m, ["D0", "D1", "D2", "D3"]) == ["D0", "D1"]
    assert present(m, ["A0", "A1", "N0", "N1"]) == []


def test_expected_return_caps_the_lease():
    m = mgr(4)
    request(m, ["A0", "A1"], session="A", cls="realtime", ttl=600, expect=-1.95)
    assert m._policy._leases["A"][1] - time.monotonic() < 0.06  # min(ttl, expect + grace)


def test_a_shared_prefix_keeps_the_strongest_live_lease():
    m = mgr(7)
    request(m, ["S0", "S1", "A0"], session="A", cls="realtime", ttl=600)
    request(m, ["S0", "S1", "E0"], session="E", cls="background", ttl=0.05)
    p = m._policy
    assert p._bucket_of[K("S0")] == 3 and p._bucket_of[K("E0")] == 1
    assert p._sessions_of[K("S0")] == {"A", "E"}
    time.sleep(0.1)
    request(m, ["N0", "N1", "N2", "N3"])  # one eviction needed: E0, expired and unshared
    check(m)
    assert present(m, ["S0", "S1", "A0"]) == ["S0", "S1", "A0"]
    assert present(m, ["E0"]) == []


def test_pending_stores_and_inflight_loads_are_never_evicted():
    m = mgr(6)
    c = ctx(session="A", cls="interactive")
    m.on_new_request(c)
    out = m.prepare_store([K("P0"), K("P1")], c)  # pending: ref_cnt == -1
    request(m, ["X0", "X1", "X2", "X3"])  # fills the tier
    lc = ctx(session="X", cls="background")
    m.on_new_request(lc)
    m.prepare_load([K("X0")], lc)  # loading: ref_cnt == 1
    up = ctx(session="A", cls="realtime")  # upgrade A while X0 is loading
    m.on_new_request(up)
    m.prepare_store([K("X0")], up)  # a ready key, recorded as reused
    m.on_request_finished(up)
    check(m)
    assert m._policy._bucket_of[K("X0")] == 3 and m._policy._bucket_of[K("P0")] == 3
    assert m.prepare_store([K(f"Y{i}") for i in range(4)], ctx()) is None  # 3 evictable
    check(m)
    m.complete_load([K("X0")], lc)
    m.complete_store(out.keys_to_store, c)
    m.on_request_finished(c)  # the latest renewal (interactive) wins for session A
    m.on_request_finished(lc)
    check(m)
    assert m._policy._bucket_of[K("X0")] == 2


def test_an_infeasible_eviction_changes_nothing():
    m = mgr(4)
    request(m, ["U0", "U1", "U2", "U3"], session="A", cls="realtime")
    # The manager's count check passes (4 evictable >= 2), but every evictable chunk is in this batch's
    # protected set, so the policy itself must refuse atomically.
    c = ctx()
    m.on_new_request(c)
    assert m.prepare_store([K(n) for n in ["U0", "U1", "U2", "U3", "N0", "N1"]], c) is None
    check(m)
    assert present(m, ["U0", "U1", "U2", "U3"]) == ["U0", "U1", "U2", "U3"]


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def fuzz(seed, n_chunks=64, steps=4000, hold_loads=False):
    """Random interleavings of stores, loads, renewals, releases and expiries, checked after every step."""
    rng = random.Random(seed)
    clock = FakeClock()
    real_time, policy_mod.time = policy_mod.time, clock
    try:
        m = mgr(n_chunks)
        pol, orig = m._policy, m._policy.evict
        stats = {"evict": 0, "evict_leased": 0, "store_none": 0}

        def evict(n, protected):
            pol._expire(clock.monotonic())  # the clock is frozen inside one call
            before = {
                k: i for i, b in enumerate(pol._buckets) for k in b._evictable if k not in protected
            }
            out = orig(n, protected)
            if out is None:
                assert len(before) < n  # fails only when plain LRU would fail too
                return None
            gone = {k for k, _ in out}
            assert len(out) == n and not (gone & protected)
            assert all(ch.ref_cnt == 0 for _, ch in out)
            top = max(before[k] for k in gone)
            assert not [k for k, b in before.items() if b < top and k not in gone]
            stats["evict"] += 1
            stats["evict_leased"] += top > 0
            return out

        pol.evict = evict
        open_stores, open_loads = [], []
        for _ in range(steps):
            clock.t += rng.choice([0.0, 0.001, 0.01])
            s = f"s{rng.randrange(12)}"
            keys = [K(f"{s}-{i}") for i in range(rng.randrange(1, 12))]
            if rng.random() < 0.3:
                keys = [K(f"shared-{i}") for i in range(rng.randrange(1, 4))] + keys
            kw = dict(
                session=s,
                cls=rng.choice(["background", "interactive", "realtime"]),
                ttl=rng.choice([0.0, 0.002, 600.0]),
                op="release" if rng.random() < 0.05 else None,
            )
            if rng.random() < 0.2:
                kw = {}
            c = ctx(**kw)
            m.on_new_request(c)
            out = m.prepare_store(keys, c)
            stats["store_none"] += out is None
            if out is not None and out.keys_to_store and rng.random() < 0.3:
                open_stores.append((out.keys_to_store, c))
            else:
                if out is not None:
                    m.complete_store(out.keys_to_store, c)
                m.on_request_finished(c)
            hits = [k for k in keys if m.lookup(k, c).name == "HIT"]
            if hits and rng.random() < 0.3:
                lc = ctx(**kw)
                m.on_new_request(lc)
                m.prepare_load(hits, lc)
                if hold_loads and rng.random() < 0.7:
                    open_loads.append((hits, lc))
                else:
                    m.complete_load(hits, lc)
                    m.on_request_finished(lc)
            while open_loads and rng.random() < 0.2:
                hk, hc = open_loads.pop(rng.randrange(len(open_loads)))
                m.complete_load(hk, hc)
                m.on_request_finished(hc)
            while open_stores and rng.random() < 0.5:
                ks, oc = open_stores.pop(rng.randrange(len(open_stores)))
                m.complete_store(ks, oc)
                m.on_request_finished(oc)
            check(m)
        for ks, oc in open_stores:
            m.complete_store(ks, oc)
            m.on_request_finished(oc)
        for hk, hc in open_loads:
            m.complete_load(hk, hc)
            m.on_request_finished(hc)
        check(m)
        assert m._num_evictable_cache_chunks == len(pol._bucket_of)  # all idle again
        return stats
    finally:
        policy_mod.time = real_time


@pytest.mark.parametrize("seed", range(6))
def test_fuzz(seed):
    stats = fuzz(seed)
    assert stats["evict_leased"] > 0


@pytest.mark.parametrize("seed", range(100, 106))
def test_fuzz_with_held_loads_exercises_refusals(seed):
    stats = fuzz(seed, n_chunks=24, hold_loads=True)
    assert stats["store_none"] > 0  # the refusal path must actually run


def test_hot_path_stays_cheap_at_a_full_size_tier():
    """prepare_store + evict runs on the scheduler's critical path; guard against complexity regressions."""
    n = 18204  # 40 GiB / (16 tokens x 144 KiB): Qwen3-4B BF16 KV, block size 16
    m = mgr(n)
    for s in range(18):
        request(m, [f"t{s}-{i}" for i in range(1011)], session=f"t{s}", cls="interactive")
    request(m, [f"u{i}" for i in range(n - 18 * 1011)])  # 6 unleased chunks
    c = ctx(session="new", cls="realtime")
    m.on_new_request(c)
    t0 = time.perf_counter()
    out = m.prepare_store([K(f"new-{i}") for i in range(128)], c)  # one 2k-token step
    elapsed = time.perf_counter() - t0
    m.complete_store(out.keys_to_store, c)
    m.on_request_finished(c)
    check(m)
    assert len(out.evicted_keys) == 128
    assert elapsed < 0.05, f"prepare_store + evict(128) took {elapsed * 1e3:.1f} ms"
