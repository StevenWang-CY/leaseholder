"""Lease-aware eviction for vLLM 0.32's native CPU offload tier.

Load it through the offloading connector's extra config:

    "kv_connector_extra_config": {"eviction_policy": "LeasePolicy",
                                  "cache_policy_module_path": "leaseholder_vllm.policy", ...}

The control plane attaches a lease hint to each request as ``kv_transfer_params.leaseholder`` (the
``EngineLeaseHint`` type in ``crates/lh-proto``). The policy reads only that hint plus recency: it never
learns about sessions any other way, and it keeps no lease table beyond what the hints imply.
"""

import heapq
import time
from typing import NamedTuple

from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, order_request_keys
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

HINT_KEY = "leaseholder"
# Eviction buckets: 0 holds chunks without a live lease; higher buckets are evicted later.
RANK = {"background": 1, "interactive": 2, "realtime": 3}
# A tool pause caps the lease at the expected return plus this grace period.
GRACE_S = 2.0


class LeaseHint(NamedTuple):
    """The parsed hint, cached on the request context."""

    session: str | None
    rank: int
    ttl_s: float
    release: bool


def lease_hint(ctx) -> LeaseHint:
    if (hint := ctx.get_state(LeaseHint)) is None:
        raw = (ctx.kv_transfer_params or {}).get(HINT_KEY) or {}
        ttl = float(raw.get("ttl_s", 0.0))
        if raw.get("expect_return_s") is not None:
            ttl = min(ttl or float("inf"), float(raw["expect_return_s"]) + GRACE_S)
        hint = LeaseHint(
            raw.get("session_id"), RANK.get(raw.get("class"), 1), ttl, raw.get("op") == "release"
        )
        ctx.set_state(hint)
    return hint


class LeasePolicy(CachePolicy):
    """Soft leases over LRU.

    Chunks without a live lease are evicted first (LRU, tail first). Leased chunks are evicted only when
    nothing else is evictable, lowest class first. A store fails only when plain LRU would also fail, because
    a refused eviction makes vLLM drop the store silently.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._buckets = [LRUCachePolicy(cache_capacity) for _ in range(len(RANK) + 1)]
        self._bucket_of = {}  # OffloadKey -> bucket index
        self._sessions_of = {}  # OffloadKey -> {session}
        self._keys_of = {}  # session -> {OffloadKey: None}, ordered head to tail
        self._leases = {}  # session -> (rank, expires_at)
        self._expiry = []  # heap of (expires_at, session)

    def _rank(self, key):
        return max(
            (self._leases[s][0] for s in self._sessions_of.get(key, ()) if s in self._leases),
            default=0,
        )

    def _move(self, keys_head_to_tail):
        """Re-bucket cached keys whose rank changed, tail first so the oldest keep the LRU order."""
        for key in reversed(list(keys_head_to_tail)):
            cur = self._bucket_of.get(key)
            if cur is not None and (new := self._rank(key)) != cur:
                chunk = self._buckets[cur].get(key)
                self._buckets[cur].remove(key)  # the chunk keeps its status and ref_cnt
                self._buckets[new].insert(key, chunk)
                self._bucket_of[key] = new

    def _drop(self, s):
        """The lease ended: untag its keys, which fall back to LRU."""
        self._leases.pop(s, None)
        keys = self._keys_of.pop(s, {})
        for k in keys:
            if (ss := self._sessions_of.get(k)) is not None:
                ss.discard(s)
                if not ss:
                    del self._sessions_of[k]
        self._move(keys)

    def _expire(self, now):
        while self._expiry and self._expiry[0][0] <= now:
            _, s = heapq.heappop(self._expiry)
            if s in self._leases and self._leases[s][1] <= now:
                self._drop(s)

    def _apply(self, ctx, keys):
        """Renew the request's lease and tag its keys."""
        hint = lease_hint(ctx)
        if (s := hint.session) is None:
            return
        now = time.monotonic()
        self._expire(now)
        if hint.release:
            return self._drop(s)
        old = self._leases.get(s, (0, 0.0))[0]
        if hint.ttl_s > 0:
            self._leases[s] = (hint.rank, now + hint.ttl_s)
            heapq.heappush(self._expiry, (now + hint.ttl_s, s))
        if s not in self._leases:
            return
        mine = self._keys_of.setdefault(s, {})
        new_keys = [k for k in keys if k not in mine]
        for k in new_keys:
            mine[k] = None
            self._sessions_of.setdefault(k, set()).add(s)
        self._move(mine if self._leases[s][0] != old else new_keys)

    def _forget(self, key):
        self._bucket_of.pop(key, None)
        for s in self._sessions_of.pop(key, ()):
            self._keys_of.get(s, {}).pop(key, None)

    # CachePolicy interface (vllm/v1/kv_offload/cpu/policies/base.py)

    def get(self, key):
        b = self._bucket_of.get(key)
        return None if b is None else self._buckets[b].get(key)

    def insert(self, key, chunk):
        self._bucket_of[key] = b = self._rank(key)
        self._buckets[b].insert(key, chunk)

    def remove(self, key):
        self._buckets[self._bucket_of[key]].remove(key)
        self._forget(key)

    def mark_evictable(self, key):
        self._buckets[self._bucket_of[key]].mark_evictable(key)

    def mark_non_evictable(self, key):
        self._buckets[self._bucket_of[key]].mark_non_evictable(key)

    def touch(self, keys, req_context):
        """Keys come head to tail: the tail ends up oldest."""
        by_bucket = {}
        for k in keys:
            if (b := self._bucket_of.get(k)) is not None:
                by_bucket.setdefault(b, []).append(k)
        for b, ks in by_bucket.items():
            self._buckets[b].touch(ks, req_context)

    def on_store_miss(self, keys, req_context):
        """Called before insert(): tag the request's keys so they land in its lease's bucket."""
        self._apply(req_context, list(keys))

    def on_request_finished(self, key_groups, insertion_only_keys, reused_keys, req_context):
        """The turn ended, so the lease clock restarts."""
        keys = [k for k in order_request_keys(key_groups, req_context) if k in self._bucket_of]
        self._apply(req_context, keys)
        self.touch(keys, req_context)

    def evict(self, n, protected):
        self._expire(time.monotonic())
        plan, left = [], n
        for b in self._buckets:  # unleased, then background, interactive, realtime
            avail = len(b._evictable) - sum(1 for k in protected if k in b._evictable)
            if left and (take := min(left, avail)) > 0:
                plan.append((b, take))
                left -= take
        if left:
            return None  # atomic: nothing changed, and the manager drops the store
        out = [kc for b, take in plan for kc in b.evict(take, protected)]
        for k, _ in out:
            self._forget(k)
        return out

    def clear(self):
        for b in self._buckets:
            b.clear()
        self._bucket_of, self._sessions_of, self._keys_of = {}, {}, {}
