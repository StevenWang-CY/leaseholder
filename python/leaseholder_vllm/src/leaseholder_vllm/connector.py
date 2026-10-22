"""vLLM's native OffloadingConnector, plus two things the control plane needs from the engine.

1. Session ids: vLLM keeps ``Request.session_id`` (from the ``X-Session-ID`` header or the body) but does not
   copy it into the offload request context. The connector mirrors it into the lease hint, so the cache policy
   sees the session even when the control plane sent no explicit ``session_id`` in the hint.
2. Hit receipts: each finished request returns how many prompt tokens were served from the GPU prefix cache and
   how many were restored from the CPU tier, as ``kv_transfer_params.leaseholder`` in the response. The
   recompute ledger needs that split per request; vLLM's own metrics give it only in aggregate.

Load it with:

    {"kv_connector": "LeaseConnector", "kv_role": "kv_both",
     "kv_connector_module_path": "leaseholder_vllm.connector", "kv_connector_extra_config": {...}}
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

from leaseholder_vllm.policy import HINT_KEY


class LeaseConnector(OffloadingConnector):
    # vLLM requires this 3-argument constructor signature.
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hits = {}  # request id -> [gpu_hit_tokens, cpu_hit_tokens]

    def on_new_request(self, request):
        if request.kv_transfer_params is None:
            request.kv_transfer_params = {}
        hint = request.kv_transfer_params.setdefault(HINT_KEY, {})
        if request.session_id:
            hint.setdefault("session_id", request.session_id)
        super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._hits[request.request_id] = [num_computed_tokens, 0]
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens and request.request_id in self._hits:
            self._hits[request.request_id][1] = num_external_tokens
        return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def request_finished_all_groups(self, request, block_ids):
        delay, _ = super().request_finished_all_groups(request, block_ids)
        gpu, cpu = self._hits.pop(request.request_id, (0, 0))
        return delay, {HINT_KEY: {"gpu_hit_tokens": gpu, "cpu_hit_tokens": cpu}}
