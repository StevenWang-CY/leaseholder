"""LeasePolicy that also records every lease hint it applies, for the live engine check."""

import json
import os

from leaseholder_vllm.policy import LeasePolicy, lease_hint


class ProbeLeasePolicy(LeasePolicy):
    def _apply(self, ctx, keys):
        record = {
            "req_id": ctx.req_id,
            "kv_transfer_params": ctx.kv_transfer_params,
            "hint": lease_hint(ctx)._asdict(),
            "keys": len(keys),
        }
        with open(os.environ["LH_PROBE_LOG"], "a") as log:
            log.write(json.dumps(record) + "\n")
        return super()._apply(ctx, keys)
