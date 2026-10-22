"""Load vLLM's CPU offload manager for the tests.

The manager and its policies are plain Python, so CI loads them from a vLLM v0.32.0 source checkout without
torch or CUDA: set VLLM_SRC to the checkout. Without VLLM_SRC the tests use an installed vLLM (the engine
environment on a GPU node); each test module skips when neither is available.
"""

import logging
import os
import sys
import types

VLLM_SRC = os.environ.get("VLLM_SRC")


def _load_from_source(src: str) -> None:
    # A bare package root, so importing vllm.v1.kv_offload does not run vllm/__init__.py (torch, CUDA).
    pkg = types.ModuleType("vllm")
    pkg.__path__ = [os.path.join(src, "vllm")]
    sys.modules["vllm"] = pkg
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch

    def init_logger(name):
        logger = logging.getLogger(name)
        logger.warning_once, logger.info_once, logger.debug_once = (
            logger.warning,
            logger.info,
            logger.debug,
        )
        return logger

    logmod = types.ModuleType("vllm.logger")
    logmod.init_logger = init_logger
    sys.modules["vllm.logger"] = logmod

    parents = [
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading",
    ]
    for name in parents:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod

    class OffloadingConnectorStats:
        def set_gauge(self, *args):
            pass

        def observe_histogram(self, *args):
            pass

        def increase_counter(self, *args):
            pass

    metrics = types.ModuleType(parents[-1] + ".metrics")
    metrics.OffloadingConnectorStats = OffloadingConnectorStats
    sys.modules[metrics.__name__] = metrics


if VLLM_SRC:
    _load_from_source(os.path.abspath(VLLM_SRC))
