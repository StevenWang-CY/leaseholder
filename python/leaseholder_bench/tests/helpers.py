"""Builders for tiny synthetic Weka traces (block size 2 unless stated)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def request(t: float, hash_ids: list[int], out: int, think: float | None = None, **extra: Any):
    """One main-agent or subagent request of a block-size-2 trace."""
    return {
        "t": t,
        "type": "n",
        "model": "m",
        "in": 2 * len(hash_ids),
        "out": out,
        "hash_ids": hash_ids,
        "think_time": think,
        **extra,
    }


def trace(trace_id: str, requests: list[dict[str, Any]], block_size: int = 2) -> dict[str, Any]:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": block_size,
        "hash_id_scope": "local",
        "requests": requests,
    }


def write_traces(directory: Path, *traces: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for t in traces:
        (directory / f"{t['id']}.json").write_text(json.dumps(t))
    return directory


def model_endpoint(**endpoint: Any):
    """AIPerf endpoint settings for the leaseholder_chat plugin."""
    from aiperf.common.enums import ModelSelectionStrategy
    from aiperf.common.models.model_endpoint_info import (
        EndpointInfo,
        ModelEndpointInfo,
        ModelInfo,
        ModelListInfo,
    )
    from aiperf.plugin.enums import EndpointType

    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="qwen")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType("leaseholder_chat"),
            base_urls=["http://127.0.0.1:8000"],
            streaming=True,
            **endpoint,
        ),
    )
