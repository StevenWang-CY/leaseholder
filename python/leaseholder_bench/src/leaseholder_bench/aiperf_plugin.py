"""AIPerf 0.13.0 plugins, registered through ``plugins.yaml`` (entry point group ``aiperf.plugins``).

``leaseholder_chat`` is AIPerf's OpenAI chat endpoint plus Leaseholder's per-session metadata (see
:mod:`leaseholder_bench.protocol`). AIPerf calls the endpoint for every request, including every
turn of a multi-turn Weka trace.

``weka_trace_scaled`` is AIPerf's Weka loader with one fix: it honours each trace file's own
``block_size``. AIPerf passes the registry default ``default_block_size: 64`` to ``weka_trace`` as an
override, so the stock loader re-expands a block-scaled trace to 64 tokens per block and synthesizes
prompts from only the first 1/32 of each request's hash ids. On a 32x-scaled corpus that reproduced
about 77% of the shared prefix and no assistant turns.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.models import RequestInfo
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.endpoints.openai_chat import ChatEndpoint

from leaseholder_bench.protocol import SessionTagger


def session_id(request_info: RequestInfo) -> str:
    """The session-tree root, so a parent agent and its subagents share one session id.

    AIPerf mints a fresh id per session play in concurrency and ``agentic_replay`` modes.
    """
    return request_info.root_correlation_id or request_info.x_correlation_id


class LeaseholderChatEndpoint(ChatEndpoint):
    """OpenAI chat completions with ``X-Session-ID``, ``X-LH-Class`` and ``priority`` per session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tagger = SessionTagger.from_env()

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        headers = super().get_endpoint_headers(request_info)
        headers.update(self.tagger.headers(session_id(request_info)))
        return headers

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        payload = super().format_payload(request_info)
        payload.update(self.tagger.body(session_id(request_info)))
        return payload


class WekaTraceScaledLoader(WekaTraceLoader):
    """``weka_trace`` that reads each trace's own ``block_size`` instead of a forced 64."""

    def __init__(self, *, default_block_size: int | None = None, **kwargs: Any) -> None:
        # AIPerf's composer passes the registry default as an override; drop it.
        del default_block_size
        super().__init__(**kwargs)
