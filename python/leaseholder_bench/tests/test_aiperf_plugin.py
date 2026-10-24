from __future__ import annotations

import pytest
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import Text, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.transports.aiohttp_transport import AioHttpTransport
from helpers import model_endpoint

from leaseholder_bench.aiperf_plugin import LeaseholderChatEndpoint, WekaTraceScaledLoader


def request_info(me: ModelEndpointInfo, correlation_id: str, root: str, conversation: str):
    return RequestInfo(
        model_endpoint=me,
        turns=[Turn(texts=[Text(contents=["hello"])], max_tokens=8)],
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id=f"req-{correlation_id}",
        x_correlation_id=correlation_id,
        root_correlation_id=root,
        conversation_id=conversation,
        source_trace_id="trace-a",
        agent_depth=0 if correlation_id == root else 1,
        parent_correlation_id=None if correlation_id == root else root,
    )


@pytest.fixture
def parent_and_child():
    me = model_endpoint()
    parent = request_info(me, "root-1", "root-1", "trace-a")
    child = request_info(me, "child-1", "root-1", "trace-a::sa:0")
    return LeaseholderChatEndpoint(model_endpoint=me), parent, child


def test_plugins_are_registered():
    assert plugins.get_class(PluginType.ENDPOINT, "leaseholder_chat") is LeaseholderChatEndpoint
    loader = plugins.get_class(PluginType.CUSTOM_DATASET_LOADER, "weka_trace_scaled")
    assert loader is WekaTraceScaledLoader
    assert issubclass(loader, WekaTraceLoader)
    # The stock entry forces 64 tokens per block; the scaled entry declares no default.
    assert plugins.get_dataset_loader_metadata("weka_trace").default_block_size == 64
    assert plugins.get_dataset_loader_metadata("weka_trace_scaled").default_block_size is None


def test_parent_and_subagent_share_the_session(parent_and_child):
    endpoint, parent, child = parent_and_child
    for info in (parent, child):
        headers = endpoint.get_endpoint_headers(info)
        assert headers["X-Session-ID"] == "root-1"
        assert headers["X-LH-Class"] == "background"
        payload = endpoint.format_payload(info)
        assert payload["prompt_cache_key"] == "root-1"
        assert payload["priority"] == 2
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["max_completion_tokens"] == 8


def test_wire_headers_keep_the_session_tree_root(parent_and_child, monkeypatch):
    endpoint, _, child = parent_and_child
    transport = AioHttpTransport(model_endpoint=child.model_endpoint)
    child.endpoint_headers = endpoint.get_endpoint_headers(child)
    headers = transport.build_headers(child)
    assert headers["X-Session-ID"] == "root-1"
    assert headers["X-Correlation-ID"] == "child-1"
    assert headers["X-LH-Class"] == "background"

    # Why aiperf.env keeps this AIPerf setting off: it replaces the root with the agent's own id.
    monkeypatch.setattr(Environment.HTTP, "X_SESSION_ID_FROM_CORRELATION_ID", True)
    assert transport.build_headers(child)["X-Session-ID"] == "child-1"


def test_class_and_priority_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LH_BENCH_CLASS", "interactive")
    monkeypatch.setenv("LH_BENCH_SEND_PRIORITY", "false")
    me = model_endpoint()
    endpoint = LeaseholderChatEndpoint(model_endpoint=me)
    info = request_info(me, "root-2", "root-2", "trace-a")
    assert endpoint.get_endpoint_headers(info)["X-LH-Class"] == "interactive"
    assert "priority" not in endpoint.format_payload(info)


def test_static_headers_and_api_key_are_kept():
    me = model_endpoint(headers=[("X-Bench-Run", "r1")], api_key="key")
    endpoint = LeaseholderChatEndpoint(model_endpoint=me)
    headers = endpoint.get_endpoint_headers(request_info(me, "root-3", "root-3", "trace-a"))
    assert headers["X-Bench-Run"] == "r1"
    assert headers["Authorization"] == "Bearer key"
    assert headers["X-Session-ID"] == "root-3"
