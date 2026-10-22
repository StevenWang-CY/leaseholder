"""The policy reads exactly what lh-proto's EngineLeaseHint defines (crates/lh-proto/schema)."""

import inspect
import json
import pathlib

import pytest

SCHEMA = (
    pathlib.Path(__file__).resolve().parents[3]
    / "crates/lh-proto/schema/engine_lease_hint.schema.json"
)


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA.read_text())


def enum_values(definition):
    return [variant["const"] for variant in definition["oneOf"]]


@pytest.fixture(scope="module")
def policy_source():
    pytest.importorskip(
        "vllm.v1.kv_offload.cpu.manager", reason="needs VLLM_SRC or an installed vLLM 0.32"
    )
    import leaseholder_vllm.policy as policy

    return policy, inspect.getsource(policy.lease_hint)


def test_class_ranks_cover_exactly_the_protocol_classes(schema, policy_source):
    policy, _ = policy_source
    classes = set(enum_values(schema["$defs"]["SloClass"]))
    assert set(policy.RANK) == classes
    assert policy.RANK["realtime"] > policy.RANK["interactive"] > policy.RANK["background"]


def test_every_field_the_policy_reads_is_in_the_schema(schema, policy_source):
    policy, source = policy_source
    assert policy.HINT_KEY == "leaseholder"
    for field in ("session_id", "class", "ttl_s", "expect_return_s", "op"):
        assert f'"{field}"' in source, f"lease_hint() no longer reads {field}"
        assert field in schema["properties"], f"{field} is not part of EngineLeaseHint"
    assert enum_values(schema["$defs"]["HintOp"]) == ["release"]
