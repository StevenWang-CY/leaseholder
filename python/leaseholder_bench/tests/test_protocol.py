from __future__ import annotations

import json

import pytest

from leaseholder_bench.protocol import (
    CLASS_HEADER,
    ENGINE_PRIORITY,
    SESSION_HEADER,
    SessionTagger,
    SloClass,
)


def test_wire_names_match_lh_proto():
    assert SESSION_HEADER == "X-Session-ID"
    assert CLASS_HEADER == "X-LH-Class"
    assert [c.value for c in SloClass] == ["realtime", "interactive", "background"]


def test_class_to_vllm_priority_mapping():
    # lh-proto SloClass::engine_priority; vLLM schedules lower values first.
    assert dict(ENGINE_PRIORITY) == {
        SloClass.REALTIME: 0,
        SloClass.INTERACTIVE: 1,
        SloClass.BACKGROUND: 2,
    }
    assert sorted(SloClass, key=ENGINE_PRIORITY.__getitem__) == [
        SloClass.REALTIME,
        SloClass.INTERACTIVE,
        SloClass.BACKGROUND,
    ]


def test_agent_replay_defaults_to_background():
    tagger = SessionTagger.from_env({})
    assert tagger.headers("s-1", "trace-a") == {
        "X-Session-ID": "s-1",
        "X-LH-Class": "background",
    }
    assert tagger.body("s-1", "trace-a") == {"prompt_cache_key": "s-1", "priority": 2}


def test_default_class_and_per_trace_classes_from_env(tmp_path):
    class_map = tmp_path / "classes.json"
    class_map.write_text(json.dumps({"trace-rt": "realtime"}))
    tagger = SessionTagger.from_env(
        {"LH_BENCH_CLASS": "interactive", "LH_BENCH_CLASS_MAP": str(class_map)}
    )
    assert tagger.headers("s", "trace-rt")["X-LH-Class"] == "realtime"
    assert tagger.body("s", "trace-rt")["priority"] == 0
    assert tagger.headers("s", "trace-other")["X-LH-Class"] == "interactive"
    assert tagger.body("s", "trace-other")["priority"] == 1


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_priority_can_be_left_out(value):
    tagger = SessionTagger.from_env({"LH_BENCH_SEND_PRIORITY": value})
    assert tagger.body("s", "t") == {"prompt_cache_key": "s"}


@pytest.mark.parametrize(
    "env",
    [{"LH_BENCH_CLASS": "urgent"}, {"LH_BENCH_SEND_PRIORITY": "maybe"}],
)
def test_bad_settings_are_rejected(env):
    with pytest.raises(ValueError):
        SessionTagger.from_env(env)


def test_bad_class_map_is_rejected(tmp_path):
    bad_class = tmp_path / "bad.json"
    bad_class.write_text(json.dumps({"trace-a": "gold"}))
    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[]")
    for path in (bad_class, not_an_object):
        with pytest.raises(ValueError):
            SessionTagger.from_env({"LH_BENCH_CLASS_MAP": str(path)})
