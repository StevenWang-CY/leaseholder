from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from leaseholder_bench.protocol import (
    CLASS_HEADER,
    ENGINE_PRIORITY,
    SESSION_HEADER,
    SessionTagger,
    SloClass,
)

LH_PROTO = Path(__file__).resolve().parents[3] / "crates/lh-proto"


def test_wire_names_match_lh_proto():
    headers = dict(re.findall(r'pub const (\w+): &str = "([^"]*)";', _source("headers.rs")))
    assert SESSION_HEADER.lower() == headers["SESSION_ID"]
    assert CLASS_HEADER.lower() == headers["CLASS"]
    schema = json.loads((LH_PROTO / "schema/slo_class.schema.json").read_text())
    classes = [variant["const"] for variant in schema["oneOf"]]
    assert [c.value for c in SloClass] == classes


def test_class_to_vllm_priority_mapping_matches_lh_proto():
    # SloClass::engine_priority; vLLM schedules lower values first.
    source = _source("class.rs")
    arms = source[source.index("fn engine_priority") :].split("}", 1)[0]
    priorities = {
        name.lower(): int(value) for name, value in re.findall(r"Self::(\w+) => (\d+)", arms)
    }
    assert {c.value: p for c, p in ENGINE_PRIORITY.items()} == priorities
    assert sorted(SloClass, key=ENGINE_PRIORITY.__getitem__) == list(SloClass)


def _source(name: str) -> str:
    return (LH_PROTO / "src" / name).read_text()


def test_agent_replay_defaults_to_background():
    tagger = SessionTagger.from_env({})
    assert tagger.headers("s-1") == {"X-Session-ID": "s-1", "X-LH-Class": "background"}
    assert tagger.body("s-1") == {"prompt_cache_key": "s-1", "priority": 2}


def test_the_class_comes_from_the_environment():
    tagger = SessionTagger.from_env({"LH_BENCH_CLASS": "interactive"})
    assert tagger.headers("s")["X-LH-Class"] == "interactive"
    assert tagger.body("s")["priority"] == 1


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_priority_can_be_left_out(value):
    tagger = SessionTagger.from_env({"LH_BENCH_SEND_PRIORITY": value})
    assert tagger.body("s") == {"prompt_cache_key": "s"}


@pytest.mark.parametrize(
    "env",
    [{"LH_BENCH_CLASS": "urgent"}, {"LH_BENCH_SEND_PRIORITY": "maybe"}],
)
def test_bad_settings_are_rejected(env):
    with pytest.raises(ValueError):
        SessionTagger.from_env(env)
