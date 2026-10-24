import json
import re
from pathlib import Path

import pytest

from leaseholder_voice.protocol import (
    CLASS_HEADER,
    REALTIME_CLASS,
    SESSION_ID_HEADER,
    realtime_headers,
    room_session_id,
)

LH_PROTO = Path(__file__).resolve().parents[3] / "crates/lh-proto"


def test_header_names_match_lh_proto() -> None:
    source = (LH_PROTO / "src/headers.rs").read_text()
    headers = dict(re.findall(r'pub const (\w+): &str = "([^"]*)";', source))
    assert SESSION_ID_HEADER.lower() == headers["SESSION_ID"]
    assert CLASS_HEADER.lower() == headers["CLASS"]
    schema = json.loads((LH_PROTO / "schema/slo_class.schema.json").read_text())
    assert REALTIME_CLASS in [variant["const"] for variant in schema["oneOf"]]


def test_room_session_id_is_scoped_to_the_room_instance() -> None:
    assert room_session_id("RM_a1B2c3") == "voice-RM_a1B2c3"
    assert room_session_id("RM_a1B2c3") != room_session_id("RM_x9Y8z7")
    with pytest.raises(ValueError):
        room_session_id("")


def test_realtime_headers() -> None:
    assert realtime_headers("voice-RM_a1B2c3") == {
        "X-Session-ID": "voice-RM_a1B2c3",
        "X-LH-Class": "realtime",
    }


def test_realtime_headers_returns_a_new_dict_each_call() -> None:
    first = realtime_headers("s")
    first["User-Agent"] = "added by the plugin"
    assert realtime_headers("s") == {"X-Session-ID": "s", "X-LH-Class": "realtime"}


@pytest.mark.parametrize("session_id", ["", "has space", "line\nbreak", "café", "a/b", "x" * 129])
def test_realtime_headers_rejects_unsafe_session_ids(session_id: str) -> None:
    with pytest.raises(ValueError):
        realtime_headers(session_id)
