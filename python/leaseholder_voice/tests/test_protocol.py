import pytest

from leaseholder_voice.protocol import (
    CLASS_HEADER,
    REALTIME_CLASS,
    SESSION_ID_HEADER,
    realtime_headers,
    room_session_id,
)


def test_header_names_match_lh_proto() -> None:
    assert SESSION_ID_HEADER.lower() == "x-session-id"
    assert CLASS_HEADER.lower() == "x-lh-class"
    assert REALTIME_CLASS == "realtime"


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
