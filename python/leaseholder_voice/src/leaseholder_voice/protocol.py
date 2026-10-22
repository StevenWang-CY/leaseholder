"""Headers the voice worker sends to the Leaseholder control plane."""

from __future__ import annotations

import re

SESSION_ID_HEADER = "X-Session-ID"
"""Routing, accounting and tracing key of one conversation (lh-proto: ``x-session-id``)."""

CLASS_HEADER = "X-LH-Class"
"""SLO class of the request (lh-proto: ``x-lh-class``)."""

REALTIME_CLASS = "realtime"

# The lh-proto SessionId rule (crates/lh-proto/src/session.rs): safe in headers and engine metadata.
_SESSION_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def room_session_id(room_sid: str) -> str:
    """Return the session id of the conversation held in a LiveKit room.

    The room SID (``RM_...``) is unique per room instance, so two conversations in
    rooms that reuse a name never share a session.
    """
    if not room_sid:
        raise ValueError("room_sid is empty")
    return f"voice-{room_sid}"


def realtime_headers(session_id: str) -> dict[str, str]:
    """Return the headers for an LLM request of a realtime voice session.

    A new dict on every call: the LiveKit LLM plugin adds its own headers to the
    dict it is given.
    """
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    return {SESSION_ID_HEADER: session_id, CLASS_HEADER: REALTIME_CLASS}
