"""A call's GPU node, when the control plane fronts several of them.

The control plane places each session on one GPU node and keeps it there, because the session's KV lives in that
node's engine, and a call's STT and TTS belong on the same node. When the worker knows several nodes
(``LH_NODES``), it asks the control plane for a new call's node (``POST /v1/sessions``, over the LLM's socket) before
the session starts, and builds the call's STT and TTS on that node's sockets: ``<node>-stt.sock`` and
``<node>-tts.sock``, beside the default ones, as the tunnels name them. The LLM path does not change: the control
plane routes the call's requests by its session.

A placement that fails (no answer within a second, a refusal, a node the worker does not know) leaves the call on
the default sockets, with a warning: the call goes on, and the control plane places it by its first request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2 as httpx

SESSIONS_URL = "http://lh.invalid/v1/sessions"
"""The control plane's placement route; requests go over its Unix socket, so the host never resolves."""

TIMEOUT_S = 1.0
"""The longest a placement may take; the control plane answers in a millisecond."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sockets:
    """Where a call's STT and TTS go, and the node they are on (``None``: the default sockets)."""

    stt: str
    tts: str
    node: str | None = None


async def place(
    session_id: str,
    room: str,
    *,
    nodes: tuple[str, ...],
    control_plane: str,
    stt: str,
    tts: str,
    timeout: float = TIMEOUT_S,
) -> Sockets:
    """The sockets of the call ``session_id`` in ``room``: with ``nodes``, those of the node the control plane on
    the socket ``control_plane`` places it on, logged as ``session placed``; without nodes, or when the placement
    fails, the default ``stt`` and ``tts``."""
    default = Sockets(stt, tts)
    if not nodes:
        return default
    context = {"session_id": session_id, "room": room}
    transport = httpx.AsyncHTTPTransport(uds=control_plane)
    async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
        try:
            body = {"session_id": session_id, "class": "realtime"}
            response = await client.post(SESSIONS_URL, json=body)
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"  # a timeout's own message can be empty
        else:
            node = _node(response)
            if node in nodes:
                logger.info("session placed", extra=context | {"node": node})
                return Sockets(
                    _beside(stt, f"{node}-stt.sock"), _beside(tts, f"{node}-tts.sock"), node
                )
            error = f"HTTP {response.status_code}: {response.text[:200]}"
    logger.warning(
        "the call has no node (%s): its STT and TTS stay on %s and %s",
        error,
        stt,
        tts,
        extra=context,
    )
    return default


def _node(response: httpx.Response) -> Any:
    """The node of a placement's answer; ``None`` for anything but a placement."""
    if response.status_code != 200:
        return None
    try:
        answer = response.json()
    except ValueError:
        return None
    return answer.get("node") if isinstance(answer, dict) else None


def _beside(socket: str, name: str) -> str:
    """The socket ``name`` in the directory of ``socket``."""
    return str(Path(socket).with_name(name))
