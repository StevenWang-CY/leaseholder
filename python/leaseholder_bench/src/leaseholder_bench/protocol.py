"""Per-session request metadata that Leaseholder's load generators send.

Names and values mirror the Rust crate ``lh-proto`` (``headers.rs`` and ``class.rs``); keep the two
in sync. The AIPerf endpoint plugin renders its requests' metadata through :class:`SessionTagger`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

SESSION_HEADER: Final = "X-Session-ID"
"""Session id: a parent agent and all of its subagents share one value."""

CLASS_HEADER: Final = "X-LH-Class"
"""SLO class of the request: ``realtime``, ``interactive`` or ``background``."""

DEFAULT_MODEL: Final = "Qwen/Qwen3-4B-Instruct-2507-FP8"
"""The frozen box's model: the probes' default, and ``configs/agentic_replay.yaml``'s."""


class SloClass(StrEnum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


ENGINE_PRIORITY: Final[Mapping[SloClass, int]] = MappingProxyType(
    {
        SloClass.REALTIME: 0,
        SloClass.INTERACTIVE: 1,
        SloClass.BACKGROUND: 2,
    }
)
"""vLLM ``priority`` per class (``SloClass::engine_priority`` in ``lh-proto``).

vLLM schedules lower values first. It rejects any value other than 0 unless the server runs with
``--scheduling-policy priority``, so turn ``send_priority`` off against a FCFS server.
"""


def parse_class(value: str) -> SloClass:
    try:
        return SloClass(value)
    except ValueError:
        valid = ", ".join(c.value for c in SloClass)
        raise ValueError(f"unknown SLO class {value!r}; expected one of: {valid}") from None


def parse_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class SessionTagger:
    """Renders the per-request metadata of a replay whose sessions share one SLO class."""

    slo_class: SloClass = SloClass.BACKGROUND
    send_priority: bool = True

    def headers(self, session_id: str) -> dict[str, str]:
        return {SESSION_HEADER: session_id, CLASS_HEADER: self.slo_class.value}

    def body(self, session_id: str) -> dict[str, Any]:
        fields: dict[str, Any] = {"prompt_cache_key": session_id}
        if self.send_priority:
            fields["priority"] = ENGINE_PRIORITY[self.slo_class]
        return fields

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> SessionTagger:
        """Configure from ``LH_BENCH_CLASS`` and ``LH_BENCH_SEND_PRIORITY``.

        AIPerf builds the endpoint plugin itself, in every worker process, so the environment is
        the plugin's only configuration channel.
        """
        send_priority = env.get("LH_BENCH_SEND_PRIORITY")
        return cls(
            slo_class=parse_class(env.get("LH_BENCH_CLASS", SloClass.BACKGROUND.value)),
            send_priority=parse_flag(send_priority) if send_priority else True,
        )
