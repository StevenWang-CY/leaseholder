"""Per-session request metadata that Leaseholder's load generators send.

Names and values mirror the Rust crate ``lh-proto`` (``headers.rs`` and ``class.rs``); keep the two
in sync. The AIPerf endpoint plugin and ``lh-bench replay`` both render requests through
:class:`SessionTagger`, so the two tools put identical metadata on the wire.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

SESSION_HEADER: Final = "X-Session-ID"
"""Session id: a parent agent and all of its subagents share one value."""

CLASS_HEADER: Final = "X-LH-Class"
"""SLO class of the request: ``realtime``, ``interactive`` or ``background``."""


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


def load_class_map(path: Path) -> dict[str, SloClass]:
    """Read a JSON object that maps trace ids to SLO classes."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of trace id -> class")
    return {str(trace_id): parse_class(cls) for trace_id, cls in raw.items()}


@dataclass(frozen=True)
class SessionTagger:
    """Assigns each replayed trace its SLO class and renders the per-request metadata."""

    default_class: SloClass = SloClass.BACKGROUND
    trace_classes: Mapping[str, SloClass] = field(default_factory=dict)
    send_priority: bool = True

    def class_for(self, trace_id: str) -> SloClass:
        return self.trace_classes.get(trace_id, self.default_class)

    def headers(self, session_id: str, trace_id: str) -> dict[str, str]:
        return {SESSION_HEADER: session_id, CLASS_HEADER: self.class_for(trace_id).value}

    def body(self, session_id: str, trace_id: str) -> dict[str, Any]:
        fields: dict[str, Any] = {"prompt_cache_key": session_id}
        if self.send_priority:
            fields["priority"] = ENGINE_PRIORITY[self.class_for(trace_id)]
        return fields

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> SessionTagger:
        """Configure from ``LH_BENCH_CLASS``, ``LH_BENCH_CLASS_MAP`` and ``LH_BENCH_SEND_PRIORITY``.

        AIPerf builds the endpoint plugin itself, in every worker process, so the environment is
        the plugin's only configuration channel.
        """
        class_map = env.get("LH_BENCH_CLASS_MAP")
        send_priority = env.get("LH_BENCH_SEND_PRIORITY")
        return cls(
            default_class=parse_class(env.get("LH_BENCH_CLASS", SloClass.BACKGROUND.value)),
            trace_classes=load_class_map(Path(class_map)) if class_map else {},
            send_priority=parse_flag(send_priority) if send_priority else True,
        )
