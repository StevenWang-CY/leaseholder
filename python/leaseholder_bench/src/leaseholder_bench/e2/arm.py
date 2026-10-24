"""E2's arm files (:func:`load_arm`).

An arm file (TOML) describes one point of E2's frontier: the calls, the agent replay beside them, the
gateway's configuration and the engine the GPU node runs. The point is named after the file.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_FLAG = re.compile(r"[A-Za-z0-9._:/=,+-]+")
"""Engine flags travel through ssh and a systemd environment line, so each is one plain word."""
_ARM = re.compile(r"(10|[1-9])[a-z]?(-[a-z0-9]+)*")
_NODE = re.compile(r"[A-Za-z0-9._-]{1,64}")
"""The gateway's rule for node names, which unit, file and socket names carry."""


class E2Error(Exception):
    """A point that cannot be measured as its arm file describes it."""


# The arm file


@dataclass(frozen=True)
class Voice:
    """``[voice]``: the measurement calls and the voice agent's turn handling."""

    callers: int = 2
    turns: int = 25
    barge_in: float = 0.25
    """Share of turns on which the caller interrupts the reply."""
    seed: int = 0
    preemptive_generation: bool = True
    """LiveKit's preemptive generation (``LH_PREEMPTIVE_GENERATION`` of the voice worker)."""
    events: bool = True
    """The voice worker's lifecycle events to the gateway (``LH_EVENTS``)."""


@dataclass(frozen=True)
class Agents:
    """``[agents]``: the AIPerf agentic replay beside the calls, through the gateway as its tenant
    ``agents``; no sessions is voice alone."""

    sessions: int = 0
    """Session trees AIPerf keeps live."""
    think_cap: float = 120.0
    """Cap on each recorded think time, in seconds."""
    warmup: float = 60.0
    """Seconds of replay between the start of AIPerf's profiling phase and the first call."""
    data: str = "weka-x32"
    """A ``scale-traces`` directory, relative to ~/leaseholder/data."""
    seed: int = 11


@dataclass(frozen=True)
class Engine:
    """``[engine]``: what the GPU node runs, as ``variant.sh use`` takes it. e2 run checks it."""

    variant: str = "frozen"
    """A directory of deploy/phastform/variants/, or ``frozen``: none."""
    flags: tuple[str, ...] = ()
    """Engine flags added to its command line, where they beat engine.yaml and the variant's."""

    def __str__(self) -> str:
        return " ".join([self.variant, *self.flags])


@dataclass(frozen=True)
class Arm:
    name: str
    arm: str
    """E2's arm, which names the frontier the point is on: 1 to 10, a letter for the variants of
    arms 7 and 8 (``7a``), a suffix for a factor (``4-no-preemptive``)."""
    label: str
    voice: Voice = Voice()
    agents: Agents = Agents()
    gateway: Mapping[str, Any] = field(default_factory=dict)
    """Merged into gateway.toml as it is (an empty table leaves gateway.toml alone)."""
    engine: Engine = Engine()
    router: str | None = None
    """``smg``: start lh-smg; the ``[gateway]`` table points the gateway at it."""


_REQUIRED = object()


class _Table:
    """One table of the arm file, read key by key; whatever is left over is a misspelling."""

    def __init__(self, table: Mapping[str, Any], where: str) -> None:
        self.rest = dict(table)
        self.where = where

    def get(self, key: str, kind: type | tuple[type, ...], default: Any = _REQUIRED) -> Any:
        if key not in self.rest:
            if default is _REQUIRED:
                raise E2Error(f"{self.where}: {key} is required")
            return default
        value = self.rest.pop(key)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if float in kinds and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
            expected = " or ".join(k.__name__ for k in kinds)
            raise E2Error(f"{self.where}: {key} must be {expected}, not {value!r}")
        return value

    def done(self) -> None:
        if self.rest:
            raise E2Error(f"{self.where}: unknown key(s): {', '.join(sorted(self.rest))}")


def _section[T](top: _Table, name: str, cls: type[T]) -> T:
    """A table of settings, each of the type of its default."""
    table = _Table(top.get(name, dict, {}), f"{top.where} [{name}]")
    defaults = cls()
    values = {
        f.name: table.get(f.name, type(getattr(defaults, f.name)), getattr(defaults, f.name))
        for f in fields(cls)  # type: ignore[arg-type]
    }
    table.done()
    return cls(**values)


def _engine(top: _Table) -> Engine:
    table = _Table(top.get("engine", dict, {}), f"{top.where} [engine]")
    variant = table.get("variant", str, "frozen")
    flags = table.get("flags", list, [])
    table.done()
    for word in [variant, *flags]:
        if not isinstance(word, str) or not _FLAG.fullmatch(word):
            raise E2Error(f"{top.where} [engine]: {word!r} is not a plain word")
    flag_values(flags)  # a value that follows no flag is an error
    return Engine(variant, tuple(flags))


def gpu_nodes(arm: Arm) -> dict[str, str]:
    """The GPU nodes of a point on several, by name, with their engines' URLs (the ``upstream`` of each of
    ``[gateway.nodes]``, the cyan end of its tunnel); empty for a point on one."""
    nodes = arm.gateway.get("nodes")
    if nodes is None:
        return {}
    where = f"{arm.name}.toml [gateway.nodes]"
    if not isinstance(nodes, Mapping) or len(nodes) < 2:
        raise E2Error(f"{where} names two GPU nodes or more, as tables")
    urls = {}
    for name, node in nodes.items():
        url = node.get("upstream") if isinstance(node, Mapping) else None
        if not _NODE.fullmatch(name) or not isinstance(url, str) or not url.startswith("http://"):
            raise E2Error(
                f"{where}: {name!r} needs a name from [A-Za-z0-9._-] and an http:// upstream"
            )
        urls[name] = url.rstrip("/")
    return urls


def load_arm(path: Path) -> Arm:
    """Read and check an arm file; the point is named after the file (``a2-stock-s16.toml``)."""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise E2Error(f"{path}: {exc}") from None
    top = _Table(raw, path.name)
    arm = Arm(
        name=path.stem,
        arm=str(top.get("arm", (int, str))),
        label=top.get("label", str),
        router=top.get("router", str, None),
        voice=_section(top, "voice", Voice),
        agents=_section(top, "agents", Agents),
        gateway=top.get("gateway", dict, {}),
        engine=_engine(top),
    )
    top.done()
    voice, agents = arm.voice, arm.agents
    for ok, what in (
        (_ARM.fullmatch(arm.arm), "arm is E2's arm: 1 to 10, then a letter (7a) or a -suffix"),
        (arm.router in (None, "smg"), "router can only be smg"),
        (voice.callers >= 1 and voice.turns >= 1, "[voice] needs a caller and a turn"),
        (0 <= voice.barge_in <= 1, "[voice] barge_in is a share of turns, 0 to 1"),
        (agents.sessions >= 0, "[agents] sessions cannot be negative"),
        (agents.think_cap > 0 and agents.warmup >= 0, "[agents] think_cap and warmup are seconds"),
    ):
        if not ok:
            raise E2Error(f"{path.name}: {what}")
    if gpu_nodes(arm) and arm.router is not None:
        raise E2Error(f"{path.name}: a router fronts one engine, not several GPU nodes")
    return arm


def flag_values(flags: Sequence[str]) -> dict[str, str]:
    """The ``--flag value`` pairs of an engine command line; a flag without a value is ``true``."""
    values: dict[str, str] = {}
    name = None
    for word in flags:
        if word.startswith("--"):
            if name is not None:
                values[name] = "true"
            name = word
        elif name is not None:
            values[name], name = word, None
        else:
            raise E2Error(f"engine flag value {word!r} follows no flag")
    if name is not None:
        values[name] = "true"
    return values
