"""E2 points (``lh-bench e2``): voice-to-voice latency beside agent load, one arm file per point.

An arm file (TOML) describes one point of E2's frontier: the calls, the agent replay beside them, the
gateway's configuration and the engine the GPU node runs. The point is named after the file.
``lh-bench e2 run`` measures it on cyan, the control host:

1. it records the engine behind the tunnel from the engine's own ``/server_info`` and checks the
   arm's ``[engine]`` against it;
2. it makes cyan's units serve the point: the router (SMG, for arm 8b), the gateway's configuration
   (``gateway.toml`` with the arm's ``[gateway]`` table merged in) and the voice agent's preemptive
   generation and lifecycle events, restarting only the units whose configuration changes;
3. it empties the engine's prefix cache, so that every point starts from the same state, and, with
   agent sessions, starts AIPerf's ``agentic_replay`` (``configs/agentic_replay.yaml``) on the
   load-generator cores and waits for its profiling phase and a warm-up;
4. it snapshots the gateway's and the engine's ``/metrics``, runs ``lh-voice bots`` inside the
   voice-plane namespace (``--save-audio`` has them save each call's audio, for the demo video) and
   snapshots again: the calls are the measurement window;
5. it stops AIPerf, keeps the gateway's request log of the calls and the voice worker's journal of
   them (its turn lines and its sessions' event counts), and writes ``summary.json``.

The GPU node's engine is switched by ``benchmarks/e2/campaign.sh`` (``variant.sh use``), never here.
A failure ends the point: ``run.json`` keeps ``"status": "failed"`` and the error, and every process
the run started is stopped by its process group. ``lh-bench e2 report`` (:mod:`.frontier`) builds the
frontier from the run directories.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import orjson

from leaseholder_bench.frontier import (
    gateway_request_lines,
    render,
    summarize_point,
    worker_lines,
)
from leaseholder_bench.kv_metrics import fetch
from leaseholder_bench.tiers import pool_tokens

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
"""The AIPerf configuration, from the source tree (``uv sync`` installs the package editable)."""

PROFILING = "Credit phase start: profiling"
"""AIPerf's log line when its measured phase begins, after the dataset is built and warmed up."""
PROFILING_TIMEOUT_S = 1200.0
"""Building the full corpus's dataset takes about 4 minutes on cyan; configs/aiperf.env allows 900 s."""
AIPERF_DURATION_S = 10800
"""AIPerf's duration: the replay is stopped when the calls end, so this only has to outlast them."""
AIPERF_STOP_S = 120.0
"""How long AIPerf may take to cancel its requests and write its exports after SIGINT."""
CALL_TURN_S = 130.0
"""The longest a call's turn can take: the line, a reply, a barge-in and a reply, the quiet wait."""
POLL_S = 1.0
"""How often a run checks on the tools and services it waits for."""
SESSIONS_END_S = 30.0
"""How long the calls' sessions may take to log their end once the callers have left: a session ends
when its room is deleted, and its job's shutdown, which sends the last events, may take 10 s."""

SMG = "lh-smg"
GATEWAY = "lh-gateway"
WORKER = "lh-voice-worker"
E2_GATEWAY = "e2-gateway"
NO_PREEMPTIVE = "no-preemptive-generation"
NO_EVENTS = "no-events"
"""Drop-ins under deploy/cyan/variants/ that e2 run links into cyan's units as ``variant-NAME.conf``."""

SGLANG = "sglang"
"""The one engine variant that serves SGLang; every other variant serves vLLM."""

_FLAG = re.compile(r"[A-Za-z0-9._:/=,+-]+")
"""Engine flags travel through ssh and a systemd environment line, so each is one plain word."""
_ARM = re.compile(r"(10|[1-9])[a-z]?(-[a-z0-9]+)*")


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
    """``[agents]``: the AIPerf agentic replay beside the calls; no sessions is voice alone."""

    sessions: int = 0
    """Session trees AIPerf keeps live."""
    target: str = "gateway"
    """``gateway`` (tenant ``agents``, with its key) or ``engine`` (straight through the tunnel)."""
    think_cap: float = 120.0
    """Cap on each recorded think time, in seconds."""
    warmup: float = 60.0
    """Seconds of replay between the start of AIPerf's profiling phase and the first call."""
    data: str = "weka-x32"
    """A ``scale-traces`` directory, relative to ~/leaseholder/data."""
    seed: int = 11
    send_priority: bool = True
    """Send the vLLM ``priority`` field; only reaches an engine with target ``engine``."""


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
        (agents.target in ("gateway", "engine"), "[agents] target is gateway or engine"),
        (agents.think_cap > 0 and agents.warmup >= 0, "[agents] think_cap and warmup are seconds"),
    ):
        if not ok:
            raise E2Error(f"{path.name}: {what}")
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


# cyan


@dataclass(frozen=True)
class Host:
    """cyan (ops/RUNBOOK.md): its service ports, its paths and how to run the tools. The tests swap
    the tools for fakes."""

    home: Path = field(default_factory=Path.home)
    gateway_url: str = "http://127.0.0.1:18080"
    engine_url: str = "http://127.0.0.1:18000"
    """The engine through the tunnel."""
    aiperf_cores: str = "16-23"
    caller_cores: str = "24-31"

    @property
    def root(self) -> Path:
        return self.home / "leaseholder"

    @property
    def config(self) -> Path:
        return self.home / ".config" / "leaseholder"

    @property
    def units(self) -> Path:
        return self.home / ".config" / "systemd" / "user"

    def aiperf(self) -> list[str]:
        """AIPerf from this environment (the bench venv) on the load generators' cores."""
        aiperf = Path(sys.executable).with_name("aiperf")
        config = CONFIGS / "agentic_replay.yaml"
        return ["taskset", "-c", self.aiperf_cores, str(aiperf), "profile", "--config", str(config)]

    def bots(self) -> list[str]:
        """``lh-voice bots`` from the voice venv, inside the voice-plane namespace, on the callers'
        cores."""
        voicenet = self.root / "src" / "deploy" / "bin" / "in-voicenet"
        lh_voice = self.root / "venvs" / "voice" / "bin" / "lh-voice"
        return ["taskset", "-c", self.caller_cores, str(voicenet), str(lh_voice), "bots"]


class Systemd:
    """cyan's user units, through ``systemctl --user`` and ``journalctl --user``; the tests use a
    fake."""

    def __call__(self, *args: str) -> str:
        """``systemctl --user ARGS``'s output; a failure is an error with systemctl's message."""
        done = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)
        if done.returncode:
            reason = done.stderr.strip() or f"exit status {done.returncode}"
            raise E2Error(f"systemctl --user {' '.join(args)}: {reason}")
        return done.stdout

    def active(self, unit: str) -> bool:
        return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0

    def environment(self, unit: str) -> dict[str, str]:
        """The environment systemd gives ``unit`` (its ``Environment=`` settings)."""
        env = {}
        for assignment in shlex.split(self("show", unit, "--property=Environment", "--value")):
            name, _, value = assignment.partition("=")
            env[name] = value
        return env

    def journal(self, unit: str, since: float, until: float | None = None) -> str:
        command = ["journalctl", "--user", "--unit", unit, "--since", f"@{int(since)}", "-o", "cat"]
        if until is not None:
            command += ["--until", f"@{math.ceil(until)}"]
        return subprocess.run(command, capture_output=True, text=True).stdout


def configure_cyan(arm: Arm, host: Host, systemd: Systemd) -> dict[str, Any]:
    """Make cyan's units serve ``arm``, restarting only those whose configuration changes; returns
    what they run, for run.json."""
    env = systemd.environment(WORKER)
    if env.get("LH_TURN_DETECTION") != "vad":
        raise E2Error(
            f"{WORKER} ends turns with LiveKit's end-of-turn model; E2 points need fixed endpointing: "
            "link deploy/cyan/variants/vad-turns (ops/RUNBOOK.md)"
        )
    applied = _apply(host, systemd, arm.router, arm.gateway, arm.voice)
    return applied | {"llm_path": env.get("LH_LLM_UDS")}


def reset_cyan(host: Host, systemd: Systemd) -> dict[str, Any]:
    """Undo what e2 run changes on cyan: SMG stopped, the gateway back on gateway.toml, preemptive
    generation and lifecycle events on."""
    with _exclusive(host):
        return _apply(host, systemd, None, {}, Voice())


def _apply(
    host: Host, systemd: Systemd, router: str | None, overlay: Mapping[str, Any], voice: Voice
) -> dict[str, Any]:
    if router == "smg":
        systemd("start", SMG)  # returns once SMG answers /health
    elif systemd.active(SMG):
        systemd("stop", SMG)
    config = _merge(_read_toml(host.config / "gateway.toml"), overlay)
    gateway = _link(host, GATEWAY, E2_GATEWAY, bool(overlay))
    if overlay:
        gateway |= _write_private(
            host.config / "gateway-e2.toml", GATEWAY_HEADER + toml_text(config)
        )
    links = [
        _link(host, WORKER, NO_PREEMPTIVE, not voice.preemptive_generation),
        _link(host, WORKER, NO_EVENTS, not voice.events),
    ]
    worker = any(links)
    if gateway or worker:
        systemd("daemon-reload")
    if gateway:
        since = time.time()
        try:
            systemd("restart", GATEWAY)  # returns once the gateway answers /health
        except E2Error as exc:
            # A configuration the gateway refuses (a key its build does not know) is in its log.
            logged = "\n".join(systemd.journal(GATEWAY, since).strip().splitlines()[-5:])
            raise E2Error(
                f"{exc}\n{GATEWAY} logged:\n{logged}\n(lh-bench e2 reset puts it back on gateway.toml)"
            ) from None
    if worker:
        since = time.time()
        systemd("restart", WORKER)
        _until(
            lambda: "registered worker" in systemd.journal(WORKER, since),
            90.0,
            f"{WORKER} did not register with LiveKit within 90 s (journalctl --user -u {WORKER})",
        )
    return {
        "router": router,
        "gateway": {"restarted": gateway, "config": _redacted(config)},
        "worker": {
            "restarted": worker,
            "preemptive_generation": voice.preemptive_generation,
            "events": voice.events,
        },
    }


GATEWAY_HEADER = (
    "# Generated by lh-bench e2 run: gateway.toml with an arm file's [gateway] table merged in.\n"
    "# lh-gateway reads it through the e2-gateway drop-in; edit gateway.toml or the arm instead.\n"
)


def _link(host: Host, unit: str, variant: str, wanted: bool) -> bool:
    """Link (or remove) one of deploy/cyan/variants/ into ``unit``; returns whether that changed."""
    link = host.units / f"{unit}.service.d" / f"variant-{variant}.conf"
    target = host.root / "src" / "deploy" / "cyan" / "variants" / variant / f"{unit}.conf"
    if wanted and link.is_symlink() and link.readlink() == target:
        return False
    if not wanted and not (link.is_symlink() or link.exists()):
        return False
    link.unlink(missing_ok=True)
    if wanted:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return True


def _write_private(path: Path, text: str) -> bool:
    """Write ``text`` to a file only its owner can read (it holds key digests); returns whether it
    changed."""
    if path.exists() and path.read_text() == text:
        return False
    staged = path.with_name(path.name + ".new")
    staged.write_text(text)
    staged.chmod(0o600)
    staged.replace(path)
    return True


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise E2Error(f"{path}: {exc}") from None


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overlay`` merged in: tables merge key by key, any other value replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


SECRETS = frozenset({"keys_sha256", "api_key", "admin_api_key", "hf_token"})
"""Settings that hold a credential or its digest: the gateway's key digests, the engines' API keys and
Hugging Face token. A run directory keeps only whether they are set."""


def _redacted(config: Any) -> Any:
    """A configuration (the gateway's, an engine's) without its credentials."""
    if isinstance(config, Mapping):
        return {
            k: (["<redacted>"] * len(v) if isinstance(v, list) else "<redacted>")
            if k in SECRETS and v
            else _redacted(v)
            for k, v in config.items()
        }
    if isinstance(config, list):
        return [_redacted(v) for v in config]
    return config


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def toml_text(table: Mapping[str, Any]) -> str:
    """TOML for a parsed TOML document of strings, numbers, booleans, arrays and tables (all the
    gateway's configuration holds)."""
    return "\n".join(_toml_lines(table, ())) + "\n"


def _toml_lines(table: Mapping[str, Any], path: tuple[str, ...]) -> list[str]:
    def is_tables(value: Any) -> bool:
        return (
            isinstance(value, list) and bool(value) and all(isinstance(v, Mapping) for v in value)
        )

    lines = [
        f"{_toml_key(k)} = {_toml_value(v)}"
        for k, v in table.items()
        if not isinstance(v, Mapping) and not is_tables(v)
    ]
    for key, value in table.items():
        header = ".".join(_toml_key(k) for k in (*path, key))
        if isinstance(value, Mapping):
            lines += ["", f"[{header}]", *_toml_lines(value, (*path, key))]
        elif is_tables(value):
            for item in value:
                lines += ["", f"[[{header}]]", *_toml_lines(item, (*path, key))]
    return lines


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a TOML basic string once DEL, which JSON leaves alone, is escaped.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise E2Error(f"cannot write {value!r} as TOML")


def read_env(path: Path) -> dict[str, str]:
    """The ``NAME=value`` lines of a shell environment file (``export`` and quotes allowed)."""
    env = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if not sep or name.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[name.strip()] = value
    return env


# The engine


@dataclass(frozen=True)
class EngineInfo:
    """The engine's own account of itself, from its ``/server_info``."""

    server: str
    """``vllm`` or ``sglang``."""
    version: str | None
    pool_tokens: int | None
    """The GPU KV pool in tokens."""
    settings: Mapping[str, Any]
    """Settings by option name: vLLM's scheduler and cache configuration, SGLang's server arguments."""
    config: Mapping[str, Any]
    """The whole configuration, for engine.json."""


SETTING_NAMES = {"--scheduling-policy": "policy"}
"""vLLM flags whose setting has another name; any other ``--a-flag`` sets ``a_flag``."""

FROZEN = (
    "--scheduling-policy", "priority",
    "--max-num-batched-tokens", "512",
    "--long-prefill-token-threshold", "256",
)  # fmt: skip
"""The scheduler settings of deploy/phastform/engine.yaml that E2's arms change (a test keeps the two
equal). A vLLM point expects them wherever its flags say nothing else, so it cannot run on what an
earlier point left on the box."""


def describe_engine(url: str) -> EngineInfo:
    """vLLM (``/server_info`` needs VLLM_SERVER_DEV_MODE=1, which lh-engine.service sets) or SGLang."""
    info = _get_json(f"{url}/server_info", params={"config_format": "json"})
    if "vllm_config" in info:
        config = info["vllm_config"]
        settings = {**config.get("cache_config", {}), **config.get("scheduler_config", {})}
        version = _get_json(f"{url}/version").get("version")
        return EngineInfo("vllm", version, pool_tokens(fetch(f"{url}/metrics")), settings, config)
    if "internal_states" in info:
        config = {k: v for k, v in info.items() if k != "internal_states"}
        return EngineInfo(
            SGLANG, info.get("version"), info.get("max_total_num_tokens"), config, config
        )
    raise E2Error(f"{url}/server_info is neither vLLM's nor SGLang's")


def check_engine(engine: Engine, info: EngineInfo) -> dict[str, list[str]]:
    """Check the arm's engine against what the engine reports: its flags and, on vLLM, the frozen
    box's scheduler settings it keeps. A flag whose setting the engine does not report is listed as
    unchecked."""
    expected = SGLANG if engine.variant == SGLANG else "vllm"
    switch = f"switch it with deploy/phastform/variant.sh use {engine}"
    if info.server != expected:
        raise E2Error(f"the engine is {info.server}, the arm file needs {expected}: {switch}")
    wanted = flag_values(engine.flags)
    if expected == "vllm":
        wanted = flag_values(FROZEN) | wanted
    checked, unchecked = [], []
    for flag, value in wanted.items():
        key = SETTING_NAMES.get(flag, flag.removeprefix("--").replace("-", "_"))
        if key not in info.settings:
            unchecked.append(flag)
        elif _same(info.settings[key], value):
            checked.append(flag)
        else:
            reported = info.settings[key]
            raise E2Error(
                f"the engine runs {key} = {reported!r}, the point needs {flag} {value}: {switch}"
            )
    return {"checked": checked, "unchecked": unchecked}


def _same(reported: Any, text: str) -> bool:
    if isinstance(reported, bool):
        return text.lower() == str(reported).lower()
    if isinstance(reported, int | float):
        try:
            return float(text) == float(reported)
        except ValueError:
            return False
    return str(reported) == text


def reset_prefix_cache(url: str, server: str, timeout: float = 120.0) -> None:
    """Empty the engine's prefix cache (and vLLM's CPU tier). Both engines refuse while requests run
    or KV transfers are in flight, so this retries."""
    path = "/reset_prefix_cache?reset_external=true" if server == "vllm" else "/flush_cache"

    def done() -> bool:
        response = httpx.post(f"{url}{path}", timeout=30)
        if server == "vllm":
            return response.status_code == 200 and response.json().get("success") is True
        return response.status_code == 200

    _until(done, timeout, f"the engine did not reset its prefix cache within {timeout:.0f} s")


def _get_json(url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = httpx.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise E2Error(f"GET {url}: {exc}") from None


# Processes


class Process:
    """A tool in its own process group, so that stopping it reaches every process it started:
    AIPerf renames its processes, so they cannot be found by name. Output goes to ``log``."""

    def __init__(
        self, name: str, command: Sequence[str], log: Path, env: Mapping[str, str]
    ) -> None:
        self.name = name
        self.log = log
        with log.open("ab") as out:
            self.popen = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=dict(env),
                start_new_session=True,
            )
        self.status: int | None = None

    def exited(self) -> bool:
        """Whether the tool has exited, without reaping it: its group id stays taken until stop()."""
        if self.status is not None:
            return True
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        return os.waitid(os.P_PID, self.popen.pid, flags) is not None

    def stop(self, grace: float = 0.0) -> int:
        """Interrupt the tool if it still runs, give it ``grace`` seconds to finish, kill what is left
        of its process group and return its exit status."""
        if self.status is None:
            pid = self.popen.pid
            if not self.exited():
                os.kill(pid, signal.SIGINT)
                deadline = time.monotonic() + grace
                while not self.exited() and time.monotonic() < deadline:
                    time.sleep(POLL_S / 5)
            # A group with nothing left but its exited leader answers ESRCH (Linux) or EPERM (macOS).
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)
            self.status = self.popen.wait()
        return self.status

    def failure(self, what: str) -> E2Error:
        """An error naming the tool's exit status and the end of its log."""
        tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-15:])
        return E2Error(f"{self.name} {what} (exit status {self.stop()}); {self.log}:\n{tail}")


def _until(done: Any, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        if time.monotonic() > deadline:
            raise E2Error(message)
        time.sleep(POLL_S)


# A point


def run_point(
    arm_file: Path,
    out: Path,
    host: Host | None = None,
    systemd: Systemd | None = None,
    *,
    save_audio: bool = False,
) -> dict[str, Any]:
    """Measure the point of ``arm_file`` into the directory ``out``; returns the summary. A
    directory with a complete point is never overwritten, and an earlier attempt that did not
    complete is kept beside it. ``save_audio``, a run option rather than the arm's, has the callers
    save each call's audio (``lh-voice bots --save-audio``)."""
    arm = load_arm(arm_file)
    host = host or Host()
    systemd = systemd or Systemd()
    with _exclusive(host):
        _fresh(out)
        shutil.copyfile(arm_file, out / "arm.toml")
        record = _Record(out, arm, host)
        record["save_audio"] = save_audio
        try:
            with contextlib.ExitStack() as running:
                summary = _measure(arm, out, host, systemd, record, running, save_audio)
        except BaseException as exc:
            record.fail(exc)
            raise
        record.finish()
        return summary


def _measure(
    arm: Arm,
    out: Path,
    host: Host,
    systemd: Systemd,
    record: _Record,
    running: contextlib.ExitStack,
    save_audio: bool,
) -> dict[str, Any]:
    info = describe_engine(host.engine_url)
    _write_json(out / "engine.json", _redacted(info.config))
    engine = {"server": info.server, "version": info.version, "pool_tokens": info.pool_tokens}
    record["engine"] = engine  # recorded before the check, which fails on a wrong engine
    record["engine"] = engine | {"flags": check_engine(arm.engine, info)}
    record.say(f"configuring cyan for {arm.name}")
    record["cyan"] = configure_cyan(arm, host, systemd)
    _until(
        lambda: _ready(host.gateway_url),
        60.0,
        f"{host.gateway_url}/ready did not answer within 60 s: the engine or the router is down",
    )
    reset_prefix_cache(host.engine_url, info.server)
    aiperf = _start_aiperf(arm, host, out, record, running) if arm.agents.sessions else None
    start = _snapshot(host, out, "start")
    end = _call(arm, host, out, record, running, aiperf, save_audio)
    _snapshot(host, out, "end")
    record["window"] = {"start": start, "end": end}
    if aiperf is not None:
        record.say("stopping AIPerf")
        record["aiperf"] = {"exit_status": aiperf.stop(AIPERF_STOP_S)}
        if aiperf.status:
            raise aiperf.failure("failed when stopped")
    _gateway_log(systemd, out, start, end)
    _worker_journal(systemd, out, start, arm.voice.callers)
    try:
        metrics = summarize_point(out, start, end, agents=bool(arm.agents.sessions))
    except ValueError as exc:
        raise E2Error(str(exc)) from None
    summary = {"name": arm.name, "arm": arm.arm, "label": arm.label} | metrics
    _write_json(out / "summary.json", summary)
    record.say(render(summary))
    return summary


def _start_aiperf(
    arm: Arm, host: Host, out: Path, record: _Record, running: contextlib.ExitStack
) -> Process:
    """Start the replay and return once it has been profiling for the warm-up."""
    agents = arm.agents
    record.say(f"AIPerf: {agents.sessions} session trees through the {agents.target}")
    aiperf = Process("AIPerf", host.aiperf(), out / "aiperf.log", _aiperf_env(arm, host, out))
    running.callback(aiperf.stop, AIPERF_STOP_S)
    deadline = time.monotonic() + PROFILING_TIMEOUT_S
    while PROFILING not in aiperf.log.read_text(errors="replace"):
        if aiperf.exited():
            raise aiperf.failure("exited before its profiling phase")
        if time.monotonic() > deadline:
            raise aiperf.failure(f"did not start profiling within {PROFILING_TIMEOUT_S:.0f} s")
        time.sleep(POLL_S)
    record.say(f"AIPerf is profiling; {agents.warmup:g} s of warm-up")
    deadline = time.monotonic() + agents.warmup
    while (left := deadline - time.monotonic()) > 0:
        if aiperf.exited():
            raise aiperf.failure("exited during the warm-up")
        time.sleep(min(POLL_S, left))
    return aiperf


def _call(
    arm: Arm,
    host: Host,
    out: Path,
    record: _Record,
    running: contextlib.ExitStack,
    aiperf: Process | None,
    save_audio: bool,
) -> float:
    """Run the calls; returns when they ended. The replay must outlast them."""
    voice = arm.voice
    record.say(f"calls: {voice.callers} callers, {voice.turns} turns each")
    options = {
        "--corpus": host.root / "data" / "callers",
        "--callers": voice.callers,
        "--turns": voice.turns,
        "--barge-in": voice.barge_in,
        "--seed": voice.seed,
        "--out": out / "bots",
    }
    command = [*host.bots(), *(str(word) for option in options.items() for word in option)]
    if save_audio:
        command.append("--save-audio")
    env = os.environ | read_env(host.config / "livekit.env")
    bots = Process("lh-voice bots", command, out / "bots.log", env)
    running.callback(bots.stop, 30.0)
    deadline = time.monotonic() + CALL_TURN_S * voice.turns + 120
    while not bots.exited():
        if aiperf is not None and aiperf.exited():
            raise aiperf.failure("exited while the calls ran, so its load did not cover them")
        if time.monotonic() > deadline:
            raise bots.failure("did not finish in time")
        time.sleep(POLL_S)
    end = time.time()
    if bots.stop() not in (0, 3):  # 3: some turns failed, which the summary counts
        raise bots.failure("failed")
    return end


def _gateway_log(systemd: Systemd, out: Path, start: float, end: float) -> None:
    """Keep the gateway's request lines of the calls (gateway.jsonl), from which the summary takes the
    first bytes exactly; none when the gateway logs text."""
    if lines := gateway_request_lines(systemd.journal(GATEWAY, start, until=end)):
        (out / "gateway.jsonl").write_text("".join(f"{line}\n" for line in lines))


def _worker_journal(systemd: Systemd, out: Path, since: float, calls: int) -> None:
    """Keep the voice worker's journal from the start of the calls (voice-worker.log): a line per
    turn, and one per session as it ends, with its event counts. Waits until every call's session
    has ended, unless the worker logs neither kind of line."""
    deadline = time.monotonic() + SESSIONS_END_S
    while True:
        journal = systemd.journal(WORKER, since, until=time.time())
        turns, sessions = worker_lines(journal)
        if len(sessions) >= calls or not (turns or sessions) or time.monotonic() > deadline:
            break
        time.sleep(POLL_S)
    (out / "voice-worker.log").write_text(journal)


def _ready(url: str) -> bool:
    with contextlib.suppress(httpx.HTTPError):
        return httpx.get(f"{url}/ready", timeout=5).status_code == 200
    return False


def _snapshot(host: Host, out: Path, when: str) -> float:
    """Save the gateway's and the engine's ``/metrics``; returns the time it finished."""
    metrics = out / "metrics"
    metrics.mkdir(exist_ok=True)
    for name, url in (("gateway", host.gateway_url), ("engine", host.engine_url)):
        try:
            (metrics / f"{name}-{when}.prom").write_text(fetch(f"{url}/metrics"))
        except httpx.HTTPError as exc:
            raise E2Error(f"GET {url}/metrics: {exc}") from None
    return time.time()


def _aiperf_env(arm: Arm, host: Host, out: Path) -> dict[str, str]:
    agents = arm.agents
    data = host.root / "data" / agents.data
    if not data.is_dir():
        raise E2Error(f"no trace corpus at {data} (lh-bench scale-traces writes one)")
    if not (CONFIGS / "agentic_replay.yaml").is_file():
        raise E2Error(
            f"no AIPerf configuration in {CONFIGS}: install the package from its source tree"
        )
    url = host.gateway_url if agents.target == "gateway" else host.engine_url
    env = os.environ | read_env(CONFIGS / "aiperf.env")
    env |= {
        "LH_BENCH_URL": f"{url}/v1/chat/completions",
        "LH_BENCH_DATA": str(data),
        "LH_BENCH_SESSIONS": str(agents.sessions),
        "LH_BENCH_DURATION": str(AIPERF_DURATION_S),
        "LH_BENCH_GRACE": "30",
        "LH_BENCH_THINK_CAP": f"{agents.think_cap:g}",
        "LH_BENCH_SEED": str(agents.seed),
        "LH_BENCH_CLASS": "background",
        "LH_BENCH_SEND_PRIORITY": str(agents.send_priority).lower(),
        "LH_BENCH_ARTIFACTS": str(out / "aiperf"),
    }
    if agents.target == "gateway":
        keys = host.config / "gateway-keys.env"
        tenant_key = read_env(keys).get("LH_GATEWAY_AGENTS_KEY") if keys.exists() else None
        if not tenant_key:
            raise E2Error(f"{keys} has no LH_GATEWAY_AGENTS_KEY, the agents tenant's key")
        env["LH_BENCH_API_KEY"] = tenant_key
    return env


@contextlib.contextmanager
def _exclusive(host: Host) -> Iterator[None]:
    """One e2 run (or reset) at a time: they share cyan's units and cores. The lock file names the
    holder's pid, to stop it by."""
    path = host.root / "runs" / ".lh-bench-e2.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Opened to append: "w" would empty the holder's pid before the lock is even tried.
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise E2Error(f"another lh-bench e2 run or reset holds {path}") from None
        lock.truncate(0)
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        yield


def _fresh(out: Path) -> None:
    run = out / "run.json"
    if run.exists():
        if orjson.loads(run.read_bytes()).get("status") == "complete":
            raise E2Error(f"{out} holds a complete point; e2 run never overwrites one")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out.rename(out.with_name(f"{out.name}.incomplete-{stamp}"))
    elif out.exists() and any(out.iterdir()):
        raise E2Error(f"{out} is not empty and holds no point")
    out.mkdir(parents=True, exist_ok=True)


class _Record:
    """run.json, rewritten at each step so that a point that dies keeps its status, and run.log."""

    def __init__(self, out: Path, arm: Arm, host: Host) -> None:
        self.path = out / "run.json"
        self.log = out / "run.log"
        self.data: dict[str, Any] = {
            "kind": "e2-point",
            "name": arm.name,
            "status": "running",
            "host": platform.node(),
            "started": time.time(),
            "arm": asdict(arm),
            "versions": _versions(host),
        }
        self._write()

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._write()

    def say(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {message}"
        with self.log.open("a") as log:
            log.write(line + "\n")
        # The point goes on when the ssh session that started it closes; run.log keeps every line.
        with contextlib.suppress(OSError):
            print(line, file=sys.stderr, flush=True)

    def fail(self, exc: BaseException) -> None:
        error = str(exc) if isinstance(exc, E2Error) else f"{type(exc).__name__}: {exc}"
        self.data |= {"status": "failed", "error": error, "finished": time.time()}
        self._write()
        self.say(f"failed: {error}")

    def finish(self) -> None:
        self.data |= {"status": "complete", "finished": time.time()}
        self._write()

    def _write(self) -> None:
        _write_json(self.path, self.data)


def _versions(host: Host) -> dict[str, str | None]:
    deployed = host.root / "src" / "DEPLOYED"  # deploy/deploy.sh records the commit it synced
    gateway = host.root / "bin" / "lh-gateway"
    lh_gateway = None
    if gateway.exists():
        shown = subprocess.run([gateway, "--version"], capture_output=True, text=True)
        lh_gateway = shown.stdout.strip() or None
    return {
        "source": deployed.read_text().strip() if deployed.exists() else None,
        "leaseholder_bench": metadata.version("leaseholder-bench"),
        "aiperf": metadata.version("aiperf"),
        "lh_gateway": lh_gateway,
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2) + b"\n")
