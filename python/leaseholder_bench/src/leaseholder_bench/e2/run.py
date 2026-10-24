"""``lh-bench e2 run`` measures one E2 point on cyan, the control host, from its arm file:

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

An arm whose ``[gateway]`` names GPU nodes (``[gateway.nodes.<name>]``, with ``[gateway.placement]``)
is a point on several nodes, each running the arm's ``[engine]``: the run starts each node's tunnel
(``lh-tunnel-<name>``), checks, resets and snapshots each node's engine, tells the voice worker the
nodes (``LH_NODES``, so that each call's STT and TTS follow its node), and reports each node and the
total.

The GPU nodes' engines are switched by ``benchmarks/e2/campaign.sh`` (``variant.sh use``), never here.
A failure ends the point: ``run.json`` keeps ``"status": "failed"`` and the error, and every process
the run started is stopped by its process group. ``lh-bench e2 report`` (:mod:`.report`) builds the
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
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import orjson

from leaseholder_bench.e2.arm import (
    NODE_NAME,
    Arm,
    E2Error,
    Engine,
    Voice,
    flag_values,
    gpu_nodes,
    load_arm,
)
from leaseholder_bench.e2.record import gateway_request_lines, render, summarize_point, worker_lines
from leaseholder_bench.kv_metrics import fetch
from leaseholder_bench.tiers import pool_tokens

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
"""The AIPerf configuration, from the source tree (``uv sync`` installs the package editable)."""
VARIANTS = Path(__file__).resolve().parents[5] / "deploy" / "phastform" / "variants"
"""The GPU node's engine variants, from the same source tree: what each adds to the engine's command
line, which the engine check expects."""

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
TUNNEL = "lh-tunnel-{}"
"""A GPU node's tunnel unit on cyan (deploy/cyan/systemd), by the node's name."""
E2_GATEWAY = "e2-gateway"
NO_PREEMPTIVE = "no-preemptive-generation"
NO_EVENTS = "no-events"
"""Drop-ins under deploy/cyan/variants/ that e2 run links into cyan's units as ``variant-NAME.conf``."""
NODES = "nodes"
"""The worker drop-in e2 run writes for a point on several GPU nodes, ``variant-nodes.conf``."""
ENGINE_UP_S = 60.0
"""How long a node's engine may take to answer through a tunnel that was just started."""

SGLANG = "sglang"
"""The one engine variant that serves SGLang; every other variant serves vLLM."""


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
    applied = _apply(host, systemd, arm.router, arm.gateway, arm.voice, tuple(gpu_nodes(arm)))
    return applied | {"llm_path": env.get("LH_LLM_UDS")}


def reset_cyan(host: Host, systemd: Systemd) -> dict[str, Any]:
    """Undo what e2 run changes on cyan: SMG stopped, the gateway back on gateway.toml, preemptive
    generation and lifecycle events on, and the voice worker on one GPU node."""
    with _exclusive(host):
        return _apply(host, systemd, None, {}, Voice(), ())


def _apply(
    host: Host,
    systemd: Systemd,
    router: str | None,
    overlay: Mapping[str, Any],
    voice: Voice,
    nodes: tuple[str, ...],
) -> dict[str, Any]:
    if router == "smg":
        systemd("start", SMG)  # returns once SMG answers /health
    elif systemd.active(SMG):
        systemd("stop", SMG)
    base = _read_toml(host.config / "gateway.toml")
    if nodes:
        base.pop("upstream", None)  # the arm's nodes replace the one engine
    config = _merge(base, overlay)
    gateway = _link(host, GATEWAY, E2_GATEWAY, bool(overlay))
    if overlay:
        gateway |= _write_private(
            host.config / "gateway-e2.toml", GATEWAY_HEADER + toml_text(config)
        )
    links = [
        _link(host, WORKER, NO_PREEMPTIVE, not voice.preemptive_generation),
        _link(host, WORKER, NO_EVENTS, not voice.events),
        _name_nodes(host, nodes),
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
    record = {
        "restarted": worker,
        "preemptive_generation": voice.preemptive_generation,
        "events": voice.events,
    }
    return {
        "router": router,
        "gateway": {"restarted": gateway, "config": _redacted(config)},
        "worker": record | ({"nodes": list(nodes)} if nodes else {}),
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


def _name_nodes(host: Host, nodes: tuple[str, ...]) -> bool:
    """Write the voice worker's drop-in that names the GPU nodes (``LH_NODES``), or remove it for a point
    on one; returns whether that changed."""
    path = host.units / f"{WORKER}.service.d" / f"variant-{NODES}.conf"
    if not nodes:
        if not (path.is_symlink() or path.exists()):
            return False
        path.unlink()
        return True
    text = (
        "# Generated by lh-bench e2 run for a point on several GPU nodes.\n"
        f"[Service]\nEnvironment=LH_NODES={','.join(nodes)}\n"
    )
    if path.is_file() and not path.is_symlink() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.write_text(text)
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


SCHEDULER = "--scheduler-cls"
ADDITIONAL = "--additional-config."
"""vLLM's flags that name the scheduler class and set one key of ``additional_config`` (make-room's
switches, ``--additional-config.make_room.reserve_tokens 0``)."""


def variant_flags(variant: str, variants: Path = VARIANTS) -> tuple[str, ...]:
    """The engine flags a variant gives its engine (frozen: none): those of the command its
    lh-engine.conf starts (the last ``ExecStart=``), from its first flag on, without ``--config`` (the
    frozen box's engine.yaml, whose settings FROZEN has) and without ``$LH_ENGINE_FLAGS``, where
    ``variant.sh use`` puts the arm's flags, after the variant's."""
    if variant == "frozen":
        return ()
    if not (variants / variant).is_dir():
        raise E2Error(f"no engine variant {variant} in {variants}")
    conf = variants / variant / "lh-engine.conf"
    if not conf.exists():
        return ()  # a variant that leaves the engine's command line alone (mps)
    # systemd joins a line that ends in a backslash with the next one.
    lines = conf.read_text().replace("\\\n", " ").splitlines()
    starts = [line.partition("=")[2] for line in lines if line.startswith("ExecStart=")]
    words = shlex.split(starts[-1]) if starts else []
    first = next((i for i, word in enumerate(words) if word.startswith("--")), len(words))
    flags: list[str] = []
    for word in words[first:]:
        if flags and flags[-1] == "--config":
            flags.pop()
        elif not word.startswith("$"):
            flags.append(word)
    return tuple(flags)


def check_engine(
    engine: Engine, info: EngineInfo, variants: Path = VARIANTS
) -> dict[str, list[str]]:
    """Check the arm's engine against what the engine reports. The point expects the engine's command
    line as ``variant.sh use`` makes it: on vLLM the frozen box's scheduler settings, then the
    variant's flags (:func:`variant_flags`), then the arm's, the last of a repeated flag winning. Each
    flag's setting must match; on vLLM the scheduler class (``scheduler_cls``) must be the one
    ``--scheduler-cls`` names, vLLM's own without it, and ``additional_config`` must be exactly what the
    ``--additional-config.KEY`` flags build, so that frozen, make-room and pr54530 engines, and
    make-room with another switch, cannot pass for one another. A flag whose setting the engine does
    not report is listed as unchecked."""
    expected = SGLANG if engine.variant == SGLANG else "vllm"
    switch = f"switch it with deploy/phastform/variant.sh use {engine}"
    if info.server != expected:
        raise E2Error(f"the engine is {info.server}, the arm file needs {expected}: {switch}")
    wanted = flag_values(variant_flags(engine.variant, variants)) | flag_values(engine.flags)
    if expected == "vllm":
        wanted = flag_values(FROZEN) | wanted
    checked, unchecked = [], []
    for flag, value in wanted.items():
        if expected == "vllm" and (flag == SCHEDULER or flag.startswith(ADDITIONAL)):
            continue  # checked as a whole below
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
    if expected == "vllm":
        scheduler = [SCHEDULER] if SCHEDULER in wanted else []
        if "scheduler_cls" not in info.settings:
            unchecked += scheduler
        elif info.settings["scheduler_cls"] != wanted.get(SCHEDULER):
            needs = f"{SCHEDULER} {wanted[SCHEDULER]}" if scheduler else "vLLM's own scheduler"
            raise E2Error(
                f"the engine runs scheduler_cls = {info.settings['scheduler_cls']!r}, the point "
                f"needs {needs}: {switch}"
            )
        else:
            checked += scheduler
        additional = [flag for flag in wanted if flag.startswith(ADDITIONAL)]
        if not isinstance(info.config.get("additional_config"), Mapping):
            unchecked += additional
        else:
            built = _additional_config({flag: wanted[flag] for flag in additional})
            if problem := _differs(info.config["additional_config"], built, ()):
                raise E2Error(f"{problem}: {switch}")
            checked += additional
    return {"checked": checked, "unchecked": unchecked}


def _additional_config(flags: Mapping[str, str]) -> dict[str, Any]:
    """The ``additional_config`` that ``--additional-config.KEY`` flags build: KEY's dots nest tables,
    and each value stays the flag's text."""
    config: dict[str, Any] = {}
    for flag, value in flags.items():
        *tables, key = flag.removeprefix(ADDITIONAL).split(".")
        table = config
        for name in tables:
            table = table.setdefault(name, {})
            if not isinstance(table, dict):
                raise E2Error(f"{flag} sets a key inside a value of another flag")
        if isinstance(table.get(key), dict):
            raise E2Error(f"{flag} sets a value where other flags set keys")
        table[key] = value
    return config


def _differs(reported: Mapping[str, Any], built: Mapping[str, Any], path: tuple[str, ...]) -> str:
    """Where the engine's ``additional_config`` (``reported``) is not the one the point's flags build,
    said for ``e2 run``'s error; empty when it is."""
    for key in [*built, *(key for key in reported if key not in built)]:
        name, flag = ".".join((*path, key)), ADDITIONAL + ".".join((*path, key))
        if key not in reported:
            needs = _flag_line(flag, built[key])
            return f"the engine's additional_config has no {name}, the point needs {needs}"
        runs = f"the engine runs additional_config {name} = {reported[key]!r}"
        if key not in built:
            return f"{runs}, which the point's flags do not set"
        if isinstance(built[key], dict) and isinstance(reported[key], Mapping):
            if problem := _differs(reported[key], built[key], (*path, key)):
                return problem
        elif isinstance(built[key], dict) or not _same(reported[key], built[key]):
            return f"{runs}, the point needs {_flag_line(flag, built[key])}"
    return ""


def _flag_line(flag: str, value: Any) -> str:
    """A built ``additional_config`` value as the flags that set it."""
    if not isinstance(value, dict):
        return f"{flag} {value}"
    return " ".join(_flag_line(f"{flag}.{key}", item) for key, item in value.items())


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


TOOL_MARK = "LH_E2_TOOL"
"""The environment variable that marks every process a tool of e2 run started, its children too."""
NEEDS_WAITID = (
    "lh-bench e2 run needs os.waitid, to watch its tools without reaping them: Linux, or macOS "
    "from Python 3.13"
)


class Process:
    """A tool in its own process group, so that stopping it reaches every process it started:
    AIPerf renames its processes, so they cannot be found by name. Each is also marked in its
    environment, which its children inherit: AIPerf starts some services in sessions of their own,
    out of the group's reach. Output goes to ``log``."""

    def __init__(
        self, name: str, command: Sequence[str], log: Path, env: Mapping[str, str]
    ) -> None:
        self.name = name
        self.log = log
        self.mark = f"{os.getpid()}-{time.monotonic_ns()}"
        with log.open("ab") as out:
            self.popen = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=dict(env) | {TOOL_MARK: self.mark},
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
            kill_marked(self.mark)
            self.status = self.popen.wait()
        return self.status

    def failure(self, what: str) -> E2Error:
        """An error naming the tool's exit status and the end of its log."""
        tail = "\n".join(self.log.read_text(errors="replace").splitlines()[-15:])
        return E2Error(f"{self.name} {what} (exit status {self.stop()}); {self.log}:\n{tail}")


def kill_marked(mark: str | None = None) -> list[int]:
    """Kill the processes whose environment carries ``mark`` under ``TOOL_MARK`` (any mark when
    None), and return their pids. It reads ``/proc``, so it finds none off Linux."""
    killed = []
    for environ in Path("/proc").glob("[0-9]*/environ"):
        try:
            variables = environ.read_bytes().split(b"\0")
        except OSError:  # another user's process, or one that just exited
            continue
        marks = [v.split(b"=", 1)[1] for v in variables if v.startswith(f"{TOOL_MARK}=".encode())]
        if marks and (mark is None or marks[0] == mark.encode()):
            pid = int(environ.parent.name)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
    return killed


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
    gpu_source: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Measure the point of ``arm_file`` into the directory ``out``; returns the summary. A
    directory with a complete point is never overwritten, and an earlier attempt that did not
    complete is kept beside it. ``save_audio``, a run option rather than the arm's, has the callers
    save each call's audio (``lh-voice bots --save-audio``). ``gpu_source`` is each GPU node's
    deployed revision (its ~/leaseholder/src/DEPLOYED, None where it has none), which campaign.sh
    reads over its SSH at the point's start and run.json keeps beside cyan's; None when unknown."""
    if not hasattr(os, "waitid"):
        raise E2Error(NEEDS_WAITID)
    arm = load_arm(arm_file)
    host = host or Host()
    systemd = systemd or Systemd()
    with _exclusive(host):
        _fresh(out)
        shutil.copyfile(arm_file, out / "arm.toml")
        record = _Record(out, arm, host, gpu_source)
        record["save_audio"] = save_audio
        # Under the lock no other run's tools are alive: a marked process is left from a run that
        # was killed before it could stop them.
        if leftovers := kill_marked():
            record["leftovers_killed"] = leftovers
            record.say(f"killed {len(leftovers)} processes left by an earlier run: {leftovers}")
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
    nodes = gpu_nodes(arm)
    engines: dict[str | None, str] = dict(nodes) or {None: host.engine_url}
    if nodes:
        _start_tunnels(nodes, systemd, record)
    servers = _check_engines(arm, engines, out, record)
    record.say(f"configuring cyan for {arm.name}")
    record["cyan"] = configure_cyan(arm, host, systemd)
    _until(
        lambda: _ready(host.gateway_url),
        60.0,
        f"{host.gateway_url}/ready did not answer within 60 s: the engine or the router is down",
    )
    for node, url in engines.items():
        reset_prefix_cache(url, servers[node])
    aiperf = _start_aiperf(arm, host, out, record, running) if arm.agents.sessions else None
    start = _snapshot(host, out, "start", engines)
    end = _call(arm, host, out, record, running, aiperf, save_audio)
    _snapshot(host, out, "end", engines)
    record["window"] = {"start": start, "end": end}
    if aiperf is not None:
        record.say("stopping AIPerf")
        record["aiperf"] = {"exit_status": aiperf.stop(AIPERF_STOP_S)}
        if aiperf.status:
            raise aiperf.failure("failed when stopped")
    _gateway_log(systemd, out, start, end)
    _worker_journal(systemd, out, start, arm.voice.callers)
    try:
        metrics = summarize_point(
            out,
            start,
            end,
            agents=bool(arm.agents.sessions),
            nodes=tuple(nodes),
            warmup_turns=arm.voice.warmup_turns,
        )
    except ValueError as exc:
        raise E2Error(str(exc)) from None
    summary = {"name": arm.name, "arm": arm.arm, "label": arm.label} | metrics
    _write_json(out / "summary.json", summary)
    record.say(render(summary))
    return summary


def _check_engines(
    arm: Arm, engines: Mapping[str | None, str], out: Path, record: _Record
) -> dict[str | None, str]:
    """Record each engine (``engine.json``, ``engine-NODE.json`` on several GPU nodes, and run.json's
    ``engine``, by node on several) and check it against the arm's; returns each engine's server."""
    recorded: dict[str | None, dict[str, Any]] = {}

    def keep(node: str | None, engine: dict[str, Any]) -> None:
        recorded[node] = engine
        record["engine"] = engine if node is None else recorded

    servers = {}
    for node, url in engines.items():
        info = describe_engine(url)
        servers[node] = info.server
        _write_json(out / _engine_file(node, "json"), _redacted(info.config))
        engine = {"server": info.server, "version": info.version, "pool_tokens": info.pool_tokens}
        keep(node, engine)  # recorded before the check, which fails on a wrong engine
        keep(node, engine | {"flags": check_engine(arm.engine, info)})
    return servers


def _start_aiperf(
    arm: Arm, host: Host, out: Path, record: _Record, running: contextlib.ExitStack
) -> Process:
    """Start the replay and return once it has been profiling for the warm-up."""
    agents = arm.agents
    record.say(f"AIPerf: {agents.sessions} session trees through the gateway")
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
    warmup = f" after {voice.warmup_turns} warm-up" if voice.warmup_turns else ""
    record.say(f"calls: {voice.callers} callers, {voice.turns} turns each{warmup}")
    options = {
        "--corpus": host.root / "data" / "callers",
        "--callers": voice.callers,
        "--turns": voice.turns,
        "--barge-in": voice.barge_in,
        "--seed": voice.seed,
        # Only when there are some, as the callers before the option took no such flag.
        **({"--warmup-turns": voice.warmup_turns} if voice.warmup_turns else {}),
        "--out": out / "bots",
    }
    command = [*host.bots(), *(str(word) for option in options.items() for word in option)]
    if save_audio:
        command.append("--save-audio")
    env = os.environ | read_env(host.config / "livekit.env")
    bots = Process("lh-voice bots", command, out / "bots.log", env)
    running.callback(bots.stop, 30.0)
    deadline = time.monotonic() + CALL_TURN_S * (voice.warmup_turns + voice.turns) + 120
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


def _ready(url: str, path: str = "/ready") -> bool:
    with contextlib.suppress(httpx.HTTPError):
        return httpx.get(f"{url}{path}", timeout=5).status_code == 200
    return False


def _start_tunnels(nodes: Mapping[str, str], systemd: Systemd, record: _Record) -> None:
    """Start each GPU node's tunnel (one that runs is left alone; none is stopped after the point) and
    wait until the node's engine answers through it."""
    record.say(f"GPU nodes: {', '.join(nodes)}")
    for node, url in nodes.items():
        systemd("start", TUNNEL.format(node))
        _until(
            lambda url=url: _ready(url, "/health"),
            ENGINE_UP_S,
            f"the engine of GPU node {node} did not answer at {url} within {ENGINE_UP_S:.0f} s: "
            f"is the node running, and {TUNNEL.format(node)} up?",
        )


def _engine_file(node: str | None, suffix: str, when: str | None = None) -> str:
    """An engine's file: ``engine[-NODE][-WHEN].SUFFIX``, without a node for a point's one engine."""
    return "-".join(part for part in ("engine", node, when) if part) + f".{suffix}"


def _snapshot(host: Host, out: Path, when: str, engines: Mapping[str | None, str]) -> float:
    """Save the gateway's and each engine's ``/metrics``; returns the time it finished."""
    metrics = out / "metrics"
    metrics.mkdir(exist_ok=True)
    files = {f"gateway-{when}.prom": host.gateway_url}
    files |= {_engine_file(node, "prom", when): url for node, url in engines.items()}
    for name, url in files.items():
        try:
            (metrics / name).write_text(fetch(f"{url}/metrics"))
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
    keys = host.config / "gateway-keys.env"
    tenant_key = read_env(keys).get("LH_GATEWAY_AGENTS_KEY") if keys.exists() else None
    if not tenant_key:
        raise E2Error(f"{keys} has no LH_GATEWAY_AGENTS_KEY, the agents tenant's key")
    env = os.environ | read_env(CONFIGS / "aiperf.env")
    env |= {
        "LH_BENCH_URL": f"{host.gateway_url}/v1/chat/completions",
        "LH_BENCH_API_KEY": tenant_key,
        "LH_BENCH_DATA": str(data),
        "LH_BENCH_SESSIONS": str(agents.sessions),
        "LH_BENCH_DURATION": str(AIPERF_DURATION_S),
        "LH_BENCH_GRACE": "30",
        "LH_BENCH_THINK_CAP": f"{agents.think_cap:g}",
        "LH_BENCH_SEED": str(agents.seed),
        "LH_BENCH_CLASS": "background",
        "LH_BENCH_ARTIFACTS": str(out / "aiperf"),
    }
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

    def __init__(
        self, out: Path, arm: Arm, host: Host, gpu_source: Mapping[str, str | None] | None
    ) -> None:
        self.path = out / "run.json"
        self.log = out / "run.log"
        self.data: dict[str, Any] = {
            "kind": "e2-point",
            "name": arm.name,
            "status": "running",
            "host": platform.node(),
            "started": time.time(),
            "arm": asdict(arm),
            "versions": _versions(host, gpu_source),
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


def _versions(host: Host, gpu_source: Mapping[str, str | None] | None) -> dict[str, Any]:
    deployed = host.root / "src" / "DEPLOYED"  # deploy/deploy.sh records the commit it synced
    gateway = host.root / "bin" / "lh-gateway"
    lh_gateway = None
    if gateway.exists():
        shown = subprocess.run([gateway, "--version"], capture_output=True, text=True)
        lh_gateway = shown.stdout.strip() or None
    return {
        "source": deployed.read_text().strip() if deployed.exists() else None,
        "gpu_source": None if gpu_source is None else dict(gpu_source),
        "leaseholder_bench": metadata.version("leaseholder-bench"),
        "aiperf": metadata.version("aiperf"),
        "lh_gateway": lh_gateway,
    }


_REVISION = re.compile(r"[A-Za-z0-9._+-]*")
"""A deployed revision as deploy/deploy.sh writes it (``git describe --always --dirty``)."""


def gpu_sources(values: Sequence[str]) -> dict[str, str | None]:
    """``NODE=REVISION`` values (``lh-bench e2 run --gpu-source``) by node, an empty revision None."""
    sources: dict[str, str | None] = {}
    for value in values:
        node, sep, revision = value.partition("=")
        if not sep or not NODE_NAME.fullmatch(node) or not _REVISION.fullmatch(revision):
            raise E2Error(f"--gpu-source {value!r} is not NODE=REVISION")
        if node in sources:
            raise E2Error(f"--gpu-source names {node} twice")
        sources[node] = revision or None
    return sources


def _write_json(path: Path, data: Any) -> None:
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2) + b"\n")
