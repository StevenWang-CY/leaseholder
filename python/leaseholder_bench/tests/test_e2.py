"""lh-bench e2 run against fakes: the gateway and the engine on loopback, a recording systemctl,
and stand-ins for AIPerf and the callers that run as real processes, so stopping them is real."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import threading
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from leaseholder_bench import e2
from leaseholder_bench.cli import main
from leaseholder_bench.e2 import (
    E2Error,
    Engine,
    EngineInfo,
    Host,
    Process,
    Systemd,
    check_engine,
    flag_values,
    load_arm,
    read_env,
    reset_cyan,
    run_point,
    toml_text,
)

FAKES = Path(__file__).parent / "fakes"
REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "deploy" / "cyan" / "gateway.example.toml"
BENCHMARKS = REPO / "benchmarks" / "e2"
PHASTFORM = REPO / "deploy" / "phastform"

ARM = """\
arm = 2
label = "stock vLLM, 4 agent sessions"

[voice]
callers = 1
turns = 3
preemptive_generation = false
events = false

[agents]
sessions = 4
warmup = 0.2
data = "corpus"

[gateway]
upstream_priority = []

[gateway.admission]
policy = "static-cap"
background_max_in_flight = 2

[engine]
flags = ["--scheduling-policy", "fcfs", "--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "0"]
"""

VOICE_ALONE = 'arm = 1\nlabel = "voice alone"\n'


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e2, "POLL_S", 0.02)
    for name in ("FAKE_AIPERF", "FAKE_BOTS"):
        monkeypatch.delenv(name, raising=False)


# The arm file


def test_an_arm_file_fills_in_the_defaults(tmp_path: Path) -> None:
    arm = load_arm(write(tmp_path, "a1-voice.toml", VOICE_ALONE))
    assert (arm.name, arm.arm, arm.label) == ("a1-voice", "1", "voice alone")
    assert arm.voice == e2.Voice(callers=2, turns=25, barge_in=0.25, preemptive_generation=True)
    assert (arm.agents.sessions, arm.agents.target, arm.agents.think_cap) == (0, "gateway", 120.0)
    assert (arm.gateway, arm.engine, arm.router) == ({}, Engine("frozen", ()), None)


def test_an_arm_file_says_everything_about_its_point(tmp_path: Path) -> None:
    arm = load_arm(write(tmp_path, "a2-stock-s4.toml", ARM))
    assert arm.voice.preemptive_generation is False and arm.voice.barge_in == 0.25
    assert arm.voice.events is False
    assert arm.agents.warmup == 0.2 and isinstance(arm.agents.think_cap, float)
    assert arm.gateway == {
        "upstream_priority": [],
        "admission": {"policy": "static-cap", "background_max_in_flight": 2},
    }
    assert str(arm.engine) == (
        "frozen --scheduling-policy fcfs --max-num-batched-tokens 2048"
        " --long-prefill-token-threshold 0"
    )
    sub = load_arm(write(tmp_path, "a7b.toml", 'arm = "7b"\nlabel = "SGLang"\nrouter = "smg"\n'))
    assert (sub.arm, sub.router) == ("7b", "smg")


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('label = "x"\n', "arm is required"),
        ("arm = 2\n", "label is required"),
        ('arm = 2\nlabel = "x"\ncallers = 2\n', "unknown key(s): callers"),
        ('arm = 2\nlabel = "x"\n[voice]\ncaller = 2\n', "[voice]: unknown key(s): caller"),
        ('arm = 2\nlabel = "x"\n[voice]\nturns = true\n', "turns must be int, not True"),
        ('arm = 2\nlabel = "x"\n[agents]\nwarmup = "60"\n', "warmup must be float"),
        ('arm = 2\nlabel = "x"\n[voice]\npreemptive_generation = 1\n', "must be bool"),
        ('arm = 2\nlabel = "x"\n[voice]\nevents = "off"\n', "events must be bool"),
        ('arm = 11\nlabel = "x"\n', "arm is E2's arm"),
        ('arm = "7z1"\nlabel = "x"\n', "arm is E2's arm"),
        ('arm = 2\nlabel = "x"\nrouter = "nginx"\n', "router can only be smg"),
        ('arm = 2\nlabel = "x"\n[agents]\ntarget = "tunnel"\n', "target is gateway or engine"),
        ('arm = 2\nlabel = "x"\n[voice]\nbarge_in = 1.5\n', "barge_in is a share"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = ["--a b"]\n', "is not a plain word"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = ["512"]\n', "follows no flag"),
        ('arm = 2\nlabel = "x"\n[engine]\nflags = "--a"\n', "flags must be list"),
        ("arm = \n", "a2.toml"),
    ],
)
def test_a_wrong_arm_file_says_what_is_wrong(tmp_path: Path, text: str, error: str) -> None:
    with pytest.raises(E2Error, match=re.escape(error)):
        load_arm(write(tmp_path, "a2.toml", text))


def test_the_campaigns_name_arm_files_that_load() -> None:
    campaigns = sorted(BENCHMARKS.glob("*.campaign"))
    assert campaigns
    for campaign in campaigns:
        lines = (line.split("#")[0].strip() for line in campaign.read_text().splitlines())
        for line in filter(None, lines):
            arm = load_arm(campaign.parent / line)
            variant = arm.engine.variant
            assert variant == "frozen" or (PHASTFORM / "variants" / variant).is_dir(), line
    assert all(load_arm(path) for path in (BENCHMARKS / "arms").glob("*.toml"))


def test_flag_values_pair_each_flag_with_its_value() -> None:
    flags = ["--max-num-seqs", "12", "--no-async-scheduling", "--watermark", "0.1", "--x"]
    assert flag_values(flags) == {
        "--max-num-seqs": "12",
        "--no-async-scheduling": "true",
        "--watermark": "0.1",
        "--x": "true",
    }


# The gateway's configuration and the environment files


def test_the_gateway_configuration_merges_the_arm_and_round_trips(tmp_path: Path) -> None:
    base = tomllib.loads(EXAMPLE.read_text())
    overlay = {
        "upstream": "http://127.0.0.1:18090",
        "upstream_priority": ["smg-header"],
        "admission": {"policy": "flow-control", "ttft_target_ms": {"realtime": 300.0}},
        "tenants": {"agents": {"max_class": "background"}},
    }
    merged = e2._merge(base, overlay)
    assert tomllib.loads(toml_text(merged)) == merged
    assert merged["tenants"]["agents"]["trusted"] is True  # tables merge key by key
    assert merged["tenants"]["agents"]["max_class"] == "background"
    assert merged["listen"] == base["listen"]  # [[listen]] stays an array of tables
    tricky = {
        "text": 'quote " slash \\ tab \t del \x7f ünï',
        "odd key": 1,
        "nested": {"a.b": [1.5, 2]},
    }
    assert tomllib.loads(toml_text(tricky)) == tricky
    redacted = e2._redacted(merged)
    assert redacted["tenants"]["agents"]["keys_sha256"] == ["<redacted>"]
    assert "<sha256" not in json.dumps(redacted)


def test_env_files_read_like_the_shell_reads_them(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "a.env",
        "# comment\nexport A=1\nB=\"two words\"\nC='x'\n\nD=a=b\n# export E=5\nnot an assignment\n",
    )
    assert read_env(path) == {"A": "1", "B": "two words", "C": "x", "D": "a=b"}


# The engine


FROZEN_SCHEDULER = {
    "policy": "priority",
    "max_num_batched_tokens": 512,
    "long_prefill_token_threshold": 256,
}
STOCK_SCHEDULER = {
    "policy": "fcfs",
    "max_num_batched_tokens": 2048,
    "long_prefill_token_threshold": 0,
}


def vllm(**scheduler: Any) -> EngineInfo:
    settings = FROZEN_SCHEDULER | {"watermark": 0.0} | scheduler
    return EngineInfo("vllm", "0.32.0", 36400, settings, {})


def test_the_engine_check_compares_each_flag_with_the_engines_setting() -> None:
    engine = Engine("frozen", ("--max-num-batched-tokens", "1024", "--watermark", "0", "--kv-x"))
    frozen = ["--scheduling-policy", "--max-num-batched-tokens", "--long-prefill-token-threshold"]
    assert check_engine(engine, vllm(max_num_batched_tokens=1024)) == {
        "checked": [*frozen, "--watermark"],
        "unchecked": ["--kv-x"],
    }
    with pytest.raises(E2Error, match="max_num_batched_tokens = 512, the point needs --max-num"):
        check_engine(engine, vllm())
    with pytest.raises(E2Error, match="the engine is vllm, the arm file needs sglang"):
        check_engine(Engine("sglang", ()), vllm())
    sglang = EngineInfo("sglang", "0.5.22", 36400, {"chunked_prefill_size": 512}, {})
    assert check_engine(Engine("sglang", ("--chunked-prefill-size", "512")), sglang) == {
        "checked": ["--chunked-prefill-size"],
        "unchecked": [],
    }


def test_a_vllm_point_needs_the_frozen_settings_its_flags_keep() -> None:
    with pytest.raises(E2Error, match="policy = 'fcfs', the point needs --scheduling-policy prio"):
        check_engine(Engine("frozen", ()), vllm(**STOCK_SCHEDULER))  # left over from arm 2
    assert check_engine(Engine("pr54530", ()), vllm())["unchecked"] == []
    engine_yaml = (PHASTFORM / "engine.yaml").read_text()
    for flag, value in flag_values(e2.FROZEN).items():
        assert re.search(rf"^{flag.removeprefix('--')}: {value}$", engine_yaml, re.M), flag


# Processes


def test_stopping_a_tool_kills_its_whole_process_group(tmp_path: Path) -> None:
    artifacts = tmp_path / "aiperf"
    env = os.environ | {"LH_BENCH_ARTIFACTS": str(artifacts), "FAKE_AIPERF": "stubborn"}
    tool = Process("AIPerf", [sys.executable, str(FAKES / "fake_aiperf.py")], tmp_path / "log", env)
    child = int(wait_for(artifacts / "child.pid").read_text())
    assert not tool.exited()
    assert tool.stop(grace=0.3) == -9  # it ignored SIGINT, so it was killed
    assert gone(child)  # and so was the process it started
    assert tool.exited() and tool.stop() == -9


def wait_for(path: Path, timeout: float = 10.0) -> Path:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)
    return path


def gone(pid: int, timeout: float = 5.0) -> bool:
    """Whether ``pid`` exits within ``timeout``: a killed orphan lasts until init reaps it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


# A point, against fakes


class Service(ThreadingHTTPServer):
    """An HTTP service on loopback; each route answers from the number of requests it has had."""

    def __init__(self, routes: dict[str, Callable[[int], tuple[int, Any]]]) -> None:
        self.routes = routes
        self.hits: Counter[str] = Counter()
        super().__init__(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: Service

    def answer(self) -> None:
        path = self.path.split("?")[0]
        self.server.hits[path] += 1
        route = self.server.routes.get(path)
        status, body = route(self.server.hits[path]) if route else (404, "")
        data = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = answer

    def log_message(self, *args: object) -> None:
        pass


def gateway_metrics(n: int) -> tuple[int, str]:
    return 200, (
        f'lh_gateway_requests_total{{class="realtime",status="200"}} {10 * n}\n'
        f'lh_gateway_first_byte_seconds_bucket{{class="realtime",le="0.1"}} {8 * n}\n'
        f'lh_gateway_first_byte_seconds_bucket{{class="realtime",le="0.5"}} {10 * n}\n'
        f'lh_gateway_first_byte_seconds_bucket{{class="realtime",le="+Inf"}} {10 * n}\n'
        f'lh_gateway_first_byte_seconds_sum{{class="realtime"}} {0.9 * n}\n'
        f'lh_gateway_first_byte_seconds_count{{class="realtime"}} {10 * n}\n'
    )


def engine_metrics(n: int) -> tuple[int, str]:
    return 200, (
        f'vllm:num_preemptions_total{{engine="0"}} {n}\n'
        f'vllm:prompt_tokens_by_source_total{{engine="0",source="local_compute"}} {100 * n}\n'
        'vllm:cache_config_info{block_size="16",num_gpu_blocks="2275"} 1\n'
    )


@dataclass(frozen=True)
class FakeHost(Host):
    def aiperf(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_aiperf.py")]

    def bots(self) -> list[str]:
        return [sys.executable, str(FAKES / "fake_bots.py")]


class FakeSystemd(Systemd):
    """Records systemctl calls; the worker runs with fixed endpointing and registers at once. A unit
    in ``failing`` does not start, and its log says why. ``journals`` gives a unit's journal from
    ``since`` to ``until``; any other unit's holds only the worker's registration."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.running: set[str] = set()
        self.failing: set[str] = set()
        self.journals: dict[str, Callable[[float, float | None], str]] = {}
        self.env = env or {"LH_TURN_DETECTION": "vad", "LH_LLM_UDS": "/run/user/1/lh/gateway.sock"}

    def __call__(self, *args: str) -> str:
        if args[0] == "show":
            return " ".join(f"{k}={v}" for k, v in self.env.items())
        self.calls.append(args)
        if args[0] in ("start", "restart") and args[1] in self.failing:
            raise E2Error(f"systemctl --user {' '.join(args)}: Job for {args[1]}.service failed")
        if args[0] == "start":
            self.running.add(args[1])
        elif args[0] == "stop":
            self.running.discard(args[1])
        return ""

    def active(self, unit: str) -> bool:
        return unit in self.running

    def journal(self, unit: str, since: float, until: float | None = None) -> str:
        if unit in self.failing:
            return f"{unit}: starting\n{unit}: config: unknown field `upstream_priority`\n"
        if unit in self.journals:
            return self.journals[unit](since, until)
        return "registered worker"


@dataclass
class Cyan:
    host: FakeHost
    systemd: FakeSystemd
    gateway: Service
    engine: Service
    scheduler: dict[str, Any]
    """What the engine's /server_info reports, as variant.sh use would leave it: the frozen box."""
    arms: Path


@pytest.fixture
def cyan(tmp_path: Path) -> Iterator[Cyan]:
    home = tmp_path / "home"
    config = home / ".config" / "leaseholder"
    config.mkdir(parents=True)
    (config / "gateway.toml").write_text(EXAMPLE.read_text())
    (config / "gateway-keys.env").write_text("LH_GATEWAY_AGENTS_KEY=agents-key\n")
    (config / "livekit.env").write_text("LIVEKIT_API_KEY=k\nLIVEKIT_API_SECRET=s\n")
    (home / "leaseholder" / "data" / "corpus").mkdir(parents=True)
    scheduler = dict(FROZEN_SCHEDULER)
    vllm_config = {"scheduler_config": scheduler, "model_config": {"hf_token": "hf_secret"}}
    gateway = Service({"/metrics": gateway_metrics, "/ready": lambda n: (200, "")})
    engine = Service(
        {
            "/metrics": engine_metrics,
            "/server_info": lambda n: (200, {"vllm_config": vllm_config}),
            "/version": lambda n: (200, {"version": "0.32.0"}),
            # Busy at first, as vLLM is while KV transfers finish.
            "/reset_prefix_cache": lambda n: (200, {"success": n > 1}),
        }
    )
    host = FakeHost(home=home, gateway_url=gateway.url, engine_url=engine.url)
    arms = tmp_path / "arms"
    arms.mkdir()
    yield Cyan(host, FakeSystemd(), gateway, engine, scheduler, arms)
    gateway.shutdown()
    engine.shutdown()


def run(cyan: Cyan, name: str, text: str, out: str | None = None) -> dict[str, Any]:
    arm_file = write(cyan.arms, f"{name}.toml", text)
    return run_point(arm_file, cyan.host.root / "runs" / (out or name), cyan.host, cyan.systemd)


def run_stock(cyan: Cyan, out: str | None = None) -> dict[str, Any]:
    """The stock arm, on an engine switched to it as variant.sh use would."""
    cyan.scheduler.update(STOCK_SCHEDULER)
    return run(cyan, "a2-stock-s4", ARM, out)


def record(cyan: Cyan, name: str) -> dict[str, Any]:
    return json.loads((cyan.host.root / "runs" / name / "run.json").read_text())


def test_voice_alone_runs_the_calls_against_the_box_as_it_is(cyan: Cyan) -> None:
    summary = run(cyan, "a1-voice", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    assert cyan.systemd.calls == []  # nothing to change, so nothing restarts
    assert cyan.engine.hits["/reset_prefix_cache"] == 2  # retried until the engine could
    assert not (out / "aiperf.log").exists()
    run_json = record(cyan, "a1-voice")
    assert run_json["status"] == "complete"
    frozen = ["--scheduling-policy", "--max-num-batched-tokens", "--long-prefill-token-threshold"]
    assert run_json["engine"] == {
        "server": "vllm",
        "version": "0.32.0",
        "pool_tokens": 36400,
        "flags": {"checked": frozen, "unchecked": []},
    }
    engine = json.loads((out / "engine.json").read_text())
    assert engine["scheduler_config"] == FROZEN_SCHEDULER
    assert engine["model_config"]["hf_token"] == "<redacted>"
    assert run_json["cyan"]["llm_path"].endswith("gateway.sock")
    assert run_json["versions"]["aiperf"] == "0.13.0"
    assert (out / "arm.toml").read_text().startswith("arm = 1")
    args = json.loads((out / "bots.args").read_text())
    assert args[args.index("--turns") + 1] == "25" and args[args.index("--callers") + 1] == "2"
    assert summary["name"] == "a1-voice" and summary["background"] is None
    assert summary["voice"]["v2v_ms"]["p95"] == 1800.0
    assert summary["engine"]["preemptions"] == 1  # one scrape apart
    # This gateway logs no JSON request lines, so the first byte comes from its histogram.
    assert not (out / "gateway.jsonl").exists()
    assert summary["gateway"]["realtime_first_byte_ms"] == {
        "n": 10, "mean": 90.0, "p50": 62.5, "p95": 400.0, "p99": 480.0, "source": "histogram",
    }  # fmt: skip
    assert summary["worker"] is None  # nor does this worker log turn lines
    assert json.loads((out / "summary.json").read_text()) == summary


def test_a_point_with_agents_configures_cyan_and_stops_aiperf_after_the_calls(cyan: Cyan) -> None:
    summary = run_stock(cyan)
    out = cyan.host.root / "runs" / "a2-stock-s4"
    units = cyan.host.units
    gateway_link = units / "lh-gateway.service.d" / "variant-e2-gateway.conf"
    worker = units / "lh-voice-worker.service.d"
    assert gateway_link.readlink().parts[-2:] == ("e2-gateway", "lh-gateway.conf")
    assert {link.readlink().parts[-2] for link in worker.glob("variant-*.conf")} == {
        "no-preemptive-generation",
        "no-events",
    }
    assert (REPO / "deploy" / "cyan" / "variants" / "no-events" / "lh-voice-worker.conf").exists()
    assert cyan.systemd.calls == [  # the worker restarts once for both of its drop-ins
        ("daemon-reload",),
        ("restart", "lh-gateway"),
        ("restart", "lh-voice-worker"),
    ]
    generated = cyan.host.config / "gateway-e2.toml"
    assert stat.S_IMODE(generated.stat().st_mode) == 0o600  # it holds the key digests
    config = tomllib.loads(generated.read_text())
    assert config["upstream_priority"] == [] and config["admission"]["policy"] == "static-cap"
    assert config["tenants"]["agents"]["keys_sha256"] == ["<sha256 of the agents key>"]
    env = json.loads((out / "aiperf" / "env.json").read_text())
    assert env["LH_BENCH_URL"] == f"{cyan.gateway.url}/v1/chat/completions"
    assert env["LH_BENCH_API_KEY"] == "agents-key"
    assert (env["LH_BENCH_SESSIONS"], env["LH_BENCH_SEND_PRIORITY"]) == ("4", "true")
    assert env["AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"] == "false"  # from configs/aiperf.env
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["aiperf"] == {"exit_status": 0}
    assert run_json["cyan"]["worker"] == {
        "restarted": True,
        "preemptive_generation": False,
        "events": False,
    }
    assert "agents-key" not in (out / "run.json").read_text()
    assert run_json["cyan"]["gateway"]["config"]["tenants"]["agents"]["keys_sha256"] == [
        "<redacted>"
    ]
    assert "Exporting all records" in (out / "aiperf.log").read_text()  # SIGINT let it finish
    background = summary["background"]
    assert background["requests"] > 0 and background["errors"] == 0
    assert background["tokens_per_s"] == pytest.approx(
        background["requests_per_s"] * 1010, rel=0.01
    )
    assert 100 <= background["ttft_ms"]["p50"] <= 109
    window = run_json["window"]
    assert 0.3 <= window["end"] - window["start"] < 5


def test_the_same_arm_again_restarts_nothing(cyan: Cyan) -> None:
    run_stock(cyan)
    cyan.systemd.calls.clear()
    run_stock(cyan, out="again")
    assert cyan.systemd.calls == []
    cyan.scheduler.update(FROZEN_SCHEDULER)
    run(cyan, "a1-voice", VOICE_ALONE)  # and back: the drop-ins go, the units restart
    assert cyan.systemd.calls == [
        ("daemon-reload",),
        ("restart", "lh-gateway"),
        ("restart", "lh-voice-worker"),
    ]
    assert not any(cyan.host.units.glob("*.service.d/variant-*.conf"))


def logged_request(
    at: float, slo_class: str | None, first_byte_ms: float | None, **fields: Any
) -> str:
    """One of the gateway's JSON request lines (log_format = "json"), for a request that ended at
    ``at``; like the gateway, it leaves out the fields it has no value for."""
    stamp = (
        datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    line = {
        "timestamp": stamp,
        "level": "INFO",
        "message": "request",
        "class": slo_class,
        "route": "chat_completions",
        "status": 200,
        "outcome": "completed",
        "first_byte_ms": first_byte_ms,
        "total_ms": 2000.0,
        "target": "lh_gateway::request",
    }
    return json.dumps({k: v for k, v in (line | fields).items() if v is not None})


def worker_line(name: str, message: str, **fields: Any) -> str:
    """One of the voice worker's JSON log lines, as LiveKit's formatter writes them."""
    line = {"message": message, "level": "INFO", "name": name, **fields, "pid": 7}
    return json.dumps(line | {"timestamp": "2026-10-23T05:00:00+00:00"})


TURNS, SESSIONS = "leaseholder_voice.turns", "leaseholder_voice.events"


def test_a_point_keeps_the_gateways_requests_and_the_workers_lines(cyan: Cyan) -> None:
    def gateway(since: float, until: float | None) -> str:
        assert until is not None and since < until  # the window of the calls
        first_bytes = (120.0, 180.0, 240.0)
        return "\n".join(
            [
                logged_request(since - 0.5, "realtime", 999.0),  # in journalctl's second, earlier
                *(
                    logged_request(since + n / 100, "realtime", b)
                    for n, b in enumerate(first_bytes, 1)
                ),
                logged_request(since + 0.05, "background", 3000.0),
                logged_request(since + 0.06, None, None, route="events", status=202, outcome=None),
                '{"timestamp":"2026-10-23T05:00:00.000000Z","message":"lifecycle event"}',
                "lh-gateway serving",
            ]
        )

    reads: list[float | None] = []
    counts = {"turns": 1, "events_sent": 10, "events_dropped": 1, "events_failed": 0}
    ended = [worker_line(SESSIONS, "session ended", session_id=f"s{n}", **counts) for n in (0, 1)]

    def worker(since: float, until: float | None) -> str:
        reads.append(until)
        return "\n".join(
            [
                "registered worker",
                worker_line(TURNS, "turn", session_id="s0", turn=0, e2e_ms=1000.0),
                worker_line(
                    TURNS, "turn", session_id="s1", turn=0, llm_ttft_ms=50.0, e2e_ms=1200.0
                ),
                *(ended if len(reads) >= 3 else []),  # the sessions end a little after the calls
            ]
        )

    cyan.systemd.journals = {"lh-gateway": gateway, "lh-voice-worker": worker}
    summary = run(cyan, "a1-voice", VOICE_ALONE)
    out = cyan.host.root / "runs" / "a1-voice"
    kept = [json.loads(line) for line in (out / "gateway.jsonl").read_text().splitlines()]
    assert len(kept) == 6 and {line["message"] for line in kept} == {"request"}
    # Exact, from the three realtime requests that ended during the calls.
    assert summary["gateway"]["realtime_first_byte_ms"] == {
        "n": 3, "mean": 180.0, "p50": 180.0, "p95": 234.0, "p99": 238.8, "source": "log",
    }  # fmt: skip
    assert summary["gateway"]["background_first_byte_ms"]["p50"] == 3000.0
    assert len(reads) == 3, "it waited for both calls' sessions to end"
    assert "session ended" in (out / "voice-worker.log").read_text()
    worker_summary = summary["worker"]
    assert (worker_summary["sessions"], worker_summary["turns"]) == (2, 2)
    assert worker_summary["events"] == {"sent": 20, "dropped": 2, "failed": 0}
    assert worker_summary["waterfall"]["e2e_ms"]["p50"] == 1100.0
    assert worker_summary["waterfall"]["llm_ttft_ms"]["n"] == 1
    assert worker_summary["waterfall"]["end_of_turn_ms"] == {"n": 0}


def test_the_worker_journal_waits_only_for_sessions_that_end(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(e2, "SESSIONS_END_S", 0.2)
    reads: list[float | None] = []

    def journal(*lines: str) -> Callable[[float, float | None], str]:
        def read(since: float, until: float | None) -> str:
            reads.append(until)
            return "\n".join(["registered worker", *lines])

        return read

    # A worker that logs no turn lines (a build from before them): nothing to wait for.
    cyan.systemd.journals = {"lh-voice-worker": journal()}
    assert run(cyan, "a1-voice", VOICE_ALONE)["worker"] is None
    assert len(reads) == 1
    # Two calls and one session that ends: the wait is bounded, and the summary says what came.
    turn = worker_line(TURNS, "turn", session_id="s", turn=0, e2e_ms=900.0)
    cyan.systemd.journals = {
        "lh-voice-worker": journal(turn, worker_line(SESSIONS, "session ended", session_id="s"))
    }
    reads.clear()
    summary = run(cyan, "a1-voice", VOICE_ALONE, out="again")
    assert len(reads) >= 3
    assert (summary["worker"]["sessions"], summary["worker"]["events"]) == (1, None)


def test_smg_runs_only_for_the_points_that_route_through_it(cyan: Cyan) -> None:
    run(cyan, "a8b", 'arm = "8b"\nlabel = "SMG"\nrouter = "smg"\n')
    assert ("start", "lh-smg") in cyan.systemd.calls and "lh-smg" in cyan.systemd.running
    run(cyan, "a1-voice", VOICE_ALONE)
    assert "lh-smg" not in cyan.systemd.running


def test_reset_puts_cyan_back(cyan: Cyan) -> None:
    run_stock(cyan)
    cyan.systemd.running.add("lh-smg")
    cyan.systemd.calls.clear()
    applied = reset_cyan(cyan.host, cyan.systemd)
    assert applied["gateway"]["restarted"] and applied["worker"]["restarted"]
    assert ("stop", "lh-smg") in cyan.systemd.calls
    assert not any(cyan.host.units.glob("*.service.d/variant-*.conf"))


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("exit-early", "AIPerf exited before its profiling phase (exit status 3)"),
        ("exit-later", "AIPerf exited while the calls ran, so its load did not cover them"),
    ],
)
def test_aiperf_failing_fails_the_point_and_stops_everything(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch, mode: str, error: str
) -> None:
    monkeypatch.setenv("FAKE_AIPERF", mode)
    monkeypatch.setenv("FAKE_BOTS_SECONDS", "5")
    with pytest.raises(E2Error, match=re.escape(error)):
        run_stock(cyan)
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["status"] == "failed" and error in run_json["error"]
    assert "failed: AIPerf" in (cyan.host.root / "runs" / "a2-stock-s4" / "run.log").read_text()
    bots_pid = cyan.host.root / "runs" / "a2-stock-s4" / "bots.pid"
    if bots_pid.exists():  # the calls had started: they were stopped too
        assert gone(int(bots_pid.read_text()))


def test_callers_crashing_fail_the_point_and_stop_aiperf(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_BOTS", "crash")
    with pytest.raises(E2Error, match=r"lh-voice bots failed \(exit status 1\)"):
        run_stock(cyan)
    out = cyan.host.root / "runs" / "a2-stock-s4"
    assert "Exporting all records" in (out / "aiperf.log").read_text()  # AIPerf got its SIGINT
    assert record(cyan, "a2-stock-s4")["status"] == "failed"


def test_a_wrong_engine_fails_before_anything_starts(cyan: Cyan) -> None:
    cyan.scheduler.update(STOCK_SCHEDULER)
    error = (
        "the engine runs max_num_batched_tokens = 2048, the point needs --max-num-batched-tokens"
        " 1024: switch it with deploy/phastform/variant.sh use frozen --scheduling-policy fcfs"
        " --max-num-batched-tokens 1024 --long-prefill-token-threshold 0"
    )
    with pytest.raises(E2Error, match=re.escape(error)):
        run(cyan, "a4", ARM.replace('"2048"', '"1024"'))
    out = cyan.host.root / "runs" / "a4"
    assert cyan.systemd.calls == [] and not (out / "aiperf.log").exists()
    run_json = record(cyan, "a4")
    assert run_json["status"] == "failed" and run_json["engine"]["server"] == "vllm"
    assert (out / "engine.json").exists()


def test_a_gateway_that_refuses_its_configuration_says_why(cyan: Cyan) -> None:
    cyan.systemd.failing.add("lh-gateway")
    with pytest.raises(E2Error, match="unknown field `upstream_priority`") as failed:
        run_stock(cyan)
    assert "Job for lh-gateway.service failed" in str(failed.value)
    assert "lh-bench e2 reset puts it back on gateway.toml" in str(failed.value)
    run_json = record(cyan, "a2-stock-s4")
    assert run_json["status"] == "failed" and "unknown field" in run_json["error"]
    assert not (cyan.host.root / "runs" / "a2-stock-s4" / "aiperf.log").exists()


def test_the_window_needs_fixed_endpointing(cyan: Cyan) -> None:
    cyan.systemd.env = {"LH_LLM_UDS": "/run/user/1/lh/gateway.sock"}
    with pytest.raises(E2Error, match="link deploy/cyan/variants/vad-turns"):
        run(cyan, "a1-voice", VOICE_ALONE)


def test_a_complete_point_is_kept_and_an_incomplete_one_moved_aside(
    cyan: Cyan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_BOTS", "crash")
    with pytest.raises(E2Error):
        run(cyan, "a1-voice", VOICE_ALONE)
    monkeypatch.delenv("FAKE_BOTS")
    run(cyan, "a1-voice", VOICE_ALONE)
    runs = cyan.host.root / "runs"
    [earlier] = runs.glob("a1-voice.incomplete-*")
    assert json.loads((earlier / "run.json").read_text())["status"] == "failed"
    with pytest.raises(E2Error, match="holds a complete point"):
        run(cyan, "a1-voice", VOICE_ALONE)
    (runs / "other").mkdir()
    (runs / "other" / "notes.txt").write_text("mine")
    with pytest.raises(E2Error, match="is not empty and holds no point"):
        run(cyan, "a1-voice", VOICE_ALONE, out="other")


def test_one_point_at_a_time(cyan: Cyan) -> None:
    lock = cyan.host.root / "runs" / ".lh-bench-e2.lock"
    with e2._exclusive(cyan.host):
        with pytest.raises(E2Error, match="another lh-bench e2 run"):
            run(cyan, "a1-voice", VOICE_ALONE)
        assert lock.read_text() == f"{os.getpid()}\n"  # the holder's, kept through the refusal


def test_the_engine_command_prints_what_variant_sh_use_takes(tmp_path: Path, capsys) -> None:
    path = write(tmp_path, "a7a.toml", 'arm = "7a"\nlabel = "x"\n[engine]\nvariant = "pr54530"\n')
    assert main(["e2", "engine", str(path)]) == 0
    assert capsys.readouterr().out == "pr54530\n"
    assert main(["e2", "engine", str(write(tmp_path, "bad.toml", "arm = 2\n"))]) == 1
    assert "label is required" in capsys.readouterr().err


def test_the_record_keeps_a_failed_step_even_when_the_engine_is_down(cyan: Cyan) -> None:
    host = replace(cyan.host, engine_url="http://127.0.0.1:9")  # nothing listens there
    with pytest.raises(E2Error, match="/server_info"):
        run_point(
            write(cyan.arms, "a1.toml", VOICE_ALONE), host.root / "runs" / "a1", host, cyan.systemd
        )
    assert record(cyan, "a1")["status"] == "failed"
