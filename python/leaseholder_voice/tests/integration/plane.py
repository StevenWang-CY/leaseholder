"""The voice plane under test: a LiveKit server and an agent worker in one namespace.

The mock STT, LLM and TTS stay outside, in the test process, and the agent reaches
them through their Unix sockets, as the deployed worker reaches the control plane and
the tunnels to the GPU node.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit import api

from tests.mocks import OpenAIMocks

from . import sockets
from .agent_app import AGENT_NAME
from .namespace import DUMMY_ADDRESS, DUMMY_INTERFACE, Namespace, our_pids, stop

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
USER_TURN_WAV = PACKAGE_ROOT / "tests" / "data" / "user_turn.wav"
BARGE_IN_WAV = PACKAGE_ROOT / "tests" / "data" / "barge_in.wav"

# Ports inside the namespace, which no other process on the host can see or take.
SIGNAL_PORT = 27880
MEDIA_PORT = 27882
API_KEY = "devkey"

LIVEKIT_CONFIG = """\
port: {signal_port}
bind_addresses: ["127.0.0.1"]
rtc:
  tcp_port: 0  # ICE over TCP listens on every interface whatever the filters say
  udp_port: {media_port}  # one UDP port carries all ICE traffic
  use_external_ip: false
  enable_loopback_candidate: true
  node_ip: 127.0.0.1
  ips:
    includes: [127.0.0.1/32]
keys:
  {api_key}: {api_secret}
logging:
  level: info
"""


@dataclass
class SocketAudit:
    namespace_sockets: list[str]
    interfaces: list[str]
    problems: list[str]

    def __str__(self) -> str:
        lines = [f"interfaces: {', '.join(self.interfaces)}", *self.namespace_sockets]
        lines += [f"PROBLEM: {problem}" for problem in self.problems]
        return "\n".join(lines)


class VoicePlane:
    """Namespace, LiveKit server and agent worker, stopped in reverse order on exit."""

    def __init__(self, run_dir: Path, mocks: OpenAIMocks, livekit_server: str) -> None:
        self.run_dir = run_dir
        self.mocks = mocks
        self.events_dir = run_dir / "agent-events"
        self.url = f"ws://127.0.0.1:{SIGNAL_PORT}"
        self._livekit_server = livekit_server
        self._api_secret = secrets.token_urlsafe(32)
        self._namespace = Namespace()
        self._stack = contextlib.ExitStack()
        self._server: subprocess.Popen | None = None
        self._worker: subprocess.Popen | None = None

    def __enter__(self) -> VoicePlane:
        self.events_dir.mkdir()
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._namespace)
            self._server = self._start_livekit()
            stack.callback(stop, self._server)
            self._worker = self._start_worker()
            stack.callback(stop, self._worker)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stack.close()

    def agent_events(self, room: str) -> list[dict[str, Any]]:
        path = self.events_dir / f"{room}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def run_bot(
        self,
        room: str,
        dispatch: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        timeout: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Run the bot in ``room`` with the agent dispatched by its token; return its events."""
        room_config = api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=json.dumps(dispatch))]
        )
        token = (
            api.AccessToken(API_KEY, self._api_secret)
            .with_identity("user-bot")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .with_room_config(room_config)
            .to_jwt()
        )
        argv = [
            sys.executable, "-m", "tests.integration.bot", "--url", self.url,
            "--user-turn", str(USER_TURN_WAV), "--barge-in", str(BARGE_IN_WAV),
        ]  # fmt: skip
        log = self.run_dir / f"bot-{room}.log"
        with log.open("ab") as stderr:
            bot = self._namespace.spawn(
                argv,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env={**os.environ, "LH_BOT_TOKEN": token},
                cwd=PACKAGE_ROOT,
            )
        assert bot.stdout is not None
        watchdog = threading.Timer(timeout, bot.kill)
        watchdog.start()
        events: list[dict[str, Any]] = []
        try:
            for line in bot.stdout:
                event = json.loads(line)
                events.append(event)
                on_event(event)
            bot.wait(timeout=30)
        finally:
            watchdog.cancel()
            stop(bot)
            bot.stdout.close()
        if bot.returncode != 0 or not events or events[-1]["event"] != "done":
            last = events[-1] if events else None
            raise AssertionError(
                f"bot failed (exit {bot.returncode}, last event {last}):\n{_tail(log)}"
            )
        return events

    def audit(self) -> SocketAudit:
        """Check where every socket of the voice plane is bound, right now."""
        assert self._server is not None and self._worker is not None
        inside = sockets.parse_ss(self._namespace.run(sockets.LISTENING))
        problems = sockets.namespace_problems(
            inside,
            loopback_only={self._server.pid, self._worker.pid},
            internal_address=DUMMY_ADDRESS,
        )
        host = sockets.parse_ss(
            subprocess.run(sockets.LISTENING, capture_output=True, text=True, check=True).stdout
        )
        problems += [f"host namespace: {line}" for line in sockets.host_problems(host, our_pids())]
        interfaces = sockets.parse_links(self._namespace.run(sockets.LINKS))
        if interfaces != {"lo", DUMMY_INTERFACE}:
            problems.append(f"unexpected interfaces in the namespace: {sorted(interfaces)}")
        details = self._namespace.run(["ip", "-d", "-o", "link", "show", DUMMY_INTERFACE])
        if " dummy " not in details:
            problems.append(f"{DUMMY_INTERFACE} is not a dummy interface: {details.strip()}")
        for family in ("-4", "-6"):
            if route := self._namespace.run(["ip", family, "route", "show", "default"]).strip():
                problems.append(f"default route in the namespace: {route}")
        return SocketAudit([s.line for s in inside], sorted(interfaces), problems)

    def _start_livekit(self) -> subprocess.Popen:
        config = self.run_dir / "livekit.yaml"
        config.touch(mode=0o600)
        config.write_text(
            LIVEKIT_CONFIG.format(
                signal_port=SIGNAL_PORT,
                media_port=MEDIA_PORT,
                api_key=API_KEY,
                api_secret=self._api_secret,
            )
        )
        log = self.run_dir / "livekit-server.log"
        server = self._namespace.spawn([self._livekit_server, "--config", str(config)], log=log)
        listening = f"127.0.0.1:{SIGNAL_PORT} "
        self._wait_for(
            lambda: listening in self._namespace.run(sockets.LISTENING), server, log, 15.0
        )
        return server

    def _start_worker(self) -> subprocess.Popen:
        env = {
            **os.environ,
            "LIVEKIT_URL": self.url,
            "LIVEKIT_API_KEY": API_KEY,
            "LIVEKIT_API_SECRET": self._api_secret,
            "LH_STT_UDS": self.mocks.stt_uds,
            "LH_LLM_UDS": self.mocks.llm_uds,
            "LH_TTS_UDS": self.mocks.tts_uds,
            "LH_AGENT_EVENTS": str(self.events_dir),
        }
        app = PACKAGE_ROOT / "tests" / "integration" / "agent_app.py"
        argv = [sys.executable, "-m", "livekit.agents", "start", str(app), "--log-format", "json"]
        log = self.run_dir / "worker.log"
        worker = self._namespace.spawn(argv, log=log, env=env, cwd=PACKAGE_ROOT)
        self._wait_for(lambda: "registered worker" in log.read_text(), worker, log, 60.0)
        return worker

    @staticmethod
    def _wait_for(
        ready: Callable[[], bool], process: subprocess.Popen, log: Path, timeout: float
    ) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            if process.poll() is not None:
                raise RuntimeError(f"{process.args} exited ({process.returncode}):\n{_tail(log)}")
            if time.monotonic() > deadline:
                stop(process)
                raise TimeoutError(f"{process.args} not ready after {timeout} s:\n{_tail(log)}")
            time.sleep(0.1)


def _tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
