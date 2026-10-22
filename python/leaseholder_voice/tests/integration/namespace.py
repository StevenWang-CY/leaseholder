"""The voice plane's network namespace, and the processes started in it.

An unprivileged user + network namespace holds ``lo`` and one dummy interface, and no
route: nothing in it can reach another host or be reached from one. libwebrtc ignores
loopback adapters, so with ``lo`` alone the WebRTC clients gather no ICE candidate and
never connect (measured on cyan, 2026-10-22). The dummy interface gives them an
address that leads nowhere.

A holder process creates the namespace and sleeps; every other process joins it with
``nsenter``. Each process gets its own process group and is killed if the test run
dies (``setpriv --pdeathsig``).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

DUMMY_INTERFACE = "lh0"
DUMMY_ADDRESS = "10.255.0.1"

TOOLS = ("unshare", "nsenter", "setpriv", "ip", "ss")

_SETUP = " && ".join(
    [
        "ip link set lo up",
        f"ip link add {DUMMY_INTERFACE} type dummy",
        f"ip link set {DUMMY_INTERFACE} addrgenmode none",  # IPv4 only, no link-local
        f"ip addr add {DUMMY_ADDRESS}/32 dev {DUMMY_INTERFACE}",
        f"ip link set {DUMMY_INTERFACE} up",
    ]
)


def holder_command() -> list[str]:
    """Create the namespace, set up its interfaces, print ``ready`` and sleep."""
    script = f"{_SETUP} && echo ready && exec sleep infinity"
    return ["unshare", "--user", "--map-root-user", "--net", "--", "sh", "-c", script]


def join_command(holder_pid: int, argv: Sequence[str]) -> list[str]:
    """Run ``argv`` inside the holder's namespace.

    ``--preserve-credentials`` is required: ``--map-root-user`` denies setgroups(2) in
    the namespace, so nsenter's default switch to uid 0 fails with EPERM.
    """
    return [
        "nsenter",
        f"--target={holder_pid}",
        "--user",
        "--net",
        "--preserve-credentials",
        "--",
        *argv,
    ]


def spawn(argv: Sequence[str], *, log: Path | None = None, **kwargs: Any) -> subprocess.Popen:
    """Start ``argv`` in a new process group, killed when this process dies.

    With ``log``, stdout and stderr are appended to that file.
    """
    command = ["setpriv", "--pdeathsig", "KILL", "--", *argv]
    if log is None:
        return subprocess.Popen(command, start_new_session=True, **kwargs)
    with log.open("ab") as output:
        return subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, **kwargs
        )


def stop(process: subprocess.Popen, timeout: float = 15.0) -> None:
    """SIGTERM the process (SIGKILL after ``timeout``), then SIGKILL its leftover children."""
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for child in descendants:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    psutil.wait_procs(descendants, timeout=5)


class Namespace:
    """The holder process of one voice-plane namespace."""

    def __init__(self) -> None:
        self._holder: subprocess.Popen | None = None

    @property
    def holder_pid(self) -> int:
        assert self._holder is not None, "namespace not started"
        return self._holder.pid

    def __enter__(self) -> Namespace:
        self._holder = spawn(holder_command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self._holder.stdout is not None and self._holder.stderr is not None
        line = self._holder.stdout.readline().decode().strip()
        if line != "ready":
            error = self._holder.stderr.read().decode().strip()
            stop(self._holder)
            raise RuntimeError(f"namespace setup failed: {error or line or 'no output'}")
        self._holder.stdout.close()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._holder is not None:
            stop(self._holder, timeout=5)
            self._holder.stderr.close()

    def command(self, argv: Sequence[str]) -> list[str]:
        return join_command(self.holder_pid, argv)

    def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen:
        return spawn(self.command(argv), **kwargs)

    def run(self, argv: Sequence[str], timeout: float = 10.0) -> str:
        """Run a short command inside the namespace and return its stdout."""
        return subprocess.run(
            self.command(argv), capture_output=True, text=True, check=True, timeout=timeout
        ).stdout


def our_pids() -> set[int]:
    """This process and all of its descendants."""
    me = psutil.Process(os.getpid())
    return {me.pid, *(child.pid for child in me.children(recursive=True))}
