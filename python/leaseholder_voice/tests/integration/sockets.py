"""Socket audit with ss(8) and ip(8).

Inside the voice-plane namespace, every listening TCP socket and every UDP socket must
be on loopback, except the WebRTC clients' ICE sockets, which may also use the
namespace's dummy address. The LiveKit server and the agent worker's HTTP server must be
on loopback only. In the host's namespace, the test's processes must own no TCP or UDP
socket at all: the mocks listen on Unix sockets and everything else runs inside.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable
from dataclasses import dataclass

LISTENING = ["ss", "-H", "-l", "-n", "-t", "-u", "-p"]  # = ss -lntup without the header
LINKS = ["ip", "-o", "link", "show"]

_PID = re.compile(r"pid=(\d+)")


@dataclass(frozen=True)
class Socket:
    netid: str
    address: str
    port: str
    pids: frozenset[int]
    line: str

    @property
    def loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.address).is_loopback
        except ValueError:  # "*"
            return False


def parse_ss(output: str) -> list[Socket]:
    """Parse ``ss -H -lntup`` output."""
    sockets = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        address, _, port = fields[4].rpartition(":")
        address = address.split("%")[0].strip("[]")  # "[fe80::1]%eth0" -> "fe80::1"
        pids = frozenset(int(pid) for pid in _PID.findall(line))
        sockets.append(Socket(fields[0], address, port, pids, " ".join(fields)))
    return sockets


def parse_links(output: str) -> set[str]:
    """Interface names from ``ip -o link show``."""
    names = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names.add(fields[1].rstrip(":").split("@")[0])
    return names


def namespace_problems(
    sockets: Iterable[Socket], *, loopback_only: set[int], internal_address: str
) -> list[str]:
    """Sockets inside the namespace that are not where they should be.

    ``loopback_only`` are the PIDs (LiveKit server, worker) that may not even use the
    namespace-internal address.
    """
    problems = []
    for socket in sockets:
        if socket.loopback:
            continue
        if socket.address == internal_address and not socket.pids & loopback_only:
            continue
        problems.append(socket.line)
    return problems


def host_problems(sockets: Iterable[Socket], ours: set[int]) -> list[str]:
    """Sockets in the host's namespace owned by the test's processes."""
    return [socket.line for socket in sockets if socket.pids & ours]
