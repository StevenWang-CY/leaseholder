"""The socket audit's rules, on ss output recorded on cyan (2026-10-22)."""

from tests.integration.sockets import (
    host_problems,
    namespace_problems,
    parse_links,
    parse_ss,
)

SERVER, CLIENT = 3079590, 3079536

# Inside the namespace: livekit-server on loopback, the client's ICE sockets on the dummy.
INSIDE = """\
udp UNCONN 0 0 10.255.0.1:49031 0.0.0.0:* users:(("python",pid=3079536,fd=28))
udp UNCONN 0 0 127.0.0.1:38963 0.0.0.0:* users:(("livekit-server",pid=3079590,fd=3))
tcp LISTEN 0 5 10.255.0.1:48505 0.0.0.0:* users:(("python",pid=3079536,fd=34))
tcp LISTEN 0 128 127.0.0.1:51683 0.0.0.0:* users:(("livekit-server",pid=3079590,fd=6))
"""

# On the host network, before the namespace: the client's ICE sockets on the public IP.
EXPOSED = """\
udp UNCONN 0 0 203.0.113.7:39948 0.0.0.0:* users:(("python",pid=3078797,fd=30))
tcp LISTEN 0 5 203.0.113.7:38155 0.0.0.0:* users:(("python",pid=3078797,fd=37))
tcp LISTEN 0 4096 127.0.0.1:44731 0.0.0.0:* users:(("livekit-server",pid=3078845,fd=6))
tcp LISTEN 0 128 *:7881 *:* users:(("livekit-server",pid=3078845,fd=6))
"""


def test_parse_ss_addresses() -> None:
    lines = (
        "udp UNCONN 0 0 [fe80::ca4b:d6ff:feab:93d5]%eno8303:546 [::]:*\n"
        "udp UNCONN 0 0 127.0.0.53%lo:53 0.0.0.0:*\n"
        "tcp LISTEN 0 128 [::]:22 [::]:*\n"
        'tcp LISTEN 0 128 *:7881 *:* users:(("livekit-server",pid=7,fd=6),("x",pid=8,fd=1))\n'
    )
    parsed = [(s.address, s.port, s.loopback, s.pids) for s in parse_ss(lines)]
    assert parsed == [
        ("fe80::ca4b:d6ff:feab:93d5", "546", False, frozenset()),
        ("127.0.0.53", "53", True, frozenset()),
        ("::", "22", False, frozenset()),
        ("*", "7881", False, frozenset({7, 8})),
    ]


def test_namespace_layout_passes() -> None:
    sockets = parse_ss(INSIDE)
    assert namespace_problems(sockets, loopback_only={SERVER}, internal_address="10.255.0.1") == []


def test_public_and_wildcard_binds_fail() -> None:
    problems = namespace_problems(
        parse_ss(EXPOSED), loopback_only={3078845}, internal_address="10.255.0.1"
    )
    assert [line.split()[4] for line in problems] == [
        "203.0.113.7:39948",
        "203.0.113.7:38155",
        "*:7881",
    ]


def test_server_may_not_use_the_dummy_address() -> None:
    problems = namespace_problems(
        parse_ss(INSIDE), loopback_only={SERVER, CLIENT}, internal_address="10.255.0.1"
    )
    assert len(problems) == 2


def test_host_problems_flag_only_our_processes() -> None:
    assert host_problems(parse_ss(EXPOSED), ours={3078797}) == [
        line.strip() for line in EXPOSED.splitlines()[:2]
    ]
    assert host_problems(parse_ss(EXPOSED), ours={1}) == []


def test_parse_links() -> None:
    output = (
        "1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN\\    link/loopback\n"
        "2: lh0: <BROADCAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc noqueue\\    link/ether 0a:61\n"
        "3: veth0@if2: <BROADCAST,MULTICAST,UP> mtu 1500\\    link/ether 0a:62\n"
    )
    assert parse_links(output) == {"lo", "lh0", "veth0"}
