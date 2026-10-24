"""The deployment: what the GPU nodes serve and the tunnels reach, and the script that authorizes cyan's
tunnel key on a GPU node, run against a stand-in for ssh."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
SERVICES = {8000, 8100, 8880}
"""A GPU node's services on its loopback: the engine, STT and TTS."""
KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMa9oQ1uNw4Z5fT1n2x0bq7d1H0mGJ1s7mYcQjv8Yw5k"
"""A tunnel key's type and blob, as ssh-keygen writes them before the comment."""

FAKE_SSH = """\
#!/bin/sh
# ssh HOST COMMAND...: runs COMMAND in HOST's home under $FAKE_HOSTS, as sshd runs it in the user's shell.
host=$1
shift
cd "$FAKE_HOSTS/$host" && exec sh -c "$*"
"""


def authorize(tmp_path: Path, public_key: str, authorized_keys: str = "") -> tuple[str, str]:
    """Runs authorize-tunnel.sh from cyan to phastform, cyan's tunnel key being ``public_key`` and
    phastform's authorized_keys ``authorized_keys``; returns what it prints and the file it leaves."""
    hosts, bin_dir = tmp_path / "hosts", tmp_path / "bin"
    for host in ("cyan", "phastform"):
        (hosts / host / ".ssh").mkdir(parents=True)
    (hosts / "cyan" / ".ssh" / "lh_tunnel_ed25519.pub").write_text(f"{public_key}\n")
    authorized = hosts / "phastform" / ".ssh" / "authorized_keys"
    authorized.write_text(authorized_keys)
    bin_dir.mkdir()
    (bin_dir / "fake-ssh").write_text(FAKE_SSH)
    (bin_dir / "fake-ssh").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOSTS": str(hosts),
        "LH_SSH": "fake-ssh",
    }
    done = subprocess.run(
        ["bash", str(DEPLOY / "authorize-tunnel.sh"), "cyan", "phastform"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout, authorized.read_text()


def permitted(authorized_keys: str) -> set[int]:
    """The loopback ports authorized_keys lets a key forward to."""
    return {int(port) for port in re.findall(r'permitopen="127\.0\.0\.1:(\d+)"', authorized_keys)}


def test_the_gpu_nodes_publish_no_kv_events_and_the_tunnels_reach_only_their_services(
    tmp_path: Path,
) -> None:
    # vLLM's KV-event publisher connects to a TCP endpoint and serves a replay of the events, which carry
    # every cached block's token ids and session id: on a shared host any user could bind or ask for them,
    # and nothing of the deployment reads them.
    engine = (DEPLOY / "phastform" / "engine.yaml").read_text()
    assert "kv-events-config" not in engine and "tcp://" not in engine
    for unit in ("lh-tunnel-phastform", "lh-tunnel-gray"):
        text = (DEPLOY / "cyan" / "systemd" / f"{unit}.service").read_text()
        forwarded = {int(forward.rsplit(":", 1)[1]) for forward in re.findall(r"-L (\S+)", text)}
        assert forwarded == SERVICES, unit
    smoke = (DEPLOY / "phastform" / "smoke.sh").read_text()
    audited = re.search(r"grep -E ':\(([\d|]+)\)\$'", smoke)
    assert audited is not None
    assert {int(port) for port in audited.group(1).split("|")} == SERVICES
    _, authorized = authorize(tmp_path, f"{KEY} lh-tunnel@cyan")
    assert permitted(authorized) == SERVICES
