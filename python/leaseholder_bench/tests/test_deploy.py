"""The deployment: what the GPU nodes serve and the tunnels reach, cyan's install, and the script that
authorizes cyan's tunnel key on a GPU node, run against stand-ins for ssh and systemd."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
SERVICES = {8000, 8100, 8880}
"""A GPU node's services on its loopback: the engine, STT and TTS."""
KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMa9oQ1uNw4Z5fT1n2x0bq7d1H0mGJ1s7mYcQjv8Yw5k"
"""A tunnel key's type and blob, as ssh-keygen writes them before the comment."""
LOGIN = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB7m2Qp9xR4tYc3vLk8wNs6fJd0aHe5uPq1zXo2iGb4M me@laptop"
"""Another key in a GPU node's authorized_keys: its user's own."""

FAKE_SSH = """\
#!/bin/sh
# ssh HOST COMMAND...: runs COMMAND in HOST's home under $FAKE_HOSTS, as sshd runs it in the user's shell.
host=$1
shift
cd "$FAKE_HOSTS/$host" && exec sh -c "$*"
"""


def authorize(
    work: Path, public_key: str, authorized_keys: str = ""
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Runs authorize-tunnel.sh from cyan to phastform, hosts kept under ``work``, cyan's tunnel key
    being ``public_key`` and phastform's authorized_keys ``authorized_keys``; returns the run and the
    file it leaves."""
    hosts, bin_dir = work / "hosts", work / "bin"
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
    )
    return done, authorized.read_text()


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
    done, authorized = authorize(tmp_path, f"{KEY} lh-tunnel@cyan")
    assert done.returncode == 0, done.stderr
    assert permitted(authorized) == SERVICES


def test_the_tunnel_key_may_forward_to_the_services_and_listen_nowhere(tmp_path: Path) -> None:
    earlier = f'restrict,port-forwarding,permitopen="127.0.0.1:5558" {KEY} lh-tunnel@cyan'
    options = (
        'restrict,port-forwarding,permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8100",'
        'permitopen="127.0.0.1:8880",permitlisten="127.0.0.1:1",command="/bin/false"'
    )
    # The key as ssh-keygen writes it, and without its comment, which is optional.
    for name, public_key in (("commented", f"{KEY} lh-tunnel@cyan"), ("bare", KEY)):
        done, authorized = authorize(tmp_path / name, public_key, f"{LOGIN}\n{earlier}\n")
        # The earlier entry for the key is replaced; every other key stays.
        assert authorized.splitlines() == [LOGIN, f"{options} {public_key}"], name
        assert done.stdout == f"authorized: {public_key.rsplit(' ', 1)[-1]} (1 entry)\n", name
    # A client without a key changes nothing.
    done, authorized = authorize(tmp_path / "empty", "", f"{LOGIN}\n")
    assert done.returncode == 2 and "no public key" in done.stderr
    assert authorized == f"{LOGIN}\n"


def test_cyans_install_makes_the_gateways_cache_salt_secret_once(tmp_path: Path) -> None:
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    for tool in ("systemctl", "loginctl"):  # cyan's user manager, which the test leaves alone
        (bin_dir / tool).write_text("#!/bin/sh\n")
        (bin_dir / tool).chmod(0o755)
    env = os.environ | {"HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    install = ["bash", str(DEPLOY / "cyan" / "install.sh")]
    subprocess.run(install, env=env, check=True, capture_output=True)
    secret = home / ".config" / "leaseholder" / "cache-salt-secret"
    made = secret.read_text()
    assert re.fullmatch(r"[0-9a-f]{64}\n", made)
    assert (
        stat.S_IMODE(secret.stat().st_mode) == 0o600
    )  # the gateway refuses any other user's access
    # A new secret would change every tenant's cache salt: the next install keeps it.
    subprocess.run(install, env=env, check=True, capture_output=True)
    assert secret.read_text() == made
    assert [p.name for p in secret.parent.iterdir()] == ["cache-salt-secret"]
