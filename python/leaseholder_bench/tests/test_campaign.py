"""benchmarks/e2/campaign.sh against stand-ins for the hosts: ssh runs each command in a directory per
host (fakes/campaign/ssh), rsync copies between them, and cyan's lh-bench and the GPU node's
variant.sh log what they are asked to do."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = REPO / "benchmarks" / "e2" / "campaign.sh"
FAKES = Path(__file__).parent / "fakes" / "campaign"
PHASTFORM = "phastform.cis.upenn.edu"
DEPLOYED = "v0.3-30-g6851e13"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="campaign.sh needs bash")

VOICE = 'arm = 1\nlabel = "voice alone"\n'


@dataclass
class Hosts:
    root: Path
    """$FAKE_HOSTS: a home directory per host, by the name campaign.sh's ssh gives."""
    campaigns: Path
    """The campaign files, with their arm files under arms/."""
    results: Path
    """$LH_E2_RESULTS."""

    def home(self, host: str) -> Path:
        return self.root / host

    def gpu(self, host: str, deployed: str | None = DEPLOYED) -> None:
        """A GPU node: its variant.sh, and the revision deploy.sh wrote unless ``deployed`` is None."""
        src = self.home(host) / "leaseholder" / "src"
        (src / "deploy" / "phastform").mkdir(parents=True)
        shutil.copy(FAKES / "variant.sh", src / "deploy" / "phastform" / "variant.sh")
        if deployed is not None:
            (src / "DEPLOYED").write_text(deployed + "\n")

    def campaign(self, name: str, lines: list[str], arms: dict[str, str]) -> Path:
        for arm, text in arms.items():
            (self.campaigns / "arms" / f"{arm}.toml").write_text(text)
        path = self.campaigns / f"{name}.campaign"
        path.write_text("".join(f"{line}\n" for line in lines))
        return path

    def run(
        self, campaign: Path, *options: str, gpus: str = PHASTFORM
    ) -> subprocess.CompletedProcess:
        env = os.environ | {
            "PATH": f"{FAKES}{os.pathsep}{os.environ['PATH']}",
            "FAKE_HOSTS": str(self.root),
            "FAKE_LH_BENCH": str(Path(sys.executable).with_name("lh-bench")),
            "LH_SSH": str(FAKES / "ssh"),
            "LH_CYAN": "cyan",
            "LH_GPU": gpus,
            "LH_E2_RESULTS": str(self.results),
        }
        return subprocess.run(
            ["bash", str(CAMPAIGN), *options, str(campaign)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def calls(self, host: str, tool: str) -> list[str]:
        """What ``tool`` (lh-bench on cyan, variant.sh on a GPU node) was asked to do, call by call."""
        log = self.home(host) / f"{tool}.log"
        return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def hosts(tmp_path: Path) -> Hosts:
    hosts = Hosts(tmp_path / "hosts", tmp_path / "campaigns", tmp_path / "results")
    bench = hosts.home("cyan") / "leaseholder" / "venvs" / "bench" / "bin"
    bench.mkdir(parents=True)
    shutil.copy(FAKES / "lh-bench", bench / "lh-bench")
    (hosts.campaigns / "arms").mkdir(parents=True)
    return hosts


def runs(hosts: Hosts) -> list[str]:
    """The points cyan's lh-bench measured, each as ``e2 run`` was called."""
    return [call for call in hosts.calls("cyan", "lh-bench") if call.startswith("lh-bench e2 run ")]


def test_each_point_runs_with_the_gpu_nodes_deployed_revision(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    campaign = hosts.campaign("c1", ["arms/a1.toml", "arms/a2.toml"], {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    runs_of = (
        "lh-bench e2 run leaseholder/runs/e2/c1/arms/{0}.toml --out leaseholder/runs/e2/c1/{0}"
    )
    assert runs(hosts) == [
        f"{runs_of.format(point)} --gpu-source phastform={DEPLOYED}" for point in ("a1", "a2")
    ]
    assert hosts.calls(PHASTFORM, "variant") == ["variant.sh use frozen"] * 2
    assert f"a1: phastform has {DEPLOYED}" in (hosts.results / "c1" / "campaign.log").read_text()


def test_a_gpu_node_without_its_revision_runs_the_point_with_none(hosts: Hosts) -> None:
    gray = "gray.cis.upenn.edu"
    hosts.gpu(PHASTFORM)
    hosts.gpu(gray, deployed=None)
    campaign = hosts.campaign("c2", ["arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, gpus=f"{PHASTFORM},{gray}")
    assert done.returncode == 0, done.stdout + done.stderr
    [run] = runs(hosts)
    assert run.endswith(f" --gpu-source phastform={DEPLOYED} --gpu-source gray=")
    assert "a1: gray has no DEPLOYED" in (hosts.results / "c2" / "campaign.log").read_text()


def test_a_campaign_sets_its_points_agents_seed(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    lines = ["# a round", "agents-seed 13  # this round's sample", "arms/a1.toml", "arms/a2.toml"]
    campaign = hosts.campaign("c3", lines, {"a1": VOICE, "a2": VOICE})
    done = hosts.run(campaign)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [run.split(" --gpu-source")[0].split()[-2:] for run in runs(hosts)] == [
        ["--agents-seed", "13"]
    ] * 2
    assert (
        "c3: 2 points, arm files at ?, the agents' seed 13"
        in (hosts.results / "c3" / "campaign.log").read_text()
    )


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        (
            ["agents-seed 11", "agents-seed 12"],
            "'agents-seed 12': a campaign sets the agents' seed once",
        ),
        (["agents-seed eleven"], "'agents-seed eleven': a campaign sets the agents' seed once"),
        (["agents-seed -1"], "'agents-seed -1': a campaign sets the agents' seed once"),
    ],
)
def test_a_campaign_refuses_a_seed_it_cannot_read(
    hosts: Hosts, lines: list[str], error: str
) -> None:
    campaign = hosts.campaign("c4", [*lines, "arms/a1.toml"], {"a1": VOICE})
    done = hosts.run(campaign, "--dry-run")
    assert done.returncode == 1 and error in done.stdout
    assert runs(hosts) == []


def test_without_a_seed_line_each_point_keeps_its_arm_files_seed(hosts: Hosts) -> None:
    hosts.gpu(PHASTFORM)
    done = hosts.run(hosts.campaign("c5", ["arms/a1.toml"], {"a1": VOICE}))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--agents-seed" not in runs(hosts)[0]
