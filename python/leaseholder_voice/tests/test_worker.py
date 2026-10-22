"""The voice agent's turn handling, as its environment sets it."""

from __future__ import annotations

import pytest
from livekit.agents import inference

from leaseholder_voice.worker import turn_handling


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LH_TURN_DETECTION", "LH_PREEMPTIVE_GENERATION"):
        monkeypatch.delenv(name, raising=False)


def test_the_worker_keeps_livekits_turn_handling_by_default() -> None:
    handling = turn_handling()
    assert isinstance(handling["turn_detection"], inference.TurnDetector)
    assert handling["interruption"] == {"mode": "vad"}
    assert handling["preemptive_generation"] == {"enabled": True}


def test_experiments_fix_the_endpointing_and_can_turn_preemptive_generation_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LH_TURN_DETECTION", "vad")
    monkeypatch.setenv("LH_PREEMPTIVE_GENERATION", "false")
    handling = turn_handling()
    assert handling["turn_detection"] == "vad"
    assert handling["preemptive_generation"] == {"enabled": False}


@pytest.mark.parametrize(
    ("name", "value"), [("LH_TURN_DETECTION", "eou"), ("LH_PREEMPTIVE_GENERATION", "off")]
)
def test_a_misspelled_setting_stops_the_worker(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        turn_handling()
