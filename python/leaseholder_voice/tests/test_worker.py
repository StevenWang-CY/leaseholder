"""The voice agent's turn handling and lifecycle events, as its environment sets them."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from livekit.agents import inference

from leaseholder_voice.worker import events_socket, turn_handling

GATEWAY = "/run/user/1007/lh/gateway.sock"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LH_TURN_DETECTION", "LH_PREEMPTIVE_GENERATION", "LH_EVENTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LH_LLM_UDS", GATEWAY)


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


def test_events_go_to_the_control_planes_socket_unless_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert events_socket() == GATEWAY
    monkeypatch.setenv("LH_EVENTS", "off")
    assert events_socket() is None
    # The bypass: the LLM goes straight to the engine, which takes no events.
    monkeypatch.setenv("LH_EVENTS", "on")
    monkeypatch.setenv("LH_LLM_UDS", "/run/user/1007/lh/phastform-engine.sock")
    assert events_socket() is None


@pytest.mark.parametrize(
    ("name", "value", "setting"),
    [
        ("LH_TURN_DETECTION", "eou", turn_handling),
        ("LH_PREEMPTIVE_GENERATION", "off", turn_handling),
        ("LH_EVENTS", "false", events_socket),
    ],
)
def test_a_misspelled_setting_stops_the_worker(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, setting: Callable[[], object]
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        setting()
