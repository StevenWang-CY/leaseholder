from __future__ import annotations

from pathlib import Path

from aiperf.config.loader import load_config

from leaseholder_bench.protocol import DEFAULT_MODEL

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_agentic_replay_template_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LH_BENCH_DATA", str(tmp_path))
    benchmark = load_config(CONFIGS / "agentic_replay.yaml").benchmark

    assert benchmark.endpoint.urls == ["http://127.0.0.1:8000/v1/chat/completions"]
    assert benchmark.endpoint.type == "leaseholder_chat"
    assert benchmark.get_model_names() == [DEFAULT_MODEL]
    assert benchmark.endpoint.extra == {"ignore_eos": True}  # decode the trace's output lengths
    dataset = benchmark.get_default_dataset()
    assert dataset.format == "weka_trace_scaled"
    assert dataset.use_think_time_only is True
    phase = benchmark.phases[0]
    assert phase.timing_mode == "agentic_replay"
    assert (phase.concurrency, phase.duration, phase.grace_period) == (16, 1800, 120)
    assert phase.burst_phase_starts is True
    assert benchmark.artifacts.raw is False


def test_agentic_replay_template_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LH_BENCH_DATA", str(tmp_path))
    monkeypatch.setenv("LH_BENCH_URL", "http://127.0.0.1:18000/v1/chat/completions")
    monkeypatch.setenv("LH_BENCH_SESSIONS", "8")
    monkeypatch.setenv("LH_BENCH_DURATION", "60")
    monkeypatch.setenv("LH_BENCH_RAW", "true")
    monkeypatch.setenv("LH_BENCH_BURST_STARTS", "false")
    monkeypatch.setenv("LH_BENCH_GRACE", "15")
    benchmark = load_config(CONFIGS / "agentic_replay.yaml").benchmark

    assert benchmark.endpoint.urls == ["http://127.0.0.1:18000/v1/chat/completions"]
    phase = benchmark.phases[0]
    assert (phase.concurrency, phase.duration, phase.grace_period) == (8, 60, 15)
    assert phase.burst_phase_starts is False
    assert benchmark.artifacts.raw is True


def test_env_file_disables_the_dataset_cache():
    settings = (CONFIGS / "aiperf.env").read_text().splitlines()
    assert "export AIPERF_DATASET_MMAP_CACHE_ENABLED=false" in settings
    assert "export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=false" in settings
