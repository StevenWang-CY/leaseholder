import math

import pytest

from leaseholder_bench.cli import main
from leaseholder_bench.kv_metrics import IDLE_BEFORE_EVICT, REUSE_GAP, Histogram, compare

LABELS = 'engine="0",model_name="m"'


def exposition(name: str, cumulative: dict[str, float]) -> str:
    lines = [f"# TYPE {name} histogram"]
    lines += [f'{name}_bucket{{{LABELS},le="{le}"}} {count}' for le, count in cumulative.items()]
    return "\n".join(lines)


def test_parse_reads_cumulative_buckets_and_ignores_other_series():
    text = (
        exposition(REUSE_GAP, {"1.0": 2, "10.0": 5, "+Inf": 6})
        + "\n"
        + exposition(IDLE_BEFORE_EVICT, {"1.0": 9, "+Inf": 9})
    )
    hist = Histogram.parse(text, REUSE_GAP)
    assert hist.bounds == (1.0, 10.0, math.inf)
    assert hist.counts == (2, 5, 6)
    assert hist.total == 6


def test_label_sets_are_summed_and_runs_are_differenced():
    before = Histogram((1.0, math.inf), (1.0, 2.0))
    after = Histogram((1.0, math.inf), (4.0, 10.0))
    assert (after - before).counts == (3.0, 8.0)
    text = (
        f'{REUSE_GAP}_bucket{{engine="0",le="1.0"}} 1\n{REUSE_GAP}_bucket{{engine="1",le="1.0"}} 2\n'
        f'{REUSE_GAP}_bucket{{engine="0",le="+Inf"}} 3\n{REUSE_GAP}_bucket{{engine="1",le="+Inf"}} 4'
    )
    assert Histogram.parse(text, REUSE_GAP).counts == (3.0, 7.0)


def test_quantiles_and_shares_interpolate_inside_buckets():
    hist = Histogram((1.0, 2.0, math.inf), (0.0, 10.0, 10.0))
    assert hist.quantile(0.5) == pytest.approx(1.5)
    assert hist.share_at_most(1.25) == pytest.approx(0.25)
    assert hist.share_at_most(5.0) == pytest.approx(1.0)
    assert Histogram((1.0, math.inf), (0.0, 0.0)).quantile(0.5) is None


def test_compare_reports_the_share_of_evictions_below_the_reuse_gap(tmp_path, capsys):
    before = (
        exposition(REUSE_GAP, {"1.0": 0, "10.0": 0, "+Inf": 0})
        + "\n"
        + exposition(IDLE_BEFORE_EVICT, {"1.0": 0, "10.0": 0, "+Inf": 0})
    )
    # Reuse gaps spread over 1-10 s; half of the evicted blocks sat idle under 1 s.
    after = (
        exposition(REUSE_GAP, {"1.0": 0, "10.0": 100, "+Inf": 100})
        + "\n"
        + exposition(IDLE_BEFORE_EVICT, {"1.0": 50, "10.0": 100, "+Inf": 100})
    )
    report = compare(before, after)
    assert f"{REUSE_GAP}: n=100" in report
    assert "reuse-gap p50 (5.5s): 75.0%" in report

    (tmp_path / "before.prom").write_text(before)
    (tmp_path / "after.prom").write_text(after)
    assert (
        main(["kv-metrics", "compare", str(tmp_path / "before.prom"), str(tmp_path / "after.prom")])
        == 0
    )
    assert "evicted blocks idle no longer than the reuse-gap p90" in capsys.readouterr().out
