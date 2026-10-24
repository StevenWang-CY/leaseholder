"""The report's figures (:mod:`.report`): a frontier per workload and the stage waterfall, as SVG
files that are the same bytes for the same report."""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from leaseholder_bench.e2.record import REPLY_TIMEOUT_MS
from leaseholder_bench.e2.report import SLO_MARGIN_MS, STACKED, arm_order, configurations

# Light chart surface, ink and the categorical palette, in its order.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_SURFACE, _INK, _MUTED, _GRID, _AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
_SECONDARY = "#52514e"

FAMILIES = (
    ("stock vLLM", 1, 5, _SECONDARY),
    ("operator answers", 6, 6, "#1baf7a"),
    ("engines", 7, 7, "#eda100"),
    ("routers", 8, 8, "#e87ba4"),
    ("Leaseholder", 9, 9, "#2a78d6"),
    ("MPS", 10, 10, "#008300"),
)
"""E2's arms by family (docs/EXPERIMENTS.md): its name, its first and last arm, and the color the
frontier figures draw it in. Any two families can meet on a scatter, and no six of the palette's
hues stay apart pair by pair, so stock vLLM, the baseline the others are read against, takes the
secondary ink, and five hues that do (with the markers and the arms' names beside them) take the
rest: blue, the first, for Leaseholder, then the others in E2's order."""
_MARKERS = (("o", 8.0), ("s", 7.1), ("^", 10.0), ("D", 7.1), ("v", 10.0), ("P", 9.5))
"""The markers of the arms of a family, counted by number (stock vLLM's 1 to 5), then by letter,
each in the size (points) that gives it the circle's area: 8 px inside its 2 px ring."""


def _save_svg(fig: Any, path: Path, **options: Any) -> None:
    """Write a figure as SVG, the same bytes for the same figure: no date, and the ids salted by
    the file's name rather than at random. ``options`` go to savefig."""
    import matplotlib

    with matplotlib.rc_context({"svg.hashsalt": path.name}):
        fig.savefig(path, format="svg", facecolor=_SURFACE, metadata={"Date": None}, **options)


def _style(arm: str) -> tuple[str, str, tuple[str, float]]:
    """An arm's family as the legend names it, its color and the arm's marker (FAMILIES, _MARKERS).
    An arm with a factor (4-no-preemptive) takes its arm's marker, and its name tells them apart."""
    found = re.match(r"(\d+)([a-z]?)", arm)
    number, letter = (int(found[1]), found[2]) if found else (0, "")
    for name, first, last, color in FAMILIES:
        if first <= number <= last:
            arms = f"arm {first}" if first == last else f"arms {first}\N{EN DASH}{last}"
            index = number - first + (ord(letter) - ord("a") if letter else 0)
            return f"{name} ({arms})", color, _MARKERS[index % len(_MARKERS)]
    return "other arms", _MUTED, _MARKERS[0]


def plot(rows: Sequence[Mapping[str, Any]], path: Path, slo: float | None = None) -> bool:
    """A workload's frontier figure: each arm's frontier in its family's color and with the arm's
    marker, joined by a line where it spans load levels (a sweep of sessions; configurations at one
    load are markers), its dominated configurations as open markers, and the arm's name by the last
    point of its frontier. A configuration measured in several campaigns sits at its runs' medians,
    with whiskers over their range on both axes: one marker per configuration keeps each frontier a
    line through what frontier.md marks, and the whiskers still show a run that collapsed. A
    configuration with failed turns, whose V2V is censored, is a cross on no frontier, and the notes
    name it. A dashed line marks the voice SLO, ``slo`` ms. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter

    placed = [g for g in configurations(rows) if g["v2v_p95_ms"] is not None]
    groups = [g for g in placed if not g["failed_turns"]]
    failed = [g for g in placed if g["failed_turns"]]
    arms = sorted({g["arm"] for g in placed}, key=arm_order)
    fence = _fence([g["v2v_p95_ms"] for g in groups])
    # The scale of the measured configurations: a censored V2V must not flatten them.
    top = fence if fence is not None else max((g["v2v_p95_ms"] for g in groups), default=None)
    if slo is not None and top is not None:
        top = max(top, slo)
    crosses = [g for g in failed if top is not None and g["v2v_p95_ms"] <= top]
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=_SURFACE, layout="constrained")
    ax.set_facecolor(_SURFACE)
    ends = []
    swept = False  # whether any arm's frontier is drawn as a line
    for arm in arms:
        _, color, (marker, size) = _style(arm)
        points = [g for g in groups if g["arm"] == arm]
        front = sorted(
            (g for g in points if g["frontier"]), key=lambda g: g["background_tokens_per_s"]
        )
        others = [g for g in points if not g["frontier"]]
        line = load_sweep(front)
        swept |= line
        ax.plot(
            [g["background_tokens_per_s"] for g in front],
            [g["v2v_p95_ms"] for g in front],
            color=color,
            linewidth=1.5,
            linestyle="-" if line else "none",
            marker=marker,
            markersize=size,
            markeredgecolor=_SURFACE,
            markeredgewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            [g["background_tokens_per_s"] for g in others],
            [g["v2v_p95_ms"] for g in others],
            s=(size - 2.75) ** 2,  # with its edge, as large as a frontier marker inside its ring
            marker=marker,
            facecolors=_SURFACE,
            edgecolors=color,
            linewidths=1.25,
            zorder=2,
        )
        for g in (g for g in points if len(g["runs"]) > 1):
            x, y = g["background_tokens_per_s"], g["v2v_p95_ms"]
            xs = [r["background_tokens_per_s"] for r in g["runs"]]
            ys = [r["v2v_p95_ms"] for r in g["runs"] if r["v2v_p95_ms"] is not None]
            ax.errorbar(
                x,
                y,
                xerr=[[x - min(xs)], [max(xs) - x]],
                yerr=[[y - min(ys)], [max(ys) - y]],
                fmt="none",
                ecolor=color,
                elinewidth=1,
                alpha=0.6,
                zorder=1,
            )
        if drawn := [g for g in front if fence is None or g["v2v_p95_ms"] <= fence]:
            ends.append((drawn[-1]["background_tokens_per_s"], drawn[-1]["v2v_p95_ms"], arm))
        if censored := [g for g in crosses if g["arm"] == arm]:
            ax.scatter(
                [g["background_tokens_per_s"] for g in censored],
                [g["v2v_p95_ms"] for g in censored],
                s=size**2,
                marker="X",
                color=color,
                linewidths=0,
                zorder=2,
            )
    notes = ["Open markers are dominated configurations"]
    if any(len(g["runs"]) > 1 for g in groups):
        notes[0] += ", whiskers span repeated runs"
    if swept:  # each note stays one line of the figure's width
        notes.insert(0, "A line joins an arm's frontier across load levels")
    if fence is not None or len(crosses) < len(failed):  # the scale of the rest and the SLO
        shown = [g["v2v_p95_ms"] for g in groups if fence is None or g["v2v_p95_ms"] <= fence]
        shown += [] if slo is None else [slo]
        if shown:
            low, high = min(shown), max(shown)
            ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    if fence is not None:
        notes += [
            f"Off the scale: {g['point']}, {g['v2v_p95_ms']:,.0f} ms at "
            f"{g['background_tokens_per_s']:,.0f} tokens/s"
            for g in groups
            if g["v2v_p95_ms"] > fence
        ]
    notes += [
        f"Failed turns: {g['point']}, {g['failed_turns']} of {g['turns']}, its V2V censored at "
        f"{REPLY_TIMEOUT_MS / 1000:.0f} s{' (a cross)' if g in crosses else ', off the scale'}"
        for g in failed
    ]
    taken = []  # what the arms' names keep clear of
    if slo is None:
        notes.append(f"No point of voice alone with {rows[0]['callers']} callers sets a voice SLO")
    else:  # said above the plot: on the line, it would cover the points that reach it
        notes.append(
            f"The dashed line is the voice SLO, {slo:,.0f} ms: voice alone + {SLO_MARGIN_MS:.0f} ms"
        )
        taken.append(ax.axhline(slo, color=_INK, linewidth=0.75, linestyle=(0, (4, 3)), zorder=1.5))
    title = f"E2 frontier: {rows[0]['workload']}"
    ax.set_title(title, color=_INK, loc="left", pad=12 + 11 * len(notes))
    note = ax.annotate(
        ".\n".join(notes) + ".",
        (0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=_SECONDARY,
    )
    ax.set_xlabel("Background goodput (tokens/s)", color=_INK)
    ax.set_ylabel("Voice V2V p95 (ms)", color=_INK)
    ax.grid(color=_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(_AXIS)
    families = list(dict.fromkeys(_style(arm)[:2] for arm in arms))
    if len(families) > 1:  # one family needs no legend: the arms' names say whose they are
        handles = [Line2D([], [], color=color, linewidth=2.5) for _, color in families]
        labels = [family for family, _ in families]
        legend = fig.legend(handles, labels, loc="outside lower left", ncols=3, frameon=False)
        for text in legend.get_texts():
            text.set_color(_INK)
    fig.draw_without_rendering()  # lays the axes out, so that the names can be placed in them
    marks = [(g["background_tokens_per_s"], g["v2v_p95_ms"]) for g in [*groups, *crosses]]
    if unnamed := _name_ends(ax, ends, marks, taken):
        notes.append(
            f"{unnamed} variants of arms crowd too closely to name; frontier.md lists them"
        )
        note.set_text(".\n".join(notes) + ".")
        ax.set_title(title, color=_INK, loc="left", pad=12 + 11 * len(notes))
    _save_svg(fig, path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return True


def load_sweep(configurations: Sequence[Mapping[str, Any]]) -> bool:
    """Whether configurations span load levels (numbers of session trees), so that a line through
    them reads as one arm swept over its load. Configurations at one load are separate markers."""
    return len({c["runs"][0]["sessions"] for c in configurations}) > 1


def _fence(values: Sequence[float]) -> float | None:
    """The V2V p95 past which a frontier figure leaves a configuration off its scale, so that one
    collapse does not flatten the rest: Tukey's fence for far outliers, three interquartile ranges
    above the upper quartile, of four or more configurations. None when none is past it."""
    if len(values) < 4:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    fence = q3 + 3 * (q3 - q1)
    return fence if max(values) > fence else None


_PLACES = (
    *((7.5, angle) for angle in (0, 35, -35, 90, -90, 145, -145, 180)),
    *((radius, angle) for radius in (20.0, 34.0) for angle in range(0, 360, 30)),
)
"""Where an arm's name may go around the last point of its frontier, in the order tried: beside it,
then farther out on a leader line. Each is a distance in points and an angle in degrees."""


def _name_ends(
    ax: Any,
    ends: Sequence[tuple[float, float, str]],
    marks: Sequence[tuple[float, float]],
    taken: Sequence[Any],
) -> int:
    """Write each arm's name by the last point of its frontier (``ends``: x, y and the name), at
    the first of _PLACES where it stays in the axes, clear of every marker (``marks``), of the texts
    ``taken`` and of the names placed before it; where none is, an arm's name goes beside the point
    anyway, and the names of an arm's variants (``9-no-headroom``) are left out. Arms whose ends
    share a marker's place share a name ("6, 8b"; past three, "9, 9-budget0 and 8 more"), and the
    most crowded go first. Returns how many variants were left unnamed."""
    import math

    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    half = 6.0 * ax.figure.dpi / 72  # the largest marker with its ring, in pixels
    boxes = [t.get_window_extent(renderer).padded(2) for t in taken]
    centers = [ax.transData.transform(mark) for mark in marks]
    for cx, cy in centers:
        boxes.append(Bbox.from_extents(cx - half, cy - half, cx + half, cy + half))
    shared: list[tuple[float, float, list[str], Any]] = []
    for x, y, name in ends:
        at = ax.transData.transform((x, y))
        same = next((s for s in shared if math.dist(s[3], at) <= half), None)
        if same is None:
            shared.append((x, y, [name], at))
        else:
            same[2].append(name)

    def crowd(end: tuple[float, float, list[str], Any]) -> int:
        """Less for an end with more markers around it."""
        return -sum(math.dist(center, end[3]) <= 6 * half for center in centers)

    def name_at(x: float, y: float, name: str, distance: float, angle: float) -> tuple[Any, Any]:
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        leader = {"arrowstyle": "-", "color": _MUTED, "linewidth": 0.6, "shrinkB": 5}
        label = ax.annotate(
            name,
            (x, y),
            xytext=(distance * dx, distance * dy),
            textcoords="offset points",
            ha="left" if dx > 0.3 else "right" if dx < -0.3 else "center",
            va="bottom" if dy > 0.3 else "top" if dy < -0.3 else "center",
            fontsize=8,
            color=_INK,
            arrowprops=None if distance == _PLACES[0][0] else leader,
        )
        label.set_in_layout(False)
        label.update_positions(renderer)
        return label, Text.get_window_extent(label, renderer).padded(1.5)

    unnamed = 0
    for x, y, names, _ in sorted(shared, key=crowd):
        name = (
            ", ".join(names)
            if len(names) <= 3
            else f"{', '.join(names[:2])} and {len(names) - 2} more"
        )
        for place in _PLACES:
            label, box = name_at(x, y, name, *place)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0
            if inside and box.y1 <= frame.y1 and not any(box.overlaps(b) for b in boxes):
                break
            label.remove()
        else:  # nowhere clear: an arm's name beside the point anyway, its variants' left out
            if all("-" in n for n in names):
                unnamed += len(names)
                continue
            label, box = name_at(x, y, name, *_PLACES[0])
        boxes.append(box)
    return unnamed


def plot_waterfall(cells: Sequence[Mapping[str, Any]], path: Path) -> bool:
    """waterfall.svg: a bar per configuration that stacks the means of its stages from the end of the
    caller's speech, so that its length is the mean V2V, with a line at the median V2V and a
    triangle at the p95, and under it a thin bar for the LLM's time to first token from its
    request. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    ys = list(range(len(cells)))
    fig, ax = plt.subplots(
        figsize=(7.5, 1.9 + 0.5 * len(cells)), facecolor=_SURFACE, layout="constrained"
    )
    ax.set_facecolor(_SURFACE)
    handles, left = [], [0.0] * len(cells)
    segments = [*STACKED, ("transport_ms", "Transport and playout")]
    for color, (key, label) in zip(_SERIES, segments, strict=False):
        widths = [max(0.0, c[key]) for c in cells]
        bars = ax.barh(ys, widths, left=left, height=0.42, color=color, edgecolor=_SURFACE)
        handles.append((bars, label))
        left = [x + width for x, width in zip(left, widths, strict=True)]
    timed = [(y, c) for y, c in zip(ys, cells, strict=True) if c["llm_ttft_ms"] is not None]
    llm = ax.barh(
        [y + 0.32 for y, _ in timed],
        [c["llm_ttft_ms"] for _, c in timed],
        left=[c["llm_request_ms"] for _, c in timed],
        height=0.08,
        color=_SECONDARY,
    )
    median = ax.vlines(
        [c["v2v_p50_ms"] for c in cells],
        [y - 0.27 for y in ys],
        [y + 0.27 for y in ys],
        colors=_INK,
        linewidth=2,
    )
    p95 = ax.scatter(
        [c["v2v_p95_ms"] for c in cells], [y - 0.34 for y in ys], s=36, marker="v", color=_INK
    )
    handles += [
        (llm, "LLM first token, from its request"),
        (median, "V2V median"),
        (p95, "V2V p95"),
    ]
    ax.set_yticks(
        ys, [c["point"] + (f" ({c['runs']} runs)" if c["runs"] > 1 else "") for c in cells]
    )
    ax.set_ylim(len(cells) - 0.5, -0.55)  # the report's order, top to bottom
    ax.set_xlabel("ms from the end of the caller's speech", color=_INK)
    ax.set_title(
        "Voice turn stages, stacked: a bar's length is the mean V2V", color=_INK, loc="left"
    )
    ax.grid(axis="x", color=_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED)
    ax.tick_params(axis="y", length=0, labelcolor=_INK)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    artists, labels = zip(*handles, strict=True)
    legend = fig.legend(artists, labels, loc="outside lower left", ncols=3, frameon=False)
    for text in legend.get_texts():
        text.set_color(_INK)
    _save_svg(fig, path, bbox_inches="tight", pad_inches=0.1)  # the rows and the legend, no more
    plt.close(fig)
    return True
