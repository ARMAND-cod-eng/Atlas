"""Run files in, the results table and chart out.

Every number the README quotes about Atlas's quality comes from here, computed
from committed run files — never typed in by hand. That is the whole point: a
results table that can drift from the data is worse than none.

Three things are reported for each system, because quality alone flatters an
expensive system: the scores, what they cost (tokens and wall clock), and how
often a run degraded or crashed. Differences from Atlas are *paired* — same
items, same corpus — with a seeded 95% bootstrap CI, the same statistics the CI
gate uses.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from atlas_eval.gate import CATEGORIES, paired_bootstrap

LABELS: dict[str, str] = {
    "atlas": "Atlas",
    "no-critic": "Atlas, no critic",
    "single-call": "Single call + search",
}
CATEGORY_LABELS: dict[str, str] = {
    "completion": "Completion",
    "grounding": "Grounding",
    "tools": "Tool use",
}
# Fixed slot per system, so colour follows the system and never its position in
# whatever list a caller passes.
SERIES_ORDER: tuple[str, ...] = ("atlas", "no-critic", "single-call")


class Interval(BaseModel):
    """A mean (or a mean difference) in percentage points, with its 95% CI."""

    value_pp: float
    low_pp: float
    high_pp: float
    items: int


class RunSummary(BaseModel):
    name: str
    variant: str
    items: int
    crashed: int
    scores: dict[str, Interval] = Field(default_factory=dict)
    tokens_per_query: float
    # Model calls whose provider reported no token count. Counted, not zeroed:
    # the per-query figure undercounts them, and the report says by how many.
    unmeasured_calls: int
    p50_latency_s: float
    degraded_share: float


class CriticEffect(BaseModel):
    """What a rejection is worth: Atlas against no-critic, on rejected items only."""

    runs: int
    rejected: int
    deltas: dict[str, Interval] = Field(default_factory=dict)


def _interval(values: Sequence[float]) -> Interval:
    low, high = paired_bootstrap(list(values))
    return Interval(
        value_pp=statistics.fmean(values) * 100, low_pp=low, high_pp=high, items=len(values)
    )


def summarize(name: str, run: Mapping[str, Any]) -> RunSummary:
    """One run file, reduced to what the table reports."""
    records: list[Mapping[str, Any]] = run["records"]
    calls = [call for record in records for call in record["metrics"]["llm_calls"]]
    tokens = [
        sum(
            (call.get("input_tokens") or 0) + (call.get("output_tokens") or 0)
            for call in record["metrics"]["llm_calls"]
        )
        for record in records
    ]
    return RunSummary(
        name=name,
        variant=str(run.get("variant", "atlas")),
        items=len(records),
        crashed=sum(1 for record in records if record.get("error")),
        scores={
            category: _interval(list(per_item.values()))
            for category, per_item in run["per_item"].items()
            if per_item
        },
        tokens_per_query=statistics.fmean(tokens) if tokens else 0.0,
        unmeasured_calls=sum(
            1
            for call in calls
            if call.get("input_tokens") is None or call.get("output_tokens") is None
        ),
        p50_latency_s=statistics.median(r["duration_s"] for r in records) if records else 0.0,
        degraded_share=(
            sum(1 for r in records if r["metrics"]["degraded"]) / len(records) if records else 0.0
        ),
    )


def paired_delta(reference: Mapping[str, float], other: Mapping[str, float]) -> Interval:
    """``other`` minus ``reference`` over their shared items, in pp."""
    shared = sorted(set(reference) & set(other))
    return _interval([other[i] - reference[i] for i in shared])


def deltas_against(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Interval]:
    """Per-category paired deltas, for categories both runs scored."""
    return {
        category: paired_delta(reference["per_item"][category], other["per_item"][category])
        for category in CATEGORIES
        if category in reference["per_item"] and category in other["per_item"]
    }


def critic_effect(atlas: Mapping[str, Any], no_critic: Mapping[str, Any]) -> CriticEffect:
    """Atlas against the same graph without its critic, on the items it sent back.

    Averaging over every item would bury the effect under the items the critic
    accepted untouched — on those, the two systems differ only by noise.
    """
    records: list[Mapping[str, Any]] = atlas["records"]
    rejected = {
        record["item_id"]
        for record in records
        if any(round_["overall"] == "revise" for round_ in record["metrics"]["critic_rounds"])
    }
    deltas: dict[str, Interval] = {}
    for category in CATEGORIES:
        if category not in atlas["per_item"] or category not in no_critic["per_item"]:
            continue
        with_critic = {k: v for k, v in atlas["per_item"][category].items() if k in rejected}
        without = {k: v for k, v in no_critic["per_item"][category].items() if k in rejected}
        if len(set(with_critic) & set(without)) >= 2:
            deltas[category] = paired_delta(without, with_critic)
    return CriticEffect(runs=len(records), rejected=len(rejected), deltas=deltas)


# --- markdown ---------------------------------------------------------------------


def _cell(summary: RunSummary, category: str, delta: Interval | None) -> str:
    score = summary.scores.get(category)
    if score is None:
        return "n/a"
    if delta is None:
        return f"{score.value_pp:.1f}%"
    return (
        f"{score.value_pp:.1f}% ({delta.value_pp:+.1f}pp, "
        f"{delta.low_pp:+.1f} to {delta.high_pp:+.1f})"
    )


def render_markdown(
    summaries: Sequence[RunSummary],
    reference: str,
    deltas: Mapping[str, Mapping[str, Interval]],
    critic: CriticEffect | None,
) -> str:
    """The results table, its key, and the critic's effect."""
    header = ["System", *(CATEGORY_LABELS[c] for c in CATEGORIES)]
    header += ["Tokens / query", "p50 latency", "Degraded", "Crashed"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for summary in summaries:
        label = LABELS.get(summary.variant, summary.variant)
        is_reference = summary.name == reference
        name = f"**{label}** (reference)" if is_reference else label
        row_deltas = {} if is_reference else deltas.get(summary.name, {})
        cells = [name, *(_cell(summary, c, row_deltas.get(c)) for c in CATEGORIES)]
        cells += [
            f"{summary.tokens_per_query:,.0f}",
            f"{summary.p50_latency_s:.0f}s",
            f"{summary.degraded_share:.0%}",
            f"{summary.crashed} / {summary.items}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Scores are means over the dataset. In brackets: the paired difference from the "
        "reference and its 95% bootstrap CI — same items, same frozen corpus. A CI that "
        "excludes zero is a real difference at this sample size; one that spans zero is not "
        "distinguishable from run-to-run noise.",
    ]

    unmeasured = sum(s.unmeasured_calls for s in summaries)
    if unmeasured:
        noun, pronoun = ("call", "it") if unmeasured == 1 else ("calls", "them")
        lines += [
            "",
            f"{unmeasured} model {noun} reported no token count; "
            f"tokens per query undercounts {pronoun}.",
        ]

    if critic is not None:
        lines += ["", f"The critic sent back {critic.rejected} of {critic.runs} runs."]
        for category, delta in critic.deltas.items():
            lines.append(
                f"- {CATEGORY_LABELS[category]}, on those items, against the same graph without "
                f"the critic: {delta.value_pp:+.1f}pp "
                f"({delta.low_pp:+.1f} to {delta.high_pp:+.1f}, n={delta.items})."
            )
    return "\n".join(lines) + "\n"


# --- the chart -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Theme:
    """Surface, ink and the three series slots for one colour scheme.

    The series steps are the reference palette's first three categorical slots,
    the only three that validate all-pairs (the scatter case) in both modes. The
    dark steps are their own validated steps, not an inversion. Light-mode aqua
    sits under 3:1 on the surface, which the direct labels and the markdown
    table — the chart's table view — relieve.
    """

    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]


LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
)
DARK = Theme(
    surface="#1a1a19",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
)

_FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def build_figure(summaries: Sequence[RunSummary], theme: Theme) -> Figure:
    """Score against cost, one panel per scored category.

    Small multiples rather than one panel with two y-scales: completion and
    grounding are different measures and get their own axes.
    """
    ordered = sorted(
        (s for s in summaries if s.variant in SERIES_ORDER),
        key=lambda s: SERIES_ORDER.index(s.variant),
    )
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": _FONTS}):
        figure = Figure(figsize=(9.0, 3.8), dpi=200, facecolor=theme.surface)
        axes = figure.subplots(1, 2)
        handles: list[PathCollection] = []
        for ax, category in zip(axes, ("completion", "grounding"), strict=True):
            ax.set_facecolor(theme.surface)
            xs: list[float] = []
            for summary in ordered:
                score = summary.scores.get(category)
                if score is None:
                    continue
                colour = theme.series[SERIES_ORDER.index(summary.variant)]
                x = summary.tokens_per_query / 1000
                xs.append(x)
                ax.errorbar(
                    [x],
                    [score.value_pp],
                    yerr=[[score.value_pp - score.low_pp], [score.high_pp - score.value_pp]],
                    fmt="none",
                    ecolor=colour,
                    elinewidth=1.5,
                    capsize=0,
                    zorder=2,
                )
                point = ax.scatter(
                    [x],
                    [score.value_pp],
                    s=70,
                    color=colour,
                    edgecolors=theme.surface,
                    linewidths=2,
                    zorder=3,
                    label=LABELS[summary.variant],
                )
                if ax is axes[0]:
                    handles.append(point)
            # Room on the right for the direct labels, a zero baseline for cost.
            right_edge = max(xs, default=1.0) * 1.45
            ax.set_xlim(0, right_edge)
            for summary, x in zip((s for s in ordered if category in s.scores), xs, strict=True):
                # Each label goes on the side with more free room, so it never
                # crosses a neighbouring point's error bar.
                left_room = x - max((other for other in xs if other < x), default=0.0)
                right_room = min((other for other in xs if other > x), default=right_edge) - x
                to_right = right_room >= left_room
                ax.annotate(
                    LABELS[summary.variant],
                    (x, summary.scores[category].value_pp),
                    xytext=(9 if to_right else -9, 0),
                    textcoords="offset points",
                    ha="left" if to_right else "right",
                    va="center",
                    fontsize=7.5,
                    color=theme.secondary,
                )
            ax.set_title(CATEGORY_LABELS[category], loc="left", fontsize=10, color=theme.ink)
            ax.set_xlabel("Tokens per query (thousands)", fontsize=8, color=theme.muted)
            ax.set_ylabel("Score (%)", fontsize=8, color=theme.muted)
            ax.grid(color=theme.grid, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=theme.muted, labelsize=7.5, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(theme.axis)
        figure.legend(
            handles=handles,
            loc="upper right",
            ncols=len(handles),
            frameon=False,
            fontsize=8,
            labelcolor=theme.secondary,
        )
        figure.text(
            0.01,
            0.97,
            "Quality against cost — means with 95% CI",
            ha="left",
            va="top",
            fontsize=10,
            color=theme.ink,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.9))
    return figure


def render_chart(summaries: Sequence[RunSummary], stem: Path) -> tuple[Path, Path]:
    """Write ``<stem>.light.png`` and ``<stem>.dark.png`` for a ``<picture>`` element."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = (stem.with_name(f"{stem.name}.light.png"), stem.with_name(f"{stem.name}.dark.png"))
    for path, theme in zip(paths, (LIGHT, DARK), strict=True):
        build_figure(summaries, theme).savefig(path, facecolor=theme.surface)
    return paths
