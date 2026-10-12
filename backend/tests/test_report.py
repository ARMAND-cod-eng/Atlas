"""The results report: every number the README quotes, computed from run files.

Nothing in the results table is typed in by hand. These tests pin the
arithmetic — token totals that keep "not measured" apart from zero, paired
deltas against Atlas, the critic's effect on exactly the items it sent back —
and that the chart gives each system the same colour whatever order it arrives
in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb

from atlas_eval import cli
from atlas_eval.report import (
    DARK,
    LIGHT,
    build_figure,
    critic_effect,
    paired_delta,
    render_chart,
    render_markdown,
    summarize,
)

ITEMS = [f"q{i}" for i in range(4)]


def _record(
    item_id: str,
    tokens: list[tuple[int | None, int | None]],
    duration_s: float,
    degraded: list[str] | None = None,
    rounds: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "error": error,
        "duration_s": duration_s,
        "metrics": {
            "llm_calls": [
                {
                    "node": "n",
                    "provider": "anthropic",
                    "model": "m",
                    "duration_s": 1.0,
                    "input_tokens": i,
                    "output_tokens": o,
                }
                for i, o in tokens
            ],
            "degraded": degraded or [],
            "critic_rounds": [
                {"iteration": n + 1, "overall": overall, "faulted_subtask_ids": []}
                for n, overall in enumerate(rounds or [])
            ],
        },
    }


def _run(
    variant: str, scores: dict[str, dict[str, float]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"variant": variant, "per_item": scores, "records": records}


def _atlas() -> dict[str, Any]:
    return _run(
        "atlas",
        {"completion": dict.fromkeys(ITEMS, 0.8), "grounding": dict.fromkeys(ITEMS, 0.9)},
        [
            _record("q0", [(1000, 500), (2000, 500)], 60.0, rounds=["revise", "accept"]),
            _record("q1", [(1000, 500)], 80.0, degraded=["researcher_failed:s1"]),
            _record("q2", [(1000, None)], 70.0),
            _record("q3", [(1000, 500)], 90.0, error="ProviderError: x"),
        ],
    )


# --- summarising one run ----------------------------------------------------------


def test_a_run_summary() -> None:
    summary = summarize("baseline", _atlas())

    assert (summary.name, summary.variant, summary.items, summary.crashed) == (
        "baseline",
        "atlas",
        4,
        1,
    )
    # 4000 + 1500 + 1000 + 1500, over four runs. The None is counted, not zeroed.
    assert summary.tokens_per_query == 2000.0
    assert summary.unmeasured_calls == 1
    assert summary.p50_latency_s == 75.0
    assert summary.degraded_share == 0.25
    assert summary.scores["completion"].value_pp == pytest.approx(80.0)


def test_paired_delta_is_other_minus_reference() -> None:
    reference = dict.fromkeys(ITEMS, 0.8)
    worse = dict.fromkeys(ITEMS, 0.7)

    delta = paired_delta(reference, worse)

    assert delta.items == 4
    assert (delta.value_pp, delta.low_pp, delta.high_pp) == pytest.approx((-10.0, -10.0, -10.0))


def test_the_critic_is_judged_on_the_items_it_sent_back() -> None:
    """q0 and q1 were rejected and gained 30pp; q2 was accepted and lost 80pp.

    Averaging over every item would bury the critic's effect under items it
    never touched. Only the rejected ones say what a rejection is worth.
    """
    atlas = _run(
        "atlas",
        {"completion": {"q0": 0.9, "q1": 0.9, "q2": 0.1}},
        [
            _record("q0", [(1, 1)], 1.0, rounds=["revise", "accept"]),
            _record("q1", [(1, 1)], 1.0, rounds=["revise", "accept"]),
            _record("q2", [(1, 1)], 1.0, rounds=["accept"]),
        ],
    )
    no_critic = _run("no-critic", {"completion": {"q0": 0.6, "q1": 0.6, "q2": 0.9}}, [])

    effect = critic_effect(atlas, no_critic)

    assert (effect.runs, effect.rejected) == (3, 2)
    assert effect.deltas["completion"].value_pp == pytest.approx(30.0)
    assert effect.deltas["completion"].items == 2


# --- the markdown -------------------------------------------------------------------


def test_the_results_table() -> None:
    atlas = summarize("baseline", _atlas())
    single = summarize(
        "single",
        _run(
            "single-call",
            {"completion": dict.fromkeys(ITEMS, 0.7), "grounding": dict.fromkeys(ITEMS, 0.9)},
            [_record(q, [(400, 100)], 10.0) for q in ITEMS],
        ),
    )
    deltas = {
        "single": {"completion": paired_delta(dict.fromkeys(ITEMS, 0.8), dict.fromkeys(ITEMS, 0.7))}
    }

    lines = render_markdown([atlas, single], "baseline", deltas, None).splitlines()

    assert lines[:4] == [
        "| System | Completion | Grounding | Tool use "
        "| Tokens / query | p50 latency | Degraded | Crashed |",
        "|---|---|---|---|---|---|---|---|",
        "| **Atlas** (reference) | 80.0% | 90.0% | n/a | 2,000 | 75s | 25% | 1 / 4 |",
        "| Single call + search | 70.0% (-10.0pp, -10.0 to -10.0) | 90.0% | n/a "
        "| 500 | 10s | 0% | 0 / 4 |",
    ]
    assert "1 model call reported no token count; tokens per query undercounts it." in lines


# --- the chart ------------------------------------------------------------------------


def _three() -> list[Any]:
    def run(variant: str, completion: float, tokens: int) -> Any:
        scores = {
            "completion": dict.fromkeys(ITEMS, completion),
            "grounding": dict.fromkeys(ITEMS, 0.9),
        }
        return summarize(
            variant, _run(variant, scores, [_record(q, [(tokens, 0)], 1.0) for q in ITEMS])
        )

    return [run("single-call", 0.6, 4000), run("no-critic", 0.75, 24000), run("atlas", 0.8, 31000)]


def test_the_chart_plots_each_system_at_its_cost_and_score() -> None:
    figure = build_figure(_three(), LIGHT)
    completion = figure.axes[0]

    assert [ax.get_title(loc="left") for ax in figure.axes] == ["Completion", "Grounding"]
    points = sorted(
        (round(float(x), 3), round(float(y), 3))
        for collection in completion.collections
        if isinstance(collection, PathCollection)
        for x, y in collection.get_offsets()
    )
    assert points == [(4.0, 60.0), (24.0, 75.0), (31.0, 80.0)]
    assert [t.get_text() for t in figure.legends[0].get_texts()] == [
        "Atlas",
        "Atlas, no critic",
        "Single call + search",
    ]


def test_colour_follows_the_system_not_its_position() -> None:
    """With the no-critic run left out, the single-call baseline is still slot 3.

    Colour by position would repaint it orange — and a reader who learned
    "the baseline is green" from one chart would be misled by the next.
    """
    single, _no_critic, atlas = _three()
    for theme in (LIGHT, DARK):
        figure = build_figure([single, atlas], theme)
        legend = figure.legends[0]
        colours = {
            text.get_text(): handle.get_facecolor()[0][:3]
            for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        }

        assert list(colours) == ["Atlas", "Single call + search"]
        assert colours["Atlas"] == pytest.approx(to_rgb(theme.series[0]))
        assert colours["Single call + search"] == pytest.approx(to_rgb(theme.series[2]))


def test_both_themes_are_written(tmp_path: Path) -> None:
    light, dark = render_chart(_three(), tmp_path / "quality-vs-cost")

    assert (light.name, dark.name) == ("quality-vs-cost.light.png", "quality-vs-cost.dark.png")
    assert [light.read_bytes()[:8], dark.read_bytes()[:8]] == [b"\x89PNG\r\n\x1a\n"] * 2


# --- the command -------------------------------------------------------------------------


def test_report_refuses_a_reference_that_is_not_atlas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "single.json").write_text(json.dumps(_run("single-call", {}, [])), encoding="utf-8")
    monkeypatch.setattr(cli, "RUNS_DIR", runs)

    args = cli.build_parser().parse_args(["report", "--runs", "single", "--reference", "single"])

    assert cli.cmd_report(args) == 2
    assert capsys.readouterr().out.splitlines() == [
        "the reference must be a run of the production system ('atlas'); 'single' is 'single-call'"
    ]
