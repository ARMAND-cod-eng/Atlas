"""The statistics the whole phase rests on.

Every number this suite reports flows through Cohen's kappa, the paired bootstrap,
or the gate arithmetic. A bug in any of them does not crash — it reports a
plausible figure forever, which is the specific failure mode Phase 5 exists to
design against. So they are checked against hand-computed values rather than
against themselves.
"""

from __future__ import annotations

import math

import pytest

from atlas_eval.calibration import KAPPA_THRESHOLD, Agreement, CalibrationReport, agreement
from atlas_eval.gate import (
    CATEGORIES,
    Baseline,
    CategoryDelta,
    NoiseEstimate,
    compare,
    noise_floor,
    paired_bootstrap,
)

# --- Cohen's kappa, against arithmetic done by hand ---------------------------


def test_kappa_matches_a_hand_computed_table() -> None:
    """The worked example, so the formula is checked and not just exercised.

    20 decisions: 8 both-true, 7 both-false, 3 judge-only, 2 human-only.
      observed = (8 + 7) / 20                       = 0.75
      judge true  = (8 + 3) / 20 = 0.55
      human true  = (8 + 2) / 20 = 0.50
      expected = 0.55*0.50 + 0.45*0.50              = 0.50
      kappa    = (0.75 - 0.50) / (1 - 0.50)         = 0.50
    """
    table = Agreement(n=20, both_true=8, both_false=7, judge_only=3, human_only=2)

    assert table.observed == 0.75
    assert table.expected == 0.5
    assert math.isclose(table.kappa, 0.5, abs_tol=1e-12)
    assert table.human_prevalence == 0.5


def test_perfect_agreement_is_kappa_one_not_zero() -> None:
    """The degenerate case the naive formula gets exactly backwards.

    When both raters mark everything the same way, chance agreement is 1.0 and
    (observed - expected) / (1 - expected) is 0/0. Reporting 0.0 there would call
    perfect agreement "no better than chance".
    """
    table = Agreement(n=10, both_true=10, both_false=0, judge_only=0, human_only=0)

    assert table.observed == 1.0
    assert table.expected == 1.0
    assert table.kappa == 1.0


def test_the_kappa_paradox_is_visible_not_hidden() -> None:
    """High raw agreement, low kappa — and both get reported.

    19 of 20 agree, but the human marked 19/20 true, so chance agreement is
    already very high and kappa collapses. Quoting only kappa would understate a
    good judge; quoting only raw agreement would oversell a bad one. Grounding
    labels are skewed exactly this way, so the suite reports all three figures.
    """
    table = Agreement(n=20, both_true=18, both_false=0, judge_only=1, human_only=1)

    assert table.observed == 0.9
    assert table.kappa < 0.5
    assert table.human_prevalence == 0.95
    # The summary carries all three, so a reader cannot see one without the others.
    assert "raw=90.0%" in table.summary()
    assert "prevalence=95.0%" in table.summary()


def test_total_disagreement_is_negative() -> None:
    table = Agreement(n=10, both_true=0, both_false=0, judge_only=5, human_only=5)
    assert table.kappa < 0


def test_agreement_builds_the_table_from_aligned_labels() -> None:
    judge_labels = [True, True, False, False, True]
    human_labels = [True, False, False, True, True]

    table = agreement(judge_labels, human_labels)

    assert (table.both_true, table.both_false, table.judge_only, table.human_only) == (2, 1, 1, 1)
    assert table.n == 5


def test_misaligned_labels_raise_rather_than_truncate() -> None:
    """Silently zipping to the shorter list would score a different denominator."""
    try:
        agreement([True, False], [True])
    except ValueError as exc:
        assert "misaligned" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


# --- the calibration report ----------------------------------------------------


def _table(kappa_target: str) -> Agreement:
    return {
        "high": Agreement(n=40, both_true=30, both_false=8, judge_only=1, human_only=1),
        "low": Agreement(n=40, both_true=20, both_false=5, judge_only=8, human_only=7),
    }[kappa_target]


def test_the_report_gates_on_the_holdout_not_the_iteration_set() -> None:
    """Iterating a prompt against the items it is scored on overfits it."""
    report = CalibrationReport(
        dimension="completion", iteration=_table("high"), holdout=_table("low")
    )

    assert report.iteration.kappa > KAPPA_THRESHOLD
    assert report.holdout.kappa < KAPPA_THRESHOLD
    # The good number exists, and is deliberately not the one that decides.
    assert report.passes is False
    assert report.overfit_warning is True
    assert "WARNING" in report.summary()


def test_a_consistent_judge_passes_without_an_overfit_warning() -> None:
    report = CalibrationReport(
        dimension="grounding", iteration=_table("high"), holdout=_table("high")
    )

    assert report.passes is True
    assert report.overfit_warning is False
    assert "PASS" in report.summary()


# --- the paired bootstrap ------------------------------------------------------


def test_bootstrap_brackets_a_known_mean() -> None:
    """A clear -10pp shift: the CI must sit around it and exclude zero."""
    differences = [-0.10] * 50

    low, high = paired_bootstrap(differences)

    # Zero variance in, zero width out.
    assert math.isclose(low, -10.0, abs_tol=1e-9)
    assert math.isclose(high, -10.0, abs_tol=1e-9)


def test_bootstrap_ci_includes_zero_for_pure_noise() -> None:
    """Symmetric noise around zero must not look like a regression."""
    differences = [0.1, -0.1] * 25

    low, high = paired_bootstrap(differences)

    assert low < 0 < high


def test_bootstrap_is_seeded_and_reproducible() -> None:
    """A gate that reports a different CI on re-run is a gate nobody trusts."""
    differences = [0.05, -0.20, 0.00, -0.05, 0.10] * 10

    assert paired_bootstrap(differences) == paired_bootstrap(differences)


def test_bootstrap_handles_the_empty_and_single_cases() -> None:
    assert paired_bootstrap([]) == (0.0, 0.0)
    assert paired_bootstrap([-0.03]) == (-3.0, -3.0)


# --- the gate ------------------------------------------------------------------


def _baseline(**means: float) -> Baseline:
    return Baseline(
        commit="abc1234",
        dataset_version="v1",
        corpus_recorded_at="2026-08-10",
        run_date="2026-08-10",
        n_items=3,
        means=dict(means),
    )


def test_a_real_regression_trips_the_gate() -> None:
    base = _baseline(completion=0.80, grounding=0.90, tools=0.95)
    baseline_items = {
        "grounding": {f"q{i}": 0.90 for i in range(50)},
    }
    # Every item down 10pp: large, consistent, unambiguous.
    current = {
        "completion": {f"q{i}": 0.80 for i in range(50)},
        "grounding": {f"q{i}": 0.80 for i in range(50)},
        "tools": {f"q{i}": 0.95 for i in range(50)},
    }

    result = compare(base, current, baseline_items)
    grounding = next(d for d in result.deltas if d.category == "grounding")

    assert result.passed is False
    assert math.isclose(grounding.delta_pp, -10.0, abs_tol=1e-9)
    assert grounding.trips(2.0) is True
    assert any("grounding regressed" in reason for reason in result.reasons)


def test_noise_below_significance_does_not_trip_the_gate() -> None:
    """The condition that stops the tripwire firing on sampling noise.

    A -3pp point delta is past the 2pp threshold, but it is driven by a handful
    of items moving in both directions, so the CI spans zero and the gate holds.
    """
    base = _baseline(completion=0.80)
    baseline_items = {"completion": {f"q{i}": 0.80 for i in range(50)}}
    noisy = [0.80] * 40 + [0.30] * 5 + [1.00] * 5  # mean 0.77, wide spread
    current = {"completion": {f"q{i}": noisy[i] for i in range(50)}}

    result = compare(base, current, baseline_items)
    completion = next(d for d in result.deltas if d.category == "completion")

    assert completion.delta_pp < -2.0
    assert completion.significant is False
    assert completion.trips(2.0) is False
    assert result.passed is True


def test_the_threshold_boundary_is_exact() -> None:
    """Exactly -2pp does not trip; -2.01pp does. Asserted, not assumed."""
    at_threshold = CategoryDelta(
        category="tools",
        baseline=0.90,
        current=0.88,
        delta_pp=-2.0,
        ci_low_pp=-3.0,
        ci_high_pp=-1.0,
        paired_sd_pp=0.5,
    )
    past_threshold = at_threshold.model_copy(update={"delta_pp": -2.01})

    assert at_threshold.trips(2.0) is False
    assert past_threshold.trips(2.0) is True


def test_an_improvement_never_trips() -> None:
    base = _baseline(completion=0.70, grounding=0.70, tools=0.70)
    baseline_items = {
        c: {f"q{i}": 0.70 for i in range(50)} for c in ("completion", "grounding", "tools")
    }
    current = {c: {f"q{i}": 0.90 for i in range(50)} for c in ("completion", "grounding", "tools")}

    result = compare(base, current, baseline_items)

    assert result.passed is True
    assert all(d.delta_pp > 0 for d in result.deltas)


def test_a_missing_per_item_baseline_fails_the_gate() -> None:
    """Without per-item scores the gate cannot decide, so it does not pass.

    It used to pass. No CI meant "not significant", which meant nothing could
    trip — a -20pp collapse included — while the reason line claimed the gate had
    fallen back to the point delta. CI never has the per-item baseline unless it
    is committed, so this was the gate's behaviour on every run.
    """
    base = _baseline(completion=0.80)
    current = {"completion": {f"q{i}": 0.60 for i in range(50)}}

    result = compare(base, current, baseline_scores=None)

    assert result.passed is False
    assert result.reasons == [
        "completion: no per-item baseline scores, so no paired comparison is possible "
        "— commit runs/baseline.json from `atlas_eval run --name baseline --write-baseline`",
        "grounding: absent from baseline, not gated",
        "tools: absent from baseline, not gated",
    ]


def test_a_per_item_baseline_with_no_overlap_fails_the_gate() -> None:
    base = _baseline(completion=0.80)
    baseline_items = {"completion": {f"old{i}": 0.80 for i in range(50)}}
    current = {"completion": {f"q{i}": 0.80 for i in range(50)}}

    result = compare(base, current, baseline_items)

    assert result.passed is False
    assert result.reasons == [
        "completion: 50 item(s) missing from the baseline run; CI computed on the overlap only",
        "completion: no item appears in both runs, so no paired comparison is possible",
        "grounding: absent from baseline, not gated",
        "tools: absent from baseline, not gated",
    ]


def test_an_unchanged_run_with_a_per_item_baseline_passes() -> None:
    """The control for the two above: the same data, with what the gate needs."""
    base = _baseline(completion=0.80)
    baseline_items = {"completion": {f"q{i}": 0.80 for i in range(50)}}
    current = {"completion": {f"q{i}": 0.80 for i in range(50)}}

    result = compare(base, current, baseline_items)

    assert result.passed is True


# --- the noise floor ------------------------------------------------------------------
#
# The gate's 2pp threshold was provisional "until the measured paired SD says
# otherwise". noise_floor is what measures it: the same code run several times,
# per-item differences between the runs, and from their SD the smallest
# regression a 50-item gate can tell apart from that run-to-run wobble.

ITEMS = [f"q{i}" for i in range(50)]


def _alternating(base: float, step: float) -> dict[str, float]:
    """Each item moved by +step or -step: zero net drift, SD of exactly ``step``-ish."""
    return {item: base + (step if i % 2 == 0 else -step) for i, item in enumerate(ITEMS)}


def test_identical_reruns_have_no_noise() -> None:
    run = {"completion": dict.fromkeys(ITEMS, 0.8)}

    [estimate] = noise_floor([run, run])

    assert (estimate.category, estimate.pairs, estimate.items) == ("completion", 1, 50)
    assert (estimate.paired_sd_pp, estimate.drift_pp, estimate.detectable_pp) == (0.0, 0.0, 0.0)


def test_the_detectable_regression_follows_from_the_paired_sd() -> None:
    """Hand-computed: diffs of +/-0.1 over 50 items have a sample SD of
    sqrt(50 * 0.01 / 49) = 10.10pp, so the standard error is 10.10 / sqrt(50) =
    1.43pp, and 80% power at two-sided 5% needs (1.96 + 0.84) x 1.43 = 4.00pp."""
    first = {"completion": dict.fromkeys(ITEMS, 0.8)}
    second = {"completion": _alternating(0.8, 0.1)}

    [estimate] = noise_floor([first, second])

    assert math.isclose(estimate.paired_sd_pp, 10.1015, abs_tol=1e-3)
    assert math.isclose(estimate.drift_pp, 0.0, abs_tol=1e-9)
    assert math.isclose(estimate.detectable_pp, 4.0023, abs_tol=1e-3)


def test_a_category_missing_from_any_run_is_not_estimated() -> None:
    """The single-call baseline has no tool use; it must not read as zero noise."""
    with_tools = {"completion": dict.fromkeys(ITEMS, 0.8), "tools": dict.fromkeys(ITEMS, 1.0)}
    without = {"completion": dict.fromkeys(ITEMS, 0.8)}

    assert [e.category for e in noise_floor([with_tools, without])] == ["completion"]


def test_noise_needs_at_least_two_runs() -> None:
    with pytest.raises(ValueError, match="at least two runs"):
        noise_floor([{"completion": dict.fromkeys(ITEMS, 0.8)}])


@pytest.mark.parametrize(
    ("detectable_pp", "verdict"),
    [
        (1.5, "the threshold binds"),
        (4.0, "raise the threshold to at least 4pp"),
    ],
)
def test_the_threshold_verdict_says_which_limit_binds(detectable_pp: float, verdict: str) -> None:
    estimate = NoiseEstimate(
        category="grounding",
        pairs=1,
        items=50,
        paired_sd_pp=1.0,
        drift_pp=0.0,
        detectable_pp=detectable_pp,
    )

    assert estimate.verdict(2.0) == verdict


def test_a_run_with_no_scores_fails_the_gate() -> None:
    """An unscored run — records saved, judge never reached — must not pass.

    Before, a category the run had no scores for was a note in the report and
    the gate passed: a run whose scoring had died would have sailed through.
    """
    base = _baseline(completion=0.80, grounding=0.90, tools=0.95)
    baseline_items = {c: {f"q{i}": 0.9 for i in range(50)} for c in CATEGORIES}

    result = compare(base, {}, baseline_items)

    assert result.passed is False
    assert result.reasons == [
        "completion: the baseline has scores, this run has none",
        "grounding: the baseline has scores, this run has none",
        "tools: the baseline has scores, this run has none",
    ]
