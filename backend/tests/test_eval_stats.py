"""The statistics the whole phase rests on.

Every number this suite reports flows through Cohen's kappa, the paired bootstrap,
or the gate arithmetic. A bug in any of them does not crash — it reports a
plausible figure forever, which is the specific failure mode Phase 5 exists to
design against. So they are checked against hand-computed values rather than
against themselves.
"""

from __future__ import annotations

import math

from atlas_eval.calibration import KAPPA_THRESHOLD, Agreement, CalibrationReport, agreement
from atlas_eval.gate import Baseline, CategoryDelta, compare, paired_bootstrap

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
    base = _baseline(completion=0.80, grounding=0.90, tools=0.95)
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


def test_a_missing_per_item_baseline_is_reported_not_papered_over() -> None:
    """Without per-item scores there is no CI, and the gate says so."""
    base = _baseline(completion=0.80)
    current = {"completion": {f"q{i}": 0.60 for i in range(50)}}

    result = compare(base, current, baseline_scores=None)

    assert any("no per-item baseline" in reason for reason in result.reasons)
    # No CI means not significant, so the tripwire alone cannot fail the build.
    assert result.passed is True
