"""The CI gate: baseline comparison with a paired bootstrap.

**What the gate honestly is.** A tripwire with a significance condition bolted on.
It is not a pre-registered hypothesis test and it should not be described as one.

**Why the obvious statistic is the wrong one.** At n=50 and p≈0.8 the standard
error on a proportion is 5.66pp, and two *independent* runs differ by ±15.68pp at
95%. Against that, a 2pp threshold is meaningless. But the suite reruns the *same
50 items* against the *same frozen corpus*, so the comparison is **paired** — and
the quantity that decides whether the gate flakes is the standard deviation of the
per-item score *difference*, which is empirical, not derivable.

That is why the gate resamples per-item differences rather than quoting a formula:
the paired bootstrap uses the correlation between runs that the paired design
actually buys, instead of throwing it away.

The gate fires when **both** hold:

* the point delta is worse than the threshold, and
* the 95% paired bootstrap CI on that delta excludes zero

The first is the gate's stated contract: it blocks on a category regression
larger than that category's threshold — the smallest regression its measured
run-to-run noise lets a 50-item run detect. The second is what stops it firing on
noise.

And it **refuses to pass** a category it cannot evaluate. Without per-item
baseline scores there is no paired CI, and "no CI" used to mean "not
significant", which meant nothing could trip: an 80pp collapse passed, while the
report claimed the gate had fallen back to the point delta. CI never has the
per-item baseline unless ``runs/baseline.json`` is committed, so that was the
gate's behaviour on every run. A gate that cannot decide must say no.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from atlas_eval.dataset import DATA_DIR

BASELINE_PATH = DATA_DIR / "baseline.json"

CATEGORIES = ("completion", "grounding", "tools")

# Percentage points, per category: the smallest regression each category's
# run-to-run noise lets the gate detect (80% power, two-sided 5%), rounded up.
# Measured by `atlas_eval noise` on two runs of identical code, 50 items each:
#
#   completion  paired SD 10.41pp  detectable >= 4.13pp
#   grounding   paired SD  8.58pp  detectable >= 3.40pp
#   tools       paired SD  6.70pp  detectable >= 2.65pp
#
# The gate started at a provisional 2pp for all three. That sat under every
# floor: a "regression" between 2pp and the floor was a coin toss the gate
# claimed to call. A test recomputes these from the runs named below, so they
# cannot drift from the measurement that justifies them.
THRESHOLDS_PP: dict[str, float] = {"completion": 5.0, "grounding": 4.0, "tools": 3.0}
THRESHOLDS_MEASURED_ON = ("baseline", "rerun1")

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260810


class Baseline(BaseModel):
    """Per-category means from a known-good commit."""

    commit: str
    dataset_version: str
    corpus_recorded_at: str
    run_date: str
    n_items: int
    means: dict[str, float]
    notes: str = ""


class CategoryDelta(BaseModel):
    category: str
    baseline: float
    current: float
    # Percentage points, negative meaning worse than baseline.
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    paired_sd_pp: float
    threshold_pp: float

    @property
    def significant(self) -> bool:
        """The bootstrap CI excludes zero."""
        return self.ci_low_pp > 0 or self.ci_high_pp < 0

    def trips(self) -> bool:
        return self.delta_pp < -self.threshold_pp and self.significant


class GateResult(BaseModel):
    passed: bool
    deltas: list[CategoryDelta] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def report(self) -> str:
        lines = [f"gate: {'PASS' if self.passed else 'FAIL'}"]
        for delta in self.deltas:
            marker = "FAIL" if delta.trips() else "ok  "
            lines.append(
                f"  {marker} {delta.category:<11} "
                f"{delta.baseline:.3f} -> {delta.current:.3f}  "
                f"delta {delta.delta_pp:+.2f}pp  "
                f"95% CI [{delta.ci_low_pp:+.2f}, {delta.ci_high_pp:+.2f}]pp  "
                f"threshold {delta.threshold_pp:g}pp"
            )
        lines.extend(f"  {reason}" for reason in self.reasons)
        return "\n".join(lines)


def paired_bootstrap(
    differences: list[float],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% CI on the mean of per-item differences, in percentage points.

    Seeded, so the same inputs give the same interval — a gate that reports a
    different CI on re-run is a gate nobody trusts.
    """
    if not differences:
        return (0.0, 0.0)
    if len(differences) == 1:
        value = differences[0] * 100
        return (value, value)

    rng = random.Random(seed)
    n = len(differences)
    means: list[float] = []
    for _ in range(samples):
        resampled = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resampled) / n * 100)
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[min(int(0.975 * samples), samples - 1)]
    return (low, high)


def compare(
    baseline: Baseline,
    current_scores: dict[str, dict[str, float]],
    baseline_scores: dict[str, dict[str, float]] | None = None,
    thresholds: Mapping[str, float] = THRESHOLDS_PP,
) -> GateResult:
    """Compare a run against the baseline, per category.

    ``current_scores`` and ``baseline_scores`` are ``{category: {item_id: score}}``.
    Per-item baseline scores are required: they are what the paired bootstrap
    runs on. A category without them, or with no item in common, fails the gate
    rather than passing it unexamined.
    """
    deltas: list[CategoryDelta] = []
    reasons: list[str] = []
    undecided: list[str] = []

    for category in CATEGORIES:
        base_mean = baseline.means.get(category)
        if base_mean is None:
            reasons.append(f"{category}: absent from baseline, not gated")
            continue

        current_items = current_scores.get(category, {})
        if not current_items:
            # A run whose scoring died — records saved, judge never reached —
            # has no scores at all. That is not a pass.
            undecided.append(category)
            reasons.append(f"{category}: the baseline has scores, this run has none")
            continue

        current_mean = statistics.fmean(current_items.values())

        if not baseline_scores or category not in baseline_scores:
            undecided.append(category)
            reasons.append(
                f"{category}: no per-item baseline scores, so no paired comparison is "
                "possible — commit runs/baseline.json from "
                "`atlas_eval run --name baseline --write-baseline`"
            )
            continue

        per_item = baseline_scores[category]
        shared = sorted(set(current_items) & set(per_item))
        if len(shared) != len(current_items):
            reasons.append(
                f"{category}: {len(current_items) - len(shared)} item(s) missing from "
                "the baseline run; CI computed on the overlap only"
            )
        if not shared:
            undecided.append(category)
            reasons.append(
                f"{category}: no item appears in both runs, so no paired comparison is possible"
            )
            continue

        differences = [current_items[i] - per_item[i] for i in shared]
        low, high = paired_bootstrap(differences)
        sd = statistics.stdev(differences) * 100 if len(differences) > 1 else 0.0

        deltas.append(
            CategoryDelta(
                category=category,
                baseline=base_mean,
                current=current_mean,
                delta_pp=(current_mean - base_mean) * 100,
                ci_low_pp=low,
                ci_high_pp=high,
                paired_sd_pp=sd,
                threshold_pp=thresholds[category],
            )
        )

    tripped = [d for d in deltas if d.trips()]
    for delta in tripped:
        reasons.append(
            f"{delta.category} regressed {abs(delta.delta_pp):.2f}pp "
            f"(CI [{delta.ci_low_pp:+.2f}, {delta.ci_high_pp:+.2f}]pp excludes zero)"
        )

    return GateResult(
        passed=not tripped and not undecided,
        deltas=deltas,
        reasons=reasons,
    )


def load_baseline(path: Path = BASELINE_PATH) -> Baseline:
    if not path.exists():
        raise FileNotFoundError(
            f"no baseline at {path}. Run `atlas_eval run --write-baseline` on a "
            "known-good commit first."
        )
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


# --- the noise floor -------------------------------------------------------------
#
# THRESHOLDS_PP come from this measurement:
# run the same code several times against the same corpus, and the per-item
# differences between those runs are pure run-to-run noise — the agent's own
# sampling, the judge's wobble. Their SD sets the smallest regression the gate
# can tell apart from that noise. Below it, a real regression and a bad roll of
# the dice look the same, and no threshold changes that.

# Two-sided 5% (the CI the gate uses) and 80% power, the usual design pair.
Z_TWO_SIDED_95 = 1.959964
Z_POWER_80 = 0.841621


class NoiseEstimate(BaseModel):
    """Run-to-run noise for one category, from reruns of identical code."""

    category: str
    pairs: int
    items: int
    # SD of per-item differences between two reruns, pooled over every pair.
    paired_sd_pp: float
    # The largest gap in the category *mean* between any two reruns — how far a
    # headline number moves when nothing changed.
    drift_pp: float
    # The smallest true regression the paired test catches 80% of the time.
    detectable_pp: float

    def verdict(self, threshold_pp: float) -> str:
        """Which limit actually decides what the gate can catch."""
        if self.detectable_pp <= threshold_pp:
            return "the threshold binds"
        return f"raise the threshold to at least {math.ceil(self.detectable_pp)}pp"


def noise_floor(runs: Sequence[Mapping[str, Mapping[str, float]]]) -> list[NoiseEstimate]:
    """Estimate per-category noise from two or more runs of the same code.

    ``runs`` are run files' ``per_item`` maps. A category absent from any run —
    tool use, for the single-call baseline — is skipped rather than reported as
    noiseless.
    """
    if len(runs) < 2:
        raise ValueError("noise needs at least two runs of the same code")

    estimates: list[NoiseEstimate] = []
    for category in CATEGORIES:
        if any(category not in run for run in runs):
            continue
        variances: list[float] = []
        drifts: list[float] = []
        sizes: list[int] = []
        for first, second in itertools.combinations(runs, 2):
            shared = sorted(set(first[category]) & set(second[category]))
            if len(shared) < 2:
                continue
            differences = [second[category][i] - first[category][i] for i in shared]
            variances.append(statistics.variance(differences))
            drifts.append(abs(statistics.fmean(differences)) * 100)
            sizes.append(len(shared))
        if not variances:
            continue
        sd_pp = math.sqrt(statistics.fmean(variances)) * 100
        items = min(sizes)
        estimates.append(
            NoiseEstimate(
                category=category,
                pairs=len(variances),
                items=items,
                paired_sd_pp=sd_pp,
                drift_pp=max(drifts),
                detectable_pp=(Z_TWO_SIDED_95 + Z_POWER_80) * sd_pp / math.sqrt(items),
            )
        )
    return estimates
