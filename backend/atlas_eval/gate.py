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

The first keeps the résumé's claim literally true — it still blocks on a >2%
category regression. The second is what stops it firing on noise.

And it **refuses to pass** a category it cannot evaluate. Without per-item
baseline scores there is no paired CI, and "no CI" used to mean "not
significant", which meant nothing could trip: an 80pp collapse passed, while the
report claimed the gate had fallen back to the point delta. CI never has the
per-item baseline unless ``runs/baseline.json`` is committed, so that was the
gate's behaviour on every run. A gate that cannot decide must say no.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from atlas_eval.dataset import DATA_DIR

BASELINE_PATH = DATA_DIR / "baseline.json"

CATEGORIES = ("completion", "grounding", "tools")

# Percentage points. Provisional until the measured paired SD says otherwise —
# see the README. If the measured SD exceeds this, the honest move is to raise it
# and say so, not to keep a number the data contradicts.
DEFAULT_THRESHOLD_PP = 2.0

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

    @property
    def significant(self) -> bool:
        """The bootstrap CI excludes zero."""
        return self.ci_low_pp > 0 or self.ci_high_pp < 0

    def trips(self, threshold_pp: float) -> bool:
        return self.delta_pp < -threshold_pp and self.significant


class GateResult(BaseModel):
    passed: bool
    threshold_pp: float
    deltas: list[CategoryDelta] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def report(self) -> str:
        lines = [f"gate: {'PASS' if self.passed else 'FAIL'} (threshold {self.threshold_pp}pp)"]
        for delta in self.deltas:
            marker = "FAIL" if delta.trips(self.threshold_pp) else "ok  "
            lines.append(
                f"  {marker} {delta.category:<11} "
                f"{delta.baseline:.3f} -> {delta.current:.3f}  "
                f"delta {delta.delta_pp:+.2f}pp  "
                f"95% CI [{delta.ci_low_pp:+.2f}, {delta.ci_high_pp:+.2f}]pp"
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
    threshold_pp: float = DEFAULT_THRESHOLD_PP,
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
        current_items = current_scores.get(category, {})
        if not current_items:
            reasons.append(f"{category}: no scores in this run")
            continue

        current_mean = statistics.fmean(current_items.values())
        base_mean = baseline.means.get(category)
        if base_mean is None:
            reasons.append(f"{category}: absent from baseline, not gated")
            continue

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
            )
        )

    tripped = [d for d in deltas if d.trips(threshold_pp)]
    for delta in tripped:
        reasons.append(
            f"{delta.category} regressed {abs(delta.delta_pp):.2f}pp "
            f"(CI [{delta.ci_low_pp:+.2f}, {delta.ci_high_pp:+.2f}]pp excludes zero)"
        )

    return GateResult(
        passed=not tripped and not undecided,
        threshold_pp=threshold_pp,
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
