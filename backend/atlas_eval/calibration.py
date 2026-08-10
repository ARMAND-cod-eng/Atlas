"""Judge-human agreement.

An LLM judge scoring against an uncalibrated rubric produces numbers, not
measurements. This module is what turns one into the other.

**Why labels are collected at the atomic level.** A human scoring "0.7" against a
rubric is making a soft, unrepeatable judgement, and agreement between two such
numbers needs an interval statistic with assumptions nobody checks. A human
ticking "is this specific fact present? y/n" is making a binary decision — so
agreement is a 2x2 table and Cohen's kappa applies without arm-waving.

**Why raw agreement and prevalence are reported alongside kappa.** Kappa corrects
for chance using the marginals, which makes it collapse when one class dominates —
the kappa paradox. Grounding labels are heavily skewed toward "supported", so 95%
raw agreement can sit beside a kappa of 0.5. Reporting either alone is how eval
numbers get oversold, so this reports all three and the README quotes all three.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Agreement(BaseModel):
    """Judge vs human over a set of binary decisions."""

    n: int
    # 2x2 counts: both true, both false, judge-only, human-only.
    both_true: int
    both_false: int
    judge_only: int
    human_only: int

    @property
    def observed(self) -> float:
        """Raw agreement — the number everyone understands."""
        if self.n == 0:
            return 0.0
        return (self.both_true + self.both_false) / self.n

    @property
    def expected(self) -> float:
        """Agreement expected by chance, from the marginals."""
        if self.n == 0:
            return 0.0
        judge_true = (self.both_true + self.judge_only) / self.n
        human_true = (self.both_true + self.human_only) / self.n
        return judge_true * human_true + (1 - judge_true) * (1 - human_true)

    @property
    def kappa(self) -> float:
        """Cohen's kappa.

        Returns 1.0 when chance agreement is already perfect — the degenerate case
        where both raters marked everything the same way. The usual formula is
        0/0 there, and reporting 0.0 would call perfect agreement "no better than
        chance", which is exactly backwards.
        """
        expected = self.expected
        if expected >= 1.0:
            return 1.0 if self.observed >= 1.0 else 0.0
        return (self.observed - expected) / (1 - expected)

    @property
    def human_prevalence(self) -> float:
        """Share of decisions the human marked true. Context for the kappa."""
        if self.n == 0:
            return 0.0
        return (self.both_true + self.human_only) / self.n

    def summary(self) -> str:
        return (
            f"n={self.n}  raw={self.observed:.1%}  kappa={self.kappa:.3f}  "
            f"prevalence={self.human_prevalence:.1%}"
        )


def agreement(judge_labels: list[bool], human_labels: list[bool]) -> Agreement:
    """Build the 2x2 table. Both lists must be aligned decision-for-decision."""
    if len(judge_labels) != len(human_labels):
        raise ValueError(
            f"misaligned labels: {len(judge_labels)} judge vs {len(human_labels)} human"
        )
    both_true = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j and h)
    both_false = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if not j and not h)
    judge_only = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j and not h)
    human_only = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if h and not j)
    return Agreement(
        n=len(judge_labels),
        both_true=both_true,
        both_false=both_false,
        judge_only=judge_only,
        human_only=human_only,
    )


# Decided before looking at any data, so the threshold cannot be chosen to make
# the result pass. Landis & Koch put 0.61-0.80 at "substantial"; 0.70 is mid-band.
# Below 0.60 the judge and a human are meaningfully disagreeing about the same
# rubric, which would make every score downstream an opinion. That banding is a
# widely used convention, not a law of nature, and the README says so.
KAPPA_THRESHOLD = 0.70


class CalibrationReport(BaseModel):
    """Agreement per scored dimension, on the iteration set and the holdout.

    The holdout exists because iterating a judge prompt against the same items it
    is then scored on overfits the prompt. If holdout kappa is materially worse,
    that is what happened, and the report says so rather than quoting the better
    number.
    """

    dimension: str
    iteration: Agreement
    holdout: Agreement
    judge_prompt_version: int = 1
    # Disagreement between repeated judge runs on identical inputs. Temperature 0
    # is not a determinism guarantee, so it is measured rather than assumed.
    judge_self_disagreement: float = 0.0

    @property
    def passes(self) -> bool:
        return self.holdout.kappa >= KAPPA_THRESHOLD

    @property
    def overfit_warning(self) -> bool:
        """True when the prompt looks tuned to the calibration set."""
        return self.iteration.kappa - self.holdout.kappa > 0.15

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "BELOW THRESHOLD"
        lines = [
            f"{self.dimension}: {verdict} (threshold kappa >= {KAPPA_THRESHOLD})",
            f"  iteration set : {self.iteration.summary()}",
            f"  holdout       : {self.holdout.summary()}",
            f"  judge self-disagreement: {self.judge_self_disagreement:.1%}",
        ]
        if self.overfit_warning:
            lines.append(
                "  WARNING: iteration kappa exceeds holdout by >0.15 — the prompt "
                "looks tuned to the calibration set"
            )
        return "\n".join(lines)


class HumanLabel(BaseModel):
    """One binary decision a human made, stored one per line."""

    item_id: str
    dimension: str  # "completion" | "grounding"
    # Rubric entry id, or the atomic claim text for grounding.
    decision_id: str
    subject: str
    human: bool
    labeled_at: str
    split: str = "iteration"  # "iteration" | "holdout"


class LabelSet(BaseModel):
    labels: list[HumanLabel] = Field(default_factory=list)

    def for_dimension(self, dimension: str, split: str) -> list[HumanLabel]:
        return [
            label for label in self.labels if label.dimension == dimension and label.split == split
        ]
