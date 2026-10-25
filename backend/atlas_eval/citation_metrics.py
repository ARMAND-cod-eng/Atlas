"""The domain eval's deterministic measures of an answer's citations of the law.

Three questions, answered against the library and the dataset's expected
citations, with no model and no judge:

* **validity**: does each provision the answer cites exist? A citation of
  "AI Act Art. 114" (the act has 113 articles) is fabricated, whatever else the
  answer gets right. The headline is the share of answers with at least one.
* **precision**: of the real provisions cited, how many are the law the
  question turns on (required or acceptable)? Real but beside-the-point law is
  how a confident answer pads itself out.
* **recall**: of the provisions an answer must cite, how many does it?

A citation matches an expected provision when one contains the other, the rule
the citation check uses for "retrieved": citing Article 6 covers 6(2), and
citing 6(3)(a) covers 6(3). Free to compute, so every variant is scored on
exactly the same terms, including plain Claude with no retrieval at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from atlas.library import Library, Unit, load_library
from atlas.library.citations import cited_labels
from atlas_eval.dataset import Dataset, ExpectedCitations
from atlas_eval.report import Interval, interval, paired_delta


class CitationScore(BaseModel):
    """One answer's citations, sorted into what they turned out to be."""

    item_id: str
    # Each distinct citation, in the order the answer first makes it.
    cited: list[str]
    # Cited, and the act has no such provision.
    nonexistent: list[str]
    # Real, but neither required nor acceptable for this question.
    unsupported: list[str]
    must_covered: list[str]
    must_missed: list[str]

    @property
    def fabricated(self) -> bool:
        return bool(self.nonexistent)

    @property
    def validity(self) -> float | None:
        """The share of citations that name real law; None for an answer citing none."""
        if not self.cited:
            return None
        return (len(self.cited) - len(self.nonexistent)) / len(self.cited)

    @property
    def precision(self) -> float | None:
        """The share of real citations that are on point; None when none are real."""
        real = len(self.cited) - len(self.nonexistent)
        if real == 0:
            return None
        return (real - len(self.unsupported)) / real

    @property
    def recall(self) -> float:
        expected = len(self.must_covered) + len(self.must_missed)
        return len(self.must_covered) / expected


def _related(library: Library, a: Unit, b: Unit) -> bool:
    return library.within(a.id, b.id) or library.within(b.id, a.id)


def score_answer(
    item_id: str, answer: str, expected: ExpectedCitations, library: Library
) -> CitationScore:
    cited = list(dict.fromkeys(cited_labels(answer)))
    units = {label: library.resolve(label) for label in cited}
    real = [unit for unit in units.values() if unit is not None]

    def expected_unit(label: str) -> Unit:
        unit = library.resolve(label)
        if unit is None:
            # The dataset's own labels are checked against the library in tests;
            # one that names nothing is a broken dataset, not a finding.
            raise ValueError(f"{item_id}: expected citation {label!r} is not in the library")
        return unit

    must = {label: expected_unit(label) for label in expected.must}
    on_point = [*must.values(), *(expected_unit(label) for label in expected.acceptable)]
    return CitationScore(
        item_id=item_id,
        cited=cited,
        nonexistent=[label for label, unit in units.items() if unit is None],
        unsupported=[
            label
            for label, unit in units.items()
            if unit is not None and not any(_related(library, unit, ok) for ok in on_point)
        ],
        must_covered=[
            label for label, unit in must.items() if any(_related(library, unit, c) for c in real)
        ],
        must_missed=[
            label
            for label, unit in must.items()
            if not any(_related(library, unit, c) for c in real)
        ],
    )


class CitationSummary(BaseModel):
    """A suite's citations: shares over every answer, quality over those that cite."""

    answers: int
    # Answers with at least one citation of a provision that does not exist.
    fabricated: Interval
    # Answers that cite no provision at all: they cannot fabricate one, so a
    # system that never cites would otherwise look flawless.
    uncited: Interval
    validity: Interval
    precision: Interval
    recall: Interval


def summarize_citations(scores: Sequence[CitationScore]) -> CitationSummary:
    def present(values: Sequence[float | None]) -> list[float]:
        return [value for value in values if value is not None]

    return CitationSummary(
        answers=len(scores),
        fabricated=interval([float(s.fabricated) for s in scores]),
        uncited=interval([float(not s.cited) for s in scores]),
        validity=interval(present([s.validity for s in scores])),
        precision=interval(present([s.precision for s in scores])),
        recall=interval([s.recall for s in scores]),
    )


def score_run(run: Mapping[str, Any], dataset: Dataset) -> list[CitationScore]:
    """Every answer in a run file scored afresh: the answers are the record, the
    scores are always today's scorer's."""
    library = load_library()
    by_id = {item.id: item for item in dataset.items}
    scores: list[CitationScore] = []
    for record in run["records"]:
        item = by_id[record["item_id"]]
        if item.citations is not None:
            scores.append(score_answer(item.id, record["answer"], item.citations, library))
    return scores


def _cell(value: Interval) -> str:
    return f"{value.value_pp:.1f}% [{value.low_pp:.1f}, {value.high_pp:.1f}]"


def _per_item(scores: Sequence[CitationScore], measure: str) -> dict[str, float]:
    if measure == "fabricated":
        return {s.item_id: float(s.fabricated) for s in scores}
    return {s.item_id: s.recall for s in scores}


def citation_table(runs: Mapping[str, Mapping[str, Any]], dataset: Dataset, reference: str) -> str:
    """Each run's citations as a Markdown table, then each against the reference.

    The comparisons are paired by item, so they ask what changed on the same
    questions, which is narrower and more honest than comparing two means.
    """
    scored = {name: score_run(run, dataset) for name, run in runs.items()}
    lines = [
        "| run | variant | answers | fabricated | uncited | validity | precision | recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, scores in scored.items():
        summary = summarize_citations(scores)
        lines.append(
            f"| {name} | {runs[name].get('variant', '?')} | {summary.answers} "
            f"| {_cell(summary.fabricated)} | {_cell(summary.uncited)} "
            f"| {_cell(summary.validity)} | {_cell(summary.precision)} "
            f"| {_cell(summary.recall)} |"
        )
    lines.append("")
    for name, scores in scored.items():
        if name == reference:
            continue
        parts = []
        for measure in ("fabricated", "recall"):
            delta = paired_delta(_per_item(scored[reference], measure), _per_item(scores, measure))
            parts.append(
                f"{measure} vs {reference}: {delta.value_pp:+.1f} pp "
                f"[{delta.low_pp:+.1f}, {delta.high_pp:+.1f}]"
            )
        lines.append(f"- {name}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"
