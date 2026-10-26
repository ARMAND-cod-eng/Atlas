"""Measuring the citation support check (atlas.library.support).

The check is a model's judgement, so its verdicts are counted against labelled
cases: a sentence, the provision it cites, and whether that provision really
says it. Half the cases are supported, in plain words rather than the act's,
so the judge has to read. Half are near misses: a real provision cited for what
a neighbouring paragraph says, or for something its own text contradicts.

What matters is counted both ways:

* **precision**: of the citations judged "does not support", the share that
  really do not. A wrong one costs a rewrite of a correct answer.
* **recall**: of the citations that do not support their sentence, the share
  caught. A miss goes out as before.
* **false-alarm rate**: of the citations that do support, the share sent back.

A support claim whose quote is not in the provision is *unverified*: it sends
nothing back, so on a case that does not support it counts as a miss.

Each case is judged in its own call, so one case never colours another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from atlas.library import load_library
from atlas.library.citations import CitationReport
from atlas.library.support import Judge, SupportReport, check_support
from atlas_eval.dataset import DATA_DIR

SUPPORT_CASES_PATH = DATA_DIR / "support_cases.v1.jsonl"

Judged = Literal["supported", "unsupported", "unverified", "unchecked"]
Outcome = Literal["caught", "missed", "false_alarm", "passed", "unverified", "unanswered"]


class SupportCase(BaseModel):
    """One labelled citation."""

    id: str
    label: str
    sentence: str
    supported: bool
    # For a supported case: words of the provision that prove it.
    evidence: str | None = None
    # For a near miss: the provision that does say it, when one does.
    correct: str | None = None
    note: str = ""
    # Who wrote the case, and who has checked it. The v1 cases were drafted by a
    # model from the provisions' text; until a person reviews them, the results
    # say so.
    drafted_by: str = "model"
    reviewed_by: str | None = None


def load_support_cases(path: Path = SUPPORT_CASES_PATH) -> list[SupportCase]:
    cases = [
        SupportCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate support case ids in {path.name}")
    return cases


def judged_as(report: SupportReport) -> Judged:
    """What the check concluded about a report's single citation."""
    if report.unsupported:
        return "unsupported"
    if report.supported:
        return "supported"
    if report.unverified:
        return "unverified"
    return "unchecked"


def outcome_of(case: SupportCase, judged: Judged) -> Outcome:
    if judged == "unchecked":
        return "unanswered"
    if case.supported:
        if judged == "supported":
            return "passed"
        return "false_alarm" if judged == "unsupported" else "unverified"
    return "caught" if judged == "unsupported" else "missed"


class SupportResult(BaseModel):
    id: str
    label: str
    supported: bool
    judged: Judged
    outcome: Outcome
    quote: str = ""
    reason: str = ""


async def run_support(cases: list[SupportCase], judge: Judge) -> list[SupportResult]:
    """Judge every case, one call each, as the critic would judge its citation."""
    library = load_library()
    results: list[SupportResult] = []
    for case in cases:
        report = await check_support(
            case.sentence, CitationReport(cited=[case.label]), library, judge
        )
        judged = judged_as(report)
        entries = report.unsupported or report.supported or report.unverified
        entry = entries[0] if entries else None
        results.append(
            SupportResult(
                id=case.id,
                label=case.label,
                supported=case.supported,
                judged=judged,
                outcome=outcome_of(case, judged),
                quote=entry.quote if entry is not None else "",
                reason=entry.reason if entry is not None else "",
            )
        )
    return results


class SupportScore(BaseModel):
    cases: int
    caught: int
    missed: int
    false_alarm: int
    passed: int
    unverified: int
    unanswered: int
    precision: float | None
    recall: float | None
    false_alarm_rate: float | None


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def score_support(outcomes: list[Outcome]) -> SupportScore:
    def count(outcome: Outcome) -> int:
        return outcomes.count(outcome)

    caught, missed, false_alarm = count("caught"), count("missed"), count("false_alarm")
    passed, unverified = count("passed"), count("unverified")
    return SupportScore(
        cases=len(outcomes),
        caught=caught,
        missed=missed,
        false_alarm=false_alarm,
        passed=passed,
        unverified=unverified,
        unanswered=count("unanswered"),
        precision=_share(caught, caught + false_alarm),
        recall=_share(caught, caught + missed),
        false_alarm_rate=_share(false_alarm, false_alarm + passed + unverified),
    )
