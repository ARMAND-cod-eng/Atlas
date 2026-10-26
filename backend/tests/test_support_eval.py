"""Measuring the citation support check: is its "does not support" right?

The cases are labelled, and the labels are themselves checked: every provision
exists, a "supports" case names the words that prove it, and those words are in
the provision. A label anyone can verify is one a measurement can rest on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage

from atlas.library import load_library
from atlas.library.citations import cited_labels, quotes
from atlas.library.support import SupportJudgement, SupportJudgements
from atlas_eval import cli
from atlas_eval.support_eval import (
    Judged,
    Outcome,
    SupportCase,
    load_support_cases,
    outcome_of,
    run_support,
    score_support,
)

CASES = load_support_cases()
LIBRARY = load_library()


def test_the_cases_are_balanced_and_every_label_names_real_law() -> None:
    assert (
        sum(c.supported for c in CASES),
        sum(not c.supported for c in CASES),
    ) == (12, 12)
    for case in CASES:
        assert LIBRARY.resolve(case.label) is not None, case.id
        assert cited_labels(case.sentence) == [case.label], case.id
        if case.correct is not None:
            assert LIBRARY.resolve(case.correct) is not None, case.id


def test_each_supports_label_is_proved_by_the_provisions_own_words() -> None:
    for case in CASES:
        unit = LIBRARY.resolve(case.label)
        assert unit is not None
        if case.supported:
            assert case.evidence is not None, case.id
            assert quotes(unit.text, case.evidence), case.id
        else:
            assert case.evidence is None, case.id


def _case(supported: bool) -> SupportCase:
    return next(c for c in CASES if c.supported is supported)


@pytest.mark.parametrize(
    ("supported", "judged", "outcome"),
    [
        (False, "unsupported", "caught"),
        (False, "supported", "missed"),
        (False, "unverified", "missed"),
        (False, "unchecked", "unanswered"),
        (True, "supported", "passed"),
        (True, "unsupported", "false_alarm"),
        (True, "unverified", "unverified"),
        (True, "unchecked", "unanswered"),
    ],
)
def test_what_each_judgement_counts_as(supported: bool, judged: Judged, outcome: Outcome) -> None:
    assert outcome_of(_case(supported), judged) == outcome


def test_the_score_counts_what_matters_both_ways() -> None:
    no, yes = _case(False), _case(True)
    outcomes: list[tuple[SupportCase, Outcome]] = [
        (no, "caught"),
        (no, "caught"),
        (no, "missed"),
        (yes, "passed"),
        (yes, "passed"),
        (yes, "false_alarm"),
        (yes, "unverified"),
    ]

    score = score_support([outcome for _, outcome in outcomes])

    assert score.model_dump() == {
        "cases": 7,
        "caught": 2,
        "missed": 1,
        "false_alarm": 1,
        "passed": 2,
        "unverified": 1,
        "unanswered": 0,
        # Of the "does not support" verdicts, how many were right; of the
        # citations that do not support, how many were caught.
        "precision": 2 / 3,
        "recall": 2 / 3,
        # Of the citations that do support, how many were sent back for nothing.
        "false_alarm_rate": 1 / 4,
    }


class ScriptedJudge:
    """Judges by the case's own label, and keeps how many times it was asked."""

    def __init__(self, cases: list[SupportCase]) -> None:
        self.by_sentence = {c.sentence: c for c in cases}
        self.calls = 0

    async def __call__(self, messages: list[BaseMessage]) -> SupportJudgements:
        self.calls += 1
        prompt = str(messages[-1].content)
        case = next(c for s, c in self.by_sentence.items() if s in prompt)
        if case.supported:
            return SupportJudgements(
                judgements=[SupportJudgement(item=1, supported=True, quote=case.evidence or "")]
            )
        return SupportJudgements(
            judgements=[SupportJudgement(item=1, supported=False, reason="no")]
        )


async def test_a_judge_that_is_always_right_scores_perfectly_one_case_a_call() -> None:
    judge = ScriptedJudge(CASES)

    results = await run_support(CASES, judge)

    assert judge.calls == len(CASES)
    assert [r.outcome for r in results] == ["passed" if c.supported else "caught" for c in CASES]
    assert (score_support([r.outcome for r in results]).precision, judge.calls) == (1.0, 24)


def test_the_command_writes_its_results_and_says_what_the_cases_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def judge_factory(metrics: Any) -> ScriptedJudge:
        return ScriptedJudge(CASES)

    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "support_judge", judge_factory)

    code = cli.main(["support", "--name", "first"])

    out = capsys.readouterr().out
    written = json.loads((tmp_path / "support-first.json").read_text(encoding="utf-8"))
    assert code == 0
    assert "24 model-drafted cases, 0 reviewed by a person" in out
    assert "precision 100.0%" in out
    assert written["score"]["caught"] == 12
    assert [r["id"] for r in written["results"]] == [c.id for c in CASES]
