"""Measuring the contract review: eight contracts with known gaps, and what it found.

The answer key must be one a run can be held to: every term of every contract
labelled, every clause it names in its contract. Candidate recall, free, is
whether the clause that meets a term is among those the model is shown. The
paid score counts what matters both ways, the gaps caught and the terms
wrongly passed, and is tested here with a scripted model.
"""

from __future__ import annotations

from typing import Any

import pytest

from atlas import models
from atlas.offline import no_fallback, scripted_structured
from atlas.review.contract import clauses_of
from atlas.review.judge import Finding, RequirementJudgement
from atlas.review.requirements import GDPR_ART_28
from atlas_eval.contracts_eval import (
    candidate_recall,
    load_contract_cases,
    review_contract,
    score_contracts,
)

CASES = load_contract_cases()
REQUIREMENTS = [requirement.id for requirement in GDPR_ART_28.requirements]


def test_the_key_labels_every_term_of_every_contract_with_clauses_it_has() -> None:
    assert [case.id for case in CASES] == [f"dpa-0{n}" for n in range(1, 9)]
    for case in CASES:
        assert list(case.expected) == REQUIREMENTS, case.id
        labels = {clause.label for clause in clauses_of(case.text)}
        for requirement, expected in case.expected.items():
            assert set(expected.clauses) <= labels, (case.id, requirement)
            # A term met or partly met names the clause that meets it; a missing one, none.
            assert bool(expected.clauses) == (expected.status != "missing"), (case.id, requirement)


def test_the_clause_that_meets_each_term_is_among_those_the_model_is_shown() -> None:
    """In-sample: the requirements' contract words were added after two misses here."""
    recall = candidate_recall(CASES)

    assert (recall.found, recall.total, recall.misses) == (56, 56, [])


def _finding(requirement: str, status: str) -> Finding:
    return Finding(
        requirement_id=requirement, label=requirement, title=requirement, status=status, reason="r"
    )


def test_the_score_counts_gaps_caught_and_terms_wrongly_passed() -> None:
    expected = {case.id: case for case in CASES[:1]}["dpa-01"]
    # dpa-01: met except sub-processors and end (partial), assistance and audits (missing).
    predicted = {
        "dpa-01": [
            _finding("details", "met"),
            _finding("instructions", "met"),
            _finding("confidentiality", "partial"),  # a false alarm
            _finding("security", "unverified"),  # flagged for a person: not a false alarm
            _finding("sub-processors", "met"),  # a gap passed
            _finding("data-subject-rights", "met"),
            _finding("assistance", "met"),  # the worst: a missing term passed
            _finding("end", "unverified"),  # flagged, not passed
            _finding("audits", "missing"),
        ]
    }

    score = score_contracts([expected], predicted)

    assert (score.terms, score.agreed) == (9, 4)
    assert (score.gaps, score.gaps_caught) == (4, 2)
    assert score.missing_passed_as_met == 1
    assert (score.met_terms, score.false_alarms) == (5, 1)
    assert score.unverified == 2


async def test_a_contract_is_reviewed_by_the_review_graph_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nothing_found(node: str, messages: list[Any], schema: type[Any], metrics: Any) -> Any:
        return RequirementJudgement(status="missing", reason="Not addressed.")

    monkeypatch.setattr(models, "get_structured_llm", scripted_structured(nothing_found))
    monkeypatch.setattr(models, "get_structured_fallback", no_fallback)
    [case] = [case for case in CASES if case.id == "dpa-02"]

    findings, _metrics = await review_contract(case)

    assert [f.requirement_id for f in findings] == REQUIREMENTS
    assert {f.status for f in findings} == {"missing"}
    # A complete contract judged to lack everything: nine false alarms, no gaps to catch.
    score = score_contracts([case], {case.id: findings})
    assert (score.false_alarms, score.gaps) == (9, 0)
