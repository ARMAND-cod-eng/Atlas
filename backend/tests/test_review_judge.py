"""Judging a contract against the law's requirements, and holding the judgment to its words.

A model decides whether a clause meets a requirement. What it may not do is say
so without the clause: a "met" or "partial" must quote the contract, the quote
is looked for in the clause it cites, and one that is not there is reported as
unverified, never as met. These tests drive the real graph and runner with a
scripted model, so what they check is what a review run does.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from atlas import eventlog, models
from atlas.events import AGENT_CALL, REQUIREMENT_JUDGED, REVIEW_STARTED, RUN_COMPLETED
from atlas.library import load_library
from atlas.metrics import RunMetrics
from atlas.offline import no_fallback, scripted_structured
from atlas.review import graph as review
from atlas.review.contract import clauses_of
from atlas.review.judge import Finding, RequirementJudgement, candidates, verified
from atlas.review.requirements import GDPR_ART_28, provision_text
from atlas.runner import execute_review
from atlas.store import InMemoryRunStore
from tests.conftest import FakeRedis

CONTRACT = """DATA PROCESSING AGREEMENT

1. Subject matter and duration
The Processor processes customer contact data of the Controller's clients, for the duration of the
services agreement, for the purpose of operating the support desk.

2. Instructions
The Processor shall process personal data only on documented instructions from the Controller.

3. Confidentiality
The Processor shall ensure that its staff are bound by an obligation of confidentiality.

4. Security
The Processor shall implement the technical and organisational measures required by Article 32 GDPR.

5. Sub-processors
The Processor shall not engage another processor without the prior written authorisation of the
Controller.

6. End of the services
At the end of the services the Processor shall delete all personal data.
"""

CLAUSES = clauses_of(CONTRACT)
BY_ID = {clause.id: clause for clause in CLAUSES}
REQUIREMENT = {requirement.id: requirement for requirement in GDPR_ART_28.requirements}
LIBRARY = load_library()


def _clause(label: str) -> str:
    return next(clause.id for clause in CLAUSES if clause.label == label)


# --- which clauses a requirement is judged against --------------------------------------


@pytest.mark.parametrize(
    ("requirement", "clause"),
    [
        ("instructions", "Clause 2"),
        ("confidentiality", "Clause 3"),
        ("security", "Clause 4"),
        ("sub-processors", "Clause 5"),
        ("end", "Clause 6"),
    ],
)
def test_the_clause_that_meets_a_requirement_is_among_its_candidates(
    requirement: str, clause: str
) -> None:
    found = candidates(
        REQUIREMENT[requirement], provision_text(REQUIREMENT[requirement], LIBRARY), CLAUSES
    )

    assert found[0].label == clause


# --- holding a judgment to the contract's words ------------------------------------------

INSTRUCTIONS = REQUIREMENT["instructions"]


def _judged(**fields: Any) -> Finding:
    return verified(RequirementJudgement.model_validate(fields), INSTRUCTIONS, BY_ID)


def test_a_met_whose_quote_is_in_the_clause_it_cites_is_met() -> None:
    finding = _judged(
        status="met",
        clause_ids=[_clause("Clause 2")],
        quote="only on documented instructions from the Controller",
        reason="Clause 2 says so.",
    )

    assert (finding.status, finding.found_in, finding.quote) == (
        "met",
        _clause("Clause 2"),
        "only on documented instructions from the Controller",
    )


def test_a_met_whose_quote_is_not_in_the_contract_is_unverified() -> None:
    finding = _judged(
        status="met",
        clause_ids=[_clause("Clause 2")],
        quote="including transfers to a third country",
        reason="Clause 2 covers transfers.",
    )

    assert (finding.status, finding.found_in) == ("unverified", None)
    assert finding.reason == (
        "Judged met, but its quote is not in the clause it cites. "
        "The model said: Clause 2 covers transfers."
    )


def test_a_met_citing_a_clause_the_contract_does_not_have_is_unverified() -> None:
    finding = _judged(status="met", clause_ids=["c99"], quote="documented instructions", reason="r")

    assert (finding.status, finding.clause_ids) == ("unverified", [])
    assert finding.reason.startswith("Judged met, but it cites no clause of this contract.")


def test_a_partial_is_held_to_its_quote_too() -> None:
    finding = verified(
        RequirementJudgement(
            status="partial",
            clause_ids=[_clause("Clause 6")],
            quote="the Processor shall delete all personal data",
            reason="Deletion, but no return.",
        ),
        REQUIREMENT["end"],
        BY_ID,
    )

    assert (finding.status, finding.found_in) == ("partial", _clause("Clause 6"))


def test_a_missing_says_nothing_of_clauses() -> None:
    finding = _judged(
        status="missing", clause_ids=[_clause("Clause 2")], quote="anything", reason="Absent."
    )

    assert (finding.status, finding.clause_ids, finding.quote, finding.found_in) == (
        "missing",
        [],
        None,
        None,
    )


# --- a review run --------------------------------------------------------------------------


def install(monkeypatch: pytest.MonkeyPatch, model: Scripted) -> None:
    """Scripted beneath the model layer, so the real call path runs and records each call."""
    monkeypatch.setattr(models, "get_structured_llm", scripted_structured(model))
    monkeypatch.setattr(models, "get_structured_fallback", no_fallback)


class Scripted:
    """The reviewer's model: missing, except where told otherwise."""

    def __init__(self, answers: dict[str, RequirementJudgement]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def __call__(
        self,
        node: str,
        messages: list[BaseMessage],
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        assert schema is RequirementJudgement
        prompt = str(messages[-1].content)
        label = prompt.split("\n", 1)[0].removeprefix("Requirement: ").split(" — ")[0]
        self.asked.append(label)
        requirement = next(r for r in GDPR_ART_28.requirements if r.label == label)
        return self.answers.get(
            requirement.id,
            RequirementJudgement(status="missing", reason=f"{label} is not addressed."),
        )


async def test_a_review_judges_every_requirement_and_keeps_the_record(
    monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    model = Scripted(
        {
            "instructions": RequirementJudgement(
                status="met",
                clause_ids=[_clause("Clause 2")],
                quote="only on documented instructions from the Controller",
                reason="Clause 2.",
            ),
            "end": RequirementJudgement(
                status="met",
                clause_ids=[_clause("Clause 6")],
                quote="delete and return",
                reason="Clause 6.",
            ),
        }
    )
    install(monkeypatch, model)
    store = InMemoryRunStore()

    result = await execute_review(
        "review-1",
        "thread-r",
        review.build_review_graph(),
        filename="dpa.txt",
        text=CONTRACT,
        store=store,
    )

    statuses = {finding["requirement_id"]: finding["status"] for finding in result["findings"]}
    # In the framework's order, one each.
    assert [f["requirement_id"] for f in result["findings"]] == [
        r.id for r in GDPR_ART_28.requirements
    ]
    assert statuses["instructions"] == "met"
    # Its quote, "delete and return", is not in Clause 6: never reported as met.
    assert statuses["end"] == "unverified"
    assert result["counts"] == {"met": 1, "partial": 0, "missing": 7, "unverified": 1}

    events = await eventlog.replay("review-1", 0)
    types = [event.type for event in events]
    assert types.count(REVIEW_STARTED) == 1
    assert types.count(REQUIREMENT_JUDGED) == len(GDPR_ART_28.requirements)
    assert types[-1] == RUN_COMPLETED
    [started] = [event for event in events if event.type == REVIEW_STARTED]
    assert [c["label"] for c in started.payload["clauses"]] == [c.label for c in CLAUSES]

    record = await store.get("review-1")
    assert record is not None
    assert (record.kind, record.query, record.status, record.domain) == (
        "review",
        "dpa.txt",
        "completed",
        "compliance",
    )
    # Every judgment's prompt is kept, under its requirement, for the inspector.
    calls = await store.calls("review-1")
    assert sorted(call.subtask_id or "" for call in calls) == sorted(
        r.id for r in GDPR_ART_28.requirements
    )
    assert AGENT_CALL in types


async def test_a_requirement_no_clause_comes_near_is_missing_without_asking_the_model(
    monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis
) -> None:
    model = Scripted({})
    install(monkeypatch, model)

    result = await execute_review(
        "review-2",
        "thread-s",
        review.build_review_graph(),
        filename="note.txt",
        text="1. Lunch\nThe parties will meet for lunch on Friday.",
    )

    assert {f["status"] for f in result["findings"]} == {"missing"}
    # Asked only about the requirements some clause shares words with.
    assert len(model.asked) < len(GDPR_ART_28.requirements)
    reasons = {f["requirement_id"]: f["reason"] for f in result["findings"]}
    assert "No clause of the contract uses the words of this requirement" in reasons["audits"]
