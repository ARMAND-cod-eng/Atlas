"""Judging one requirement against a contract: a model reads, the contract's words decide.

For each requirement, the clauses most like it are found by ranking (BM25, as
the library ranks the law), and a model judges from those clauses alone whether
the contract meets it, meets part of it, or does not.

A "met" or a "partial" must quote the clause that meets it. The quote is then
looked for in the clause it cites (``atlas.library.citations.quotes``, the same
test the law's quotations get), and a finding whose quote is not there is
reported as **unverified**, never as met: a clause claimed that is not in the
contract is the failure this check exists to catch. A requirement no clause
shares a word with is missing without a model call, and says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from atlas import prompts
from atlas.library.citations import quotes
from atlas.library.glossary import expand
from atlas.library.search import legal_terms
from atlas.review.contract import Clause
from atlas.review.requirements import Requirement
from atlas.tools.rank import BM25

# Enough for the clause that meets a requirement to be among them when the
# contract splits it ("Deletion" in one clause, "return" in the next).
CANDIDATES = 6

Status = Literal["met", "partial", "missing", "unverified"]


class RequirementJudgement(BaseModel):
    """What the model returns for one requirement."""

    status: Literal["met", "partial", "missing"]
    clause_ids: list[str] = Field(
        default_factory=list, description="The ids of the clauses that meet it; none when missing."
    )
    quote: str | None = Field(
        default=None, description="The contract's words that meet it, exactly; none when missing."
    )
    reason: str = Field(description="One sentence a lawyer can check against the clause.")


class Finding(BaseModel):
    """One requirement, as a reviewer sees it: the status a reviewer can rely on."""

    requirement_id: str
    label: str
    title: str
    status: Status
    clause_ids: list[str] = Field(default_factory=list)
    quote: str | None = None
    # The clause the quote was found in, when it was.
    found_in: str | None = None
    reason: str


def candidates(requirement: Requirement, provision: str, clauses: Sequence[Clause]) -> list[Clause]:
    """The clauses most like the requirement, best first; only those sharing a word with it."""
    if not clauses:
        return []
    index = BM25([legal_terms(expand(clause.words)) for clause in clauses])
    terms = legal_terms(expand(f"{requirement.title} {provision}"))
    scored = [(index.score(position, terms), position) for position in range(len(clauses))]
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
    return [clauses[position] for _, position in ranked[:CANDIDATES]]


def judge_messages(
    requirement: Requirement, provision: str, shown: Sequence[Clause]
) -> list[BaseMessage]:
    clauses = "\n\n".join(
        f"[{clause.id}] {clause.label}"
        + (f" — {clause.heading}" if clause.heading else "")
        + f"\n{clause.text}"
        for clause in shown
    )
    return [
        SystemMessage(prompts.REVIEWER),
        HumanMessage(
            f"Requirement: {requirement.label} — {requirement.title}\n"
            f"The law: {provision}\n\nClauses:\n{clauses}"
        ),
    ]


def unaddressed(requirement: Requirement) -> Finding:
    """A requirement no clause shares a word with: missing, and no model was asked."""
    return Finding(
        requirement_id=requirement.id,
        label=requirement.label,
        title=requirement.title,
        status="missing",
        reason=(
            "No clause of the contract uses the words of this requirement, so no model was asked."
        ),
    )


def verified(
    judgement: RequirementJudgement, requirement: Requirement, clauses: Mapping[str, Clause]
) -> Finding:
    """The model's judgment, held to the contract's words."""
    base = {
        "requirement_id": requirement.id,
        "label": requirement.label,
        "title": requirement.title,
    }
    if judgement.status == "missing":
        return Finding(**base, status="missing", reason=judgement.reason)

    cited = [clause_id for clause_id in judgement.clause_ids if clause_id in clauses]
    quote = (judgement.quote or "").strip()
    found_in = next(
        (clause_id for clause_id in cited if quote and quotes(clauses[clause_id].words, quote)),
        None,
    )
    if found_in is None:
        why = (
            "it cites no clause of this contract"
            if not cited
            else "it quotes nothing"
            if not quote
            else "its quote is not in the clause it cites"
        )
        return Finding(
            **base,
            status="unverified",
            clause_ids=cited,
            quote=quote or None,
            reason=f"Judged {judgement.status}, but {why}. The model said: {judgement.reason}",
        )
    return Finding(
        **base,
        status=judgement.status,
        clause_ids=cited,
        quote=quote,
        found_in=found_in,
        reason=judgement.reason,
    )
