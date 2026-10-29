"""Measuring the contract review against contracts whose gaps are known.

Eight processor contracts (``data/contracts/dpa-0N.md``), written for this eval,
each with the status a careful reader would give each GDPR Art. 28(3) term and
the clauses that meet the terms met or partly met (``expected.json``). They are
model-drafted and not yet reviewed by a lawyer, and the results say so.

Two measurements:

* **Candidate recall**, free: whether the clause that meets a term is among
  the clauses the model is shown for it. A clause never shown cannot be found,
  so this bounds everything after it.
* **The review**, paid (``--model``): the review graph itself, one model call
  per term, scored against the key. Counted both ways, because both cost:

  - **gaps caught**: of the terms partly met or missing, how many the review
    did not pass as met. A gap passed goes into a signed contract.
  - **missing terms passed as met**: the worst error, counted on its own.
  - **false alarms**: of the terms met, how many it said were partial or
    missing. A false alarm costs a lawyer's time, not a gap.
  - **unverified**: findings whose quote is not in the contract. Not passed,
    and not wrong either: a person has to read the clause.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from atlas.library import load_library
from atlas.metrics import RunMetrics
from atlas.review.contract import clauses_of, text_of
from atlas.review.graph import build_review_graph, review_result
from atlas.review.judge import Finding, candidates
from atlas.review.requirements import GDPR_ART_28, provision_text
from atlas_eval.dataset import DATA_DIR
from atlas_eval.runner import install_offline_budget

CONTRACTS_DIR = DATA_DIR / "contracts"
KEY_PATH = CONTRACTS_DIR / "expected.json"


class Expected(BaseModel):
    status: Literal["met", "partial", "missing"]
    clauses: list[str] = []


class ContractCase(BaseModel):
    id: str
    filename: str
    text: str
    why: str
    # In the framework's order.
    expected: dict[str, Expected]


def load_contract_cases(key_path: Path = KEY_PATH) -> list[ContractCase]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    order = [requirement.id for requirement in GDPR_ART_28.requirements]
    cases = []
    for contract_id, entry in sorted(key["contracts"].items()):
        path = key_path.parent / f"{contract_id}.md"
        expected = {rid: Expected.model_validate(entry["expected"][rid]) for rid in order}
        cases.append(
            ContractCase(
                id=contract_id,
                filename=path.name,
                text=text_of(path.name, path.read_bytes()),
                why=entry["why"],
                expected=expected,
            )
        )
    return cases


class Recall(BaseModel):
    found: int
    total: int
    misses: list[str]


def candidate_recall(cases: Sequence[ContractCase]) -> Recall:
    """Of the terms met or partly met, how many have their clause among the candidates."""
    library = load_library()
    found = total = 0
    misses: list[str] = []
    for case in cases:
        clauses = clauses_of(case.text)
        for requirement in GDPR_ART_28.requirements:
            gold = case.expected[requirement.id].clauses
            if not gold:
                continue
            total += 1
            shown = {
                c.label
                for c in candidates(requirement, provision_text(requirement, library), clauses)
            }
            if shown & set(gold):
                found += 1
            else:
                misses.append(f"{case.id} {requirement.id}")
    return Recall(found=found, total=total, misses=misses)


async def review_contract(case: ContractCase) -> tuple[list[Finding], RunMetrics]:
    """The review graph on one contract, in this process: its findings and its metrics."""
    install_offline_budget()
    final = await build_review_graph().ainvoke(
        {
            "filename": case.filename,
            "text": case.text,
            "framework": GDPR_ART_28.id,
            "clauses": [],
            "findings": [],
            "run_metrics": RunMetrics(),
        },
        config={
            "configurable": {
                "thread_id": f"eval-{case.id}",
                "run_id": f"eval-{case.id}-{uuid4().hex[:8]}",
                "domain": "compliance",
            }
        },
    )
    review = review_result(dict(final), f"eval-{case.id}")
    return review.findings, review.metrics


class ContractScore(BaseModel):
    terms: int
    agreed: int
    gaps: int
    gaps_caught: int
    missing_passed_as_met: int
    met_terms: int
    false_alarms: int
    unverified: int
    # The terms the review and the key disagree on: "dpa-04 security: key partial, review met".
    disagreements: list[str]


def score_contracts(
    cases: Sequence[ContractCase], predicted: Mapping[str, Sequence[Finding]]
) -> ContractScore:
    terms = agreed = gaps = caught = passed_missing = met_terms = alarms = unverified = 0
    disagreements: list[str] = []
    for case in cases:
        found = {finding.requirement_id: finding.status for finding in predicted.get(case.id, [])}
        for requirement_id, expected in case.expected.items():
            # A term with no finding was not judged: counted as flagged, never as met.
            got = found.get(requirement_id, "unverified")
            terms += 1
            agreed += got == expected.status
            if got != expected.status:
                disagreements.append(
                    f"{case.id} {requirement_id}: key {expected.status}, review {got}"
                )
            if expected.status != "met":
                gaps += 1
                caught += got != "met"
            passed_missing += expected.status == "missing" and got == "met"
            if expected.status == "met":
                met_terms += 1
                alarms += got in ("partial", "missing")
            unverified += got == "unverified"
    return ContractScore(
        terms=terms,
        agreed=agreed,
        gaps=gaps,
        gaps_caught=caught,
        missing_passed_as_met=passed_missing,
        met_terms=met_terms,
        false_alarms=alarms,
        unverified=unverified,
        disagreements=disagreements,
    )
