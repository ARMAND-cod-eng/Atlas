"""A contract review, as a graph of its own: read the contract, then judge each requirement.

``reader`` cuts the contract into clauses and says what the review is against;
``reviewer`` runs once per requirement, in parallel (a ``Send`` each, as the
researchers fan out), and emits its finding as it lands. A graph rather than a
loop so a review is run by the same machinery as a research run
(``atlas.runner``): its events stream and replay, every model call is recorded
for the inspector, and the run is kept in the record like any other.

No checkpointer: a review has no conversation to continue, and its result is
kept in the run record.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel

from atlas.events import REQUIREMENT_JUDGED, REVIEW_STARTED, emit, node_span
from atlas.library import load_library
from atlas.metrics import RunMetrics, merge_run_metrics
from atlas.models import call_structured
from atlas.result import validate_list, validate_metrics
from atlas.review.contract import Clause, clauses_of
from atlas.review.judge import (
    Finding,
    RequirementJudgement,
    candidates,
    judge_messages,
    unaddressed,
    verified,
)
from atlas.review.requirements import FRAMEWORKS, Requirement, provision_text

logger = logging.getLogger(__name__)

READER = "reader"
REVIEWER = "reviewer"

STATUSES = ("met", "partial", "missing", "unverified")


def merge_findings(left: list[Finding], right: list[Finding]) -> list[Finding]:
    """Upsert by requirement: each requirement has one finding, however often it is written."""
    merged = {finding.requirement_id: finding for finding in validate_list(left, Finding)}
    for finding in validate_list(right, Finding):
        merged[finding.requirement_id] = finding
    return list(merged.values())


class ReviewState(TypedDict):
    filename: str
    text: str
    framework: str
    clauses: list[Clause]
    findings: Annotated[list[Finding], merge_findings]
    run_metrics: Annotated[RunMetrics, merge_run_metrics]


class ReviewerInput(TypedDict):
    requirement: Requirement
    clauses: list[Clause]


async def reader(state: ReviewState) -> dict[str, Any]:
    framework = FRAMEWORKS[state["framework"]]
    async with node_span(READER) as span:
        clauses = clauses_of(state["text"])
        metrics = span.metrics
    emit(
        REVIEW_STARTED,
        READER,
        framework=framework.id,
        title=framework.title,
        filename=state["filename"],
        requirements=[requirement.model_dump() for requirement in framework.requirements],
        clauses=[clause.model_dump() for clause in clauses],
    )
    return {"clauses": clauses, "run_metrics": metrics}


def fan_out(state: ReviewState) -> list[Send]:
    """One reviewer per requirement, in parallel: one judgment never colours another."""
    framework = FRAMEWORKS[state["framework"]]
    return [
        Send(REVIEWER, ReviewerInput(requirement=requirement, clauses=state["clauses"]))
        for requirement in framework.requirements
    ]


async def reviewer(state: ReviewerInput) -> dict[str, Any]:
    """Judge one requirement. Input arrives from a ``Send``, not the review's state."""
    requirement = state["requirement"]
    clauses = validate_list(state["clauses"], Clause)
    provision = provision_text(requirement, load_library())
    async with node_span(REVIEWER, subtask_id=requirement.id) as span:
        shown = candidates(requirement, provision, clauses)
        if not shown:
            finding = unaddressed(requirement)
        else:
            try:
                judgement = await call_structured(
                    REVIEWER,
                    judge_messages(requirement, provision, shown),
                    RequirementJudgement,
                    span.metrics,
                )
            except Exception as exc:
                # One requirement unjudged, said as such; never the whole review lost.
                logger.warning(
                    "a requirement could not be judged",
                    extra={"requirement": requirement.id, "error_type": type(exc).__name__},
                )
                finding = Finding(
                    requirement_id=requirement.id,
                    label=requirement.label,
                    title=requirement.title,
                    status="unverified",
                    reason=(
                        f"Not judged: the model call failed ({type(exc).__name__}). "
                        "Check this by hand."
                    ),
                )
            else:
                finding = verified(
                    judgement, requirement, {clause.id: clause for clause in clauses}
                )
        metrics = span.metrics
    emit(REQUIREMENT_JUDGED, REVIEWER, **finding.model_dump())
    return {"findings": [finding], "run_metrics": metrics}


ReviewGraph = CompiledStateGraph[ReviewState, Any, ReviewState, ReviewState]


def build_review_graph() -> ReviewGraph:
    builder: StateGraph[ReviewState, Any, ReviewState, ReviewState] = StateGraph(ReviewState)
    builder.add_node(READER, reader)
    builder.add_node(REVIEWER, reviewer)
    builder.add_edge(START, READER)
    builder.add_conditional_edges(READER, fan_out, [REVIEWER])
    builder.add_edge(REVIEWER, END)
    return builder.compile()


class ReviewResult(BaseModel):
    """A finished review, as the record keeps it and ``GET /reviews/{id}`` serves it."""

    framework: str
    title: str
    filename: str
    thread_id: str
    clauses: list[Clause]
    # In the framework's order.
    findings: list[Finding]
    counts: dict[str, int]
    metrics: RunMetrics


def review_result(state: dict[str, Any], thread_id: str) -> ReviewResult:
    framework = FRAMEWORKS[str(state["framework"])]
    order = {
        requirement.id: position for position, requirement in enumerate(framework.requirements)
    }
    findings = sorted(
        validate_list(state.get("findings"), Finding),
        key=lambda finding: order[finding.requirement_id],
    )
    return ReviewResult(
        framework=framework.id,
        title=framework.title,
        filename=str(state["filename"]),
        thread_id=thread_id,
        clauses=validate_list(state.get("clauses"), Clause),
        findings=findings,
        counts={
            status: sum(finding.status == status for finding in findings) for status in STATUSES
        },
        metrics=validate_metrics(state.get("run_metrics")),
    )
