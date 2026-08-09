"""The LangGraph state schema and the data contracts that travel through it.

The state is the contract between nodes: everything one node needs from another
travels through here, which is what lets nodes stay independently testable.

Pydantic models rather than dicts throughout — a typo in a dict key is a runtime
surprise three nodes later; a typo in a field name is a type error now.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from atlas.metrics import RunMetrics, merge_run_metrics


class Source(BaseModel):
    """One citable web result.

    Tavily's raw shape never escapes ``atlas.tools.search`` — nodes only ever see
    this, so swapping search providers touches one module.
    """

    url: str
    title: str
    snippet: str


class Subtask(BaseModel):
    """One independent question the planner decomposed the query into."""

    id: str
    question: str
    rationale: str


class PlannerOutput(BaseModel):
    """Structured-output schema for the planner.

    A wrapper rather than a bare list because providers constrain to an object
    schema; a top-level array is not universally supported.
    """

    subtasks: list[Subtask] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """What one researcher produced — including the case where it failed.

    ``error`` being a field rather than an exception is the point: a branch that
    fails returns one of these with ``error`` set, so the run continues and the
    synthesizer can say which subtask went missing.
    """

    subtask_id: str
    findings: str
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None


class AtlasState(TypedDict):
    """State threaded through every node in the graph."""

    query: str
    subtasks: list[Subtask]
    # operator.add is what makes fan-in work at all. Researchers run concurrently
    # and every one of them writes this key; without a reducer LangGraph raises
    # InvalidUpdateError rather than merging, because it has no way to know how
    # two simultaneous writes should combine. `add` on lists means "concatenate",
    # which is exactly right here — order across branches is not meaningful.
    research_results: Annotated[list[ResearchResult], operator.add]
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]
    # Same reasoning as research_results: every parallel branch writes metrics.
    run_metrics: Annotated[RunMetrics, merge_run_metrics]


class ResearcherInput(TypedDict):
    """What a ``Send`` payload delivers to one researcher.

    Deliberately *not* AtlasState. A Send's argument becomes the target node's
    entire input rather than a patch over the graph state, so the researcher sees
    only its own subtask plus the original query for context. Its return value is
    a normal state update and merges back through the reducers above.
    """

    subtask: Subtask
    query: str
