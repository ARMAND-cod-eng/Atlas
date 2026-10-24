"""The LangGraph state schema and the data contracts that travel through it.

The state is the contract between nodes: everything one node needs from another
travels through here, which is what lets nodes stay independently testable.

Pydantic models rather than dicts throughout — a typo in a dict key is a runtime
surprise three nodes later; a typo in a field name is a type error now.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from atlas.metrics import RunMetrics, merge_run_metrics


class Source(BaseModel):
    """One citable source: a web result, or a provision of the law.

    Tavily's raw shape never escapes ``atlas.tools.search`` — nodes only ever see
    this, so swapping search providers touches one module. A provision from the
    regulation library arrives in the same shape (atlas.library.sources), with
    its citation label and ``primary_law`` authority.
    """

    url: str
    title: str
    snippet: str
    # The page's text, present only when the search asked for it. Nothing in the
    # graph asks today; the eval corpus records it so a reader that uses more
    # than the snippet can be measured later without recording again.
    content: str | None = None
    # How much weight the source carries. Only primary law establishes what the
    # law requires; official guidance interprets it; anything else is secondary.
    # Defaulted, so every corpus recorded before this field still validates.
    authority: Literal["primary_law", "official_guidance", "secondary"] = "secondary"
    # A provision's citation label, "AI Act Art. 6(2)". None for a web source.
    label: str | None = None


class Subtask(BaseModel):
    """One independent question the planner decomposed the query into."""

    id: str
    question: str
    rationale: str


# --- tolerating the shapes structured output actually arrives in ------------------
#
# Observed live against claude-sonnet-5, each with correct content inside:
#
#   a nested list double-encoded as a string     {"subtasks": "[...]"}
#   that string holding the whole object         {"subtasks": "{\"subtasks\": [...]}"}
#   the whole answer under one wrapper key       {"$PARAMETER_VALUE": {...}}
#
# Strict validation rejected all three, and a rejected answer goes to the
# fallback — a GPT-4o call, a structured_output_fallback mark, and on the critic a
# different model doing the reviewing. In the first live eval runs that was five
# structured answers in eight. They are unwrapped at the schema, narrowly: each
# rule fires only when the expected field is absent or arrives as a string, so a
# payload genuinely missing its content still fails.


def _unwrap_single_key(data: object, required: str) -> object:
    """``{"<anything>": {...}}`` -> ``{...}``, when ``required`` is missing."""
    if isinstance(data, dict) and required not in data and len(data) == 1:
        (inner,) = data.values()
        if isinstance(inner, dict):
            return inner
    return data


def _decode_json_list(value: object, field: str) -> object:
    """A list sent as a JSON string — possibly a string of the whole object."""
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        # Not JSON either — let pydantic report the real problem.
        return value
    if isinstance(decoded, dict) and field in decoded:
        return decoded[field]
    return decoded


class PlannerOutput(BaseModel):
    """Structured-output schema for the planner.

    A wrapper rather than a bare list because providers constrain to an object
    schema; a top-level array is not universally supported.
    """

    subtasks: list[Subtask] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    # `object` rather than Any: a "before" validator receives whatever the
    # provider sent, and object states that honestly without disabling checking.
    def _unwrap(cls, data: object) -> object:
        return _unwrap_single_key(data, "subtasks")

    @field_validator("subtasks", mode="before")
    @classmethod
    def _accept_json_encoded_list(cls, value: object) -> object:
        """Tolerate ``subtasks`` arriving as a JSON string instead of an array.

        Losing it would degrade the run to a single subtask — the whole fan-out —
        over an encoding wobble, with the right content sitting in the payload.
        """
        return _decode_json_list(value, "subtasks")


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


class SubtaskVerdict(BaseModel):
    """The critic's judgement on one subtask's research."""

    subtask_id: str
    verdict: Literal["accept", "revise"]
    reason: str
    guidance: str


class CriticVerdict(BaseModel):
    """One full critic pass.

    Structured rather than free text so routing can be derived mechanically: the
    presence of faulted subtasks selects a targeted retry, synthesis feedback
    alone selects a re-write, and neither means accept.
    """

    overall: Literal["accept", "revise"]
    per_subtask: list[SubtaskVerdict] = Field(default_factory=list)
    synthesis_feedback: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data: object) -> object:
        return _unwrap_single_key(data, "overall")

    @field_validator("per_subtask", mode="before")
    @classmethod
    def _accept_json_encoded_list(cls, value: object) -> object:
        return _decode_json_list(value, "per_subtask")

    def faulted_ids(self) -> list[str]:
        """Subtask ids the critic wants re-researched."""
        return [v.subtask_id for v in self.per_subtask if v.verdict == "revise"]

    def guidance_for(self, subtask_id: str) -> str | None:
        """The critic's instruction for one subtask, if it faulted it."""
        for verdict in self.per_subtask:
            if verdict.subtask_id == subtask_id and verdict.verdict == "revise":
                return verdict.guidance
        return None


class Revision(BaseModel):
    """A person's edit to one branch of a finished run: the question it now asks.

    Kept in state, not just in the request that caused it, so the checkpoint
    history records that a human redirected this branch and from what — the run's
    own audit trail — and so the routing that dispatches the edited branch can
    read which one it is.
    """

    parent_run_id: str
    subtask_id: str
    before: str
    after: str


def merge_research_results(
    left: list[ResearchResult], right: list[ResearchResult]
) -> list[ResearchResult]:
    """Upsert by ``subtask_id``, preserving first-seen order.

    This replaces ``operator.add`` and is the load-bearing change of the critic
    loop. Appending was correct while every subtask ran exactly once. Once a
    faulted subtask can be re-researched, appending produces **two** results for
    the same ``subtask_id`` — the synthesizer then sees one subtask twice with
    contradictory findings, and ``_dedupe_sources`` keeps whichever URL it met
    first, which may be the stale one.

    Correct in both regimes, which is the point:

    * **Initial fan-out** — every ``subtask_id`` is new, so every result appends.
      Behaviourally identical to ``operator.add``, so the Phase 1 merge and
      concurrency tests keep their meaning.
    * **Targeted retry** — the id already exists, so the new result replaces it in
      place. Order is preserved, so subtask ordering in the answer does not
      shuffle between iterations.

    ``test_targeted_retry`` proves the difference by running ``operator.add`` and
    this function over the same input and asserting 6 against 5.
    """
    by_id = {result.subtask_id: result for result in left}
    order = [result.subtask_id for result in left]
    for result in right:
        if result.subtask_id not in by_id:
            order.append(result.subtask_id)
        by_id[result.subtask_id] = result
    return [by_id[subtask_id] for subtask_id in order]


class AtlasState(TypedDict):
    """State threaded through every node in the graph."""

    query: str
    subtasks: list[Subtask]
    # Upsert by subtask_id rather than append — see merge_research_results. A
    # reducer is mandatory regardless: researchers run concurrently and all write
    # this key, and without one LangGraph raises InvalidUpdateError rather than
    # guessing how two simultaneous writes combine.
    research_results: Annotated[list[ResearchResult], merge_research_results]
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]
    # Same reasoning as research_results: every parallel branch writes metrics.
    run_metrics: Annotated[RunMetrics, merge_run_metrics]
    # Written only by the critic, which never runs in parallel with itself, so
    # these need no conflict handling. operator.add is right on critic_feedback
    # precisely because accumulating the history is the point — a retried
    # researcher must be able to see what was wrong last time.
    iteration: int
    critic_feedback: Annotated[list[CriticVerdict], operator.add]
    # Set only on a run that revises another (see atlas.revision). No reducer, so
    # a new question's input (``revision: None``) replaces the last edit outright.
    revision: Revision | None


class ResearcherInput(TypedDict):
    """What a ``Send`` payload delivers to one researcher.

    Deliberately *not* AtlasState. A Send's argument becomes the target node's
    entire input rather than a patch over the graph state, so the researcher sees
    only its own subtask plus the original query for context. Its return value is
    a normal state update and merges back through the reducers above.
    """

    subtask: Subtask
    query: str
    # The critic's instruction from the previous iteration, when this is a retry.
    # None on the first pass. This is the whole mechanism by which feedback
    # reaches the researcher that earned it.
    guidance: str | None
