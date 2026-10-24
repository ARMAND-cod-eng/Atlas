"""Per-run metrics.

Deliberately small. Langfuse replaces this in a later phase, so the interface is
kept to three models, one reducer, and one context manager — swapping it out
should mean deleting this file, not unpicking it from everywhere.

What it exists to answer, all of which Phase 0 could not:
  * which provider actually answered each call (a dead primary hid behind a
    working fallback for an entire session)
  * how long each node took (the live check that fan-out is really parallel)
  * whether the run degraded, and how
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

# Recorded in RunMetrics.degraded. Strings rather than an enum so a node can add
# a qualifier (the failing subtask id) without a schema change.
DEGRADED_PLANNER_FALLBACK = "planner_fallback_single_subtask"
DEGRADED_BUDGET_EXCEEDED = "budget_exceeded"
DEGRADED_RESEARCHER_FAILED = "researcher_failed"
# The synthesizer cited a URL that appeared in no source. Prompting against this
# is necessary but not sufficient — a fabricated citation looks checkable and is
# not, which makes it the worst failure mode a research system has.
DEGRADED_UNGROUNDED_CITATION = "ungrounded_citation"
# The critic still wanted changes when the iteration cap was reached. The run
# returns its best answer rather than looping or failing.
DEGRADED_CRITIC_MAX_ITERATIONS = "critic_max_iterations"
# Redis was unreachable, so spend could not be checked. Fails open — a dead cache
# must not break queries — but the client is told the guard was not running.
DEGRADED_BUDGET_CHECK_UNAVAILABLE = "budget_check_unavailable"
# The critic said "revise" without faulting any subtask or giving synthesis
# feedback. Not actionable, so the loop exits rather than burning an iteration.
DEGRADED_CRITIC_NOT_ACTIONABLE = "critic_revise_not_actionable"
# The critic call itself failed and the answer was accepted unreviewed. Found in
# live verification: the fail-open path recorded a round of overall="accept",
# which is byte-for-byte what a genuine approval looks like. An unreviewed answer
# and a reviewed one must never be indistinguishable in the metrics.
DEGRADED_CRITIC_UNAVAILABLE = "critic_unavailable"
# The synthesizer stopped because it hit SYNTHESIS_MAX_TOKENS rather than because
# it finished, so the answer is cut mid-sentence. Also found live: the cap is
# meant to catch outliers, and a run that trips it is evidence the cap is set
# below the prompt's typical output rather than above it.
DEGRADED_SYNTHESIS_TRUNCATED = "synthesis_truncated"
# The critic faulted a subtask id that is not in the plan. Its concern cannot be
# routed to any branch, so it is dropped — and said so, rather than leaving a
# "revise" verdict that quietly went nowhere.
DEGRADED_CRITIC_UNKNOWN_SUBTASK = "critic_unknown_subtask"
# The primary answered a structured call (plan, verdict) in the wrong shape and
# the fallback was asked instead. A primary that keeps doing this looks, from the
# answers alone, exactly like a healthy one.
DEGRADED_STRUCTURED_OUTPUT_FALLBACK = "structured_output_fallback"
# The run's record could not be written to the durable store (atlas.store). The
# answer is unaffected; what is missing is the trail that says it was produced —
# and a missing audit record looks, from the answer, like nothing at all.
DEGRADED_AUDIT_INCOMPLETE = "audit_incomplete"
# A compliance researcher's web search failed, and the branch went ahead on the
# provisions it had read from the library. The law is all an answer needs to
# cite; the guidance that would have sat beside it is missing.
DEGRADED_WEB_SEARCH_FAILED = "web_search_failed"


class LLMCall(BaseModel):
    """One model invocation."""

    node: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_s: float


class NodeTiming(BaseModel):
    """Wall-clock time for one node execution."""

    node: str
    duration_s: float
    # Set for researchers. One timing per execution, so this is the only record
    # of how many times a subtask actually ran — research_results keeps one
    # entry per subtask however often it was retried.
    subtask_id: str | None = None


class CriticRound(BaseModel):
    """One pass of the critic.

    Recorded so the accept rate is visible. A critic that never accepts is a cost
    multiplier; a critic that always accepts is a no-op that looks perfect on
    every other metric. Both are invisible without this.
    """

    iteration: int
    overall: str
    faulted_subtask_ids: list[str] = Field(default_factory=list)


class RunMetrics(BaseModel):
    """Everything measured during a single graph run."""

    llm_calls: list[LLMCall] = Field(default_factory=list)
    node_timings: list[NodeTiming] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    critic_rounds: list[CriticRound] = Field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens or 0 for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens or 0 for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


def merge_run_metrics(left: RunMetrics, right: RunMetrics) -> RunMetrics:
    """State reducer for ``run_metrics``.

    Required for exactly the same reason ``research_results`` needs
    ``operator.add``: parallel researchers all write this key, and LangGraph
    raises ``InvalidUpdateError`` when concurrent branches write an un-reduced
    key. Concatenating the inner lists is the whole job — ordering across
    branches is not meaningful, so no sorting is attempted.
    """
    return RunMetrics(
        llm_calls=[*left.llm_calls, *right.llm_calls],
        node_timings=[*left.node_timings, *right.node_timings],
        degraded=[*left.degraded, *right.degraded],
        critic_rounds=[*left.critic_rounds, *right.critic_rounds],
    )


@asynccontextmanager
async def track_node(node: str, subtask_id: str | None = None) -> AsyncIterator[RunMetrics]:
    """Time a node and give it a fresh :class:`RunMetrics` to fill in.

    Yields metrics scoped to this node only — the reducer merges it back into the
    run. Each node returns ``{"run_metrics": m, ...}`` *after* the block exits,
    so the timing is already recorded by the time it is returned.

    The timing is written in ``finally``, so a node that raises still reports how
    long it ran before failing.
    """
    metrics = RunMetrics()
    started = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.node_timings.append(
            NodeTiming(node=node, duration_s=time.perf_counter() - started, subtask_id=subtask_id)
        )
