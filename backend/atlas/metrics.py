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


class LLMCall(BaseModel):
    """One model invocation."""

    node: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_s: float


class NodeTiming(BaseModel):
    """Wall-clock time for one node."""

    node: str
    duration_s: float


class RunMetrics(BaseModel):
    """Everything measured during a single graph run."""

    llm_calls: list[LLMCall] = Field(default_factory=list)
    node_timings: list[NodeTiming] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)

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
    )


@asynccontextmanager
async def track_node(node: str) -> AsyncIterator[RunMetrics]:
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
        metrics.node_timings.append(NodeTiming(node=node, duration_s=time.perf_counter() - started))
