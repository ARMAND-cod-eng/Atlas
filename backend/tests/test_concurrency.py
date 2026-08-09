"""Do the researchers actually run in parallel?

**This is the load-bearing test of Phase 1.**

``test_parallel_merge`` proves results merge without loss — and a fully
sequential implementation would pass it identically. It verifies the outcome, not
the mechanism. That is the Phase 0 lesson repeating: a fallback that always fires
looks exactly like a working system, and results that merge correctly look
exactly like results that merged *in parallel*.

Parallelism is the feature of this phase, so it gets a test that fails when
parallelism breaks — if someone replaces ``Send`` with a loop, or a synchronous
call sneaks into the researcher body and blocks the event loop, the wall clock
says so immediately.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.metrics import RunMetrics
from tests.conftest import initial_state

BRANCHES = 4
DELAY_S = 0.2
# Sequential execution would take BRANCHES * DELAY_S = 0.8s. The ceiling sits well
# above one delay (so a loaded CI machine does not flake) and well below the
# sequential floor (so sequential execution cannot sneak under it).
CEILING_S = 0.5


@pytest.fixture
def slow_parallel_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real graph whose only slow part is the search inside each researcher."""

    async def slow_search(query: str, max_results: int | None = None) -> list[Source]:
        await asyncio.sleep(DELAY_S)
        return [Source(url=f"https://example.com/{query}", title="T", snippet="S")]

    async def fake_structured(
        node: str,
        messages: object,
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> PlannerOutput:
        return PlannerOutput(
            subtasks=[
                Subtask(id=f"s{i}", question=f"question {i}", rationale="r")
                for i in range(BRANCHES)
            ]
        )

    async def fast_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> AIMessage:
        return AIMessage(content="findings")

    monkeypatch.setattr(nodes, "search", slow_search)
    monkeypatch.setattr(nodes, "call_structured", fake_structured)
    monkeypatch.setattr(nodes, "call_model", fast_model)


async def test_researchers_run_concurrently(slow_parallel_graph: None) -> None:
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "concurrency"}}

    started = time.perf_counter()
    result = await graph.ainvoke(initial_state(), config=config)
    elapsed = time.perf_counter() - started

    # Sanity: every branch really did run. Without this, a graph that skipped the
    # researchers entirely would trivially satisfy the timing assertion.
    assert len(result["research_results"]) == BRANCHES

    sequential = BRANCHES * DELAY_S
    assert elapsed < CEILING_S, (
        f"researchers took {elapsed:.2f}s; sequential execution would be "
        f"~{sequential:.1f}s and the ceiling is {CEILING_S}s. "
        "The fan-out is running serially."
    )


async def test_timing_metrics_show_overlap(slow_parallel_graph: None) -> None:
    """The per-node timings corroborate the wall clock.

    Summing the researcher timings gives roughly the sequential cost; the run as a
    whole took far less. That gap *is* the parallelism, and it is what the live
    verification checks from the client via the metrics in the response.
    """
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "timings"}}

    started = time.perf_counter()
    result = await graph.ainvoke(initial_state(), config=config)
    elapsed = time.perf_counter() - started

    metrics = result["run_metrics"]
    researcher_time = sum(t.duration_s for t in metrics.node_timings if t.node == nodes.RESEARCHER)

    assert researcher_time > elapsed, (
        f"summed researcher time ({researcher_time:.2f}s) should exceed wall clock "
        f"({elapsed:.2f}s) when branches overlap"
    )
