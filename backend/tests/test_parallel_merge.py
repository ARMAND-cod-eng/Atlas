"""Fan-in: every branch's result survives the merge.

This exercises the `operator.add` reducer on `research_results` and
`merge_run_metrics` on `run_metrics`. Without them, concurrent branches writing
the same key raise InvalidUpdateError rather than combining.

Note this proves *merging*, not *parallelism* — see test_concurrency.py for the
test that fails when the fan-out goes serial.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import Source
from atlas.metrics import RunMetrics
from tests.conftest import initial_state, make_call_model, make_call_structured, make_plan

BRANCHES = 5


@pytest.fixture
def graph_with_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct search results per subtask so losses are detectable."""

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://example.com/{query}", title=query, snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(BRANCHES)))
    monkeypatch.setattr(nodes, "call_model", make_call_model())


async def _run(thread: str) -> dict[str, object]:
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    return dict(await graph.ainvoke(initial_state(), config=config))


async def test_every_branch_result_survives(graph_with_fakes: None) -> None:
    result = await _run("merge-results")
    results = result["research_results"]
    assert isinstance(results, list)
    assert len(results) == BRANCHES

    # Every subtask id appears exactly once — no drops, no duplicates.
    ids = sorted(r.subtask_id for r in results)
    assert ids == sorted(f"s{i}" for i in range(BRANCHES))


async def test_metrics_merge_across_branches(graph_with_fakes: None) -> None:
    """run_metrics needs its own reducer for exactly the same reason."""
    result = await _run("merge-metrics")
    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)

    researcher_timings = [t for t in metrics.node_timings if t.node == nodes.RESEARCHER]
    assert len(researcher_timings) == BRANCHES

    # Planner, synthesizer and critic each contribute one timing on top of the
    # branches.
    nodes_seen = {t.node for t in metrics.node_timings}
    assert nodes_seen == {nodes.PLANNER, nodes.RESEARCHER, nodes.SYNTHESIZER, nodes.CRITIC}


async def test_run_reaches_an_answer(graph_with_fakes: None) -> None:
    result = await _run("merge-answer")
    assert isinstance(result["answer"], str)
    assert result["answer"]
