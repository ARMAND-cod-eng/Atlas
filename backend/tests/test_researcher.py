"""Researcher resilience: a failed branch must not kill the run."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from atlas.errors import SearchError
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import ResearcherInput, Source, Subtask
from atlas.metrics import DEGRADED_RESEARCHER_FAILED, RunMetrics
from tests.conftest import initial_state, make_call_structured, make_plan

SUBTASK = Subtask(id="s0", question="what is RAG?", rationale="r")


def _input() -> ResearcherInput:
    return {"subtask": SUBTASK, "query": "original", "guidance": None}


async def _ok_model(
    node: str,
    messages: object,
    metrics: RunMetrics,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AIMessage:
    return AIMessage(content="findings with a citation")


async def test_happy_path_returns_findings_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url="https://a.example", title="A", snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    result = await nodes.researcher(_input())
    research = result["research_results"][0]

    assert research.error is None
    assert research.findings == "findings with a citation"
    assert [s.url for s in research.sources] == ["https://a.example"]


async def test_search_failure_becomes_a_result_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The most important error boundary in the phase.

    An exception escaping the researcher aborts the whole graph run — one flaky
    search would discard four healthy branches' completed work.
    """

    async def boom(query: str, max_results: int | None = None) -> list[Source]:
        raise SearchError("tavily is down", query=query)

    monkeypatch.setattr(nodes, "search", boom)
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    result = await nodes.researcher(_input())
    research = result["research_results"][0]

    assert research.error is not None
    assert "SearchError" in research.error
    assert research.subtask_id == "s0"


async def test_failure_is_recorded_in_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(query: str, max_results: int | None = None) -> list[Source]:
        raise SearchError("down", query=query)

    monkeypatch.setattr(nodes, "search", boom)
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    result = await nodes.researcher(_input())
    degraded = result["run_metrics"].degraded

    assert any(d.startswith(DEGRADED_RESEARCHER_FAILED) for d in degraded)
    assert any("s0" in d for d in degraded)


async def test_empty_search_results_are_reported_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Nothing found" is information, not an error."""

    async def empty(query: str, max_results: int | None = None) -> list[Source]:
        return []

    monkeypatch.setattr(nodes, "search", empty)
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    result = await nodes.researcher(_input())
    research = result["research_results"][0]

    assert research.error is None
    assert "No sources" in research.findings


async def test_retries_with_looser_phrasing_on_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded by MAX_TOOL_CALLS_PER_RESEARCHER (default 3)."""
    attempts: list[str] = []

    async def fussy_search(query: str, max_results: int | None = None) -> list[Source]:
        attempts.append(query)
        if len(attempts) < 2:
            return []
        return [Source(url="https://a.example", title="A", snippet="s")]

    monkeypatch.setattr(nodes, "search", fussy_search)
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    result = await nodes.researcher(_input())

    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert result["research_results"][0].error is None


async def test_a_failing_branch_still_reaches_the_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: one branch dies, the run still produces an answer."""

    async def half_broken(query: str, max_results: int | None = None) -> list[Source]:
        if "q1" in query:
            raise SearchError("this one is down", query=query)
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", half_broken)
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(3)))
    monkeypatch.setattr(nodes, "call_model", _ok_model)

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "partial-failure"}}
    result = await graph.ainvoke(initial_state(), config=config)

    results = result["research_results"]
    assert len(results) == 3
    assert sum(1 for r in results if r.error is not None) == 1
    # The run completed despite the failure.
    assert result["answer"]
