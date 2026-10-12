"""MAX_TOKENS_PER_QUERY, enforced.

The counter lives in Redis rather than graph state because parallel branches all
read the same pre-fan-out snapshot — five researchers holding a state-local
counter would each believe they were the only one spending.

Two failure directions are tested, and they pull opposite ways. A breach must
actually stop the loop, or the guard is decorative. A Redis outage must *not* stop
the loop, because a dead cache breaking every query is a far worse outcome than an
overspent one — the run stays bounded by MAX_CRITIC_ITERATIONS and MAX_SUBTASKS
regardless. Fail closed on breach, fail open on outage.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from atlas import budget
from atlas.api.main import create_app
from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import AtlasState, CriticVerdict, ResearchResult, Source, Subtask
from atlas.metrics import (
    DEGRADED_BUDGET_CHECK_UNAVAILABLE,
    DEGRADED_BUDGET_EXCEEDED,
    RunMetrics,
)
from atlas.runner import execute_run
from tests.conftest import (
    FakeArq,
    FakeRedis,
    initial_state,
    make_call_model,
    make_call_structured,
    make_plan,
    run_config,
)

THREAD = "budget-thread"
RUN = "budget-run"
KEY = f"atlas:budget:{RUN}"
BRANCHES = 3


@pytest.fixture
def in_a_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend we are inside graph run RUN.

    ``current_run_id`` reads LangGraph's ambient config, which does not exist
    when a node function is called directly.
    """
    monkeypatch.setattr(budget, "current_run_id", lambda: RUN)


# --- the counter itself -------------------------------------------------------


async def test_spend_accumulates_on_the_runs_key(in_a_run: None, fake_redis: FakeRedis) -> None:
    await budget.record_spend(1200)
    await budget.record_spend(800)

    assert fake_redis.values == {KEY: 2000}
    assert await budget.spent() == 2000


async def test_the_key_gets_a_ttl_so_abandoned_runs_do_not_leak(
    in_a_run: None, fake_redis: FakeRedis
) -> None:
    await budget.record_spend(10)
    assert fake_redis.expiries == {KEY: 3600}


async def test_parallel_branches_share_one_counter(in_a_run: None, fake_redis: FakeRedis) -> None:
    """The reason this is not in graph state.

    Five concurrent writers, one total. A counter carried in state would show 100
    here, because every branch would have started from the same snapshot.
    """
    await asyncio.gather(*(budget.record_spend(100) for _ in range(5)))

    assert fake_redis.values == {KEY: 500}
    assert await budget.spent() == 500


async def test_an_unspent_run_reads_zero_not_unknown(in_a_run: None) -> None:
    """0 and None mean different things: "nothing yet" versus "could not check"."""
    assert await budget.spent() == 0
    assert await budget.is_exceeded() == (False, 0)


async def test_breach_is_strictly_above_the_limit(in_a_run: None, fake_redis: FakeRedis) -> None:
    limit = get_settings().max_tokens_per_query
    fake_redis.values[KEY] = limit
    assert await budget.is_exceeded() == (False, limit)

    fake_redis.values[KEY] = limit + 1
    assert await budget.is_exceeded() == (True, limit + 1)


# --- failing open -------------------------------------------------------------


async def test_a_redis_outage_reports_unknown_rather_than_zero(
    in_a_run: None, fake_redis: FakeRedis
) -> None:
    fake_redis.fail = True

    assert await budget.spent() is None
    assert await budget.is_exceeded() == (False, None)


async def test_recording_spend_never_raises_when_redis_is_down(
    in_a_run: None, fake_redis: FakeRedis
) -> None:
    """A dead cache must not propagate an exception into a node."""
    fake_redis.fail = True
    await budget.record_spend(500)  # must not raise


async def test_outside_a_graph_there_is_nothing_to_track() -> None:
    """No run id, no counter — and no crash. Unit tests call nodes directly."""
    assert budget.current_run_id() is None
    assert await budget.spent() is None
    assert await budget.is_exceeded() == (False, None)


# --- the checkpoint in the critic node ----------------------------------------


def _reviewable_state() -> AtlasState:
    state = initial_state("what is RAG?")
    state["subtasks"] = [Subtask(id="s0", question="q0", rationale="r")]
    state["research_results"] = [
        ResearchResult(
            subtask_id="s0",
            findings="findings",
            sources=[Source(url="https://a.example", title="A", snippet="s")],
        )
    ]
    state["answer"] = "the best answer so far"
    return state


async def test_a_breach_accepts_without_calling_the_critic(
    in_a_run: None, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over budget means stop spending — including on the review itself."""
    called = False

    async def should_not_run(*args: object, **kwargs: object) -> CriticVerdict:
        nonlocal called
        called = True
        return CriticVerdict(overall="revise")

    monkeypatch.setattr(nodes, "call_structured", should_not_run)
    fake_redis.values[KEY] = get_settings().max_tokens_per_query + 1

    result = await nodes.critic(_reviewable_state())

    assert called is False
    assert result["critic_feedback"][0].overall == "accept"
    assert DEGRADED_BUDGET_EXCEEDED in result["run_metrics"].degraded


async def test_one_breach_is_reported_once_not_twice(
    in_a_run: None, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two checks, one signal.

    The synthesizer runs an in-state estimate a node earlier and writes the same
    marker; a live run reported ``budget_exceeded`` twice for a single breach.
    Both checks are worth keeping — the critic's is authoritative and Redis-backed,
    the synthesizer's still works when Redis is down — but a duplicated flag tells
    the client nothing it did not already know.
    """

    async def unused(*args: object, **kwargs: object) -> CriticVerdict:
        raise AssertionError("the critic must not be called over budget")

    monkeypatch.setattr(nodes, "call_structured", unused)
    fake_redis.values[KEY] = get_settings().max_tokens_per_query + 1

    state = _reviewable_state()
    # What the synthesizer left behind when its own estimate tripped.
    state["run_metrics"] = RunMetrics(degraded=[DEGRADED_BUDGET_EXCEEDED])

    result = await nodes.critic(state)
    merged = state["run_metrics"].degraded + result["run_metrics"].degraded

    assert merged.count(DEGRADED_BUDGET_EXCEEDED) == 1


async def test_an_unavailable_check_is_recorded_and_the_run_continues(
    in_a_run: None, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degraded, not stopped. The client is told the guard was not running."""
    fake_redis.fail = True
    monkeypatch.setattr(
        nodes,
        "call_structured",
        make_call_structured(make_plan(1), [CriticVerdict(overall="accept")]),
    )

    result = await nodes.critic(_reviewable_state())
    degraded = result["run_metrics"].degraded

    assert DEGRADED_BUDGET_CHECK_UNAVAILABLE in degraded
    assert DEGRADED_BUDGET_EXCEEDED not in degraded
    assert result["iteration"] == 1


# --- end to end ---------------------------------------------------------------


@pytest.fixture
def budget_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graph whose critic would reject forever if the budget let it."""

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    verdict = CriticVerdict(overall="revise", synthesis_feedback="do it again")
    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the best answer so far"))
    monkeypatch.setattr(
        nodes, "call_structured", make_call_structured(make_plan(BRANCHES), [verdict])
    )


async def test_a_breached_run_stops_at_one_iteration_with_an_answer(
    budget_graph: None, fake_redis: FakeRedis
) -> None:
    """The critic would have kept rejecting. The budget is what stops it."""
    fake_redis.values[KEY] = get_settings().max_tokens_per_query + 1

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config = run_config(THREAD, RUN)
    result = await graph.ainvoke(initial_state(), config=config)

    assert result["iteration"] == 1
    assert result["answer"] == "the best answer so far"

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert DEGRADED_BUDGET_EXCEEDED in metrics.degraded
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [(1, "accept")]


def test_a_breach_returns_200_not_an_error(budget_graph: None, fake_redis: FakeRedis) -> None:
    """Overspending is a degraded result, not a failed request.

    A 5xx here would throw away an answer the user already paid for. Still
    asserted through the HTTP route even though execution moved to the worker in
    Phase 3 — the client-visible contract is what this test is about, and it did
    not change.
    """
    compiled = build_state_graph().compile(checkpointer=MemorySaver())

    async def over_budget_run(run_id: str, query: str, thread_id: str) -> dict[str, Any]:
        # The route mints the run id, so the breach is planted once it exists.
        fake_redis.values[f"atlas:budget:{run_id}"] = get_settings().max_tokens_per_query + 1
        return await execute_run(run_id, query, thread_id, compiled)

    app = create_app()
    app.state.arq = FakeArq(over_budget_run)

    response = TestClient(app).post("/query", json={"query": "what is RAG?", "thread_id": THREAD})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "the best answer so far"
    assert body["iterations"] == 1
    assert DEGRADED_BUDGET_EXCEEDED in body["metrics"]["degraded"]
