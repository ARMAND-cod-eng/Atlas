"""What a run emits, in what order, and who each event belongs to.

The truthiness trap of this phase is "events arrived". A stream that emits the
right *number* of events in the wrong order, or that cannot say which of five
concurrent branches an event came from, is useless to a client and passes any
assertion phrased as ``len(events) > 0``. So every assertion here compares an
exact sequence or an exact set.

The interleaving test carries the same weight ``test_concurrency`` does for the
fan-out: it is written so that a serialised implementation fails it. Branch
completion order is staggered to be the *reverse* of dispatch order, which is
impossible if the branches ran one after another.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.errors import ProviderError, SearchError
from atlas.events import (
    CRITIC_VERDICT,
    FAN_OUT,
    ITERATION_STARTED,
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_STARTED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    AgentEvent,
    emit,
)
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import Source
from atlas.metrics import RunMetrics
from atlas.runner import execute_run
from tests.conftest import FakeRedis, make_call_model, make_call_structured, make_plan

BRANCHES = 5


async def _events(run_id: str) -> list[AgentEvent]:
    return await eventlog.replay(run_id, after_seq=0)


async def _run(run_id: str, thread: str = "t") -> dict[str, object]:
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    return await execute_run(run_id, "what is RAG?", thread, graph)


# --- the guard that protects every pre-Phase-3 test ---------------------------


def test_emit_outside_a_graph_is_a_no_op() -> None:
    """``get_stream_writer`` raises when no graph is running.

    Every test written before this phase calls node functions directly, with no
    graph anywhere — so an unguarded emit would not fail *this* module, it would
    fail 126 tests in the others. Asserting the no-op here is what makes that a
    deliberate property rather than an accident of import order.
    """
    emit(NODE_STARTED, "planner", subtask_id="s0")  # must not raise


# --- a full run, exactly -------------------------------------------------------


@pytest.fixture
def one_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))


async def test_a_full_run_emits_this_exact_sequence(one_branch: None) -> None:
    """The whole contract, as one equality.

    Written out rather than spot-checked because the order *is* the product: a
    client renders the plan when fan_out lands and a branch card when
    node_started lands, so fan_out arriving after the branches would be a bug
    that no per-event assertion would notice.
    """
    await _run("exact")

    assert [(e.seq, e.type, e.node) for e in await _events("exact")] == [
        (1, RUN_STARTED, None),
        (2, NODE_STARTED, nodes.PLANNER),
        (3, NODE_COMPLETED, nodes.PLANNER),
        (4, FAN_OUT, nodes.PLANNER),
        (5, NODE_STARTED, nodes.RESEARCHER),
        (6, NODE_COMPLETED, nodes.RESEARCHER),
        (7, NODE_STARTED, nodes.SYNTHESIZER),
        (8, NODE_COMPLETED, nodes.SYNTHESIZER),
        (9, ITERATION_STARTED, nodes.CRITIC),
        (10, NODE_STARTED, nodes.CRITIC),
        (11, NODE_COMPLETED, nodes.CRITIC),
        (12, CRITIC_VERDICT, nodes.CRITIC),
        (13, RUN_COMPLETED, None),
    ]


async def test_seq_is_dense_from_one(one_branch: None) -> None:
    """Monotonic *and* gapless, asserted as one thing.

    Monotonic alone would be satisfied by 1, 5, 9 — and a client comparing
    ``Last-Event-ID`` against what it has would then think it had missed events
    it never sent. Denseness is what makes "I have everything up to N" sayable.
    """
    await _run("dense")
    events = await _events("dense")

    assert [e.seq for e in events] == list(range(1, len(events) + 1))


async def test_the_terminal_event_is_last_and_unique(one_branch: None) -> None:
    await _run("terminal")
    events = await _events("terminal")

    assert events[-1].type == RUN_COMPLETED
    assert [e.type for e in events].count(RUN_COMPLETED) == 1
    assert RUN_FAILED not in [e.type for e in events]


async def test_run_completed_carries_a_summary_not_the_whole_result(one_branch: None) -> None:
    """A pointer, not a payload — the full result comes from GET /runs/{id}.

    Replaying the entire response on every reconnect would make the reconnection
    this phase exists to support the most expensive thing a client can do.
    """
    await _run("summary")
    completed = (await _events("summary"))[-1]

    assert completed.payload["answer"] == "the answer"
    assert completed.payload["iterations"] == 1
    assert completed.payload["subtask_count"] == 1
    assert "sources" not in completed.payload
    assert "metrics" not in completed.payload


# --- five branches at once -----------------------------------------------------


@pytest.fixture
def staggered_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Branches whose completion order is the reverse of their dispatch order.

    s0 sleeps longest and s4 shortest, so concurrent execution finishes s4 first.
    Run serially, each branch would complete before the next began and completion
    order would equal dispatch order — which is exactly what the test rejects.
    """

    async def staggered_search(query: str, max_results: int | None = None) -> list[Source]:
        index = int(query[1]) if len(query) > 1 and query[1].isdigit() else 0
        await asyncio.sleep(0.02 * (BRANCHES - index))
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", staggered_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(BRANCHES)))


async def test_branch_events_interleave_rather_than_running_in_series(
    staggered_branches: None,
) -> None:
    await _run("interleave")
    events = await _events("interleave")

    started = [
        e.payload["subtask_id"]
        for e in events
        if e.type == NODE_STARTED and e.node == nodes.RESEARCHER
    ]
    completed = [
        e.payload["subtask_id"]
        for e in events
        if e.type == NODE_COMPLETED and e.node == nodes.RESEARCHER
    ]

    assert started == [f"s{i}" for i in range(BRANCHES)]
    # The discriminating part. Serial execution gives completed == started; only
    # genuinely concurrent branches can finish in the opposite order.
    assert completed == [f"s{i}" for i in reversed(range(BRANCHES))]
    assert completed != started


async def test_every_branch_event_is_attributable_to_its_subtask(
    staggered_branches: None,
) -> None:
    """Interleaved events are worthless if you cannot tell whose they are."""
    await _run("attribution")
    events = await _events("attribution")

    researcher_events = [
        (e.type, e.payload["subtask_id"]) for e in events if e.node == nodes.RESEARCHER
    ]

    assert sorted(researcher_events) == sorted(
        [(NODE_STARTED, f"s{i}") for i in range(BRANCHES)]
        + [(NODE_COMPLETED, f"s{i}") for i in range(BRANCHES)]
    )
    # No branch reports twice and none is missing.
    assert len(researcher_events) == BRANCHES * 2
    assert {sid for _, sid in researcher_events} == {f"s{i}" for i in range(BRANCHES)}


async def test_fan_out_announces_the_branches_before_any_of_them_start(
    staggered_branches: None,
) -> None:
    """A client sizes its UI from this event, so it must precede the branches."""
    await _run("fanout")
    events = await _events("fanout")

    fan_out_seq = next(e.seq for e in events if e.type == FAN_OUT)
    first_branch_seq = min(e.seq for e in events if e.node == nodes.RESEARCHER)

    assert fan_out_seq < first_branch_seq
    payload = next(e.payload for e in events if e.type == FAN_OUT)
    assert payload["count"] == BRANCHES
    assert payload["subtask_ids"] == [f"s{i}" for i in range(BRANCHES)]


# --- failure -------------------------------------------------------------------


async def test_a_failed_branch_reports_node_failed_not_node_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead branch does not raise — Phase 1 turns it into data so the other
    four survive. It must still surface as a failure to a watching client."""

    async def half_broken(query: str, max_results: int | None = None) -> list[Source]:
        if query == "q1":
            raise SearchError("this one is down", query=query)
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", half_broken)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(3)))

    await _run("branch-failure")
    events = await _events("branch-failure")

    failed = [e for e in events if e.type == NODE_FAILED]
    assert [e.payload["subtask_id"] for e in failed] == ["s1"]
    assert "SearchError" in failed[0].payload["error"]

    completed = {
        e.payload["subtask_id"]
        for e in events
        if e.type == NODE_COMPLETED and e.node == nodes.RESEARCHER
    }
    assert completed == {"s0", "s2"}
    # One branch died; the run did not.
    assert events[-1].type == RUN_COMPLETED


async def test_a_raising_node_still_terminates_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client must learn the run ended, even when it ended badly.

    Without a terminal event on the failure path a client sits on an open
    connection forever, unable to distinguish a dead run from a slow one.
    """

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url="https://ex.example/x", title="T", snippet="s")]

    async def dies_in_synthesis(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node == nodes.SYNTHESIZER:
            raise ProviderError(provider="anthropic", message="everything is down", node=node)
        return AIMessage(content="findings")

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", dies_in_synthesis)
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))

    with pytest.raises(ProviderError):
        await _run("run-failure")

    events = await _events("run-failure")

    assert events[-1].type == RUN_FAILED
    assert events[-1].payload["error_type"] == "ProviderError"
    assert "everything is down" in events[-1].payload["error"]
    # The node that died is named, so a client can point at it.
    assert [(e.type, e.node) for e in events if e.type == NODE_FAILED] == [
        (NODE_FAILED, nodes.SYNTHESIZER)
    ]
    assert [e.seq for e in events] == list(range(1, len(events) + 1))


async def test_a_telemetry_failure_does_not_kill_the_run(
    one_branch: None, fake_redis: FakeRedis
) -> None:
    """Redis dying must not turn a working run into a failed one.

    The run still answers; the events are simply lost. That is the right trade —
    the alternative is telemetry that can fail the thing it observes.
    """
    fake_redis.fail = True

    result = await _run("telemetry-down")

    assert result["answer"] == "the answer"
    assert eventlog.events_key("telemetry-down") not in fake_redis.streams
