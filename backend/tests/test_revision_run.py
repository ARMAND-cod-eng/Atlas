"""A revision as the worker runs it: events, the run record, and failure.

``test_revision`` holds the graph to the fork's contract. This file holds the
runner to its own: a revision streams like any run, tells a watching client which
branches it carried over, is recorded against its parent, and ends in a terminal
event and a released thread on every path, the refused ones included.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.events import (
    AGENT_CALL,
    CRITIC_VERDICT,
    ITERATION_STARTED,
    NODE_COMPLETED,
    NODE_STARTED,
    REVISION_STARTED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
)
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import PlannerOutput, Revision, Source
from atlas.metrics import DEGRADED_AUDIT_INCOMPLETE, RunMetrics
from atlas.runner import execute_revision, execute_run
from atlas.store import InMemoryRunStore, RunRecord, RunStatusName
from tests.conftest import FakeRedis, make_call_structured, make_plan

THREAD = "t1"


@pytest.fixture
def planner_inputs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fake search and models; returns what each planner call was shown."""
    seen: list[list[str]] = []

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        prompt = str(messages[-1].content)
        if node == nodes.SYNTHESIZER:
            # Which research the answer was written from, so two versions'
            # answers can be told apart in the conversation that follows.
            return AIMessage(content="answer (edited)" if "edited q1" in prompt else "answer")
        return AIMessage(content=f"{node}: {prompt.splitlines()[0]}")

    plan = make_call_structured(make_plan(3))

    async def recording_structured(
        node: str,
        messages: list[BaseMessage],
        schema: type[Any],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> object:
        if schema is PlannerOutput:
            seen.append(
                [str(m.content) for m in messages if isinstance(m, HumanMessage | AIMessage)]
            )
        return await plan(node, messages, schema, metrics, temperature)

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    monkeypatch.setattr(nodes, "call_structured", recording_structured)
    return seen


def _graph() -> Any:
    return build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))


async def _events(run_id: str) -> list[tuple[str, str | None, dict[str, Any]]]:
    return [(e.type, e.node, e.payload) for e in await eventlog.replay(run_id, 0)]


# --- the stream ----------------------------------------------------------------------------


async def test_a_revision_streams_the_edited_branch_alone(
    planner_inputs: list[list[str]],
) -> None:
    graph = _graph()
    await execute_run("run-1", "first question", THREAD, graph)

    await execute_revision(
        "run-2", THREAD, graph, parent_run_id="run-1", subtask_id="s1", question="edited q1"
    )
    events = await _events("run-2")

    assert [(kind, node) for kind, node, _ in events] == [
        (RUN_STARTED, None),
        (REVISION_STARTED, None),
        (NODE_STARTED, nodes.RESEARCHER),
        # The edited branch's search: the only one the revision makes.
        (AGENT_CALL, nodes.RESEARCHER),
        (NODE_COMPLETED, nodes.RESEARCHER),
        (NODE_STARTED, nodes.SYNTHESIZER),
        (NODE_COMPLETED, nodes.SYNTHESIZER),
        (ITERATION_STARTED, nodes.CRITIC),
        (NODE_STARTED, nodes.CRITIC),
        (NODE_COMPLETED, nodes.CRITIC),
        (CRITIC_VERDICT, nodes.CRITIC),
        (RUN_COMPLETED, None),
    ]
    assert events[0][2] == {"query": "first question", "thread_id": THREAD}
    assert events[2][2]["subtask_id"] == "s1"


async def test_revision_started_says_which_branches_were_carried_over(
    planner_inputs: list[list[str]],
) -> None:
    """Without it a client knows only the edited branch, and draws the run as one
    branch wide: the other two did their work in the parent and send no events."""
    graph = _graph()
    await execute_run("run-1", "first question", THREAD, graph)

    await execute_revision(
        "run-2", THREAD, graph, parent_run_id="run-1", subtask_id="s1", question="edited q1"
    )
    [started] = [payload for kind, _, payload in await _events("run-2") if kind == REVISION_STARTED]

    assert started == {
        "parent_run_id": "run-1",
        "subtask_id": "s1",
        "before": "q1",
        "after": "edited q1",
        "branches": [
            {
                "subtask_id": "s0",
                "question": "q0",
                "origin": "carried",
                "sources": [{"url": "https://ex.example/q0", "title": "q0"}],
                "error": None,
            },
            {
                "subtask_id": "s1",
                "question": "edited q1",
                "origin": "edited",
                "sources": [],
                "error": None,
            },
            {
                "subtask_id": "s2",
                "question": "q2",
                "origin": "carried",
                "sources": [{"url": "https://ex.example/q2", "title": "q2"}],
                "error": None,
            },
        ],
    }


async def test_a_refused_revision_still_ends_its_stream_and_frees_the_thread(
    planner_inputs: list[list[str]], fake_redis: FakeRedis
) -> None:
    graph = _graph()
    await execute_run("run-1", "first question", THREAD, graph)
    fake_redis.values[eventlog.thread_key(THREAD)] = "run-2"

    with pytest.raises(Exception, match="s9"):
        await execute_revision(
            "run-2", THREAD, graph, parent_run_id="run-1", subtask_id="s9", question="x"
        )

    events = await _events("run-2")
    assert [kind for kind, _, _ in events] == [RUN_FAILED]
    assert events[0][2]["error_type"] == "UnknownSubtaskError"
    assert eventlog.thread_key(THREAD) not in fake_redis.values


# --- the run record --------------------------------------------------------------------------


async def test_both_kinds_of_run_are_recorded_with_how_they_ended(
    planner_inputs: list[list[str]],
) -> None:
    graph = _graph()
    store = InMemoryRunStore()

    first = await execute_run("run-1", "first question", THREAD, graph, store=store)
    second = await execute_revision(
        "run-2",
        THREAD,
        graph,
        parent_run_id="run-1",
        subtask_id="s1",
        question="edited q1",
        store=store,
    )
    runs = await store.thread(THREAD)

    assert [(r.run_id, r.kind, r.query, r.parent_run_id, r.revision, r.status) for r in runs] == [
        ("run-1", "question", "first question", None, None, "completed"),
        (
            "run-2",
            "revision",
            "first question",
            "run-1",
            Revision(parent_run_id="run-1", subtask_id="s1", before="q1", after="edited q1"),
            "completed",
        ),
    ]
    assert [r.result for r in runs] == [first, second]


async def test_a_refused_revision_is_recorded_as_failed(
    planner_inputs: list[list[str]],
) -> None:
    graph = _graph()
    store = InMemoryRunStore()
    await execute_run("run-1", "first question", THREAD, graph, store=store)

    with pytest.raises(Exception, match="s9"):
        await execute_revision(
            "run-2",
            THREAD,
            graph,
            parent_run_id="run-1",
            subtask_id="s9",
            question="x",
            store=store,
        )

    record = await store.get("run-2")
    assert record is not None
    assert (record.kind, record.parent_run_id, record.status) == ("revision", "run-1", "failed")
    assert record.error is not None and "s9" in record.error


async def test_a_store_that_cannot_write_degrades_the_run_but_does_not_fail_it(
    planner_inputs: list[list[str]],
) -> None:
    """The answer is still good; the audit trail is not complete, and it says so."""

    class BrokenStore(InMemoryRunStore):
        async def started(self, record: RunRecord) -> None:
            raise ConnectionError("postgres is down")

        async def finished(
            self,
            run_id: str,
            *,
            status: RunStatusName,
            result: Any = None,
            error: str | None = None,
            at: Any = None,
        ) -> None:
            raise ConnectionError("postgres is down")

    result = await execute_run("run-1", "first question", THREAD, _graph(), store=BrokenStore())

    assert result["metrics"]["degraded"] == [DEGRADED_AUDIT_INCOMPLETE]
    assert result["answer"] != ""


# --- follow-ups from a chosen version -------------------------------------------------------


async def test_a_follow_up_continues_from_the_version_it_names(
    planner_inputs: list[list[str]],
) -> None:
    """After a revision the thread's latest state is the revision's. A person
    reading the original and asking a follow-up there means the original."""
    graph = _graph()
    await execute_run("run-1", "first question", THREAD, graph)
    await execute_revision(
        "run-2", THREAD, graph, parent_run_id="run-1", subtask_id="s1", question="edited q1"
    )

    await execute_run("run-3", "from the latest", THREAD, graph)
    await execute_run("run-4", "from v1", THREAD, graph, parent_run_id="run-1")

    # The revision never plans; these are the two follow-ups' planners.
    assert planner_inputs[1:] == [
        ["first question", "answer (edited)", "from the latest"],
        ["first question", "answer", "from v1"],
    ]


async def test_the_result_carries_each_branchs_research_whole(
    planner_inputs: list[list[str]],
) -> None:
    """The inspector's Output tab for a finished run: what each branch found."""
    graph = _graph()

    result = await execute_run("run-1", "first question", THREAD, graph)

    assert [(r["subtask_id"], r["findings"]) for r in result["research_results"]] == [
        ("s0", "researcher: Question: q0"),
        ("s1", "researcher: Question: q1"),
        ("s2", "researcher: Question: q2"),
    ]
    assert result["research_results"][0]["sources"][0]["url"] == "https://ex.example/q0"
