"""A question outside the AI Act and the GDPR is said to be, and not researched.

The planner judges it, in the plan it writes anyway: no extra call. Three things
hold that judgment to account. A question that names an act, cites a provision
or asks what a term the acts define means is planned whatever the planner says.
Only a conversation's first question is judged: a follow-up reads in the light
of what came before, and asking again in the conversation is how a person
researches a question anyway. And only the domain with a library has a scope.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.domains import using_domain
from atlas.events import NODE_STARTED, SCOPE_DECLINED
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.library.scope import declined_answer, names_the_law
from atlas.library.sources import library_search
from atlas.metrics import RunMetrics
from atlas.offline import plan_of
from atlas.result import build_response
from atlas.runner import execute_run, initial_state
from tests.conftest import FakeRedis, make_call_model, make_call_structured

CAMEROON = "What is the history of Cameroon?"
REASON = "The question is about the history of Cameroon."
SYNTHESIS = "The synthesized answer."


@pytest.fixture
def judged_out(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A planner that judges every question out of scope; the questions researchers got."""
    researched: list[str] = []

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        researched.append(query)
        return []

    async def fake_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        return AIMessage(content=SYNTHESIS if node == nodes.SYNTHESIZER else "findings")

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    monkeypatch.setattr(
        nodes, "call_structured", make_call_structured(PlannerOutput(out_of_scope=REASON))
    )
    return researched


async def _run(
    query: str, *, check_scope: bool, domain: str = "compliance"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    config = {"configurable": {"thread_id": "t", "run_id": "r", "domain": domain}}
    events: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        initial_state(query, check_scope=check_scope),
        config=config,
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            events.append(dict(chunk))
        else:
            final = dict(chunk)
    return final, events


async def test_a_question_judged_out_of_scope_ends_at_the_planner(judged_out: list[str]) -> None:
    final, events = await _run(CAMEROON, check_scope=True)

    assert [(e["type"], e["reason"]) for e in events if e["type"] == SCOPE_DECLINED] == [
        (SCOPE_DECLINED, REASON)
    ]
    # Nothing researched, nothing written, nothing reviewed.
    assert [e["node"] for e in events if e["type"] == NODE_STARTED] == [nodes.PLANNER]
    assert judged_out == []
    assert final["declined"] == REASON
    assert final["answer"] == declined_answer(REASON)
    response = build_response(final, "t")
    assert (response.declined, response.answer, response.subtasks) == (
        REASON,
        declined_answer(REASON),
        [],
    )


async def test_a_question_that_names_the_law_is_planned_whatever_the_planner_says(
    judged_out: list[str],
) -> None:
    question = "Does the GDPR say anything about the history of Cameroon?"

    final, events = await _run(question, check_scope=True)

    assert [e for e in events if e["type"] == SCOPE_DECLINED] == []
    # The planner gave no branches, so the question is researched as one.
    assert len(judged_out) == 1
    assert final["declined"] is None
    assert final["answer"] == SYNTHESIS


async def test_a_question_not_judged_is_researched_as_one_branch(judged_out: list[str]) -> None:
    final, events = await _run(CAMEROON, check_scope=False)

    assert [e for e in events if e["type"] == SCOPE_DECLINED] == []
    assert len(judged_out) == 1
    assert (final["declined"], final["answer"]) == (None, SYNTHESIS)


async def test_the_general_domain_has_no_scope(judged_out: list[str]) -> None:
    final, events = await _run(CAMEROON, check_scope=True, domain="general")

    assert [e for e in events if e["type"] == SCOPE_DECLINED] == []
    assert (final["declined"], final["answer"]) == (None, SYNTHESIS)


async def test_a_plan_without_a_judgment_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the planner's subtasks are researched exactly as before."""
    researched: list[str] = []

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        researched.append(query)
        return []

    plan = PlannerOutput(
        subtasks=[
            Subtask(id="a", question="q one", rationale="r"),
            Subtask(id="b", question="q two", rationale="r"),
        ]
    )
    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model(SYNTHESIS))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(plan))

    final, events = await _run(
        "What must a provider of a high-risk AI system do?", check_scope=True
    )

    assert [e for e in events if e["type"] == SCOPE_DECLINED] == []
    assert final["declined"] is None
    assert [s.question for s in final["subtasks"]] == ["q one", "q two"]
    assert sorted(researched) == ["q one", "q two"]


# --- the runner: only a conversation's first question is judged -------------------


async def test_only_the_first_question_on_a_thread_is_judged(
    judged_out: list[str], fake_redis: FakeRedis
) -> None:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))

    with using_domain("compliance"):
        first = await execute_run("run-1", CAMEROON, "thread-1", graph)
        # Asked again in the conversation: researched, as "Research anyway" does.
        again = await execute_run("run-2", CAMEROON, "thread-1", graph, parent_run_id="run-1")

    first_events = [e.type for e in await eventlog.replay("run-1", 0)]
    again_events = [e.type for e in await eventlog.replay("run-2", 0)]
    assert SCOPE_DECLINED in first_events
    assert SCOPE_DECLINED not in again_events
    assert (first["declined"], first["answer"]) == (REASON, declined_answer(REASON))
    assert (again["declined"], again["answer"]) == (None, SYNTHESIS)
    assert len(judged_out) == 1


async def test_a_follow_up_on_the_latest_state_is_not_judged_either(
    judged_out: list[str], fake_redis: FakeRedis
) -> None:
    """Without a parent named: the thread's own history is what makes it a follow-up."""
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))

    with using_domain("compliance"):
        await execute_run("run-1", CAMEROON, "thread-2", graph)
        again = await execute_run("run-2", "And its geography?", "thread-2", graph)

    assert again["declined"] is None


# --- what holds the judgment to account ----------------------------------------------


@pytest.mark.parametrize(
    ("question", "named"),
    [
        ("What does Article 22 GDPR say?", True),
        ("Is this covered by the AI Act?", True),
        ("Summarise Annex III.", True),
        ("history of cameroon", False),
        ("How do I make sourdough bread at home?", False),
    ],
)
def test_naming_the_law(question: str, named: bool) -> None:
    assert names_the_law(question) is named


# --- offline mode's stand-in for the planner's judgment ------------------------------


@pytest.mark.parametrize(
    ("question", "declined"),
    [
        ("history of cameroon", True),
        ("Who painted the Mona Lisa?", True),
        ("what is a DPIA", False),
        ("right to be forgotten", False),
        ("Is a CV-screening tool high-risk under the AI Act?", False),
    ],
)
def test_offline_mode_declines_only_what_the_law_has_nothing_near(
    question: str, declined: bool
) -> None:
    plan = plan_of(question)

    assert (plan.out_of_scope is not None) is declined
    # Branches either way: a judgment not acted on (a follow-up) still researches.
    assert plan.subtasks != []


def test_a_dpia_is_found_by_its_abbreviation() -> None:
    hits = library_search().search("what is a DPIA", limit=3)

    # Article 35 is the data protection impact assessment; before, nothing was found.
    assert [hit.unit.label.split("(")[0] for hit in hits] == ["GDPR Art. 35"] * 3
