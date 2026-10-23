"""Editing one branch's task re-runs that branch, and only that branch.

A revision forks the thread from the parent run's final checkpoint: the edited
subtask is researched again, every other branch's research is carried over, and
the synthesizer and critic run as usual. The fork has to start clean. The parent's
answer drafts, its critic's feedback, its metrics and its iteration count all sit
in that checkpoint, and each one, carried into the revision, produces a plausible
answer built on someone else's state — the failure that looks like success.

Every test here runs the real graph against an in-memory checkpointer, with only
search and the models faked.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import CriticVerdict, Revision, Source, SubtaskVerdict
from atlas.metrics import RunMetrics
from atlas.revision import ParentRunError, UnknownSubtaskError, fork_for_revision
from tests.conftest import initial_state, make_call_structured, make_plan, run_config

THREAD = "conversation"
FEEDBACK = "the write-up omits the 2024 benchmark"


class Calls:
    """What the fakes saw: every search query, and every prompt by node."""

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.prompts: dict[str, list[list[BaseMessage]]] = {}

    def clear(self) -> None:
        self.searches.clear()
        self.prompts.clear()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    seen = Calls()

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        seen.searches.append(query)
        return [Source(url=f"https://ex.example/{query}", title=query, snippet=f"about {query}")]

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        seen.prompts.setdefault(node, []).append(list(messages))
        question = str(messages[-1].content).split("\n", 1)[0]
        return AIMessage(content=f"{node}: {question}")

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    return seen


def _critic(monkeypatch: pytest.MonkeyPatch, *verdicts: CriticVerdict) -> None:
    plan = make_plan(3)
    monkeypatch.setattr(
        nodes, "call_structured", make_call_structured(plan, list(verdicts) or None)
    )


def _graph() -> Any:
    return build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))


async def _parent(graph: Any, monkeypatch: pytest.MonkeyPatch, *verdicts: CriticVerdict) -> Any:
    _critic(monkeypatch, *verdicts)
    return await graph.ainvoke(initial_state("first question"), config=run_config(THREAD, "run-1"))


async def _revise(
    graph: Any,
    run_id: str,
    subtask_id: str,
    question: str,
    parent_run_id: str = "run-1",
) -> dict[str, Any]:
    config = await fork_for_revision(
        graph,
        thread_id=THREAD,
        parent_run_id=parent_run_id,
        run_id=run_id,
        subtask_id=subtask_id,
        question=question,
    )
    return dict(await graph.ainvoke(None, config=config))


# --- what runs ------------------------------------------------------------------------


async def test_only_the_edited_branch_is_researched_again(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    parent = await _parent(graph, monkeypatch)
    calls.clear()

    revised = await _revise(graph, "run-2", "s1", "edited q1")

    assert calls.searches == ["edited q1"]
    assert [(s.id, s.question) for s in revised["subtasks"]] == [
        ("s0", "q0"),
        ("s1", "edited q1"),
        ("s2", "q2"),
    ]
    by_id = {r.subtask_id: r for r in revised["research_results"]}
    assert [r.subtask_id for r in revised["research_results"]] == ["s0", "s1", "s2"]
    # Carried over untouched, not re-researched to the same result.
    assert by_id["s0"] == parent["research_results"][0]
    assert by_id["s2"] == parent["research_results"][2]
    assert by_id["s1"].findings == "researcher: Question: edited q1"
    assert revised["revision"] == Revision(
        parent_run_id="run-1", subtask_id="s1", before="q1", after="edited q1"
    )


async def test_the_edited_branch_is_not_told_a_reviewer_rejected_it(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A human edit is a new question, not a critique of the old answer."""
    graph = _graph()
    await _parent(graph, monkeypatch)
    calls.clear()

    await _revise(graph, "run-2", "s1", "edited q1")

    [prompt] = calls.prompts[nodes.RESEARCHER]
    assert str(prompt[-1].content) == (
        "Question: edited q1\n\nSources:\n"
        "[1] edited q1\nURL: https://ex.example/edited q1\nabout edited q1"
    )


async def test_the_synthesizer_sees_neither_the_parents_drafts_nor_its_feedback(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent wrote two drafts, the first rejected for its write-up.

    Both drafts sit in ``messages`` and the rejection in ``critic_feedback``. Left
    there, the revision's synthesizer would replay the old answers as prior turns
    and be told "a reviewer rejected your previous answer" about an answer it
    never wrote.
    """
    graph = _graph()
    await _parent(
        graph,
        monkeypatch,
        CriticVerdict(overall="revise", synthesis_feedback=FEEDBACK),
        CriticVerdict(overall="accept"),
    )
    calls.clear()
    _critic(monkeypatch)

    await _revise(graph, "run-2", "s1", "edited q1")

    [prompt] = calls.prompts[nodes.SYNTHESIZER]
    assert [type(m) for m in prompt] == [SystemMessage, HumanMessage, HumanMessage]
    assert str(prompt[1].content) == "first question"
    assert FEEDBACK not in str(prompt[-1].content)


async def test_the_conversation_keeps_the_question_and_only_the_new_answer(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    await _parent(
        graph,
        monkeypatch,
        CriticVerdict(overall="revise", synthesis_feedback=FEEDBACK),
        CriticVerdict(overall="accept"),
    )
    _critic(monkeypatch)

    revised = await _revise(graph, "run-2", "s1", "edited q1")

    assert [(type(m), m.content) for m in revised["messages"]] == [
        (HumanMessage, "first question"),
        (AIMessage, revised["answer"]),
    ]


# --- what is reset -------------------------------------------------------------------------


async def test_the_fork_starts_without_the_parents_answer(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revision that failed before synthesis must not report the parent's answer."""
    graph = _graph()
    parent = await _parent(graph, monkeypatch)
    config = await fork_for_revision(
        graph,
        thread_id=THREAD,
        parent_run_id="run-1",
        run_id="run-2",
        subtask_id="s1",
        question="edited q1",
    )

    snapshot = await graph.aget_state(config)
    forked = snapshot.values

    assert snapshot.metadata["run_id"] == "run-2"
    assert parent["answer"] != ""
    assert forked["answer"] == ""
    assert forked["iteration"] == 0
    assert forked["critic_feedback"] == []
    assert forked["run_metrics"] == RunMetrics()


async def test_the_revisions_metrics_describe_the_revision_alone(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    await _parent(graph, monkeypatch)

    revised = await _revise(graph, "run-2", "s1", "edited q1")
    metrics = revised["run_metrics"]

    assert [(t.node, t.subtask_id) for t in metrics.node_timings] == [
        (nodes.RESEARCHER, "s1"),
        (nodes.SYNTHESIZER, None),
        (nodes.CRITIC, None),
    ]
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [(1, "accept")]
    assert metrics.degraded == []


async def test_the_critic_gets_a_fresh_budget_of_iterations(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent spent both review rounds. The revision is a new piece of work.

    Inheriting ``iteration`` would end the revision at its first rejection, with
    the parent's exhaustion stamped on it.
    """
    graph = _graph()
    faulted = CriticVerdict(
        overall="revise",
        per_subtask=[SubtaskVerdict(subtask_id="s0", verdict="revise", reason="r", guidance="g")],
    )
    parent = await _parent(graph, monkeypatch, faulted, faulted)
    assert parent["iteration"] == 2

    _critic(monkeypatch, faulted, CriticVerdict(overall="accept"))
    calls.clear()
    revised = await _revise(graph, "run-2", "s1", "edited q1")

    assert [(r.iteration, r.overall) for r in revised["run_metrics"].critic_rounds] == [
        (1, "revise"),
        (2, "accept"),
    ]
    # The edit, then the critic's own retry of s0.
    assert calls.searches == ["edited q1", "q0"]
    assert revised["run_metrics"].degraded == []


# --- forks of forks, and refusals -----------------------------------------------------------


async def test_the_same_parent_can_be_revised_twice(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each revision forks the parent, not the previous revision."""
    graph = _graph()
    await _parent(graph, monkeypatch)
    await _revise(graph, "run-2", "s1", "first edit")
    calls.clear()

    second = await _revise(graph, "run-3", "s2", "second edit")

    assert calls.searches == ["second edit"]
    assert [s.question for s in second["subtasks"]] == ["q0", "q1", "second edit"]


async def test_a_revision_can_itself_be_revised(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    await _parent(graph, monkeypatch)
    await _revise(graph, "run-2", "s1", "first edit")
    calls.clear()

    third = await _revise(graph, "run-3", "s2", "second edit", parent_run_id="run-2")

    assert calls.searches == ["second edit"]
    assert [s.question for s in third["subtasks"]] == ["q0", "first edit", "second edit"]


async def test_the_revisions_checkpoints_carry_its_own_run_id(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    await _parent(graph, monkeypatch)
    await _revise(graph, "run-2", "s1", "edited q1")

    latest = await graph.aget_state({"configurable": {"thread_id": THREAD}})

    assert latest.metadata["run_id"] == "run-2"


async def test_a_follow_up_after_a_revision_is_not_an_edit(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new question on the thread plans afresh and carries no revision marker."""
    graph = _graph()
    await _parent(graph, monkeypatch)
    await _revise(graph, "run-2", "s1", "edited q1")

    follow_up = await graph.ainvoke(initial_state("follow-up"), config=run_config(THREAD, "run-3"))

    assert follow_up["revision"] is None
    assert [s.question for s in follow_up["subtasks"]] == ["q0", "q1", "q2"]


async def test_an_unknown_subtask_is_refused(calls: Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph()
    await _parent(graph, monkeypatch)

    with pytest.raises(UnknownSubtaskError, match="s9"):
        await _revise(graph, "run-2", "s9", "edited")


async def test_a_run_with_no_checkpoint_on_the_thread_is_refused(
    calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    await _parent(graph, monkeypatch)

    with pytest.raises(ParentRunError, match="run-404"):
        await _revise(graph, "run-2", "s1", "edited", parent_run_id="run-404")
