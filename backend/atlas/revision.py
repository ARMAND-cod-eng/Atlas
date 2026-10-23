"""Revising one branch of a finished run.

A person reads a finished answer, sees that one branch asked the wrong thing, and
rewrites that branch's question. The revision is a new run that forks the thread
from the parent run's final checkpoint, researches the edited branch alone, keeps
every other branch's research, and then synthesises and reviews as usual.

The fork is written as an update from the ``human_edit`` node, whose routing edge
dispatches only the edited branch. Not from the critic, which already has an edge
that sends branches back: a human edit dressed up as a critic verdict would be
appended to ``critic_feedback``, and the synthesizer reads the latest write-up
critique from that list, which would then be the parent's. It would also reach
the researcher as a reviewer's rejection, which an edit is not.

What the parent's checkpoint holds that the revision must not inherit, each one a
plausible answer built on someone else's state if it leaked:

* its answer drafts, in ``messages``, which the synthesizer replays as prior turns;
* its critic's verdicts, which the synthesizer would address as its own;
* its metrics, which the budget check sums;
* its iteration count, which would leave the critic no rounds to spend;
* its answer, which a revision that failed before synthesis would report as its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from atlas.graph.nodes import HUMAN_EDIT
from atlas.graph.state import AtlasState, Revision, Subtask
from atlas.metrics import RunMetrics

Graph = CompiledStateGraph[AtlasState, Any, AtlasState, AtlasState]


class RevisionError(Exception):
    """A revision that cannot be made, for a reason the caller can report."""


class ParentRunError(RevisionError):
    """The run to revise has no finished state on the thread."""


class UnknownSubtaskError(RevisionError):
    """The edit names a branch the parent run did not have."""


async def final_checkpoint(graph: Graph, thread_id: str, run_id: str) -> RunnableConfig:
    """The last checkpoint a run wrote on its thread.

    LangGraph copies the run's ``configurable`` keys into each checkpoint's
    metadata, so the checkpointer itself knows which run wrote what — no second
    record of it to drift out of step.
    """
    checkpointer = graph.checkpointer
    if checkpointer is None or checkpointer is True or checkpointer is False:
        raise ParentRunError("revising a run needs a graph compiled with a checkpointer")
    async for checkpoint in checkpointer.alist(
        {"configurable": {"thread_id": thread_id}}, filter={"run_id": run_id}, limit=1
    ):
        return checkpoint.config
    raise ParentRunError(f"run {run_id!r} has no checkpoint on thread {thread_id!r}")


def _drafts_of_the_last_turn(messages: list[BaseMessage]) -> list[RemoveMessage]:
    """Every message after the turn's question: one answer per synthesis pass.

    All of them, not only the last — a parent the critic sent back once wrote two.
    """
    questions = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if not questions:
        raise ParentRunError("the parent run's conversation has no question in it")
    drafts = messages[questions[-1] + 1 :]
    if any(message.id is None for message in drafts):
        # add_messages assigns every message an id, so this means the state was
        # written some other way — and a draft that cannot be removed would be
        # replayed to the revision's synthesizer.
        raise RevisionError("a parent answer draft has no id and cannot be removed")
    return [RemoveMessage(id=str(message.id)) for message in drafts]


async def fork_for_revision(
    graph: Graph,
    *,
    thread_id: str,
    parent_run_id: str,
    run_id: str,
    subtask_id: str,
    question: str,
) -> RunnableConfig:
    """Fork the thread for a revision and return the config that runs it.

    Resume with ``graph.astream(None, config)``: the edited branch is researched
    first, then the run continues to the synthesizer and the critic.
    """
    parent = await final_checkpoint(graph, thread_id, parent_run_id)
    snapshot = await graph.aget_state(parent)
    if snapshot.next:
        raise ParentRunError(f"run {parent_run_id!r} did not finish, so it cannot be revised")

    values = snapshot.values
    subtasks: list[Subtask] = values["subtasks"]
    original = next((s for s in subtasks if s.id == subtask_id), None)
    if original is None:
        known = ", ".join(s.id for s in subtasks) or "none"
        raise UnknownSubtaskError(
            f"run {parent_run_id!r} has no subtask {subtask_id!r} (it has: {known})"
        )

    edited = [
        s.model_copy(update={"question": question}) if s.id == subtask_id else s for s in subtasks
    ]
    update: dict[str, Any] = {
        "subtasks": edited,
        "revision": Revision(
            parent_run_id=parent_run_id,
            subtask_id=subtask_id,
            before=original.question,
            after=question,
        ),
        "answer": "",
        "iteration": 0,
        "critic_feedback": Overwrite([]),
        "run_metrics": Overwrite(RunMetrics()),
        "messages": _drafts_of_the_last_turn(list(values["messages"])),
    }
    # The run_id goes in so the fork checkpoint is filed under the revision, like
    # every checkpoint after it, rather than under nobody.
    forked = await graph.aupdate_state(
        {"configurable": {**parent["configurable"], "run_id": run_id}},
        update,
        as_node=HUMAN_EDIT,
    )
    return {"configurable": {**forked["configurable"], "run_id": run_id}}


def revision_started_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    """What a client needs to draw a revision before its first branch reports.

    Every branch, in plan order, marked edited or carried over. Carried branches
    send no events in the revision's stream, so their sources travel here.
    """
    revision: Revision = values["revision"]
    results = {result.subtask_id: result for result in values["research_results"]}
    branches: list[dict[str, Any]] = []
    for subtask in values["subtasks"]:
        edited = subtask.id == revision.subtask_id
        carried = None if edited else results.get(subtask.id)
        branches.append(
            {
                "subtask_id": subtask.id,
                "question": subtask.question,
                "origin": "edited" if edited else "carried",
                "sources": [
                    {"url": source.url, "title": source.title}
                    for source in (carried.sources if carried is not None else [])
                ],
                "error": carried.error if carried is not None else None,
            }
        )
    return {
        "parent_run_id": revision.parent_run_id,
        "subtask_id": revision.subtask_id,
        "before": revision.before,
        "after": revision.after,
        "branches": branches,
    }
