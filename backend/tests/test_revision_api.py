"""The HTTP surface of revisions and versions.

``POST /runs/{run_id}/revisions`` edits one branch of a finished run and queues the
revision. ``GET /threads/{thread_id}/versions`` lists every run on a thread, so a
client can draw the original and its revisions side by side. Both read the
durable run record, which outlives the hour Redis keeps a run.

The queue is the inline FakeArq, so by the time a request returns its run has
happened; the graph runs against an in-memory checkpointer.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from atlas.api.main import create_app
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import Source
from atlas.retention import forget_thread
from atlas.runner import execute_revision, execute_run
from atlas.store import InMemoryRunStore
from atlas.worker import FORGET_THREAD, RUN_RESEARCH, RUN_REVISION
from tests.conftest import FakeArq, FakeRedis, make_call_model, make_call_structured, make_plan


@pytest.fixture(autouse=True)
def fast_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(3)))


class Harness:
    def __init__(self) -> None:
        saver = MemorySaver(serde=checkpoint_serde())
        graph = build_state_graph().compile(checkpointer=saver)
        self.store = InMemoryRunStore()

        async def research(
            run_id: str, query: str, thread_id: str, parent_run_id: str | None = None
        ) -> dict[str, Any]:
            return await execute_run(
                run_id, query, thread_id, graph, parent_run_id=parent_run_id, store=self.store
            )

        async def revise(
            run_id: str, thread_id: str, parent_run_id: str, subtask_id: str, question: str
        ) -> dict[str, Any]:
            return await execute_revision(
                run_id,
                thread_id,
                graph,
                parent_run_id=parent_run_id,
                subtask_id=subtask_id,
                question=question,
                store=self.store,
            )

        async def forget(thread_id: str) -> list[str]:
            return await forget_thread(thread_id, self.store, saver)

        self.arq = FakeArq(research, {RUN_REVISION: revise, FORGET_THREAD: forget})
        app = create_app()
        app.state.arq = self.arq
        app.state.store = self.store
        self.client = TestClient(app)

    def ask(self, query: str = "first question", **extra: str) -> str:
        response = self.client.post("/runs", json={"query": query, **extra})
        assert response.status_code == 202, response.text
        return str(response.json()["run_id"])


@pytest.fixture
def harness() -> Harness:
    return Harness()


# --- revising -----------------------------------------------------------------------------


def test_a_revision_is_queued_on_the_parents_thread(harness: Harness) -> None:
    parent = harness.ask(thread_id="t1")

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": "edited q1"}
    )

    assert response.status_code == 202
    body = response.json()
    assert body["thread_id"] == "t1"
    assert body["events_url"] == f"/runs/{body['run_id']}/events"
    assert harness.arq.calls[-1] == (RUN_REVISION, body["run_id"], "t1", parent, "s1", "edited q1")
    result = harness.client.get(f"/runs/{body['run_id']}").json()["result"]
    assert [s["question"] for s in result["subtasks"]] == ["q0", "edited q1", "q2"]


def test_revising_an_unknown_run_is_404(harness: Harness) -> None:
    response = harness.client.post(
        "/runs/nope/revisions", json={"subtask_id": "s1", "question": "edited"}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown run: nope"}


def test_an_unknown_branch_is_422_and_nothing_is_queued(harness: Harness) -> None:
    parent = harness.ask()
    queued = len(harness.arq.calls)

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s9", "question": "edited"}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": f"run {parent} has no subtask s9 (it has: s0, s1, s2)"}
    assert len(harness.arq.calls) == queued


def test_an_unchanged_question_is_422(harness: Harness) -> None:
    """Re-running a branch on the same question is a retry, not an edit."""
    parent = harness.ask()

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": "  q1 "}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "the question for s1 is unchanged"}


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_an_empty_or_overlong_question_is_422(harness: Harness, question: str) -> None:
    parent = harness.ask()

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": question}
    )

    assert response.status_code == 422


def test_a_run_that_failed_cannot_be_revised(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(nodes, "call_model", boom)
    parent = harness.ask()

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": "edited"}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": f"run {parent} failed, so it has no answer to revise"}


def test_a_busy_thread_refuses_a_revision(harness: Harness, fake_redis: FakeRedis) -> None:
    parent = harness.ask(thread_id="t1")
    fake_redis.values["atlas:thread:t1"] = "run-in-flight"

    response = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": "edited"}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "thread t1 already has a run in progress: run-in-flight"}


# --- follow-ups from a version ------------------------------------------------------------


def test_a_follow_up_can_name_the_version_it_continues(harness: Harness) -> None:
    parent = harness.ask(thread_id="t1")

    response = harness.client.post("/runs", json={"query": "more", "parent_run_id": parent})

    assert response.status_code == 202
    assert response.json()["thread_id"] == "t1"
    assert harness.arq.calls[-1] == (RUN_RESEARCH, response.json()["run_id"], "more", "t1", parent)


def test_a_follow_up_naming_another_threads_run_is_422(harness: Harness) -> None:
    parent = harness.ask(thread_id="t1")

    response = harness.client.post(
        "/runs", json={"query": "more", "thread_id": "t2", "parent_run_id": parent}
    )

    assert response.status_code == 422
    assert response.json() == {"detail": f"run {parent} belongs to thread t1, not t2"}


# --- versions ----------------------------------------------------------------------------------


def test_a_thread_lists_its_runs_with_what_each_revised(harness: Harness) -> None:
    parent = harness.ask(thread_id="t1")
    revision = harness.client.post(
        f"/runs/{parent}/revisions", json={"subtask_id": "s1", "question": "edited q1"}
    ).json()["run_id"]
    follow_up = harness.ask("more", thread_id="t1")

    body = harness.client.get("/threads/t1/versions").json()

    assert body["thread_id"] == "t1"
    assert [
        (r["run_id"], r["kind"], r["query"], r["parent_run_id"], r["revision"], r["status"])
        for r in body["runs"]
    ] == [
        (parent, "question", "first question", None, None, "completed"),
        (
            revision,
            "revision",
            "first question",
            parent,
            {"parent_run_id": parent, "subtask_id": "s1", "before": "q1", "after": "edited q1"},
            "completed",
        ),
        (follow_up, "question", "more", None, None, "completed"),
    ]


def test_an_unknown_thread_is_404(harness: Harness) -> None:
    response = harness.client.get("/threads/nope/versions")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown thread: nope"}


# --- the record outlives Redis -------------------------------------------------------------------


def test_a_finished_run_is_served_from_the_record_once_redis_has_forgotten_it(
    harness: Harness, fake_redis: FakeRedis
) -> None:
    run_id = harness.ask()
    expected = harness.client.get(f"/runs/{run_id}").json()

    # Redis keeps a run for an hour; simulate the hour passing.
    fake_redis.values.clear()
    fake_redis.streams.clear()

    response = harness.client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert response.json() == expected


# --- forgetting a thread -----------------------------------------------------------------


def test_deleting_a_thread_queues_its_forgetting(harness: Harness) -> None:
    first = harness.ask(thread_id="t1")
    second = harness.ask("more", thread_id="t1")

    response = harness.client.delete("/threads/t1")

    assert response.status_code == 202
    assert response.json() == {"thread_id": "t1", "runs": [first, second]}
    assert harness.arq.calls[-1] == (FORGET_THREAD, "t1")
    assert harness.client.get("/threads/t1/versions").status_code == 404


def test_deleting_an_unknown_thread_is_404(harness: Harness) -> None:
    response = harness.client.delete("/threads/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown thread: nope"}


def test_a_thread_with_a_run_in_progress_is_not_deleted(
    harness: Harness, fake_redis: FakeRedis
) -> None:
    harness.ask(thread_id="t1")
    fake_redis.values["atlas:thread:t1"] = "run-in-flight"
    queued = len(harness.arq.calls)

    response = harness.client.delete("/threads/t1")

    assert response.status_code == 409
    assert response.json() == {"detail": "thread t1 has a run in progress: run-in-flight"}
    assert len(harness.arq.calls) == queued
