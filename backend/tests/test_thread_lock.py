"""One run at a time per thread, and a bounded question.

Two runs on one thread write the same checkpoint concurrently: both start from
the same state, and whichever finishes second silently overwrites the other's
turn of the conversation. Nothing raises. So the API claims the thread before
enqueueing and refuses a second run with 409 until the first ends.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.api.main import create_app
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import Source
from atlas.runner import execute_run
from atlas.worker import JOB_TIMEOUT_SECONDS
from tests.conftest import FakeArq, FakeRedis, make_call_model, make_call_structured, make_plan

THREAD_KEY = "atlas:thread:t1"


@pytest.fixture
def fast_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str, max_results: int | None = None, include_content: bool = False
    ) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))


def _client() -> tuple[TestClient, FakeArq]:
    compiled = build_state_graph().compile(checkpointer=MemorySaver())
    arq = FakeArq(
        lambda run_id, query, thread_id, domain=None: execute_run(
            run_id, query, thread_id, compiled, domain=domain
        )
    )
    app = create_app()
    app.state.arq = arq
    return TestClient(app), arq


# --- the claim --------------------------------------------------------------------


def test_a_second_run_on_a_busy_thread_is_refused(fast_graph: None, fake_redis: FakeRedis) -> None:
    fake_redis.values[THREAD_KEY] = "run-in-flight"
    client, arq = _client()

    response = client.post("/runs", json={"query": "q", "thread_id": "t1"})

    assert response.status_code == 409
    assert response.json() == {"detail": "thread t1 already has a run in progress: run-in-flight"}
    # Refused before anything was queued or spent.
    assert arq.calls == []
    assert fake_redis.values[THREAD_KEY] == "run-in-flight"


def test_another_thread_is_unaffected(fast_graph: None, fake_redis: FakeRedis) -> None:
    """The control: the claim is per thread, not global."""
    fake_redis.values[THREAD_KEY] = "run-in-flight"
    client, _ = _client()

    assert client.post("/runs", json={"query": "q", "thread_id": "t2"}).status_code == 202


def test_the_thread_is_free_again_once_the_run_ends(
    fast_graph: None, fake_redis: FakeRedis
) -> None:
    client, arq = _client()

    first = client.post("/runs", json={"query": "q", "thread_id": "t1"})
    second = client.post("/runs", json={"query": "follow-up", "thread_id": "t1"})

    assert (first.status_code, second.status_code) == (202, 202)
    assert [call[3] for call in arq.calls] == ["t1", "t1"]
    assert THREAD_KEY not in fake_redis.values


def test_a_failed_run_releases_the_thread_too(
    fast_graph: None, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(nodes, "call_model", boom)
    client, _ = _client()

    client.post("/runs", json={"query": "q", "thread_id": "t1"})

    assert THREAD_KEY not in fake_redis.values


def test_a_failed_enqueue_releases_the_claim(fast_graph: None, fake_redis: FakeRedis) -> None:
    """Otherwise the thread stays locked for the claim's whole TTL by a run that
    never existed."""

    class RefusingArq:
        async def enqueue_job(self, *args: object, **kwargs: object) -> None:
            return None

    app = create_app()
    app.state.arq = RefusingArq()

    response = TestClient(app).post("/runs", json={"query": "q", "thread_id": "t1"})

    assert response.status_code == 503
    assert THREAD_KEY not in fake_redis.values


def test_the_claim_outlives_the_longest_possible_run() -> None:
    """It expires on its own if a worker dies hard, but never under a live run."""
    assert eventlog.THREAD_CLAIM_SECONDS > JOB_TIMEOUT_SECONDS


async def test_a_claim_is_held_with_that_expiry(fake_redis: FakeRedis) -> None:
    assert await eventlog.claim_thread("t1", "run-a") is None

    assert fake_redis.values == {THREAD_KEY: "run-a"}
    assert fake_redis.expiries == {THREAD_KEY: eventlog.THREAD_CLAIM_SECONDS}


async def test_starting_the_run_renews_its_own_claim_only(fake_redis: FakeRedis) -> None:
    """Time spent queued must not eat into the claim the run then needs."""
    fake_redis.values[THREAD_KEY] = "run-a"
    fake_redis.expiries[THREAD_KEY] = 5

    await eventlog.renew_thread("t1", "run-b")
    assert fake_redis.expiries == {THREAD_KEY: 5}

    await eventlog.renew_thread("t1", "run-a")
    assert fake_redis.expiries == {THREAD_KEY: eventlog.THREAD_CLAIM_SECONDS}


async def test_releasing_never_frees_another_runs_claim(fake_redis: FakeRedis) -> None:
    fake_redis.values[THREAD_KEY] = "run-b"

    await eventlog.release_thread("t1", "run-a")

    assert fake_redis.values == {THREAD_KEY: "run-b"}


async def test_a_claim_reports_who_holds_the_thread(fake_redis: FakeRedis) -> None:
    await eventlog.claim_thread("t1", "run-a")

    assert await eventlog.claim_thread("t1", "run-b") == "run-a"


# --- the question -------------------------------------------------------------------


@pytest.mark.parametrize(("length", "status"), [(2000, 202), (2001, 422)])
def test_the_question_is_bounded(
    fast_graph: None, fake_redis: FakeRedis, length: int, status: int
) -> None:
    """Every character is multiplied across the fan-out and every prompt after it."""
    client, _ = _client()

    response = client.post("/runs", json={"query": "q" * length})

    assert response.status_code == status
