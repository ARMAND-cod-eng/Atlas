"""The durable run record: one contract, two stores.

Redis keeps a run's events and result for an hour. A revision needs its parent
long after that, and a thread's versions list needs every run on it, so each run
is also recorded in Postgres. The in-memory store serves the tests and the eval;
both are held to the same contract here, so a test that passes against one says
something about the other.

The Postgres half runs only under ``-m postgres`` with ``ATLAS_TEST_DATABASE_URL``
set (CI starts a Postgres service for it). It is deselected by default rather than
skipped when the database is missing: a skip reads as a pass.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from atlas.events import AgentEvent
from atlas.graph.state import Revision
from atlas.store import InMemoryRunStore, PostgresRunStore, RunRecord, RunStore
from atlas.trace import AgentCall

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[RunStore]:
    if request.param == "memory":
        yield InMemoryRunStore()
        return

    from sqlalchemy.ext.asyncio import create_async_engine

    url = os.environ["ATLAS_TEST_DATABASE_URL"]
    engine = create_async_engine(url)
    postgres = PostgresRunStore(engine)
    await postgres.setup()
    try:
        yield postgres
    finally:
        await engine.dispose()


def _record(thread_id: str, offset_s: int = 0, **fields: object) -> RunRecord:
    defaults: dict[str, object] = {
        "run_id": f"run-{uuid4()}",
        "thread_id": thread_id,
        "kind": "question",
        "query": "what is RAG?",
        "created_at": T0 + timedelta(seconds=offset_s),
    }
    return RunRecord.model_validate({**defaults, **fields})


async def test_a_started_run_reads_back_as_running(store: RunStore) -> None:
    record = _record(f"t-{uuid4()}")

    await store.started(record)

    assert await store.get(record.run_id) == record
    assert record.status == "running"


async def test_a_completed_run_keeps_its_result(store: RunStore) -> None:
    record = _record(f"t-{uuid4()}")
    result = {"answer": "RAG retrieves, then generates.", "thread_id": record.thread_id}
    await store.started(record)

    await store.finished(record.run_id, status="completed", result=result, at=T0)

    assert await store.get(record.run_id) == record.model_copy(
        update={"status": "completed", "result": result, "finished_at": T0}
    )


async def test_a_failed_run_keeps_its_error(store: RunStore) -> None:
    record = _record(f"t-{uuid4()}")
    await store.started(record)

    await store.finished(record.run_id, status="failed", error="ProviderError: down", at=T0)

    stored = await store.get(record.run_id)
    assert stored is not None
    assert (stored.status, stored.error, stored.result) == ("failed", "ProviderError: down", None)


async def test_a_revision_round_trips_what_it_changed(store: RunStore) -> None:
    thread = f"t-{uuid4()}"
    parent = _record(thread)
    revision = _record(
        thread,
        offset_s=60,
        kind="revision",
        parent_run_id=parent.run_id,
        revision=Revision(
            parent_run_id=parent.run_id, subtask_id="s1", before="q1", after="edited q1"
        ),
    )
    await store.started(parent)
    await store.started(revision)

    assert await store.get(revision.run_id) == revision


async def test_a_thread_lists_its_runs_oldest_first_and_only_its_own(store: RunStore) -> None:
    thread, other = f"t-{uuid4()}", f"t-{uuid4()}"
    later = _record(thread, offset_s=120, query="follow-up")
    first = _record(thread, offset_s=0)
    elsewhere = _record(other, offset_s=60)
    for record in (later, elsewhere, first):
        await store.started(record)

    assert [r.run_id for r in await store.thread(thread)] == [first.run_id, later.run_id]


async def test_an_unknown_run_is_none(store: RunStore) -> None:
    assert await store.get(f"run-{uuid4()}") is None


@pytest.mark.postgres
async def test_postgres_setup_is_idempotent() -> None:
    """The worker runs it on every start; the second start must not fail."""
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["ATLAS_TEST_DATABASE_URL"])
    try:
        await PostgresRunStore(engine).setup()
        await PostgresRunStore(engine).setup()
    finally:
        await engine.dispose()


# --- agent calls ------------------------------------------------------------------------


def _call(offset_s: int, **fields: object) -> AgentCall:
    defaults: dict[str, object] = {
        "span_id": "span-1",
        "node": "researcher",
        "subtask_id": "s0",
        "kind": "model",
        "name": "claude-sonnet-5",
        "started_at": T0 + timedelta(seconds=offset_s),
        "duration_s": 1.5,
        "request": [{"role": "system", "content": "You are a researcher."}],
        "response": "findings [GDPR Art. 33(1)]",
        "input_tokens": 120,
        "output_tokens": 40,
    }
    return AgentCall.model_validate({**defaults, **fields})


async def test_a_runs_calls_read_back_whole_in_the_order_they_started(store: RunStore) -> None:
    run_id = f"run-{uuid4()}"
    later = _call(5, kind="tool", name="library", request={"query": "breach"}, response=[])
    first = _call(0)
    await store.add_call(run_id, later)
    await store.add_call(run_id, first)
    await store.add_call(f"run-{uuid4()}", _call(1))

    assert await store.calls(run_id) == [first, later]
    assert await store.call(run_id, later.call_id) == later


async def test_a_failed_call_keeps_its_error(store: RunStore) -> None:
    run_id = f"run-{uuid4()}"
    failed = _call(0, response=None, error="ProviderError: all providers failed")
    await store.add_call(run_id, failed)

    assert await store.call(run_id, failed.call_id) == failed


async def test_a_call_of_another_run_is_not_found(store: RunStore) -> None:
    run_id = f"run-{uuid4()}"
    call = _call(0)
    await store.add_call(run_id, call)

    assert await store.call(f"run-{uuid4()}", call.call_id) is None


# --- the event archive ------------------------------------------------------------------


async def test_a_runs_events_are_archived_and_read_back_in_order(store: RunStore) -> None:
    run_id = f"run-{uuid4()}"
    events = [
        AgentEvent(run_id=run_id, seq=1, ts=T0, type="run_started", payload={"query": "q"}),
        AgentEvent(
            run_id=run_id,
            seq=2,
            ts=T0 + timedelta(seconds=1),
            type="node_started",
            node="planner",
            payload={"subtask_id": None, "span_id": "abc"},
        ),
        AgentEvent(
            run_id=run_id, seq=3, ts=T0 + timedelta(seconds=9), type="run_completed", payload={}
        ),
    ]

    await store.archive_events(run_id, list(reversed(events)))

    assert await store.events(run_id) == events
    assert await store.events(f"run-{uuid4()}") == []
