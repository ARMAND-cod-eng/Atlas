"""Forgetting a conversation, on request and on a schedule.

Atlas keeps a thread's questions, every prompt its agents were sent, and the
graph's checkpoints. People's questions are personal data, and a tool that
answers questions about the GDPR should practise it: a thread can be deleted
outright, and threads left idle past the retention period are deleted nightly.

"Deleted" means every place a thread lives: the run records, the recorded
calls and archived events, the checkpoints, and anything still in Redis.
Nothing is deleted from under a running run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import Source
from atlas.retention import ThreadBusyError, forget_thread, purge_expired
from atlas.runner import execute_run
from atlas.store import InMemoryRunStore
from tests.conftest import FakeRedis, make_call_model, make_call_structured, make_plan

NOW = datetime(2026, 10, 24, 3, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fast_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))


class World:
    def __init__(self) -> None:
        self.store = InMemoryRunStore()
        self.saver = MemorySaver(serde=checkpoint_serde())
        self.graph = build_state_graph().compile(checkpointer=self.saver)

    async def ask(self, run_id: str, thread_id: str, at: datetime | None = None) -> None:
        await execute_run(run_id, "q", thread_id, self.graph, store=self.store)
        if at is not None:
            record = self.store.records[run_id]
            self.store.records[run_id] = record.model_copy(update={"created_at": at})

    async def checkpointed(self, thread_id: str) -> bool:
        return await self.saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is not None


@pytest.fixture
def world() -> World:
    return World()


async def test_forgetting_a_thread_removes_it_everywhere(
    world: World, fake_redis: FakeRedis
) -> None:
    await world.ask("run-1", "t1")
    await world.ask("run-2", "t1")
    await world.ask("run-3", "t2")

    forgotten = await forget_thread("t1", world.store, world.saver)

    assert forgotten == ["run-1", "run-2"]
    assert await world.store.thread("t1") == []
    assert [await world.store.calls(r) for r in forgotten] == [[], []]
    assert [await world.store.events(r) for r in forgotten] == [[], []]
    assert await world.checkpointed("t1") is False
    assert [await eventlog.exists(r) for r in forgotten] == [False, False]
    assert all("run-1" not in key and "run-2" not in key for key in fake_redis.streams)


async def test_forgetting_one_thread_leaves_the_others(world: World) -> None:
    await world.ask("run-1", "t1")
    await world.ask("run-3", "t2")

    await forget_thread("t1", world.store, world.saver)

    assert [r.run_id for r in await world.store.thread("t2")] == ["run-3"]
    assert await world.checkpointed("t2") is True
    assert await eventlog.exists("run-3") is True


async def test_a_thread_with_a_run_in_progress_is_not_deleted(
    world: World, fake_redis: FakeRedis
) -> None:
    await world.ask("run-1", "t1")
    fake_redis.values[eventlog.thread_key("t1")] = "run-in-flight"

    with pytest.raises(ThreadBusyError, match="run-in-flight"):
        await forget_thread("t1", world.store, world.saver)

    assert [r.run_id for r in await world.store.thread("t1")] == ["run-1"]
    assert fake_redis.values[eventlog.thread_key("t1")] == "run-in-flight"


async def test_the_nightly_purge_forgets_threads_idle_past_the_retention_period(
    world: World,
) -> None:
    await world.ask("run-old", "old", at=NOW - timedelta(days=31))
    await world.ask("run-recent", "kept", at=NOW - timedelta(days=31))
    await world.ask("run-follow-up", "kept", at=NOW - timedelta(days=2))
    await world.ask("run-new", "new", at=NOW - timedelta(hours=1))

    purged = await purge_expired(world.store, world.saver, retention_days=30, now=NOW)

    assert purged == ["old"]
    assert await world.checkpointed("old") is False
    assert [r.run_id for r in await world.store.thread("kept")] == ["run-recent", "run-follow-up"]


async def test_a_busy_thread_is_skipped_by_the_purge_not_failed(
    world: World, fake_redis: FakeRedis
) -> None:
    await world.ask("run-a", "busy", at=NOW - timedelta(days=40))
    await world.ask("run-b", "idle", at=NOW - timedelta(days=40))
    fake_redis.values[eventlog.thread_key("busy")] = "run-in-flight"

    purged = await purge_expired(world.store, world.saver, retention_days=30, now=NOW)

    assert purged == ["idle"]


async def test_a_retention_period_of_zero_keeps_everything(world: World) -> None:
    await world.ask("run-old", "old", at=NOW - timedelta(days=400))

    assert await purge_expired(world.store, world.saver, retention_days=0, now=NOW) == []


def test_the_worker_runs_the_forget_job_by_the_name_the_api_queues_it_under() -> None:
    """A renamed job would leave every DELETE queued for a function nobody runs."""
    from arq.worker import Function

    from atlas.worker import FORGET_THREAD, WorkerSettings, purge_expired_job

    names = [f.name for f in WorkerSettings.functions if isinstance(f, Function)]
    [nightly] = WorkerSettings.cron_jobs

    assert names == [FORGET_THREAD]
    assert (nightly.coroutine, nightly.hour, nightly.minute) == (purge_expired_job, {3}, {0})
