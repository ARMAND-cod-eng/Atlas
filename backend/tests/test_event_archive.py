"""A run's timeline outlives Redis.

Redis keeps a run's event stream for an hour, which is right for watching it
live. A version opened tomorrow from the versions list still needs its
timeline, so the runner archives the stream when the run ends, and the SSE
route replays the archive once Redis has forgotten the run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog
from atlas.api.main import create_app
from atlas.events import RUN_COMPLETED
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import Source
from atlas.runner import execute_run
from atlas.store import InMemoryRunStore
from tests.conftest import FakeRedis, make_call_model, make_call_structured, make_plan


@pytest.fixture(autouse=True)
def fast_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))


async def _finished_run(store: InMemoryRunStore) -> list[Any]:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    await execute_run("run-1", "q", "t1", graph, store=store)
    return await eventlog.replay("run-1", 0)


def _frames(body: str) -> list[dict[str, Any]]:
    """The data of each SSE frame, in order."""
    return [
        json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")
    ]


async def test_the_whole_stream_is_archived_when_the_run_ends() -> None:
    store = InMemoryRunStore()
    live = await _finished_run(store)

    assert await store.events("run-1") == live
    assert live[-1].type == RUN_COMPLETED


async def test_once_redis_forgets_the_run_its_timeline_replays_from_the_archive(
    fake_redis: FakeRedis,
) -> None:
    store = InMemoryRunStore()
    live = await _finished_run(store)
    fake_redis.values.clear()
    fake_redis.streams.clear()
    app = create_app()
    app.state.store = store

    response = TestClient(app).get("/runs/run-1/events")

    assert response.status_code == 200
    assert [(f["seq"], f["type"]) for f in _frames(response.text)] == [
        (e.seq, e.type) for e in live
    ]


async def test_a_reconnect_resumes_the_archive_where_it_left_off(fake_redis: FakeRedis) -> None:
    store = InMemoryRunStore()
    live = await _finished_run(store)
    fake_redis.values.clear()
    fake_redis.streams.clear()
    app = create_app()
    app.state.store = store

    response = TestClient(app).get("/runs/run-1/events", headers={"Last-Event-ID": "5"})

    assert [f["seq"] for f in _frames(response.text)] == [e.seq for e in live if e.seq > 5]


def test_a_run_in_neither_redis_nor_the_archive_is_404() -> None:
    app = create_app()
    app.state.store = InMemoryRunStore()

    response = TestClient(app).get("/runs/nope/events")

    assert response.status_code == 404
