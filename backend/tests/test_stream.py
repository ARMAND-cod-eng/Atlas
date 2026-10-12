"""The SSE relay: replay, heartbeats, termination, and surviving a disconnect.

``event_stream`` is tested directly rather than through HTTP. Everything worth
asserting lives in that generator, and driving it as a coroutine makes the two
hardest properties — what an *idle* stream does, and what happens when a client
vanishes — deterministic instead of dependent on socket timing.

The disconnect test is the one that states the architecture's central claim. The
run is owned by the worker, not by the connection, so a client dropping at 40s of
a 67s run must not affect it. That is asserted explicitly, as instructed: not
"the relay cleaned up" but "the run reached its terminal event with nobody
listening".
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from sse_starlette.event import ServerSentEvent

from atlas import eventlog
from atlas.api.main import create_app
from atlas.api.streams import event_stream, parse_last_event_id
from atlas.errors import ProviderError
from atlas.events import RUN_COMPLETED, RUN_STARTED, AgentEvent
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import Source
from atlas.metrics import RunMetrics
from atlas.runner import execute_run
from tests.conftest import FakeArq, FakeRedis, make_call_model, make_call_structured, make_plan

HEARTBEAT_S = 0.05


async def _append(run_id: str, seq: int, event_type: str, **payload: object) -> None:
    await eventlog.append(
        AgentEvent(run_id=run_id, seq=seq, type=event_type, node="planner", payload=payload)
    )


async def _take(
    stream: AsyncIterator[ServerSentEvent], count: int, limit_s: float = 2.0
) -> list[ServerSentEvent]:
    """Pull exactly ``count`` frames, then stop consuming.

    The deadline is a test guard, not part of the behaviour: a relay bug that
    yields nothing should fail here rather than hang the suite.
    """
    frames: list[ServerSentEvent] = []
    async with asyncio.timeout(limit_s):
        async for frame in stream:
            frames.append(frame)
            if len(frames) == count:
                break
    return frames


async def _drain(
    stream: AsyncIterator[ServerSentEvent], limit_s: float = 2.0
) -> list[ServerSentEvent]:
    """Consume until the generator ends on its own."""
    async with asyncio.timeout(limit_s):
        return [frame async for frame in stream]


# --- Last-Event-ID parsing -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("7", 7), ("0", 0), (None, 0), ("", 0), ("not-a-number", 0), ("-3", 0)],
)
def test_last_event_id_parsing(raw: str | None, expected: int) -> None:
    """Anything unparseable replays everything.

    Re-sending a short run costs nothing; silently skipping events a client never
    received is unrecoverable, because seq is how it would have noticed.
    """
    assert parse_last_event_id(raw) == expected


# --- replay --------------------------------------------------------------------


async def _seed_finished_run(run_id: str, last_seq: int = 5) -> None:
    for seq in range(1, last_seq):
        await _append(run_id, seq, "node_started", index=seq)
    await _append(run_id, last_seq, RUN_COMPLETED, answer="done")


async def test_a_finished_run_replays_in_full_and_closes() -> None:
    await _seed_finished_run("done-run")

    frames = await _drain(event_stream("done-run", after_seq=0, heartbeat_s=HEARTBEAT_S))

    assert [f.id for f in frames] == ["1", "2", "3", "4", "5"]
    assert frames[-1].event == RUN_COMPLETED


async def test_reconnect_replays_exactly_the_events_after_that_seq() -> None:
    """The exact set, not a count and not "some events arrived".

    Off-by-one here is the whole failure mode: replaying from ``seq`` rather than
    ``seq + 1`` duplicates an event the client already rendered, and replaying
    from ``seq + 2`` loses one silently.
    """
    await _seed_finished_run("resume-run")

    frames = await _drain(event_stream("resume-run", after_seq=3, heartbeat_s=HEARTBEAT_S))

    assert [f.id for f in frames] == ["4", "5"]
    assert [f.event for f in frames] == ["node_started", RUN_COMPLETED]


async def test_a_client_that_already_has_everything_gets_only_the_terminal() -> None:
    await _seed_finished_run("caught-up")

    frames = await _drain(event_stream("caught-up", after_seq=4, heartbeat_s=HEARTBEAT_S))

    assert [f.id for f in frames] == ["5"]


async def test_replay_from_zero_equals_the_whole_log() -> None:
    await _seed_finished_run("whole-log")

    frames = await _drain(event_stream("whole-log", after_seq=0, heartbeat_s=HEARTBEAT_S))
    logged = await eventlog.replay("whole-log", 0)

    assert [f.id for f in frames] == [str(e.seq) for e in logged]


# --- live tailing and heartbeats ------------------------------------------------


async def test_events_appended_after_subscribing_are_delivered() -> None:
    """The live path: a subscriber attached before the run produced anything."""
    await _append("live", 1, RUN_STARTED, query="q")

    stream = event_stream("live", after_seq=0, heartbeat_s=HEARTBEAT_S)
    first = await _take(stream, 1)
    assert [f.id for f in first] == ["1"]

    async def produce() -> None:
        await asyncio.sleep(0.01)
        await _append("live", 2, "node_started")
        await _append("live", 3, RUN_COMPLETED, answer="done")

    producer = asyncio.create_task(produce())
    rest = await _drain(stream)
    await producer

    assert [f.id for f in rest] == ["2", "3"]
    assert rest[-1].event == RUN_COMPLETED


async def test_subscribing_before_the_first_event_waits_rather_than_404s() -> None:
    """The gap between enqueue and the worker's first write.

    Found by live verification, not by these tests: POST /runs returns the moment
    the job is queued, and a client subscribing immediately arrives before
    ``run_started`` exists. The unit tests missed it because the arq double runs
    jobs inline, so events were always already there — a fake that is *too* fast
    hides a race the real thing has.
    """
    await eventlog.register("not-yet")
    assert await eventlog.exists("not-yet") is True

    stream = event_stream("not-yet", after_seq=0, heartbeat_s=HEARTBEAT_S)

    async def produce_late() -> None:
        await asyncio.sleep(HEARTBEAT_S * 2)
        await _append("not-yet", 1, RUN_STARTED, query="q")
        await _append("not-yet", 2, RUN_COMPLETED, answer="done")

    producer = asyncio.create_task(produce_late())
    frames = await _drain(stream)
    await producer

    # Heartbeats while waiting, then the real events — and no gap at seq 1.
    assert [f.id for f in frames if f.id is not None] == ["1", "2"]
    assert any(f.comment == "heartbeat" for f in frames)


async def test_an_unregistered_run_is_still_unknown() -> None:
    """The marker must not make every id look valid."""
    assert await eventlog.exists("never-heard-of-it") is False


async def test_an_idle_stream_emits_heartbeats() -> None:
    """Proxies kill quiet connections, and a synthesis call is 22s of quiet."""
    await _append("idle", 1, RUN_STARTED, query="q")

    stream = event_stream("idle", after_seq=0, heartbeat_s=HEARTBEAT_S)
    frames = await _take(stream, 3)
    await stream.aclose()

    assert frames[0].id == "1"
    # Everything after is a heartbeat: no data, no event name.
    assert [f.comment for f in frames[1:]] == ["heartbeat", "heartbeat"]


async def test_heartbeats_carry_no_id_so_they_cannot_move_the_resume_point() -> None:
    """A comment frame with an id would corrupt reconnection.

    The browser stores the last id it saw. If a heartbeat set one, a client that
    reconnected straight after would resume past events it never received.
    """
    await _append("hb-id", 1, RUN_STARTED, query="q")

    stream = event_stream("hb-id", after_seq=0, heartbeat_s=HEARTBEAT_S)
    frames = await _take(stream, 2)
    await stream.aclose()

    assert frames[1].comment == "heartbeat"
    assert frames[1].id is None
    assert frames[1].data is None


async def test_the_stream_stops_at_a_terminal_event_and_not_before() -> None:
    await _append("stops", 1, RUN_STARTED, query="q")
    await _append("stops", 2, RUN_COMPLETED, answer="done")
    # Anything appended after a terminal event is not the client's problem.
    await _append("stops", 3, "node_started")

    frames = await _drain(event_stream("stops", after_seq=0, heartbeat_s=HEARTBEAT_S))

    assert [f.id for f in frames] == ["1", "2"]


# --- the disconnect claim -------------------------------------------------------


@pytest.fixture
def slow_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_search(query: str, max_results: int | None = None) -> list[Source]:
        await asyncio.sleep(0.15)
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", slow_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))


async def test_a_disconnecting_client_does_not_stop_the_run(slow_graph: None) -> None:
    """The central claim of the decoupled design, asserted rather than implied.

    The client is killed while the run is genuinely mid-flight — the assertion
    that it had not yet finished is what stops this passing vacuously against a
    run that was already over.
    """
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    run = asyncio.create_task(execute_run("abandoned", "what is RAG?", "t", graph))

    stream = event_stream("abandoned", after_seq=0, heartbeat_s=HEARTBEAT_S)
    seen = await _take(stream, 1)
    assert seen[0].event == RUN_STARTED

    # The client vanishes.
    await stream.aclose()

    mid_flight = await eventlog.replay("abandoned", 0)
    assert [e.type for e in mid_flight][-1] != RUN_COMPLETED, "the run was already over"

    result = await asyncio.wait_for(run, timeout=5)

    assert result["answer"] == "the answer"
    events = await eventlog.replay("abandoned", 0)
    assert events[-1].type == RUN_COMPLETED
    assert [e.seq for e in events] == list(range(1, len(events) + 1))


async def test_reconnecting_after_a_disconnect_recovers_the_missed_events(
    slow_graph: None,
) -> None:
    """Drop, let the run continue, come back with Last-Event-ID, miss nothing."""
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    run = asyncio.create_task(execute_run("reconnect", "what is RAG?", "t", graph))

    stream = event_stream("reconnect", after_seq=0, heartbeat_s=HEARTBEAT_S)
    before = await _take(stream, 2)
    await stream.aclose()
    last_seen = int(before[-1].id or "0")

    await asyncio.wait_for(run, timeout=5)

    resumed = await _drain(event_stream("reconnect", after_seq=last_seen, heartbeat_s=HEARTBEAT_S))
    everything = await eventlog.replay("reconnect", 0)

    # Exactly the tail, with no overlap and no hole at the join.
    assert [int(f.id or "0") for f in resumed] == [e.seq for e in everything if e.seq > last_seen]
    assert [f.id for f in before] + [f.id for f in resumed] == [str(e.seq) for e in everything]
    assert resumed[-1].event == RUN_COMPLETED


# --- the route ------------------------------------------------------------------


def _app_with_inline_worker() -> TestClient:
    compiled = build_state_graph().compile(checkpointer=MemorySaver())
    app = create_app()
    app.state.arq = FakeArq(
        lambda run_id, query, thread_id: execute_run(run_id, query, thread_id, compiled)
    )
    return TestClient(app)


@pytest.fixture
def fast_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))


def test_unknown_run_is_404_not_an_empty_stream(fast_graph: None) -> None:
    """An empty stream would be indistinguishable from a run that had not started."""
    response = _app_with_inline_worker().get("/runs/does-not-exist/events")
    assert response.status_code == 404


def test_the_stream_disables_proxy_buffering(fast_graph: None) -> None:
    """Without this nginx delivers every event at once when the run ends."""
    client = _app_with_inline_worker()
    started = client.post("/runs", json={"query": "what is RAG?"})
    assert started.status_code == 202

    run_id = started.json()["run_id"]
    with client.stream("GET", f"/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["cache-control"] == "no-cache"


def test_post_runs_returns_the_ids_and_the_events_url(fast_graph: None) -> None:
    body = _app_with_inline_worker().post("/runs", json={"query": "what is RAG?"}).json()

    assert body["events_url"] == f"/runs/{body['run_id']}/events"
    # Different lifetimes: one thread accumulates many runs.
    assert body["run_id"] != body["thread_id"]


def test_get_run_returns_the_full_result(fast_graph: None) -> None:
    """The fetch point run_completed's summary points at."""
    client = _app_with_inline_worker()
    run_id = client.post("/runs", json={"query": "what is RAG?"}).json()["run_id"]

    body = client.get(f"/runs/{run_id}").json()

    assert body["status"] == "completed"
    assert body["result"]["answer"] == "the answer"
    assert body["result"]["iterations"] == 1
    # The things deliberately absent from the terminal event are present here.
    assert len(body["result"]["sources"]) == 2
    assert len(body["result"]["subtasks"]) == 2


def test_get_run_is_404_for_an_unknown_id(fast_graph: None) -> None:
    assert _app_with_inline_worker().get("/runs/nope").status_code == 404


def test_get_run_reports_a_failed_run_as_failed(
    fast_graph: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to say "running" — for as long as the event log lived.

    A result is stored only on success, so a failed run fell through to the
    "known but unfinished" branch. The stream had already delivered
    ``run_failed``; the status endpoint contradicted it for an hour.
    """

    async def synthesis_fails(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node == nodes.SYNTHESIZER:
            raise ProviderError(provider="openai", message="all providers failed", node=node)
        return AIMessage(content="findings")

    monkeypatch.setattr(nodes, "call_model", synthesis_fails)
    client = _app_with_inline_worker()
    run_id = client.post("/runs", json={"query": "what is RAG?"}).json()["run_id"]

    body = client.get(f"/runs/{run_id}").json()

    assert body == {
        "run_id": run_id,
        "status": "failed",
        "result": None,
        "error": "ProviderError: all providers failed",
    }


def test_a_completed_run_whose_result_was_lost_is_not_reported_running(
    fast_graph: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``store_result`` fails open. The terminal event still says how the run ended."""

    async def lost(run_id: str, result: dict[str, object]) -> None:
        return None

    monkeypatch.setattr(eventlog, "store_result", lost)
    client = _app_with_inline_worker()
    run_id = client.post("/runs", json={"query": "what is RAG?"}).json()["run_id"]

    body = client.get(f"/runs/{run_id}").json()

    assert body == {"run_id": run_id, "status": "completed", "result": None, "error": None}


def test_a_run_with_no_terminal_event_is_running(fast_graph: None) -> None:
    """The control: registered, nothing finished, so neither completed nor failed."""
    client = _app_with_inline_worker()
    asyncio.run(eventlog.register("queued-run"))

    body = client.get("/runs/queued-run").json()

    assert body == {"run_id": "queued-run", "status": "running", "result": None, "error": None}


def test_query_still_returns_the_phase_2_shape(fast_graph: None, fake_redis: FakeRedis) -> None:
    """POST /query is unchanged from the client's side.

    Its execution moved to the queue, but a client that calls it must not have to
    care.
    """
    body = _app_with_inline_worker().post("/query", json={"query": "what is RAG?"}).json()

    assert body["answer"] == "the answer"
    assert body["iterations"] == 1
    assert len(body["subtasks"]) == 2
    assert body["metrics"]["degraded"] == []
