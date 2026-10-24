"""Reading a run's agent calls back: a list to browse, and one call in full.

The list carries what an inspector needs to lay the calls out (who made each,
when, at what cost); a prompt and its output are fetched only when someone opens
the call. Prompts run to tens of kilobytes and most are never looked at.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from atlas.api.main import create_app
from atlas.store import InMemoryRunStore, RunRecord
from atlas.trace import AgentCall

T0 = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


@pytest.fixture
def store() -> InMemoryRunStore:
    return InMemoryRunStore()


@pytest.fixture
def client(store: InMemoryRunStore) -> TestClient:
    app = create_app()
    app.state.store = store
    return TestClient(app)


def _call(offset_s: int, node: str, **fields: object) -> AgentCall:
    return AgentCall.model_validate(
        {
            "span_id": f"span-{node}",
            "node": node,
            "kind": "model",
            "name": "claude-sonnet-5",
            "started_at": T0 + timedelta(seconds=offset_s),
            "duration_s": 2.0,
            "request": [{"role": "system", "content": f"You are the {node}."}],
            "response": f"{node} output",
            "input_tokens": 100,
            "output_tokens": 20,
            **fields,
        }
    )


async def _seed(store: InMemoryRunStore) -> list[AgentCall]:
    await store.started(
        RunRecord(run_id="run-1", thread_id="t1", kind="question", query="q", created_at=T0)
    )
    calls = [_call(0, "planner"), _call(3, "researcher", subtask_id="s0")]
    for call in calls:
        await store.add_call("run-1", call)
    return calls


async def test_the_list_lays_out_every_call_without_its_contents(
    client: TestClient, store: InMemoryRunStore
) -> None:
    planner, researcher = await _seed(store)

    body = client.get("/runs/run-1/calls").json()

    assert body["run_id"] == "run-1"
    assert [(c["call_id"], c["node"], c["subtask_id"]) for c in body["calls"]] == [
        (planner.call_id, "planner", None),
        (researcher.call_id, "researcher", "s0"),
    ]
    assert all("request" not in c for c in body["calls"])
    assert body["calls"][0]["preview"] == "planner output"


async def test_one_call_comes_back_whole(client: TestClient, store: InMemoryRunStore) -> None:
    planner, _ = await _seed(store)

    response = client.get(f"/runs/run-1/calls/{planner.call_id}")

    assert response.status_code == 200
    assert AgentCall.model_validate(response.json()) == planner


async def test_an_unknown_run_or_call_is_404(client: TestClient, store: InMemoryRunStore) -> None:
    await _seed(store)

    assert client.get("/runs/nope/calls").json() == {"detail": "unknown run: nope"}
    assert client.get("/runs/run-1/calls/nope").status_code == 404
