"""A question chooses its domain: the law, with every citation checked, or general research.

Chosen when a conversation starts, and kept for the whole of it: a follow-up or
a revision runs in its conversation's domain, and cannot switch it, because the
answer it builds on was researched under the other one's rules. The domain
reaches the job, the record, the stream and every node.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from atlas import eventlog
from atlas.api.main import create_app
from atlas.events import RUN_STARTED
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import CriticVerdict, Source
from atlas.library.support import SupportJudgements
from atlas.metrics import RunMetrics
from atlas.runner import execute_revision, execute_run
from atlas.store import InMemoryRunStore
from atlas.worker import RUN_RESEARCH, RUN_REVISION
from tests.conftest import FakeArq, FakeRedis, make_call_model, make_plan


@pytest.fixture
def planner_prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The planner's system prompt, each time it is asked to plan."""
    seen: list[str] = []

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    async def fake_structured(
        node: str,
        messages: list[BaseMessage],
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        if schema is CriticVerdict:
            return CriticVerdict(overall="accept")
        if schema is SupportJudgements:
            return SupportJudgements()
        seen.append(str(messages[0].content))
        return make_plan(2)

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the answer"))
    monkeypatch.setattr(nodes, "call_structured", fake_structured)
    return seen


class Harness:
    def __init__(self) -> None:
        graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
        self.store = InMemoryRunStore()

        async def research(
            run_id: str,
            query: str,
            thread_id: str,
            parent_run_id: str | None = None,
            *,
            domain: str | None = None,
        ) -> dict[str, Any]:
            return await execute_run(
                run_id,
                query,
                thread_id,
                graph,
                parent_run_id=parent_run_id,
                store=self.store,
                domain=domain,
            )

        async def revise(
            run_id: str,
            thread_id: str,
            parent_run_id: str,
            subtask_id: str,
            question: str,
            *,
            domain: str | None = None,
        ) -> dict[str, Any]:
            return await execute_revision(
                run_id,
                thread_id,
                graph,
                parent_run_id=parent_run_id,
                subtask_id=subtask_id,
                question=question,
                store=self.store,
                domain=domain,
            )

        self.arq = FakeArq(research, {RUN_REVISION: revise})
        app = create_app()
        app.state.arq = self.arq
        app.state.store = self.store
        self.client = TestClient(app)

    def ask(self, **body: object) -> dict[str, Any]:
        response = self.client.post("/runs", json=body)
        assert response.status_code == 202, response.text
        return dict(response.json())

    def domains_queued(self) -> list[tuple[object, object]]:
        return [
            (call[0], kwargs.get("domain"))
            for call, kwargs in zip(self.arq.calls, self.arq.kwargs, strict=True)
        ]


COMPLIANCE_PLANNER = "You are the planning stage of a compliance research"


async def test_a_new_conversation_runs_in_the_domain_it_asked_for(
    planner_prompts: list[str], fake_redis: FakeRedis
) -> None:
    harness = Harness()

    run = harness.ask(query="What must a deployer do?", domain="compliance")

    assert harness.domains_queued() == [(RUN_RESEARCH, "compliance")]
    record = await harness.store.get(run["run_id"])
    assert record is not None and record.domain == "compliance"
    [started] = [e for e in await eventlog.replay(run["run_id"], 0) if e.type == RUN_STARTED]
    assert started.payload["domain"] == "compliance"
    # And the nodes ran under it: the planner was given the compliance prompt.
    assert [prompt.startswith(COMPLIANCE_PLANNER) for prompt in planner_prompts] == [True]


async def test_without_a_domain_the_configured_one_applies(
    planner_prompts: list[str], fake_redis: FakeRedis
) -> None:
    """The control: tests run with ATLAS_DOMAIN=general (tests/conftest.py)."""
    harness = Harness()

    run = harness.ask(query="What is RAG?")

    assert harness.domains_queued() == [(RUN_RESEARCH, "general")]
    record = await harness.store.get(run["run_id"])
    assert record is not None and record.domain == "general"
    assert [prompt.startswith(COMPLIANCE_PLANNER) for prompt in planner_prompts] == [False]


def test_a_follow_up_keeps_its_conversations_domain(
    planner_prompts: list[str], fake_redis: FakeRedis
) -> None:
    harness = Harness()
    first = harness.ask(query="What must a deployer do?", domain="compliance")

    harness.ask(query="And from when?", thread_id=first["thread_id"])
    harness.ask(query="And a provider?", parent_run_id=first["run_id"])

    assert [domain for _, domain in harness.domains_queued()] == ["compliance"] * 3
    assert [p.startswith(COMPLIANCE_PLANNER) for p in planner_prompts] == [True] * 3


def test_a_follow_up_cannot_switch_domains(
    planner_prompts: list[str], fake_redis: FakeRedis
) -> None:
    harness = Harness()
    first = harness.ask(query="What must a deployer do?", domain="compliance")

    response = harness.client.post(
        "/runs", json={"query": "q", "thread_id": first["thread_id"], "domain": "general"}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            f"thread {first['thread_id']} researches in the compliance domain; "
            "start a new conversation to ask in the general one"
        )
    }
    assert len(harness.arq.calls) == 1


async def test_a_revision_keeps_its_parents_domain(
    planner_prompts: list[str], fake_redis: FakeRedis
) -> None:
    harness = Harness()
    first = harness.ask(query="What must a deployer do?", domain="compliance")

    response = harness.client.post(
        f"/runs/{first['run_id']}/revisions", json={"subtask_id": "s1", "question": "a new one"}
    )

    assert response.status_code == 202, response.text
    assert harness.domains_queued()[-1] == (RUN_REVISION, "compliance")
    record = await harness.store.get(response.json()["run_id"])
    assert record is not None and record.domain == "compliance"
