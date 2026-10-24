"""Every agent's exact prompt and output, recorded, never reconstructed.

The inspector shows what each agent was told and what it said. Rebuilding a
prompt from state and today's templates would show what the prompt *would*
have been, and differ silently the day a template changes; so the model layer
records the messages it actually sent, at the one place every call passes
through. Searches are recorded the same way: the query, and what came back,
which is the evidence behind "the cited provision was retrieved".

The full record goes to the run store. The live event stream gets a summary,
because replay re-sends the whole stream to every reconnecting client.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from atlas import eventlog, models
from atlas.events import AGENT_CALL, NODE_STARTED
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import CriticVerdict, PlannerOutput, Source, Subtask
from atlas.prompts import GENERAL_PLANNER
from atlas.runner import execute_run
from atlas.store import InMemoryRunStore
from atlas.trace import PREVIEW_CHARS

PLAN = PlannerOutput(
    subtasks=[
        Subtask(id="s0", question="What is RAG?", rationale="r"),
        Subtask(id="s1", question="What is BM25?", rationale="r"),
    ]
)


def _envelope(parsed: Any) -> dict[str, Any]:
    raw = AIMessage(content="{}", response_metadata={"model_name": "claude-sonnet-5"})
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


@pytest.fixture
def real_model_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake providers *under* atlas.models, so call_model itself runs."""

    def llm(temperature: float | None = None, max_tokens: int | None = None) -> Any:
        return FakeListChatModel(responses=["x" * (PREVIEW_CHARS + 50)])

    def structured(schema: type[Any], temperature: float | None = None) -> Any:
        parsed = PLAN if schema is PlannerOutput else CriticVerdict(overall="accept")
        return RunnableLambda(lambda _messages: _envelope(parsed))

    async def web(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title=query, snippet="s")]

    monkeypatch.setattr(models, "get_llm", llm)
    monkeypatch.setattr(models, "get_structured_llm", structured)
    monkeypatch.setattr(nodes, "search", web)
    # The node functions call these by name; point them back at the real ones.
    monkeypatch.setattr(nodes, "call_model", models.call_model)
    monkeypatch.setattr(nodes, "call_structured", models.call_structured)


async def _run(store: InMemoryRunStore | None) -> list[Any]:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    await execute_run("run-1", "first question", "t1", graph, store=store)
    return await eventlog.replay("run-1", 0)


async def test_every_model_call_is_recorded_with_what_was_sent(real_model_layer: None) -> None:
    store = InMemoryRunStore()
    await _run(store)

    calls = [c for c in await store.calls("run-1") if c.kind == "model"]

    assert sorted((c.node, c.subtask_id) for c in calls) == [
        (nodes.CRITIC, None),
        (nodes.PLANNER, None),
        (nodes.RESEARCHER, "s0"),
        (nodes.RESEARCHER, "s1"),
        (nodes.SYNTHESIZER, None),
    ]
    [planner] = [c for c in calls if c.node == nodes.PLANNER]
    assert planner.request[0] == {
        "role": "system",
        "content": GENERAL_PLANNER.format(max_subtasks=5),
    }
    assert planner.request[-1] == {"role": "human", "content": "first question"}
    assert planner.response == PLAN.model_dump()


async def test_every_search_is_recorded_with_what_came_back(real_model_layer: None) -> None:
    store = InMemoryRunStore()
    await _run(store)

    searches = [c for c in await store.calls("run-1") if c.kind == "tool"]

    assert sorted((c.subtask_id, c.name, c.request, c.response) for c in searches) == [
        (
            "s0",
            "web",
            {"query": "What is RAG?"},
            [{"url": "https://ex.example/What is RAG?", "title": "What is RAG?"}],
        ),
        (
            "s1",
            "web",
            {"query": "What is BM25?"},
            [{"url": "https://ex.example/What is BM25?", "title": "What is BM25?"}],
        ),
    ]


async def test_a_call_belongs_to_the_span_that_made_it(real_model_layer: None) -> None:
    """The inspector lists a branch's attempts by span; a call must name its own."""
    store = InMemoryRunStore()
    events = await _run(store)

    spans = {
        (e.node, e.payload["subtask_id"]): e.payload["span_id"]
        for e in events
        if e.type == NODE_STARTED
    }
    calls = await store.calls("run-1")

    assert {(c.node, c.subtask_id): c.span_id for c in calls} == {
        key: span for key, span in spans.items() if key in {(c.node, c.subtask_id) for c in calls}
    }
    assert len(set(spans.values())) == len(spans)


async def test_the_live_stream_carries_a_summary_not_the_prompt(real_model_layer: None) -> None:
    store = InMemoryRunStore()
    events = await _run(store)

    summaries = [e.payload for e in events if e.type == AGENT_CALL]
    full = {c.call_id: c for c in await store.calls("run-1")}

    assert sorted(summaries[0]) == [
        "call_id",
        "duration_s",
        "error",
        "input_tokens",
        "kind",
        "name",
        "output_tokens",
        "preview",
        "span_id",
        "subtask_id",
    ]
    assert {s["call_id"] for s in summaries} == set(full)
    synthesis = next(s for s in summaries if full[s["call_id"]].node == nodes.SYNTHESIZER)
    assert synthesis["preview"] == "x" * PREVIEW_CHARS + "..."


async def test_a_call_can_be_read_back_by_id(real_model_layer: None) -> None:
    store = InMemoryRunStore()
    await _run(store)
    [first, *_] = await store.calls("run-1")

    assert await store.call("run-1", first.call_id) == first
    assert await store.call("run-1", "nope") is None


async def test_without_a_store_the_stream_still_has_its_summaries(
    real_model_layer: None,
) -> None:
    events = await _run(None)

    assert len([e for e in events if e.type == AGENT_CALL]) == 7
