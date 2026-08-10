"""Capture event-stream fixtures for the frontend reducer tests.

    uv run python scripts/capture_fixtures.py            # deterministic scenarios
    uv run python scripts/capture_fixtures.py <run_id>   # dump a real run from Redis

Why this exists: the reducer must be tested against event sequences the backend
actually produces, not against hand-written ones. Hand-written fixtures test the
reducer against my beliefs about the backend; recorded ones test it against the
backend.

Two sources, and the difference is worth being explicit about.

**Live runs** (``<run_id>`` form) are dumped from the Redis event log after a real
query. They are the ground truth for the happy path and for a real targeted retry.

**Deterministic scenarios** (no argument) run the *same* real graph, runner and
nodes with the model and search faked to fail in specific ways. Every event is
still produced by production code — only the inputs are controlled. This is the
only way to get the paths that matter most and that a successful run can never
produce: a branch that fails while its siblings succeed, a critic that exhausts
its iterations, and a run that dies outright. A reducer tested only on happy
paths is a reducer tested on the runs that did not need it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from atlas import eventlog
from atlas.errors import ProviderError, SearchError
from atlas.events import AgentEvent
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import CriticVerdict, PlannerOutput, Source, Subtask, SubtaskVerdict
from atlas.metrics import RunMetrics
from atlas.runner import execute_run

# Spelled out so the fakes match the real signatures they stand in for, rather
# than being annotated Any and silently drifting when a signature changes.
SearchFn = Callable[..., Awaitable[list[Source]]]
ModelFn = Callable[..., Awaitable[AIMessage]]
StructuredFn = Callable[..., Awaitable[BaseModel]]

FIXTURES = Path(__file__).resolve().parents[2] / "frontend" / "tests" / "fixtures"

QUESTIONS = [
    "How accurate is RAG for enterprise document QA?",
    "How accurate are long-context LLMs on the same task?",
    "What are the cost characteristics of each?",
    "What are the latency characteristics of each?",
    "When is each the better choice?",
]


def plan(count: int) -> PlannerOutput:
    return PlannerOutput(
        subtasks=[Subtask(id=f"s{i}", question=QUESTIONS[i], rationale="r") for i in range(count)]
    )


def fake_model(answer: str, fail_in: str | None = None) -> ModelFn:
    async def call(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if fail_in is not None and node == fail_in:
            raise ProviderError(provider="anthropic", message="everything is down", node=node)
        return AIMessage(content=answer if node == nodes.SYNTHESIZER else "findings")

    return call


def fake_structured(count: int, verdicts: list[CriticVerdict]) -> StructuredFn:
    remaining = list(verdicts)

    async def call(
        node: str,
        messages: object,
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        if schema is CriticVerdict:
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return plan(count)

    return call


def fake_search(broken: str | None = None) -> SearchFn:
    async def call(query: str, max_results: int | None = None) -> list[Source]:
        if broken is not None and query == broken:
            raise SearchError("this source is unreachable", query=query)
        slug = query.lower().split()[0].strip("?")
        return [
            Source(url=f"https://example.com/{slug}-{i}", title=f"{slug} result {i}", snippet="s")
            for i in range(2)
        ]

    return call


def revise(subtask_id: str) -> CriticVerdict:
    return CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(
                subtask_id=subtask_id,
                verdict="revise",
                reason="the claims are not supported by the listed sources",
                guidance="Search for primary benchmark data rather than vendor posts.",
            )
        ],
    )


ACCEPT = CriticVerdict(overall="accept")


async def capture(
    name: str,
    *,
    count: int,
    verdicts: list[CriticVerdict],
    broken: str | None = None,
    fail_in: str | None = None,
) -> None:
    """Run the real graph with controlled inputs and record every event."""
    collected: list[AgentEvent] = []

    async def record(event: AgentEvent) -> None:
        collected.append(event)

    async def no_store(run_id: str, result: dict[str, Any]) -> None:
        """The fixtures are the event stream; the stored result is not needed."""

    # The nodes look these names up at call time, so rebinding the module
    # attribute is what swaps the implementation.
    saved: dict[str, Any] = {
        "append": eventlog.append,
        "store_result": eventlog.store_result,
        "search": nodes.search,
        "call_model": nodes.call_model,
        "call_structured": nodes.call_structured,
    }
    eventlog.append = record
    eventlog.store_result = no_store
    nodes.search = fake_search(broken)
    nodes.call_model = fake_model("## Answer\n\nSynthesised from the findings.", fail_in)
    nodes.call_structured = fake_structured(count, verdicts)

    try:
        graph = build_state_graph().compile(checkpointer=MemorySaver())
        try:
            await execute_run(name, "Compare RAG with long-context LLMs.", "thread", graph)
        except Exception as exc:
            print(f"  ({name} ended with {type(exc).__name__}, as intended)")
    finally:
        eventlog.append = saved["append"]
        eventlog.store_result = saved["store_result"]
        nodes.search = saved["search"]
        nodes.call_model = saved["call_model"]
        nodes.call_structured = saved["call_structured"]

    write(name, collected)


def write(name: str, events: list[AgentEvent]) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{name}.json"
    payload = [json.loads(event.model_dump_json()) for event in events]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    kinds = ", ".join(sorted({event.type for event in events}))
    print(f"{path.name:<24} {len(events):>3} events  [{kinds}]")


async def dump_live(run_id: str) -> None:
    """Dump a real run's events straight from the Redis log."""
    events = await eventlog.replay(run_id, 0)
    if not events:
        raise SystemExit(f"no events for run {run_id} — expired, or wrong id?")
    write("live-run", events)
    await eventlog.close_client()


async def main() -> None:
    if len(sys.argv) > 1:
        await dump_live(sys.argv[1])
        return

    # The happy path, deterministic. The live capture is the ground truth; this
    # one keeps the reducer tests runnable without Redis or a provider.
    await capture("accepted", count=5, verdicts=[ACCEPT])

    # One branch dies while four succeed. Phase 1 turns a failed branch into a
    # ResearchResult rather than raising so the siblings survive, and Phase 3
    # reports node_failed for it anyway — so the UI must show one lane failed,
    # four done, and a run that still completes with an answer. A green run that
    # quietly answered from four fifths of the research is the failure this
    # fixture exists to make impossible.
    await capture("partial-failure", count=5, verdicts=[ACCEPT], broken=QUESTIONS[1])

    # A targeted retry: the critic faults one subtask, then accepts.
    await capture("retry", count=5, verdicts=[revise("s2"), ACCEPT])

    # The critic never accepts. Terminates at MAX_CRITIC_ITERATIONS with the
    # answer it has, degraded rather than failed.
    await capture("exhaustion", count=3, verdicts=[revise("s1"), revise("s1"), revise("s1")])

    # The run dies outright. The stream must still terminate.
    await capture("run-failed", count=3, verdicts=[ACCEPT], fail_in=nodes.SYNTHESIZER)


if __name__ == "__main__":
    asyncio.run(main())
