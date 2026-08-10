"""Turning final graph state into the API's response shape.

Extracted from ``atlas.api.routes`` in Phase 3 because the worker now needs it
too, and the worker cannot import the API (the API imports the worker's client).
Keeping one definition means a streamed run and a synchronous ``POST /query``
cannot drift into reporting different things about the same run.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from atlas.graph.state import ResearchResult, Source, Subtask
from atlas.metrics import RunMetrics


class QueryResponse(BaseModel):
    """Answer, the thread it belongs to, and how it was produced.

    Additive across every phase — anything reading ``.answer`` still works. The
    rest exists because a research platform's citations are the product, because
    a fan-out you cannot see is a fan-out you cannot debug, and because
    ``metrics.degraded`` makes a partial run visible from the client rather than
    only in server logs.
    """

    answer: str
    thread_id: str
    subtasks: list[Subtask] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    # How many critic passes ran. With metrics.critic_rounds this makes the
    # quality gate visible: 1 with an "accept" round means it reviewed and
    # approved, which is very different from it never having run.
    iterations: int = 0


def validate_list[ModelT: BaseModel](raw: object, model: type[ModelT]) -> list[ModelT]:
    """Coerce a state value into typed models.

    Graph state round-trips through the checkpointer, which may hand back plain
    dicts rather than model instances. ``model_validate`` accepts either, so this
    is correct on a fresh run and on a resumed thread alike.
    """
    if not isinstance(raw, list):
        return []
    return [model.model_validate(item) for item in raw]


def validate_metrics(raw: object) -> RunMetrics:
    """Coerce state's run_metrics, tolerating an absent or resumed value."""
    if raw is None:
        return RunMetrics()
    return RunMetrics.model_validate(raw)


def dedupe_sources(results: list[ResearchResult]) -> list[Source]:
    """Flatten every researcher's sources, first occurrence of each URL winning.

    Parallel researchers frequently land on the same page, and a citation list
    that repeats one URL five times is noise.
    """
    seen: set[str] = set()
    sources: list[Source] = []
    for result in results:
        for source in result.sources:
            if source.url in seen:
                continue
            seen.add(source.url)
            sources.append(source)
    return sources


def build_response(state: dict[str, Any], thread_id: str) -> QueryResponse:
    """Build the response from a run's final state."""
    subtasks = validate_list(state.get("subtasks"), Subtask)
    results = validate_list(state.get("research_results"), ResearchResult)
    raw_iterations = state.get("iteration")
    return QueryResponse(
        answer=str(state.get("answer") or ""),
        thread_id=thread_id,
        subtasks=subtasks,
        sources=dedupe_sources(results),
        metrics=validate_metrics(state.get("run_metrics")),
        iterations=raw_iterations if isinstance(raw_iterations, int) else 0,
    )


def response_payload(response: QueryResponse) -> dict[str, Any]:
    """JSON-safe dict, for the arq job result and the stored run result.

    ``mode="json"`` rather than a plain dump: the result crosses a process
    boundary as JSON, so datetimes have to be strings by the time it gets there.
    """
    return response.model_dump(mode="json")
