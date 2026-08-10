"""Runs dataset items through the real graph, in-process, against frozen search.

In-process rather than through ``POST /query``: the eval scores the *agent*, and
the queue, SSE relay and HTTP layer are Phase 3's concerns with 161 tests of their
own. Driving the endpoint would mean booting four processes to exercise code this
suite does not score, and paying ~1.0s of queue pickup fifty times over. With
``MemorySaver`` and a frozen corpus the suite needs no services at all, which is
what makes the CI job simple enough to trust.

The graph itself is untouched — same nodes, same prompts, same critic loop. Only
``search`` is substituted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, recursion_limit
from atlas.graph.state import ResearchResult, Source, Subtask
from atlas.metrics import RunMetrics
from atlas.result import build_response, validate_list, validate_metrics
from atlas.runner import initial_state
from atlas_eval.corpus import FrozenSearch, load_corpus
from atlas_eval.dataset import EvalItem

logger = logging.getLogger(__name__)

# The frozen search for the item currently executing. A ContextVar rather than a
# module global because items run concurrently: asyncio tasks inherit the context
# they were created in, so each run sees its own corpus. A plain global would let
# five parallel items answer each other's searches — the kind of bug that does not
# crash, it just quietly scores the wrong thing.
_ACTIVE: ContextVar[FrozenSearch | None] = ContextVar("atlas_eval_search", default=None)

_installed = False


async def _dispatch(query: str, max_results: int | None = None) -> list[Source]:
    """Stands in for ``atlas.graph.nodes.search`` for the whole eval process."""
    active = _ACTIVE.get()
    if active is None:
        raise RuntimeError(
            "search was called outside an eval run — the frozen corpus is not "
            "installed, and the eval must never reach the network"
        )
    return await active.search(query, max_results)


def install_frozen_search() -> None:
    """Replace the graph's search with the corpus dispatcher, once."""
    global _installed
    if not _installed:
        nodes.search = _dispatch
        _installed = True


class RunRecord(BaseModel):
    """Everything one scored run produced.

    Persisted so scoring, calibration and the gate all read the same artefact
    rather than each re-running the agent.
    """

    item_id: str
    query: str
    answer: str
    subtasks: list[Subtask] = Field(default_factory=list)
    research_results: list[ResearchResult] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    iterations: int = 0
    # Every search string the agent issued, in order. Tool-call validity reads
    # this to find redundant duplicate searches.
    search_calls: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    # Set when the run raised. A failed run is scored as zero rather than dropped
    # — dropping it would quietly change the denominator.
    error: str | None = None

    @property
    def retrieved_context(self) -> str:
        """Every retrieved snippet, for the grounding scorer to verify against."""
        blocks = []
        for result in self.research_results:
            for source in result.sources:
                blocks.append(f"[{source.url}] {source.title}\n{source.snippet}")
        return "\n\n".join(blocks)


async def run_item(item: EvalItem, corpus_root: Path | None = None) -> RunRecord:
    """Execute one dataset item and capture what it produced."""
    install_frozen_search()
    corpus = load_corpus(item.id) if corpus_root is None else load_corpus(item.id, corpus_root)
    frozen = FrozenSearch(corpus)
    token = _ACTIVE.set(frozen)

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {
        "configurable": {"thread_id": f"eval-{item.id}"},
        "recursion_limit": recursion_limit(),
    }

    started = time.perf_counter()
    try:
        final = await graph.ainvoke(initial_state(item.query), config=config)
        elapsed = time.perf_counter() - started
        response = build_response(dict(final), thread_id=f"eval-{item.id}")
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer=response.answer,
            subtasks=response.subtasks,
            research_results=validate_list(final.get("research_results"), ResearchResult),
            sources=response.sources,
            metrics=validate_metrics(final.get("run_metrics")),
            iterations=response.iterations,
            search_calls=list(frozen.calls),
            duration_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.exception("eval run failed", extra={"item_id": item.id})
        # Recorded, not raised. One item blowing up must not abort a 50-item run,
        # and it must not silently shrink the denominator either.
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer="",
            search_calls=list(frozen.calls),
            duration_s=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _ACTIVE.reset(token)


async def run_items(items: list[EvalItem], concurrency: int = 5) -> list[RunRecord]:
    """Run items concurrently, preserving dataset order in the output.

    Concurrency is bounded because each item is itself a five-branch fan-out; the
    real limit is the provider's rate limit, not ours.
    """
    limiter = asyncio.Semaphore(concurrency)

    async def guarded(item: EvalItem) -> RunRecord:
        async with limiter:
            return await run_item(item)

    return list(await asyncio.gather(*(guarded(item) for item in items)))
