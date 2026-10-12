"""Recording the frozen corpus. The only part of the eval that goes online.

Kept in its own module so the import graph makes the boundary obvious: nothing
that runs during scoring imports this, and this is the only place
``atlas.tools.search`` is reached for real.

Multiple passes per item on purpose. The planner rephrases between runs, so one
pass captures one decomposition's worth of searches; three passes widen the pool
enough that replay has real documents for whatever wording turns up later.
"""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from typing import Any, Protocol
from uuid import uuid4

from atlas.graph import nodes
from atlas.graph.state import Source
from atlas.tools.search import search as live_search
from atlas_eval.corpus import Recorder, corpus_path, save_corpus
from atlas_eval.dataset import EvalItem
from atlas_eval.runner import run_item

logger = logging.getLogger(__name__)


class SearchFn(Protocol):
    """The shape of :func:`atlas.tools.search.search`, defaults included, so a
    substitute for ``nodes.search`` type-checks as one."""

    def __call__(
        self, query: str, max_results: int | None = None, include_content: bool = False
    ) -> Coroutine[Any, Any, list[Source]]: ...


def capturing_search(recorder: Recorder) -> SearchFn:
    """Live search for the agent, with every result kept by ``recorder``.

    Page text is always fetched, because the corpus is recorded once and a later
    reader may need it. It is stripped before the agent sees it unless the agent
    asked: recording must not change the run it is recording.
    """

    async def search(
        query: str, max_results: int | None = None, include_content: bool = False
    ) -> list[Source]:
        sources = await live_search(query, max_results, include_content=True)
        recorder.absorb(query, sources)
        if include_content:
            return sources
        return [source.model_copy(update={"content": None}) for source in sources]

    return search


async def record_corpus(items: list[EvalItem], passes: int = 3) -> None:
    """Run each item against live search, keeping every document retrieved."""
    total_calls = 0

    for position, item in enumerate(items, start=1):
        recorder = Recorder(item.id)

        original = nodes.search
        nodes.search = capturing_search(recorder)
        try:
            for attempt in range(passes):
                try:
                    # Reuse the ordinary run path so the searches recorded are the
                    # ones the agent actually issues, not ones invented here.
                    await _run_online(item)
                except Exception:
                    logger.exception(
                        "recording pass failed",
                        extra={"item_id": item.id, "pass": attempt + 1},
                    )
        finally:
            nodes.search = original

        existing = None
        if corpus_path(item.id).exists():
            from atlas_eval.corpus import load_corpus

            existing = load_corpus(item.id)
        corpus = recorder.finish(existing)
        save_corpus(corpus)
        total_calls += recorder.tavily_calls
        print(
            f"[{position}/{len(items)}] {item.id}: {len(corpus.documents)} documents "
            f"from {recorder.tavily_calls} live searches"
        )

    print(f"\n{total_calls} live Tavily calls in total")


async def _run_online(item: EvalItem) -> None:
    """One full agent run with live search installed.

    Deliberately reuses ``run_item``'s sibling path rather than duplicating the
    graph setup: recording has to exercise the same code that will later be
    replayed, or the pool will not match what the agent asks for.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from atlas.graph.builder import build_state_graph, checkpoint_serde, recursion_limit
    from atlas.runner import initial_state

    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    await graph.ainvoke(
        initial_state(item.query),
        config={
            # One run id per pass: each pass is its own run with its own budget.
            "configurable": {"thread_id": f"record-{item.id}", "run_id": str(uuid4())},
            "recursion_limit": recursion_limit(),
        },
    )


__all__ = ["record_corpus", "run_item"]
