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

from atlas.graph import nodes
from atlas.graph.state import Source
from atlas.tools.search import search as live_search
from atlas_eval.corpus import Recorder, corpus_path, save_corpus
from atlas_eval.dataset import EvalItem
from atlas_eval.runner import run_item

logger = logging.getLogger(__name__)


async def record_corpus(items: list[EvalItem], passes: int = 3) -> None:
    """Run each item against live search, keeping every document retrieved."""
    total_calls = 0

    for position, item in enumerate(items, start=1):
        recorder = Recorder(item.id)

        async def capturing_search(
            query: str,
            max_results: int | None = None,
            _recorder: Recorder = recorder,
        ) -> list[Source]:
            sources = await live_search(query, max_results)
            _recorder.absorb(query, sources)
            return sources

        original = nodes.search
        nodes.search = capturing_search
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

    from atlas.graph.builder import build_state_graph, recursion_limit
    from atlas.runner import initial_state

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    await graph.ainvoke(
        initial_state(item.query),
        config={
            "configurable": {"thread_id": f"record-{item.id}"},
            "recursion_limit": recursion_limit(),
        },
    )


__all__ = ["record_corpus", "run_item"]
