"""The single-call baseline: what Atlas has to beat to justify its machinery.

One search on the question as asked, answered by Atlas's own synthesizer — the
same system prompt, the same output cap, the same citation check, the same
frozen corpus. No planner, no parallel researchers, no critic. Everything the
two systems share is literally shared code, so the comparison isolates what the
full graph adds: decomposition, per-subtask research, and review.

If the graph cannot beat this, the fan-out and the critic are cost, not quality,
and the eval should say so.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from atlas.graph import nodes
from atlas.graph.state import ResearchResult
from atlas.metrics import RunMetrics
from atlas.runner import initial_state
from atlas_eval.corpus import FrozenSearch, load_corpus
from atlas_eval.dataset import EvalItem
from atlas_eval.runner import RunRecord

logger = logging.getLogger(__name__)

# Documents the single call reads. Atlas reads up to SEARCH_RESULTS_PER_QUERY (5)
# per subtask across up to MAX_SUBTASKS (5), so one search cannot match its
# breadth — but ten documents is a fair share of evidence for one request, and
# each system's token cost is reported beside its scores, so neither is judged
# on quality alone.
SINGLE_CALL_SOURCES = 10


async def run_single_call(item: EvalItem, corpus_root: Path | None = None) -> RunRecord:
    """Answer one item with one search and one synthesis.

    A missing corpus raises, exactly as :func:`atlas_eval.runner.run_item` does:
    it is a setup error, not a finding. A failure after that is recorded and
    scored as zero.
    """
    corpus = load_corpus(item.id) if corpus_root is None else load_corpus(item.id, corpus_root)
    frozen = FrozenSearch(corpus)
    metrics = RunMetrics()
    started = time.perf_counter()

    try:
        sources = await frozen.search(item.query, SINGLE_CALL_SOURCES)
        # The snippets stand in for a researcher's findings, so the synthesizer
        # receives them in exactly the shape it receives the graph's research.
        evidence = ResearchResult(
            subtask_id="s0", findings=nodes.render_sources(sources), sources=sources
        )
        state = initial_state(item.query)
        state["research_results"] = [evidence]
        answer = await nodes._synthesize(state, metrics)
        nodes._check_citations(answer, [evidence], metrics)
    except Exception as exc:
        logger.exception("single-call run failed", extra={"item_id": item.id})
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer="",
            variant="single-call",
            metrics=metrics,
            search_calls=list(frozen.calls),
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RunRecord(
        item_id=item.id,
        query=item.query,
        answer=answer,
        variant="single-call",
        research_results=[evidence],
        sources=sources,
        metrics=metrics,
        search_calls=list(frozen.calls),
        duration_s=time.perf_counter() - started,
    )
