"""The single-call baseline: what Atlas has to beat to justify its machinery.

One search on the question as asked, answered by Atlas's own synthesizer — the
same system prompt, the same output cap, the same citation check, the same
frozen corpus. No planner, no parallel researchers, no critic. Everything the
two systems share is literally shared code, so the comparison isolates what the
full graph adds: decomposition, per-subtask research, and review.

If the graph cannot beat this, the fan-out and the critic are cost, not quality,
and the eval should say so.

In the compliance domain the one search is of the law library, and there is a
second baseline below it: **plain**, a model answering from memory with no
retrieval at all, which is what a team gets by asking a model directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from atlas import models
from atlas.config import get_settings
from atlas.domains import DomainName, using_domain
from atlas.graph import nodes
from atlas.graph.state import ResearchResult, Source, Subtask
from atlas.library.sources import law_sources
from atlas.metrics import RunMetrics
from atlas.runner import initial_state
from atlas_eval.dataset import EvalItem
from atlas_eval.runner import EvalSearch, RunRecord, eval_search

logger = logging.getLogger(__name__)

# Documents the single call reads. Atlas reads up to SEARCH_RESULTS_PER_QUERY (5)
# per subtask across up to MAX_SUBTASKS (5), so one search cannot match its
# breadth — but ten documents is a fair share of evidence for one request, and
# each system's token cost is reported beside its scores, so neither is judged
# on quality alone.
SINGLE_CALL_SOURCES = 10


async def run_single_call(
    item: EvalItem, corpus_root: Path | None = None, domain: DomainName = "general"
) -> RunRecord:
    """Answer one item with one search and one synthesis.

    A missing corpus raises, exactly as :func:`atlas_eval.runner.run_item` does:
    it is a setup error, not a finding. A failure after that is recorded and
    scored as zero.
    """
    web = eval_search(item, domain, corpus_root)
    metrics = RunMetrics()
    started = time.perf_counter()

    try:
        sources = await _one_search(item.query, domain, web)
        # The sources stand in for a researcher's findings, so the synthesizer
        # receives them in the shape a researcher in this domain reads them.
        rendered = (
            nodes.render_law_and_guidance(sources)
            if domain == "compliance"
            else nodes.render_sources(sources)
        )
        evidence = ResearchResult(subtask_id="s0", findings=rendered, sources=sources)
        state = initial_state(item.query)
        # One branch asking the question itself: the shape the graph's prompt
        # takes, so the comparison measures the pipeline and not the prompt.
        state["subtasks"] = [Subtask(id="s0", question=item.query, rationale="single call")]
        state["research_results"] = [evidence]
        with using_domain(domain):
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
            search_calls=list(web.calls),
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
        search_calls=list(web.calls),
        duration_s=time.perf_counter() - started,
    )


async def _one_search(query: str, domain: DomainName, web: EvalSearch) -> list[Source]:
    """The single call's evidence: the law library in the compliance domain (the
    web is off there), the frozen corpus otherwise."""
    if domain == "compliance":
        return law_sources(query, SINGLE_CALL_SOURCES)
    return await web.search(query, SINGLE_CALL_SOURCES)


# The compliance synthesizer's rules, less the ones about findings: the same
# citation form, the same separation of law from guidance, the same economy.
PLAIN_COMPLIANCE = """You are a compliance analyst answering a question about EU \
law: the AI Act (Regulation (EU) 2024/1689) or the GDPR \
(Regulation (EU) 2016/679). Answer from what you know; you \
have no documents to read.

Rules:
- Cite every statement about the law with its provision label in square \
brackets, exactly in this form: [AI Act Art. 26(6)], [AI Act Annex III(4)(a)], \
[GDPR Art. 33(1)], [GDPR Recital 71].
- Keep what the law requires (cite the provision) apart from guidance or \
interpretation (say whose).
- If the question rests on a false premise, correct it first.
- If the answer turns on facts the question does not give (the user's role, what \
a system is used for), say what it turns on rather than assuming.
- Answer directly and economically. Every sentence that does not add information \
costs the reader time.
"""


async def run_plain(item: EvalItem) -> RunRecord:
    """Answer one item from the model's memory: one call, nothing retrieved.

    The same output cap as the synthesizer, and a prompt asking for the same
    citation form, so the citation metrics read its answers on equal terms.
    """
    metrics = RunMetrics()
    started = time.perf_counter()
    try:
        response = await models.call_model(
            nodes.SYNTHESIZER,
            [SystemMessage(PLAIN_COMPLIANCE), HumanMessage(item.query)],
            metrics,
            max_tokens=get_settings().synthesis_max_tokens,
        )
    except Exception as exc:
        logger.exception("plain run failed", extra={"item_id": item.id})
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer="",
            variant="plain",
            metrics=metrics,
            duration_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return RunRecord(
        item_id=item.id,
        query=item.query,
        answer=response.text,
        variant="plain",
        metrics=metrics,
        duration_s=time.perf_counter() - started,
    )
