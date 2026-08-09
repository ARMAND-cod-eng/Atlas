"""Node functions for the Atlas research graph.

Each node takes state and returns a *partial* state — only the keys it changed.
LangGraph merges the partial in using each key's reducer.

The error philosophy across this module: a node that fails should degrade the run,
never end it. The planner falls back to a single subtask, a researcher records its
failure as data, and the synthesizer reports what went missing. Only a total
provider outage produces an actual error response.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from atlas.config import get_settings
from atlas.graph.state import (
    AtlasState,
    PlannerOutput,
    ResearcherInput,
    ResearchResult,
    Source,
    Subtask,
)
from atlas.metrics import (
    DEGRADED_BUDGET_EXCEEDED,
    DEGRADED_PLANNER_FALLBACK,
    DEGRADED_RESEARCHER_FAILED,
    DEGRADED_UNGROUNDED_CITATION,
    RunMetrics,
    track_node,
)
from atlas.models import call_model, call_structured
from atlas.tools.search import search

logger = logging.getLogger(__name__)

# Node names, referenced by the builder and by fan_out's return value. Constants
# so a rename cannot silently desynchronise the edge from the node.
PLANNER = "planner"
RESEARCHER = "researcher"
SYNTHESIZER = "synthesizer"

PLANNER_SYSTEM = """You are the planning stage of a research system.

Break the user's question into INDEPENDENT subtasks that can be researched in \
parallel. This is the hard constraint: no subtask may depend on the answer to \
another. If you find yourself writing "based on the previous subtask", merge them \
into one instead.

Rules:
- Produce between 1 and {max_subtasks} subtasks.
- Return FEWER subtasks for simple questions. Do not pad to reach a number; a \
straightforward factual question needs exactly one subtask.
- Each subtask must be a self-contained question that makes sense on its own, \
without reference to the others or to the original phrasing.
- Give each a short rationale explaining what it contributes to the whole answer.
"""

RESEARCHER_SYSTEM = """You are a researcher answering ONE specific question from \
web search results.

Rules:
- Use ONLY the provided sources. If they do not answer the question, say so \
plainly rather than filling the gap from memory.
- Cite the URL inline after each claim it supports.
- Be concise and factual. You are producing input for a synthesis step, not a \
finished answer for a reader.
"""

SYNTHESIZER_SYSTEM = """You are the synthesis stage of a research system. Merge \
the research findings into one coherent answer to the user's original question.

Rules:
- Cite sources inline as URLs. You may ONLY cite URLs that appear in the findings \
given to you. Never write a URL that is not in that list — an invented citation \
is worse than no citation, because it looks checkable and is not.
- If any subtask is reported as FAILED, state clearly in the answer which aspects \
could not be researched. Do not present partial findings as if they were complete.
- Answer the question directly. Do not describe your process or the subtasks \
unless a failure makes that necessary.
"""


def _single_subtask(query: str) -> Subtask:
    """The degraded plan: research the original question as one unit."""
    return Subtask(
        id=str(uuid.uuid4()),
        question=query,
        rationale="Planning failed; researching the original question directly.",
    )


async def _plan(query: str, history: list[Any], metrics: RunMetrics) -> list[Subtask]:
    """Decompose the query, or degrade to a single subtask.

    A planner failure must not end the run: it is one structured-output parse
    ahead of the entire fan-out, and answering the question as a single unit is
    far better than returning an error. The degradation is recorded so it reaches
    the client in metrics rather than only the logs.
    """
    settings = get_settings()
    prompt = [
        SystemMessage(PLANNER_SYSTEM.format(max_subtasks=settings.max_subtasks)),
        # Prior turns give the planner conversational context, so a follow-up like
        # "and what about the second one?" decomposes against what came before.
        *history,
        HumanMessage(query),
    ]
    try:
        output = await call_structured(PLANNER, prompt, PlannerOutput, metrics)
    except Exception as exc:
        logger.warning(
            "planner failed; degrading to a single subtask",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:300]},
        )
        metrics.degraded.append(DEGRADED_PLANNER_FALLBACK)
        return [_single_subtask(query)]

    # Truncate rather than reject: a planner that returns seven good subtasks
    # should not fail the run over a config limit.
    return output.subtasks[: settings.max_subtasks]


async def planner(state: AtlasState) -> dict[str, Any]:
    """Decompose the query into independent, parallelizable subtasks."""
    async with track_node(PLANNER) as metrics:
        subtasks = await _plan(state["query"], list(state["messages"]), metrics)
    logger.info("planned subtasks", extra={"subtask_count": len(subtasks)})
    return {
        "subtasks": subtasks,
        "run_metrics": metrics,
        "messages": [HumanMessage(content=state["query"])],
    }


def fan_out(state: AtlasState) -> list[Send] | str:
    """Conditional edge: one parallel researcher per subtask.

    Returning a list of ``Send`` is what actually creates the parallelism — each
    one schedules an independent invocation of the researcher node, and LangGraph
    runs them concurrently.

    An empty subtask list routes straight to the synthesizer instead. Returning an
    empty list would mean no researcher ever runs and the synthesizer receives an
    empty ``research_results`` with no indication of why — a silent no-op rather
    than a stated outcome.
    """
    subtasks = state["subtasks"]
    if not subtasks:
        logger.warning("no subtasks to research; routing straight to synthesis")
        return SYNTHESIZER
    return [
        Send(RESEARCHER, ResearcherInput(subtask=subtask, query=state["query"]))
        for subtask in subtasks
    ]


async def _search_with_retries(subtask: Subtask, max_attempts: int) -> list[Source]:
    """Search for the subtask, retrying with looser phrasing if nothing comes back.

    The retries are cheap textual variations, not model calls. Asking a model to
    rewrite the query would add an LLM round trip to every branch — multiplied
    across the fan-out — for a marginal recall gain. If the phrasing turns out to
    matter more than expected, this is the seam where a ReAct loop would go.
    """
    attempts = [subtask.question]
    if max_attempts > 1:
        attempts.append(f"{subtask.question} overview")
    if max_attempts > 2:
        # Last resort: the first few keywords, dropping any narrowing qualifiers.
        attempts.append(" ".join(subtask.question.split()[:6]))

    for attempt in attempts[:max_attempts]:
        sources = await search(attempt)
        if sources:
            return sources
    return []


async def _summarize(subtask: Subtask, sources: list[Source], metrics: RunMetrics) -> str:
    """Turn raw sources into findings with citations."""
    rendered = "\n\n".join(
        f"[{i + 1}] {s.title}\nURL: {s.url}\n{s.snippet}" for i, s in enumerate(sources)
    )
    response = await call_model(
        RESEARCHER,
        [
            SystemMessage(RESEARCHER_SYSTEM),
            HumanMessage(f"Question: {subtask.question}\n\nSources:\n{rendered}"),
        ],
        metrics,
    )
    return response.text


async def _research(subtask: Subtask, metrics: RunMetrics) -> ResearchResult:
    """Research one subtask. Never raises.

    This is the most important error boundary in the phase. Researchers run
    concurrently, and an exception escaping here would abort the whole graph run —
    one flaky search would lose four healthy branches' work. A failure becomes a
    ResearchResult carrying the error, which the synthesizer reports.
    """
    settings = get_settings()
    try:
        sources = await _search_with_retries(subtask, settings.max_tool_calls_per_researcher)
        if not sources:
            # Not an error: "nothing found" is a real, reportable outcome.
            return ResearchResult(
                subtask_id=subtask.id,
                findings="No sources were found for this question.",
                sources=[],
            )
        findings = await _summarize(subtask, sources, metrics)
        return ResearchResult(subtask_id=subtask.id, findings=findings, sources=sources)
    except Exception as exc:
        logger.exception("researcher failed", extra={"subtask_id": subtask.id})
        metrics.degraded.append(f"{DEGRADED_RESEARCHER_FAILED}:{subtask.id}")
        return ResearchResult(
            subtask_id=subtask.id,
            findings="",
            sources=[],
            error=f"{type(exc).__name__}: {exc}",
        )


async def researcher(state: ResearcherInput) -> dict[str, Any]:
    """Research a single subtask. Input arrives from a ``Send``, not AtlasState."""
    subtask = state["subtask"]
    async with track_node(RESEARCHER) as metrics:
        result = await _research(subtask, metrics)
    # A one-element list because the reducer on research_results concatenates.
    return {"research_results": [result], "run_metrics": metrics}


def _render_findings(results: list[ResearchResult]) -> str:
    """Format results for the synthesizer, marking failures explicitly."""
    blocks: list[str] = []
    for result in results:
        if result.error is not None:
            blocks.append(f"Subtask {result.subtask_id}: FAILED — {result.error}")
            continue
        urls = "\n".join(f"- {s.url}" for s in result.sources) or "- (none)"
        blocks.append(f"Subtask {result.subtask_id}:\n{result.findings}\n\nSources:\n{urls}")
    return "\n\n---\n\n".join(blocks)


# Trailing punctuation is excluded so "see https://a.example." does not capture
# the sentence's full stop as part of the URL.
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\]]+")


def allowed_urls(results: list[ResearchResult]) -> set[str]:
    """Every URL the synthesizer is permitted to cite."""
    return {source.url for result in results for source in result.sources}


def ungrounded_urls(answer: str, results: list[ResearchResult]) -> set[str]:
    """URLs in the answer that appear in no source.

    Prompting the model not to invent citations is necessary but not sufficient —
    it is an instruction, not a guarantee. This is the check that actually knows.
    """
    permitted = allowed_urls(results)
    cited = {match.rstrip(".,;:") for match in _URL_PATTERN.findall(answer)}
    return cited - permitted


def _check_citations(answer: str, results: list[ResearchResult], metrics: RunMetrics) -> None:
    """Flag any citation the research did not support.

    Recorded rather than raised: an answer with one bad link is still worth
    returning, but the client must be told. A fabricated citation is the worst
    failure mode here precisely because it looks verifiable.
    """
    invented = ungrounded_urls(answer, results)
    if invented:
        logger.warning(
            "answer cites URLs absent from all sources",
            extra={"ungrounded_urls": sorted(invented)},
        )
        metrics.degraded.append(DEGRADED_UNGROUNDED_CITATION)


def _check_budget(state: AtlasState, metrics: RunMetrics) -> None:
    """Warn if the run exceeded its token budget.

    Measurement only in this phase — see the config comment. Enforcement lands
    with the critic loop, which is what makes spend genuinely unbounded. Recording
    it in ``degraded`` means the client sees it without reading server logs.
    """
    settings = get_settings()
    accumulated = state.get("run_metrics") or RunMetrics()
    total = accumulated.total_tokens + metrics.total_tokens
    if total > settings.max_tokens_per_query:
        logger.warning(
            "token budget exceeded",
            extra={"total_tokens": total, "budget": settings.max_tokens_per_query},
        )
        metrics.degraded.append(DEGRADED_BUDGET_EXCEEDED)


async def _synthesize(state: AtlasState, metrics: RunMetrics) -> str:
    """Merge findings into one cited answer."""
    results = state["research_results"]
    if not results:
        # No model call. Answering from priors here would be the worst outcome:
        # a confident answer with no research behind it, indistinguishable from
        # a researched one.
        return (
            "No research was performed for this question, so no answer can be "
            "given from sources. This usually means the planner produced no "
            "subtasks — try rephrasing the question."
        )

    response = await call_model(
        SYNTHESIZER,
        [
            SystemMessage(SYNTHESIZER_SYSTEM),
            *state["messages"],
            HumanMessage(
                f"Original question: {state['query']}\n\n"
                f"Research findings:\n\n{_render_findings(results)}"
            ),
        ],
        metrics,
    )
    return response.text


async def synthesizer(state: AtlasState) -> dict[str, Any]:
    """Produce the final answer from all research results."""
    async with track_node(SYNTHESIZER) as metrics:
        answer = await _synthesize(state, metrics)
        _check_citations(answer, state["research_results"], metrics)
        _check_budget(state, metrics)
    return {
        "answer": answer,
        "run_metrics": metrics,
        "messages": [AIMessage(content=answer)],
    }
