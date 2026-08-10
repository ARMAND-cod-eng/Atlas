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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Send

from atlas.budget import is_exceeded
from atlas.config import get_settings
from atlas.events import (
    CRITIC_VERDICT,
    FAN_OUT,
    ITERATION_STARTED,
    emit,
    node_span,
)
from atlas.graph.state import (
    AtlasState,
    CriticVerdict,
    PlannerOutput,
    ResearcherInput,
    ResearchResult,
    Source,
    Subtask,
)
from atlas.metrics import (
    DEGRADED_BUDGET_CHECK_UNAVAILABLE,
    DEGRADED_BUDGET_EXCEEDED,
    DEGRADED_CRITIC_MAX_ITERATIONS,
    DEGRADED_CRITIC_NOT_ACTIONABLE,
    DEGRADED_CRITIC_UNAVAILABLE,
    DEGRADED_PLANNER_FALLBACK,
    DEGRADED_RESEARCHER_FAILED,
    DEGRADED_SYNTHESIS_TRUNCATED,
    DEGRADED_UNGROUNDED_CITATION,
    CriticRound,
    RunMetrics,
)
from atlas.models import call_model, call_structured
from atlas.tools.search import search

logger = logging.getLogger(__name__)

# Node names, referenced by the builder and by fan_out's return value. Constants
# so a rename cannot silently desynchronise the edge from the node.
PLANNER = "planner"
RESEARCHER = "researcher"
SYNTHESIZER = "synthesizer"
CRITIC = "critic"

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
- Be economical. Cover the substance and stop. Do not restate the question, \
recap what you are about to say, or add a summary of your own summary. Length is \
the single biggest driver of how long the user waits, so every sentence that does \
not add information costs them time for nothing.
"""

CRITIC_SYSTEM = """You are reviewing the research behind a drafted answer, before \
it reaches the user.

You are given each subtask's question, the findings written for it, and THE \
SOURCES THAT WERE ACTUALLY RETRIEVED. Judge the findings against those sources — \
not against your own knowledge of the topic. If a claim is correct but not \
supported by any listed source, that is a grounding failure and you should say so.

For each subtask, check three things:
1. Did the findings actually answer the subtask's question?
2. Is every substantive claim traceable to one of the listed sources?
3. Are there obvious gaps — an aspect of the question the sources cover but the \
findings ignore?

ACCEPTING SOUND WORK IS THE CORRECT OUTCOME. You are a quality gate, not a proof \
of your own usefulness. Every rejection costs the user another search, another \
model call, and roughly forty more seconds of waiting. Reject only when a \
specific, nameable problem would leave the final answer wrong, unsupported, or \
materially incomplete. Do not reject for style, for phrasing you would have \
chosen differently, or to appear thorough. If the research is adequate, return \
"accept" and stop.

When you do reject a subtask, "guidance" must be a concrete instruction the \
researcher can act on — say what to look for, not that the answer was weak.

If the research is sound but the final write-up misrepresents it, omits a \
finding, or cites something absent from the sources, set synthesis_feedback \
instead of faulting subtasks. That fixes the answer without re-running searches.
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
    async with node_span(PLANNER) as span:
        subtasks = await _plan(state["query"], list(state["messages"]), span.metrics)
        metrics = span.metrics
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
        emit(FAN_OUT, PLANNER, subtask_ids=[], count=0)
        return SYNTHESIZER
    # Emitted here rather than in the planner because this is where the width of
    # the fan-out is actually decided — a client can size its UI before the first
    # branch reports anything.
    emit(
        FAN_OUT,
        PLANNER,
        subtask_ids=[subtask.id for subtask in subtasks],
        count=len(subtasks),
        questions={subtask.id: subtask.question for subtask in subtasks},
    )
    return [
        Send(
            RESEARCHER,
            ResearcherInput(subtask=subtask, query=state["query"], guidance=None),
        )
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


async def _summarize(
    subtask: Subtask,
    sources: list[Source],
    metrics: RunMetrics,
    guidance: str | None = None,
) -> str:
    """Turn raw sources into findings with citations."""
    rendered = "\n\n".join(
        f"[{i + 1}] {s.title}\nURL: {s.url}\n{s.snippet}" for i, s in enumerate(sources)
    )
    prompt = f"Question: {subtask.question}\n\nSources:\n{rendered}"
    if guidance is not None:
        # A retry. Telling the researcher what was wrong last time is the entire
        # value of the loop — without it the second attempt repeats the first.
        prompt = (
            f"{prompt}\n\nA reviewer rejected your previous answer to this "
            f"question. Address this specifically:\n{guidance}"
        )
    response = await call_model(
        RESEARCHER,
        [SystemMessage(RESEARCHER_SYSTEM), HumanMessage(prompt)],
        metrics,
    )
    return response.text


async def _research(
    subtask: Subtask, metrics: RunMetrics, guidance: str | None = None
) -> ResearchResult:
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
        findings = await _summarize(subtask, sources, metrics, guidance)
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
    # subtask_id on the span is what makes five interleaved branches attributable.
    async with node_span(RESEARCHER, subtask_id=subtask.id) as span:
        result = await _research(subtask, span.metrics, state.get("guidance"))
        if result.error is not None:
            # A failed branch does not raise — Phase 1 turns it into a result so
            # the other four survive — but it has still failed, and a watching
            # client should see node_failed rather than node_completed.
            span.failed(result.error)
        metrics = span.metrics
    # A one-element list because the reducer on research_results upserts by id.
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

    settings = get_settings()
    prompt = (
        f"Original question: {state['query']}\n\nResearch findings:\n\n{_render_findings(results)}"
    )
    # On a synthesis-only revision the critic faulted the write-up, not the
    # research. Passing its feedback is the only thing that makes the second
    # attempt different from the first.
    feedback = _latest_synthesis_feedback(state)
    if feedback is not None:
        prompt = f"{prompt}\n\nA reviewer rejected your previous answer:\n{feedback}"

    response = await call_model(
        SYNTHESIZER,
        [
            SystemMessage(SYNTHESIZER_SYSTEM),
            *state["messages"],
            HumanMessage(prompt),
        ],
        metrics,
        max_tokens=settings.synthesis_max_tokens,
    )
    if _hit_the_output_cap(response):
        logger.warning(
            "synthesis stopped at the output cap; the answer is truncated",
            extra={"max_tokens": settings.synthesis_max_tokens},
        )
        metrics.degraded.append(DEGRADED_SYNTHESIS_TRUNCATED)
    return response.text


# Anthropic reports the stop condition as stop_reason, OpenAI as finish_reason.
# Both are checked because either model can serve this call through the fallback.
_TRUNCATION_MARKERS: dict[str, str] = {"stop_reason": "max_tokens", "finish_reason": "length"}


def _hit_the_output_cap(response: BaseMessage) -> bool:
    """Did the model stop because it ran out of budget rather than out of things to say?

    Worth distinguishing: a cap that never trips is a backstop, and a cap that
    trips on ordinary runs is a truncator. Only the response metadata knows which
    one is deployed, and a truncated answer is otherwise silent — it just ends.
    """
    metadata = response.response_metadata
    return any(metadata.get(field) == value for field, value in _TRUNCATION_MARKERS.items())


def _latest_synthesis_feedback(state: AtlasState) -> str | None:
    """The most recent critic note about the write-up, if any."""
    for verdict in reversed(state.get("critic_feedback") or []):
        if verdict.synthesis_feedback:
            return verdict.synthesis_feedback
    return None


async def synthesizer(state: AtlasState) -> dict[str, Any]:
    """Produce the final answer from all research results."""
    async with node_span(SYNTHESIZER) as span:
        metrics = span.metrics
        answer = await _synthesize(state, metrics)
        _check_citations(answer, state["research_results"], metrics)
        _check_budget(state, metrics)
    return {
        "answer": answer,
        "run_metrics": metrics,
        "messages": [AIMessage(content=answer)],
    }


def _render_for_critic(state: AtlasState) -> str:
    """Findings *with their sources*, so the critic judges against evidence.

    Deliberately richer than ``_render_findings``: the synthesizer only needs
    URLs to cite, but the critic needs the snippets to tell a supported claim
    from a plausible-sounding one. Ungrounded self-critique degrades answers about
    as often as it improves them; this is what makes the review grounded.
    """
    by_id = {s.id: s for s in state["subtasks"]}
    blocks: list[str] = []
    for result in state["research_results"]:
        subtask = by_id.get(result.subtask_id)
        question = subtask.question if subtask else "(unknown subtask)"
        if result.error is not None:
            blocks.append(
                f"### Subtask {result.subtask_id}\nQuestion: {question}\nFAILED — {result.error}"
            )
            continue
        sources = (
            "\n".join(f"- {s.url} — {s.title}\n  {s.snippet}" for s in result.sources)
            or "- (no sources retrieved)"
        )
        blocks.append(
            f"### Subtask {result.subtask_id}\nQuestion: {question}\n"
            f"Findings:\n{result.findings}\n\nSources retrieved:\n{sources}"
        )
    return "\n\n".join(blocks)


async def _review(state: AtlasState, metrics: RunMetrics) -> CriticVerdict:
    """Ask the critic for a verdict. A failure here accepts rather than blocks.

    If the critic itself errors, the run must not die — an answer already exists.
    Treating a broken reviewer as approval is the right default: the alternative
    is failing a query because the optional quality gate fell over.
    """
    try:
        return await call_structured(
            CRITIC,
            [
                SystemMessage(CRITIC_SYSTEM),
                HumanMessage(
                    f"Original question: {state['query']}\n\n"
                    f"Drafted answer:\n{state['answer']}\n\n"
                    f"Research under review:\n\n{_render_for_critic(state)}"
                ),
            ],
            CriticVerdict,
            metrics,
        )
    except Exception as exc:
        logger.warning(
            "critic failed; accepting the current answer",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:300]},
        )
        # Recorded, not just logged. Live verification caught this returning a
        # round of overall="accept" that was identical in every visible respect to
        # a genuine approval — the answer went out unreviewed and nothing said so.
        metrics.degraded.append(DEGRADED_CRITIC_UNAVAILABLE)
        return CriticVerdict(overall="accept")


async def critic(state: AtlasState) -> dict[str, Any]:
    """Review the answer against the evidence, and decide whether to send it back.

    Also the budget checkpoint. This is the only node from which a new iteration
    can start — the retry fan-out is dispatched by this node's routing edge — so
    one check here covers both "before a new iteration" and "before a fan-out",
    and keeps the routing function synchronous.
    """
    iteration = state["iteration"] + 1
    emit(ITERATION_STARTED, CRITIC, iteration=iteration)
    async with node_span(CRITIC) as span:
        metrics = span.metrics
        exceeded, total = await is_exceeded()
        if total is None:
            metrics.degraded.append(DEGRADED_BUDGET_CHECK_UNAVAILABLE)
        if exceeded:
            logger.warning(
                "token budget exhausted; accepting current answer",
                extra={"spent": total, "budget": get_settings().max_tokens_per_query},
            )
            # The synthesizer runs its own in-state estimate one node earlier and
            # writes the same marker, so a live run reported budget_exceeded
            # twice. Two checks are worth keeping — this one is authoritative and
            # Redis-backed, the other still works when Redis is down — but one
            # breach should say so once.
            if DEGRADED_BUDGET_EXCEEDED not in (state.get("run_metrics") or RunMetrics()).degraded:
                metrics.degraded.append(DEGRADED_BUDGET_EXCEEDED)
            verdict = CriticVerdict(overall="accept")
        else:
            verdict = await _review(state, metrics)

        metrics.critic_rounds.append(
            CriticRound(
                iteration=iteration,
                overall=verdict.overall,
                faulted_subtask_ids=verdict.faulted_ids(),
            )
        )

        # The router cannot write state, so anything the *answer* or metrics need
        # about how this loop ends has to be decided here.
        update: dict[str, Any] = {}
        if verdict.overall == "revise":
            actionable = bool(verdict.faulted_ids()) or bool(verdict.synthesis_feedback)
            if not actionable:
                metrics.degraded.append(DEGRADED_CRITIC_NOT_ACTIONABLE)
            elif iteration >= get_settings().max_critic_iterations:
                # Out of iterations with concerns outstanding. Keep this pass's
                # answer and tell the user what remains unverified — re-synthesising
                # would add 20-34s to a run that has already spent its budget.
                metrics.degraded.append(DEGRADED_CRITIC_MAX_ITERATIONS)
                note = _exhaustion_note(verdict)
                if note:
                    update["answer"] = state["answer"] + note

    emit(
        CRITIC_VERDICT,
        CRITIC,
        iteration=iteration,
        overall=verdict.overall,
        faulted_subtask_ids=verdict.faulted_ids(),
        has_synthesis_feedback=verdict.synthesis_feedback is not None,
    )
    logger.info(
        "critic verdict",
        extra={
            "iteration": iteration,
            "overall": verdict.overall,
            "faulted": verdict.faulted_ids(),
        },
    )
    return {
        "iteration": iteration,
        "critic_feedback": [verdict],
        "run_metrics": metrics,
        **update,
    }


def _exhaustion_note(verdict: CriticVerdict) -> str:
    """A short note naming what the critic still wanted, for the final answer."""
    parts = [v.reason for v in verdict.per_subtask if v.verdict == "revise"]
    if verdict.synthesis_feedback:
        parts.append(verdict.synthesis_feedback)
    if not parts:
        return ""
    joined = " ".join(f"- {p}" for p in parts)
    return (
        "\n\n---\n\n**Unverified:** review limits were reached with concerns "
        f"outstanding, so these points remain unaddressed: {joined}"
    )


def route_after_critic(state: AtlasState) -> list[Send] | str:
    """Decide whether to finish, retry specific branches, or just re-write.

    Synchronous by design — the budget check that could block a new iteration
    already ran inside the critic node, so this only has to read the verdict.
    """
    settings = get_settings()
    history = state.get("critic_feedback") or []
    if not history:
        return END
    verdict = history[-1]

    if verdict.overall == "accept":
        return END

    if state["iteration"] >= settings.max_critic_iterations:
        # Never loop forever. The answer from this pass stands, annotated.
        logger.warning(
            "critic iteration cap reached with concerns outstanding",
            extra={"iteration": state["iteration"]},
        )
        return END

    faulted = verdict.faulted_ids()
    if faulted:
        # (b) targeted retry: only the branches the critic actually faulted.
        by_id = {s.id: s for s in state["subtasks"]}
        sends = [
            Send(
                RESEARCHER,
                ResearcherInput(
                    subtask=by_id[subtask_id],
                    query=state["query"],
                    guidance=verdict.guidance_for(subtask_id),
                ),
            )
            for subtask_id in faulted
            if subtask_id in by_id
        ]
        if sends:
            return sends

    if verdict.synthesis_feedback:
        # (c) the research is sound, the write-up is not. Cheapest revision there
        # is: no searches, one synthesis call.
        return SYNTHESIZER

    # "revise" with nothing actionable. Looping would burn a full iteration to
    # produce the same answer, so exit and record it.
    logger.warning("critic returned revise with no actionable feedback; exiting")
    return END
