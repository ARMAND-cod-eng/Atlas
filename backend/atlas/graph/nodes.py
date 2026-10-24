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
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Overwrite, Send

from atlas.budget import is_exceeded
from atlas.config import get_settings
from atlas.domains import current_domain
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
from atlas.library.sources import law_sources, with_authority
from atlas.metrics import (
    DEGRADED_BUDGET_CHECK_UNAVAILABLE,
    DEGRADED_BUDGET_EXCEEDED,
    DEGRADED_CRITIC_MAX_ITERATIONS,
    DEGRADED_CRITIC_NOT_ACTIONABLE,
    DEGRADED_CRITIC_UNAVAILABLE,
    DEGRADED_CRITIC_UNKNOWN_SUBTASK,
    DEGRADED_PLANNER_FALLBACK,
    DEGRADED_RESEARCHER_FAILED,
    DEGRADED_SYNTHESIS_TRUNCATED,
    DEGRADED_UNGROUNDED_CITATION,
    DEGRADED_WEB_SEARCH_FAILED,
    CriticRound,
    RunMetrics,
)
from atlas.models import call_model, call_structured

# Re-exported explicitly (`as search`) rather than imported plainly. Under mypy's
# no-implicit-reexport this name is part of the module's declared interface, and
# it genuinely is one: it is the seam the tests, the fixture recorder and the eval
# suite all substitute to keep themselves off the network.
from atlas.tools.search import search as search

logger = logging.getLogger(__name__)

# Node names, referenced by the builder and by fan_out's return value. Constants
# so a rename cannot silently desynchronise the edge from the node.
PLANNER = "planner"
RESEARCHER = "researcher"
SYNTHESIZER = "synthesizer"
CRITIC = "critic"
# Never executed: the entry point a revision writes through (see atlas.revision).
HUMAN_EDIT = "human_edit"


def _single_subtask(query: str) -> Subtask:
    """The degraded plan: research the original question as one unit."""
    return Subtask(
        id="s0",
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
        SystemMessage(current_domain().planner.format(max_subtasks=settings.max_subtasks)),
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
    return _with_server_ids(output.subtasks[: settings.max_subtasks])


def _with_server_ids(subtasks: list[Subtask]) -> list[Subtask]:
    """Key each subtask by its position, whatever id the model wrote.

    ``subtask_id`` keys the upsert reducer, the retry ``Send`` payloads and the
    critic's verdicts, so it has to be unique — and the model is not a reliable
    source of unique keys. Two subtasks it labelled alike merged into one result,
    and the second branch vanished from the answer without a trace.
    """
    return [
        subtask.model_copy(update={"id": f"s{index}"}) for index, subtask in enumerate(subtasks)
    ]


async def planner(state: AtlasState) -> dict[str, Any]:
    """Decompose the query into independent, parallelizable subtasks."""
    async with node_span(PLANNER) as span:
        subtasks = await _plan(state["query"], list(state["messages"]), span.metrics)
        metrics = span.metrics
    logger.info("planned subtasks", extra={"subtask_count": len(subtasks)})
    return {
        "subtasks": subtasks,
        # A thread holds many runs, and these three keys belong to one of them.
        # Graph input cannot clear them: LangGraph applies input through each
        # key's reducer, and merging [] into the previous run's results keeps
        # them. A follow-up was synthesised from the last question's research and
        # read the last critic's feedback as its own. The planner runs exactly
        # once, first, in every run, so it replaces them outright. ``messages``
        # is left to its reducer on purpose — that is the conversation.
        "research_results": Overwrite([]),
        "critic_feedback": Overwrite([]),
        "run_metrics": Overwrite(metrics),
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


async def human_edit(state: AtlasState) -> dict[str, Any]:
    """The node a person's edit is written as. It never runs.

    A revision is applied with ``aupdate_state(..., as_node=HUMAN_EDIT)``, which
    runs only this node's writers and its routing edge — never the function. It
    has no incoming edge, so the graph cannot reach it either. Raising rather than
    passing keeps it that way: a refactor that wires an edge here fails loudly
    instead of running a no-op step in the middle of a run.
    """
    raise RuntimeError("human_edit is applied with aupdate_state, never executed")


def route_after_edit(state: AtlasState) -> list[Send]:
    """Routing edge for a human edit: research the edited branch, nothing else.

    No guidance. A person changing the question is not a reviewer rejecting the
    old answer, and the researcher frames guidance as exactly that rejection.
    """
    revision = state["revision"]
    if revision is None:
        raise RuntimeError("a human edit was applied without a revision")
    by_id = {subtask.id: subtask for subtask in state["subtasks"]}
    edited = by_id[revision.subtask_id]
    return [Send(RESEARCHER, ResearcherInput(subtask=edited, query=state["query"], guidance=None))]


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


async def _law_then_guidance(subtask: Subtask, metrics: RunMetrics) -> list[Source]:
    """The compliance researcher's reading: provisions of the law, then the web.

    A web search that fails costs the branch its guidance, not its law: with
    provisions already in hand the branch goes on, marked degraded. Without any,
    the failure is the branch's, as in the general domain.
    """
    settings = get_settings()
    law = law_sources(subtask.question, settings.library_results)
    if settings.compliance_web_results <= 0:
        return law
    try:
        web = await search(subtask.question, max_results=settings.compliance_web_results)
    except Exception:
        if not law:
            raise
        logger.warning("web search failed; continuing on the law", extra={"subtask_id": subtask.id})
        metrics.degraded.append(f"{DEGRADED_WEB_SEARCH_FAILED}:{subtask.id}")
        return law
    return law + [with_authority(source) for source in web]


def render_sources(sources: list[Source]) -> str:
    """Numbered sources as a model reads them: title, URL, snippet.

    Shared with the eval's single-call baseline so the two systems present
    evidence to the model identically, and differ only in their structure.
    """
    return "\n\n".join(
        f"[{i + 1}] {s.title}\nURL: {s.url}\n{s.snippet}" for i, s in enumerate(sources)
    )


def render_law_and_guidance(sources: list[Source]) -> str:
    """The compliance researcher's sources: the law first, by label, then the rest.

    Kept apart because they are not equals. A provision says what the law
    requires and is cited by its label; a web page, even the regulator's own,
    can only explain it, and is cited by URL.
    """
    law = [s for s in sources if s.authority == "primary_law"]
    other = [s for s in sources if s.authority != "primary_law"]
    blocks: list[str] = []
    if law:
        provisions = "\n\n".join(f"[{s.label}] {s.title}\n{s.snippet}" for s in law)
        blocks.append(f"PRIMARY LAW (cite by the bracketed label):\n\n{provisions}")
    if other:
        pages = "\n\n".join(
            f"[{i + 1}] {s.title} ({s.authority.replace('_', ' ')})\nURL: {s.url}\n{s.snippet}"
            for i, s in enumerate(other)
        )
        blocks.append(f"SECONDARY SOURCES (guidance only; cite by URL):\n\n{pages}")
    return "\n\n".join(blocks)


async def _summarize(
    subtask: Subtask,
    sources: list[Source],
    metrics: RunMetrics,
    guidance: str | None = None,
) -> str:
    """Turn raw sources into findings with citations."""
    domain = current_domain()
    rendered = render_law_and_guidance(sources) if domain.library else render_sources(sources)
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
        [SystemMessage(domain.researcher), HumanMessage(prompt)],
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
        if current_domain().library:
            sources = await _law_then_guidance(subtask, metrics)
        else:
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
        # Reported on the branch's own event rather than waiting for the run to
        # end. These sources exist the moment the branch finishes, and the
        # synthesizer then spends 20-45s producing nothing observable — this is
        # what a watching client reads during that silence.
        span.report(
            sources=[{"url": source.url, "title": source.title} for source in result.sources],
            question=subtask.question,
        )
        if result.error is not None:
            # A failed branch does not raise — Phase 1 turns it into a result so
            # the other four survive — but it has still failed, and a watching
            # client should see node_failed rather than node_completed.
            span.failed(result.error)
        metrics = span.metrics
    # A one-element list because the reducer on research_results upserts by id.
    return {"research_results": [result], "run_metrics": metrics}


def _render_findings(results: list[ResearchResult], subtasks: list[Subtask]) -> str:
    """Format results for the synthesizer, each under the question it answers.

    The question is shown because a person can now rewrite it: without it the
    synthesizer had only the findings to guess from what a branch was asked.
    """
    questions = {subtask.id: subtask.question for subtask in subtasks}
    blocks: list[str] = []
    for result in results:
        question = questions.get(result.subtask_id, "(unknown subtask)")
        header = f"Subtask {result.subtask_id}\nQuestion: {question}"
        if result.error is not None:
            blocks.append(f"{header}\nFAILED — {result.error}")
            continue
        cited = "\n".join(f"- [{s.label}]" if s.label else f"- {s.url}" for s in result.sources)
        blocks.append(f"{header}\n\n{result.findings}\n\nSources:\n{cited or '- (none)'}")
    return "\n\n---\n\n".join(blocks)


_URL_PATTERN = re.compile(r"https?://[^\s<>\"\]]+")

# Trailing punctuation is trimmed so "see https://a.example." does not capture the
# sentence's full stop. Parentheses and apostrophes are legal in URLs and common —
# Wikipedia's ".../Raft_(algorithm)", ".../Cohen's_kappa". The pattern used to stop
# at both, so a correctly cited, retrieved URL was read as ".../Raft_(algorithm"
# and reported as invented: most of the "ungrounded" citations in the first full
# recording were this. The match now takes them, and _trim removes only what
# belongs to the surrounding prose.
_TRAILING_PROSE = ".,;:!?'\""


def _trim(url: str) -> str:
    """Drop prose punctuation and unbalanced closing parentheses from the end."""
    while url:
        if url[-1] in _TRAILING_PROSE:
            url = url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            url = url[:-1]
        else:
            break
    return url


def _same_page(url: str) -> str:
    """Comparison key: a fragment or a trailing slash does not make another page."""
    return url.split("#", 1)[0].rstrip("/")


def allowed_urls(results: list[ResearchResult]) -> set[str]:
    """Every URL the synthesizer is permitted to cite."""
    return {source.url for result in results for source in result.sources}


def ungrounded_urls(answer: str, results: list[ResearchResult]) -> set[str]:
    """URLs in the answer that appear in no source.

    Prompting the model not to invent citations is necessary but not sufficient —
    it is an instruction, not a guarantee. This is the check that actually knows.
    """
    permitted = {_same_page(url) for url in allowed_urls(results)}
    cited = {_trim(match) for match in _URL_PATTERN.findall(answer)}
    return {url for url in cited if _same_page(url) not in permitted}


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
    findings = _render_findings(results, state["subtasks"])
    prompt = f"Original question: {state['query']}\n\nResearch findings:\n\n{findings}"
    # On a synthesis-only revision the critic faulted the write-up, not the
    # research. Passing its feedback is the only thing that makes the second
    # attempt different from the first.
    feedback = _latest_synthesis_feedback(state)
    if feedback is not None:
        prompt = f"{prompt}\n\nA reviewer rejected your previous answer:\n{feedback}"

    response = await call_model(
        SYNTHESIZER,
        [
            SystemMessage(current_domain().synthesizer),
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
            "\n".join(
                f"- [{s.label}] {s.title}\n  {s.snippet}"
                if s.label
                else f"- {s.url} — {s.title}\n  {s.snippet}"
                for s in result.sources
            )
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
                SystemMessage(current_domain().critic),
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
            verdict = _drop_unknown_subtasks(
                await _review(state, metrics), state["subtasks"], metrics
            )

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


def _drop_unknown_subtasks(
    verdict: CriticVerdict, subtasks: list[Subtask], metrics: RunMetrics
) -> CriticVerdict:
    """Keep only the critic's verdicts about subtasks that exist.

    An id the critic invents — "subtask-1" for ``s0`` — names nothing routing can
    send work to. Left in, it made a "revise" look actionable, so nothing was
    recorded, routing found no branch to retry, and the run ended on an
    unaddressed rejection with ``degraded`` empty. Dropped here, everything
    downstream — the actionability check, the recorded round, the router — sees
    the same verdict, and the loss is reported once.
    """
    known = {subtask.id for subtask in subtasks}
    lost = [
        v.subtask_id
        for v in verdict.per_subtask
        if v.subtask_id not in known and v.verdict == "revise"
    ]
    if lost:
        logger.warning(
            "critic faulted subtasks that are not in the plan; dropping them",
            extra={"unknown_subtask_ids": lost, "known_subtask_ids": sorted(known)},
        )
        metrics.degraded.append(DEGRADED_CRITIC_UNKNOWN_SUBTASK)
    return verdict.model_copy(
        update={"per_subtask": [v for v in verdict.per_subtask if v.subtask_id in known]}
    )


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
