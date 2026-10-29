"""Offline mode: the whole product, with the models and the web scripted.

``ATLAS_OFFLINE=true`` runs the real API, worker, graph, regulation library,
citation checks, run store and workspace with no API key. Only what a model or
a web search would have returned is scripted, beneath ``atlas.models``, so every
call is traced as a live one is and every check runs on what the scripts write.
The scripts say so: every call reports the model ``scripted``, and every answer
opens by saying a model did not write it.

What the scripts write is built from the run's own research, not canned:

* the planner splits the question into its clauses, one branch each;
* each researcher reports the provisions the library returned for its branch;
* the synthesizer cites the provisions its researchers read, one per line.

So the citation check and the support check see real citations of retrieved
provisions, a rewritten branch reads different law, and the answer changes with
it. The critic accepts. The support check judges nothing: whether a provision
says what a sentence says is a model's judgement, and a script that said "yes"
would be a judgement nobody made. Every citation is left unchecked, and the
chips show what the deterministic check against the law found, which is real.

The same wrappers script the frontend's recorded fixtures
(``scripts/capture_fixtures.py``), with each scenario's own scripts.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from atlas import models, trace
from atlas.graph import nodes
from atlas.graph.state import CriticVerdict, PlannerOutput, Source, Subtask
from atlas.library.scope import OFFLINE_FLOOR, OFFLINE_REASON, closest_match, names_the_law
from atlas.library.support import SupportJudgements
from atlas.metrics import RunMetrics

SCRIPTED = "scripted"

ModelFn = Callable[..., Awaitable[AIMessage]]
StructuredFn = Callable[..., Awaitable[BaseModel]]

NOTICE = "*Offline mode: this answer was scripted from the research below, not written by a model.*"


def _node() -> str:
    """The node making the call, from the span the runner opened for it."""
    span = trace.current_span()
    return span.node if span is not None else "unknown"


def scripted_llm(fn: ModelFn) -> Callable[..., Runnable[LanguageModelInput, AIMessage]]:
    """``models.get_llm`` answering as ``fn`` scripts it, node by node."""

    async def answer(messages: list[Any]) -> AIMessage:
        reply = await fn(_node(), messages, RunMetrics())
        return AIMessage(content=reply.content, response_metadata={"model_name": SCRIPTED})

    def get_llm(
        temperature: float | None = None, max_tokens: int | None = None
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return RunnableLambda(answer)

    return get_llm


def scripted_structured(
    fn: StructuredFn,
) -> Callable[..., Runnable[LanguageModelInput, dict[str, Any]]]:
    """``models.get_structured_llm`` answering as ``fn`` scripts it, in the envelope
    ``with_structured_output(include_raw=True)`` returns."""

    def get_structured_llm(
        schema: type[BaseModel], temperature: float | None = None
    ) -> Runnable[LanguageModelInput, dict[str, Any]]:
        async def answer(messages: list[Any]) -> dict[str, Any]:
            parsed = await fn(_node(), messages, schema, RunMetrics())
            raw = AIMessage(content="", response_metadata={"model_name": SCRIPTED})
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(answer)

    return get_structured_llm


def no_fallback(*args: object, **kwargs: object) -> Runnable[LanguageModelInput, Any]:
    """A scripted answer that failed to parse would be retried on the structured
    fallback, a real provider. Scripted models must never reach one."""
    raise RuntimeError("a scripted model reached the structured fallback")


async def no_web(
    query: str, max_results: int | None = None, include_content: bool = False
) -> list[Source]:
    """Web search, off: offline, the law comes from the library alone."""
    return []


@contextmanager
def scripted(model: ModelFn, structured: StructuredFn) -> Iterator[None]:
    """Run with the given scripts in place of every model and the web."""
    saved = (
        models.get_llm,
        models.get_structured_llm,
        models.get_structured_fallback,
        nodes.search,
    )
    install(model, structured)
    try:
        yield
    finally:
        (
            models.get_llm,
            models.get_structured_llm,
            models.get_structured_fallback,
            nodes.search,
        ) = saved


def install(model: ModelFn, structured: StructuredFn) -> None:
    """Put the scripts in place, for the life of the process (the worker's)."""
    models.get_llm = scripted_llm(model)
    models.get_structured_llm = scripted_structured(structured)
    models.get_structured_fallback = no_fallback
    nodes.search = no_web


# --- the offline scripts ----------------------------------------------------------------

# "..., and what must its provider do?": a new question starts after ", and".
_CLAUSE = re.compile(
    r",?\s+and\s+(?=(?:what|which|how|when|who|whether|does|do|is|are|must|should|can)\b)",
    re.IGNORECASE,
)
_LAW = re.compile(r"^\[((?:AI Act|GDPR) [^\]]+)\] (.*)$", re.MULTILINE)
_FOUND = re.compile(r"^- \[((?:AI Act|GDPR) [^\]]+)\] (.*)$", re.MULTILINE)
MAX_BRANCHES = 3
MAX_CITED = 4
PER_BRANCH = 2


def plan_of(question: str) -> PlannerOutput:
    """One branch per clause of the question, as asked.

    And the planner's scope judgment, scripted: a question that neither names the
    law nor comes near anything in it is judged to be about neither act (see
    atlas.library.scope.OFFLINE_FLOOR). Branches either way, so a judgment the
    graph does not act on, a follow-up's, still researches.
    """
    clauses = [c.strip(" ,?") for c in _CLAUSE.split(question) if c.strip(" ,?")][:MAX_BRANCHES]
    off_topic = not names_the_law(question) and closest_match(question) < OFFLINE_FLOOR
    return PlannerOutput(
        subtasks=[
            Subtask(id=f"s{i}", question=clause[0].upper() + clause[1:] + "?", rationale=SCRIPTED)
            for i, clause in enumerate(clauses or [question])
        ],
        out_of_scope=OFFLINE_REASON if off_topic else None,
    )


def _last(messages: object) -> str:
    return str(messages[-1].content) if isinstance(messages, list) and messages else ""


async def offline_model(node: str, messages: object, metrics: RunMetrics, **_: object) -> AIMessage:
    """A researcher lists the law it was given; the synthesizer cites it."""
    prompt = _last(messages)
    if node == nodes.SYNTHESIZER:
        # Each branch's law, so a rewritten branch shows in the answer.
        branches = re.split(r"(?m)^Subtask \S+$", prompt)[1:]
        found = list(
            dict.fromkeys(
                pair for branch in branches for pair in _FOUND.findall(branch)[:PER_BRANCH]
            )
        )
        if not found:
            return AIMessage(content=f"{NOTICE}\n\nThe research found no provisions to cite.")
        lines = "\n".join(f"- {title} [{label}]." for label, title in found)
        return AIMessage(
            content=f"{NOTICE}\n\nThe research for this question read these provisions:\n\n{lines}"
        )
    law = _LAW.findall(prompt)[:MAX_CITED]
    lines = "\n".join(f"- [{label}] {title}" for label, title in law)
    return AIMessage(
        content=f"Scripted findings (offline mode). The library returned:\n{lines}"
        if law
        else "Scripted findings (offline mode). The library returned nothing for this branch."
    )


async def offline_structured(
    node: str, messages: object, schema: type[BaseModel], metrics: RunMetrics, **_: object
) -> BaseModel:
    """The plan, an accepting critic, and a support check that judges nothing."""
    prompt = _last(messages)
    if schema is PlannerOutput:
        return plan_of(prompt)
    if schema is CriticVerdict:
        return CriticVerdict(overall="accept")
    if schema is SupportJudgements:
        return SupportJudgements()
    raise TypeError(f"offline mode has no script for {schema.__name__}")


def install_offline() -> None:
    """Offline mode, for the worker: every model and the web, scripted."""
    install(offline_model, offline_structured)
