"""Capture event-stream fixtures for the frontend reducer tests.

    uv run python scripts/capture_fixtures.py            # deterministic scenarios
    uv run python scripts/capture_fixtures.py <run_id>   # dump a real run from Redis

Why this exists: the reducer must be tested against event sequences the backend
actually produces, not against hand-written ones. Hand-written fixtures test the
reducer against my beliefs about the backend; recorded ones test it against the
backend.

Two sources, and the difference is worth being explicit about.

**Live runs** (``<run_id>`` form) are dumped from the Redis event log after a real
query. They are the ground truth for the happy path and for a real targeted retry.

**Deterministic scenarios** (no argument) run the *same* real graph, runner and
nodes with the model and search faked to fail in specific ways. Every event is
still produced by production code — only the inputs are controlled. This is the
only way to get the paths that matter most and that a successful run can never
produce: a branch that fails while its siblings succeed, a critic that exhausts
its iterations, and a run that dies outright. A reducer tested only on happy
paths is a reducer tested on the runs that did not need it.

The models are scripted *underneath* ``atlas.models``: the real ``call_model``
and ``call_structured`` run, so every call is traced as a live one is, with the
exact prompt the node built. Scripted answers say so: they report the model
``scripted``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from atlas import eventlog, models
from atlas.api.topology import build_topology
from atlas.domains import DomainName, using_domain
from atlas.errors import ProviderError, SearchError
from atlas.events import AgentEvent
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import CriticVerdict, PlannerOutput, Source, Subtask, SubtaskVerdict
from atlas.library.support import SupportJudgement, SupportJudgements
from atlas.metrics import RunMetrics
from atlas.offline import no_fallback, scripted_llm, scripted_structured
from atlas.runner import execute_revision, execute_run
from atlas.store import InMemoryRunStore
from atlas_eval.runner import install_offline_budget

# Spelled out so the fakes match the real signatures they stand in for, rather
# than being annotated Any and silently drifting when a signature changes.
SearchFn = Callable[..., Awaitable[list[Source]]]
ModelFn = Callable[..., Awaitable[AIMessage]]
StructuredFn = Callable[..., Awaitable[BaseModel]]

FIXTURES = Path(__file__).resolve().parents[2] / "frontend" / "tests" / "fixtures"
DEMO = Path(__file__).resolve().parents[2] / "frontend" / "src" / "demo"

ANSWER = "## Answer\n\nSynthesised from the findings."

QUESTION: dict[str, str] = {
    "general": "Compare RAG with long-context LLMs.",
    "compliance": (
        "Is a CV-screening tool high-risk under the AI Act, and what must its provider do?"
    ),
}

QUESTIONS: dict[str, list[str]] = {
    "general": [
        "How accurate is RAG for enterprise document QA?",
        "How accurate are long-context LLMs on the same task?",
        "What are the cost characteristics of each?",
        "What are the latency characteristics of each?",
        "When is each the better choice?",
    ],
    "compliance": [
        "Is an AI system that filters job applications high-risk under the AI Act?",
        "What must a provider of a high-risk AI system do under the AI Act?",
    ],
}


def plan(count: int, domain: DomainName = "general") -> PlannerOutput:
    return PlannerOutput(
        subtasks=[
            Subtask(id=f"s{i}", question=QUESTIONS[domain][i], rationale="r") for i in range(count)
        ]
    )


def citing_model() -> ModelFn:
    """A synthesizer that cites the provisions it was given. Its first draft adds
    an invented article, which the check against the law catches, and cites a
    real one for what it does not say, which the support check catches. The
    rewrite keeps the citations that hold."""
    drafts = 0

    async def call(
        node: str,
        messages: list[Any],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        nonlocal drafts
        if node != nodes.SYNTHESIZER:
            return AIMessage(content="findings")
        labels = re.findall(r"- \[((?:AI Act|GDPR)[^\]]+)\]", str(messages[-1].content))
        drafts += 1
        if drafts == 1:
            cited = " ".join(f"[{label}]" for label in labels[:3])
            return AIMessage(
                content=f"## Answer\n\nYes: it is high-risk {cited}. "
                "It must also register it [AI Act Art. 114]."
            )
        kept = " ".join(f"[{label}]" for label in labels[:3] if SCRIPTED_SUPPORT[label][0])
        return AIMessage(
            content="## Answer\n\nYes: a tool that analyses and filters job applications is "
            f"high-risk {kept}."
        )

    return call


def fake_model(answer: str, fail_in: str | None = None) -> ModelFn:
    async def call(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if fail_in is not None and node == fail_in:
            raise ProviderError(provider="anthropic", message="everything is down", node=node)
        return AIMessage(content=answer if node == nodes.SYNTHESIZER else "findings")

    return call


def fake_structured(
    count: int,
    verdicts: list[CriticVerdict],
    domain: DomainName = "general",
    out_of_scope: str | None = None,
) -> StructuredFn:
    remaining = list(verdicts)

    async def call(
        node: str,
        messages: object,
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        if schema is CriticVerdict:
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if schema is SupportJudgements:
            return supporting_judge(messages)
        return plan(count, domain).model_copy(update={"out_of_scope": out_of_scope})

    return call


_ITEM = re.compile(
    r"^\[(\d+)\] Sentence: .*?\nCited: (.*?)\nProvision text:\n(.*?)(?=\n\n\[\d+\] Sentence: |\Z)",
    re.S | re.M,
)

# How the scripted support check reads the demo's first draft, which calls a tool
# high-risk citing three provisions. Only Annex III(4)(a) says so; the other two
# are real, retrieved, and about something else.
SCRIPTED_SUPPORT: dict[str, tuple[bool, str]] = {
    "AI Act Annex III(4)(a)": (True, "to analyse and filter job applications"),
    "AI Act Art. 6(2)": (
        True,
        "AI systems referred to in Annex III shall be considered to be high-risk",
    ),
    "AI Act Art. 6(1)": (
        False,
        "Art. 6(1) makes high-risk an AI system that is a safety component of a product "
        "under the Union harmonisation law in Annex I; a recruitment tool is not one.",
    ),
}


def supporting_judge(messages: object) -> SupportJudgements:
    """The support check, scripted as a careful reader would answer it, citation
    by citation, as SCRIPTED_SUPPORT says. A citation it has no answer for stops
    the recording: a default of "supported" once passed Art. 6(1) for a CV tool,
    after better retrieval changed what the demo's first draft cites."""
    prompt = str(messages[-1].content) if isinstance(messages, list) and messages else ""
    judgements = []
    for n, label, _text in _ITEM.findall(prompt):
        if label not in SCRIPTED_SUPPORT:
            raise RuntimeError(
                f"the demo's first draft cites {label}, which SCRIPTED_SUPPORT does not "
                "judge: say how a careful reader would, then record again"
            )
        supported, said = SCRIPTED_SUPPORT[label]
        judgements.append(
            SupportJudgement(item=int(n), supported=True, quote=said)
            if supported
            else SupportJudgement(item=int(n), supported=False, reason=said)
        )
    return SupportJudgements(judgements=judgements)


def fake_search(broken: str | None = None) -> SearchFn:
    async def call(query: str, max_results: int | None = None) -> list[Source]:
        if broken is not None and query == broken:
            raise SearchError("this source is unreachable", query=query)
        slug = query.lower().split()[0].strip("?")
        return [
            Source(url=f"https://example.com/{slug}-{i}", title=f"{slug} result {i}", snippet="s")
            for i in range(2)
        ]

    return call


def revise(subtask_id: str) -> CriticVerdict:
    return CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(
                subtask_id=subtask_id,
                verdict="revise",
                reason="the claims are not supported by the listed sources",
                guidance="Search for primary benchmark data rather than vendor posts.",
            )
        ],
    )


ACCEPT = CriticVerdict(overall="accept")


async def capture(
    name: str,
    *,
    count: int,
    verdicts: list[CriticVerdict],
    broken: str | None = None,
    fail_in: str | None = None,
    domain: DomainName = "general",
    model: ModelFn | None = None,
    revise_branch: tuple[str, str] | None = None,
    question: str | None = None,
    out_of_scope: str | None = None,
) -> InMemoryRunStore:
    """Run the real graph with controlled inputs and record every event.

    ``revise_branch`` (a subtask id, its new question) then revises the run, and
    the fixture is the revision's stream: what a client sees after an edit.
    """
    collected: list[AgentEvent] = []

    async def record(event: AgentEvent) -> None:
        collected.append(event)

    async def no_store(run_id: str, result: dict[str, Any]) -> None:
        """The fixtures are the event stream; the stored result is not needed."""

    # The nodes look these names up at call time, so rebinding the module
    # attribute is what swaps the implementation.
    saved: dict[str, Any] = {
        "append": eventlog.append,
        "store_result": eventlog.store_result,
        "search": nodes.search,
        "call_model": nodes.call_model,
        "call_structured": nodes.call_structured,
        "get_llm": models.get_llm,
        "get_structured_llm": models.get_structured_llm,
        "get_structured_fallback": models.get_structured_fallback,
    }
    eventlog.append = record
    eventlog.store_result = no_store
    nodes.search = fake_search(broken)
    # The nodes call the real model layer; only the providers under it are scripted.
    nodes.call_model = models.call_model
    nodes.call_structured = models.call_structured
    models.get_llm = scripted_llm(model or fake_model(ANSWER, fail_in))
    models.get_structured_llm = scripted_structured(
        fake_structured(count, verdicts, domain, out_of_scope)
    )
    # A scripted answer that fails to parse would be retried on the structured
    # fallback, a real provider. A recording must never reach one.
    models.get_structured_fallback = no_fallback
    store = InMemoryRunStore()

    try:
        graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
        with using_domain(domain):
            try:
                await execute_run(
                    name, question or QUESTION[domain], "thread", graph, store=store
                )
                if revise_branch is not None:
                    collected.clear()
                    subtask_id, question = revise_branch
                    await execute_revision(
                        f"{name}-v2",
                        "thread",
                        graph,
                        parent_run_id=name,
                        subtask_id=subtask_id,
                        question=question,
                        store=store,
                    )
            except Exception as exc:
                print(f"  ({name} ended with {type(exc).__name__}, as intended)")
    finally:
        eventlog.append = saved["append"]
        eventlog.store_result = saved["store_result"]
        nodes.search = saved["search"]
        nodes.call_model = saved["call_model"]
        nodes.call_structured = saved["call_structured"]
        models.get_llm = saved["get_llm"]
        models.get_structured_llm = saved["get_structured_llm"]
        models.get_structured_fallback = saved["get_structured_fallback"]

    write(name, collected)
    return store


def write(name: str, events: list[AgentEvent]) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{name}.json"
    payload = [json.loads(event.model_dump_json()) for event in events]
    # LF on every platform: the fixtures are committed, and a diff should show
    # events, not line endings.
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    kinds = ", ".join(sorted({event.type for event in events}))
    print(f"{path.name:<24} {len(events):>3} events  [{kinds}]")


async def dump_live(run_id: str) -> None:
    """Dump a real run's events straight from the Redis log."""
    events = await eventlog.replay(run_id, 0)
    if not events:
        raise SystemExit(f"no events for run {run_id} — expired, or wrong id?")
    write("live-run", events)
    await eventlog.close_client()


async def main() -> None:
    if len(sys.argv) > 1:
        await dump_live(sys.argv[1])
        return

    # The budget in this process's memory, as the eval keeps it. Without Redis the
    # budget check fails open and stamps budget_check_unavailable on every run:
    # an artefact of where the fixtures were captured, not of any scenario.
    install_offline_budget()

    # The happy path, deterministic. The live capture is the ground truth; this
    # one keeps the reducer tests runnable without Redis or a provider.
    await capture("accepted", count=5, verdicts=[ACCEPT])

    # One branch dies while four succeed. Phase 1 turns a failed branch into a
    # ResearchResult rather than raising so the siblings survive, and Phase 3
    # reports node_failed for it anyway — so the UI must show one lane failed,
    # four done, and a run that still completes with an answer. A green run that
    # quietly answered from four fifths of the research is the failure this
    # fixture exists to make impossible.
    await capture("partial-failure", count=5, verdicts=[ACCEPT], broken=QUESTIONS["general"][1])

    # A targeted retry: the critic faults one subtask, then accepts.
    await capture("retry", count=5, verdicts=[revise("s2"), ACCEPT])

    # The critic never accepts. Terminates at MAX_CRITIC_ITERATIONS with the
    # answer it has, degraded rather than failed.
    await capture("exhaustion", count=3, verdicts=[revise("s1"), revise("s1"), revise("s1")])

    # The run dies outright. The stream must still terminate.
    await capture("run-failed", count=3, verdicts=[ACCEPT], fail_in=nodes.SYNTHESIZER)

    # A person rewrites branch s2 of an accepted run. The fixture is the
    # revision's stream: revision_started names the branches carried over, and
    # only the edited one researches again.
    await capture(
        "revision",
        count=5,
        verdicts=[ACCEPT],
        revise_branch=("s2", "What are the per-token costs of long-context models?"),
    )

    # The compliance domain: researchers read the regulation library, sources
    # carry citation labels and authority, and every draft's citations are
    # checked against the law. The first draft cites an article that does not
    # exist; the check sends it back whatever the critic said, and the second
    # draft is clean.
    compliance = await capture(
        "compliance", count=2, verdicts=[ACCEPT], domain="compliance", model=citing_model()
    )

    await capture_declined()

    await write_demo(compliance)
    write_citation_cases()


async def capture_declined() -> None:
    """A compliance question the planner judges to be about neither act: the run
    ends at the plan, with the reason, and nothing is researched."""
    await capture(
        "declined",
        count=0,
        verdicts=[ACCEPT],
        domain="compliance",
        question="What is the history of Cameroon?",
        out_of_scope="The question is about the history of Cameroon.",
    )


async def write_demo(store: InMemoryRunStore) -> None:
    """What the frontend's /demo page replays, offline: a recorded stream, every
    call it made in full (the inspector's prompts, which a live page fetches from
    the API), and the graph's real topology, all from the backend rather than
    written by hand."""
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "compliance.json").write_text(
        (FIXTURES / "compliance.json").read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    calls = [call.model_dump(mode="json") for call in await store.calls("compliance")]
    (DEMO / "calls.json").write_text(
        json.dumps(calls, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    topology = build_topology().model_dump()
    (DEMO / "topology.json").write_text(
        json.dumps(topology, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"demo/                    compliance.json, calls.json ({len(calls)}), topology.json")


# Citations as models write them, and some that are not citations. The frontend
# marks citations in an answer with its own port of the parser, held to these.
CITATION_INPUTS = [
    "[AI Act Art. 6(2)]",
    "[AI Act Article 6 (2)]",
    "[ai act art 6(2)(a)]",
    "[AI Act Art. 5(1)(A)]",
    "[GDPR Art. 33(1); GDPR Art. 34(1)]",
    "[GDPR Art. 6(1)(a), GDPR Art. 9(2)(a)]",
    "[GDPR Art. 6(1)(a), (b)]",
    "[AI Act Annex iii(4)(a)]",
    "[AI Act Recital 85]",
    "[GDPR Recital 071]",
    "[AI Act Art. 6(2)] and [GDPR Art. 35(1)]",
    "[1]",
    "[see the guidance]",
    "[https://example.com]",
    "[The AI Act Art. 6]",
    "[GDPR Article 4(11)",
]


def write_citation_cases() -> None:
    """What the backend makes of each citation input, for the frontend's parser."""
    from atlas.library.citations import cited_labels

    cases = [{"text": text, "labels": cited_labels(text)} for text in CITATION_INPUTS]
    (FIXTURES / "citation-cases.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"citation-cases.json       {len(cases)} cases")


if __name__ == "__main__":
    asyncio.run(main())
