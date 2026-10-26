"""Does a cited provision say what the sentence citing it says?

The deterministic check (test_citations.py) knows a provision exists and was
read. It cannot know whether the provision says what the answer attributes to
it: a live run cited AI Act Art. 2(3), the national-security exclusion, for the
act reaching providers outside the Union, which is Art. 2(1)(c). That takes
reading, so a model judges it, and its judgement is held to evidence: a
"supported" verdict must quote the provision word for word, and the quote is
checked against the text.

The judge is faked here; what is tested is what it is asked, and what its
answers are allowed to mean.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from atlas.config import get_settings
from atlas.errors import ProviderError
from atlas.events import CITATION_SUPPORT
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import PlannerOutput, Source, Subtask
from atlas.library import load_library
from atlas.library.citations import CitationReport
from atlas.library.sources import as_source
from atlas.library.support import (
    MAX_CITATIONS,
    CitedSentence,
    SupportJudgement,
    SupportJudgements,
    check_support,
    cited_sentences,
)
from atlas.metrics import (
    DEGRADED_CITATION_SUPPORT_UNAVAILABLE,
    DEGRADED_CRITIC_MAX_ITERATIONS,
    DEGRADED_UNSUPPORTED_CITATION,
    RunMetrics,
)
from tests.conftest import initial_state, make_call_structured

LIBRARY = load_library()
SCOPE = "providers and deployers of AI systems that have their place of establishment"
WRONG = (
    "The AI Act applies to providers outside the EU when the output of their AI system "
    "is used in the EU [AI Act Art. 2(3)]."
)
RIGHT = (
    "The AI Act applies to providers outside the EU when the output of their AI system "
    "is used in the EU [AI Act Art. 2(1)(c)]."
)


def _report(
    *cited: str, unresolved: tuple[str, ...] = (), unretrieved: tuple[str, ...] = ()
) -> CitationReport:
    return CitationReport(
        cited=list(cited), unresolved=list(unresolved), unretrieved=list(unretrieved)
    )


class Judge:
    """A judge that answers from a script, and keeps what it was asked."""

    def __init__(self, *judgements: SupportJudgement) -> None:
        self.judgements = list(judgements)
        self.asked: list[list[BaseMessage]] = []

    async def __call__(self, messages: list[BaseMessage]) -> SupportJudgements:
        self.asked.append(messages)
        return SupportJudgements(judgements=self.judgements)


def test_each_citation_is_paired_with_the_sentence_it_sits_in() -> None:
    answer = (
        "## Scope\n\nIt is high-risk [AI Act Art. 6(2)]. Providers must register it "
        "[AI Act Art. 49(1)] [AI Act Art. 6(2)].\n- A bullet with no full stop [GDPR Art. 5]\n"
        "- Another [GDPR Art. 35(1)]"
    )

    pairs = cited_sentences(
        answer, _report("AI Act Art. 6(2)", "AI Act Art. 49(1)", "GDPR Art. 5", "GDPR Art. 35(1)")
    )

    assert pairs == [
        CitedSentence(label="AI Act Art. 6(2)", sentence="It is high-risk [AI Act Art. 6(2)]."),
        CitedSentence(
            label="AI Act Art. 49(1)",
            sentence="Providers must register it [AI Act Art. 49(1)] [AI Act Art. 6(2)].",
        ),
        CitedSentence(
            label="AI Act Art. 6(2)",
            sentence="Providers must register it [AI Act Art. 49(1)] [AI Act Art. 6(2)].",
        ),
        CitedSentence(label="GDPR Art. 5", sentence="- A bullet with no full stop [GDPR Art. 5]"),
        CitedSentence(label="GDPR Art. 35(1)", sentence="- Another [GDPR Art. 35(1)]"),
    ]


def test_only_citations_that_passed_the_law_check_are_judged() -> None:
    answer = "One [AI Act Art. 6(2)]. Two [AI Act Art. 114]. Three [GDPR Art. 5]."

    pairs = cited_sentences(
        answer,
        _report(
            "AI Act Art. 6(2)",
            "AI Act Art. 114",
            "GDPR Art. 5",
            unresolved=("AI Act Art. 114",),
            unretrieved=("GDPR Art. 5",),
        ),
    )

    assert [p.label for p in pairs] == ["AI Act Art. 6(2)"]


async def test_the_judge_reads_the_sentence_and_the_provision_text() -> None:
    judge = Judge(SupportJudgement(item=1, supported=False, reason="national security"))

    await check_support(WRONG, _report("AI Act Art. 2(3)"), LIBRARY, judge)

    [messages] = judge.asked
    prompt = str(messages[-1].content)
    assert "[1] Sentence: " + WRONG in prompt
    assert "Cited: AI Act Art. 2(3)" in prompt
    assert "does not apply to areas outside the scope of Union law" in prompt


async def test_a_citation_judged_not_to_support_its_sentence_is_unsupported() -> None:
    judge = Judge(
        SupportJudgement(item=1, supported=False, reason="Art. 2(3) excludes national security")
    )

    report = await check_support(WRONG, _report("AI Act Art. 2(3)"), LIBRARY, judge)

    assert [(s.label, s.reason) for s in report.unsupported] == [
        ("AI Act Art. 2(3)", "Art. 2(3) excludes national security")
    ]
    assert (report.ok, report.supported, report.unverified, report.unchecked) == (False, [], [], 0)
    assert report.feedback() == (
        "Citations that do not say what they are cited for, judged against each "
        "provision's text:\n"
        f'- [AI Act Art. 2(3)] is cited for "{WRONG}" but does not say that: Art. 2(3) '
        "excludes national security. Cite the provision that does, or drop the claim."
    )


async def test_support_counts_only_with_the_provisions_own_words() -> None:
    quoted = Judge(SupportJudgement(item=1, supported=True, quote=SCOPE))
    invented = Judge(SupportJudgement(item=1, supported=True, quote="applies to the whole world"))

    good = await check_support(RIGHT, _report("AI Act Art. 2(1)(c)"), LIBRARY, quoted)
    bad = await check_support(RIGHT, _report("AI Act Art. 2(1)(c)"), LIBRARY, invented)

    assert [(s.label, s.quote) for s in good.supported] == [("AI Act Art. 2(1)(c)", SCOPE)]
    # Not evidence the citation is wrong, so not sent back; not evidence it is
    # right either, so not counted as supported.
    assert ([s.label for s in bad.unverified], bad.supported, bad.ok) == (
        ["AI Act Art. 2(1)(c)"],
        [],
        True,
    )


async def test_a_citation_the_judge_skipped_is_unchecked_not_supported() -> None:
    answer = "One [AI Act Art. 6(2)]. Two [GDPR Art. 5]."
    judge = Judge(SupportJudgement(item=2, supported=False, reason="principles, not this"))

    report = await check_support(answer, _report("AI Act Art. 6(2)", "GDPR Art. 5"), LIBRARY, judge)

    assert ([s.label for s in report.unsupported], report.unchecked) == (["GDPR Art. 5"], 1)


async def test_the_judge_is_asked_about_at_most_the_cap() -> None:
    answer = " ".join(f"Point {n} [GDPR Art. 5]." for n in range(MAX_CITATIONS + 3))
    judge = Judge()

    report = await check_support(answer, _report("GDPR Art. 5"), LIBRARY, judge)

    prompt = str(judge.asked[0][-1].content)
    assert f"[{MAX_CITATIONS}] Sentence" in prompt
    assert f"[{MAX_CITATIONS + 1}] Sentence" not in prompt
    assert report.unchecked == MAX_CITATIONS + 3


async def test_nothing_to_judge_asks_nothing() -> None:
    judge = Judge()

    report = await check_support("No citations at all.", _report(), LIBRARY, judge)

    assert (judge.asked, report.ok, report.unchecked) == ([], True, 0)


# --- in the graph: the critic enforces it --------------------------------------------

QUESTION = "Does the AI Act apply to a provider established outside the EU?"


class Drafts:
    """A synthesizer writing the given drafts in turn, keeping each prompt."""

    def __init__(self, *drafts: str) -> None:
        self.drafts = drafts
        self.prompts: list[str] = []


def _install_graph(
    monkeypatch: pytest.MonkeyPatch,
    drafts: Drafts,
    supports: list[SupportJudgements],
    judge_fails: bool = False,
) -> list[type[BaseModel]]:
    """Article 2 retrieved, the model critic approving everything; the schemas
    every structured call asked for are returned, to see who was asked."""
    asked: list[type[BaseModel]] = []
    library = load_library()
    article_2 = library.resolve("AI Act Art. 2")
    assert article_2 is not None

    def law(question: str, limit: int) -> list[Source]:
        return [as_source(library, article_2)]

    async def no_web(query: str, max_results: int | None = None) -> list[Source]:
        return []

    async def model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node != nodes.SYNTHESIZER:
            return AIMessage(content="findings [AI Act Art. 2]")
        drafts.prompts.append(str(messages[-1].content))
        return AIMessage(content=drafts.drafts[min(len(drafts.prompts), len(drafts.drafts)) - 1])

    plan = PlannerOutput(subtasks=[Subtask(id="s0", question=QUESTION, rationale="r")])
    structured = make_call_structured(plan, supports=supports)

    async def recording(
        node: str, messages: Any, schema: type[BaseModel], metrics: RunMetrics, **kw: Any
    ) -> BaseModel:
        asked.append(schema)
        if schema is SupportJudgements and judge_fails:
            raise ProviderError(provider="anthropic", message="down", node=node)
        return await structured(node, messages, schema, metrics, **kw)

    monkeypatch.setattr(nodes, "law_sources", law)
    monkeypatch.setattr(nodes, "search", no_web)
    monkeypatch.setattr(nodes, "call_model", model)
    monkeypatch.setattr(nodes, "call_structured", recording)
    return asked


async def _run_graph(domain: str = "compliance") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    config = {"configurable": {"thread_id": "t", "run_id": "r", "domain": domain}}
    final: dict[str, Any] = {}
    async for mode, chunk in graph.astream(
        initial_state(QUESTION), config=config, stream_mode=["custom", "values"]
    ):
        if mode == "custom" and chunk["type"] == CITATION_SUPPORT:
            events.append(dict(chunk))
        elif mode == "values":
            final = dict(chunk)
    return final, events


NOT_SUPPORTED = SupportJudgements(
    judgements=[SupportJudgement(item=1, supported=False, reason="it excludes national security")]
)
SUPPORTED = SupportJudgements(judgements=[SupportJudgement(item=1, supported=True, quote=SCOPE)])


async def test_a_citation_that_does_not_say_it_is_sent_back_even_when_the_critic_approves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drafts = Drafts(WRONG, RIGHT)
    _install_graph(monkeypatch, drafts, [NOT_SUPPORTED, SUPPORTED])

    final, events = await _run_graph()

    rounds = [(r.iteration, r.overall) for r in final["run_metrics"].critic_rounds]
    assert rounds == [(1, "revise"), (2, "accept")]
    assert final["answer"] == RIGHT
    assert final["run_metrics"].degraded == []
    assert "[AI Act Art. 2(3)] is cited for" in drafts.prompts[1]
    assert "it excludes national security" in drafts.prompts[1]
    assert [
        (
            e["iteration"],
            [s["label"] for s in e["unsupported"]],
            [s["label"] for s in e["supported"]],
        )
        for e in events
    ] == [(1, ["AI Act Art. 2(3)"], []), (2, [], ["AI Act Art. 2(1)(c)"])]


async def test_out_of_rounds_an_unsupported_citation_goes_out_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_graph(monkeypatch, Drafts(WRONG), [NOT_SUPPORTED])

    final, _ = await _run_graph()

    assert final["run_metrics"].degraded == [
        DEGRADED_UNSUPPORTED_CITATION,
        DEGRADED_CRITIC_MAX_ITERATIONS,
    ]
    assert "[AI Act Art. 2(3)] is cited for" in final["answer"]


async def test_a_failed_judge_costs_the_check_not_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_graph(monkeypatch, Drafts(WRONG), [NOT_SUPPORTED], judge_fails=True)

    final, events = await _run_graph()

    assert final["answer"] == WRONG
    assert final["run_metrics"].degraded == [DEGRADED_CITATION_SUPPORT_UNAVAILABLE]
    assert events == []


async def test_the_general_domain_and_the_setting_turn_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _install_graph(monkeypatch, Drafts(WRONG), [NOT_SUPPORTED])
    general, events = await _run_graph(domain="general")
    assert SupportJudgements not in asked
    assert (general["answer"], events) == (WRONG, [])
    assert [r.overall for r in general["run_metrics"].critic_rounds] == ["accept"]

    monkeypatch.setenv("CITATION_SUPPORT_CHECK", "false")
    get_settings.cache_clear()
    try:
        asked = _install_graph(monkeypatch, Drafts(WRONG), [NOT_SUPPORTED])
        final, _ = await _run_graph()
    finally:
        get_settings.cache_clear()
    assert SupportJudgements not in asked
    assert final["answer"] == WRONG
