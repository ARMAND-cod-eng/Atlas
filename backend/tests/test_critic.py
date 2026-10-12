"""The critic: it must reject, and it must accept.

Testing only one direction proves nothing. A critic that never accepts is a cost
multiplier wearing a quality gate's clothes — every query pays for the maximum
number of iterations. A critic that always accepts is a twenty-second no-op with a
Pydantic schema, and it looks *perfect* on every metric: fast, cheap, one
iteration, no degraded flags. That is the Phase 0 shape exactly, where a component
that never fired was indistinguishable from one that worked.

So the accept path is asserted as strictly as the reject path — exactly one critic
call, exactly one synthesis call — and the loop is asserted to terminate on its
own rather than on the recursion limit.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from pydantic import BaseModel

from atlas.errors import ProviderError
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import (
    AtlasState,
    CriticVerdict,
    ResearchResult,
    Source,
    Subtask,
    SubtaskVerdict,
)
from atlas.metrics import (
    DEGRADED_CRITIC_MAX_ITERATIONS,
    DEGRADED_CRITIC_NOT_ACTIONABLE,
    DEGRADED_CRITIC_UNAVAILABLE,
    DEGRADED_CRITIC_UNKNOWN_SUBTASK,
    DEGRADED_SYNTHESIS_TRUNCATED,
    RunMetrics,
)
from tests.conftest import (
    initial_state,
    make_call_model,
    make_call_structured,
    make_plan,
    run_config,
)

BRANCHES = 3
SNIPPET = "the 2024 benchmark reports 91.4 percent recall"


def _accept() -> CriticVerdict:
    return CriticVerdict(overall="accept")


def _revise(subtask_id: str, guidance: str = "look for the primary source") -> CriticVerdict:
    return CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(
                subtask_id=subtask_id,
                verdict="revise",
                reason="the claim is not in any listed source",
                guidance=guidance,
            )
        ],
    )


@pytest.fixture
def graph_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet=SNIPPET)]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", make_call_model("the synthesized answer"))


def _with_verdicts(monkeypatch: pytest.MonkeyPatch, verdicts: list[CriticVerdict]) -> None:
    monkeypatch.setattr(
        nodes, "call_structured", make_call_structured(make_plan(BRANCHES), verdicts)
    )


async def _run(thread: str) -> dict[str, object]:
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    return dict(await graph.ainvoke(initial_state(), config=run_config(thread)))


def _calls(result: dict[str, object], node: str) -> int:
    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    return sum(1 for t in metrics.node_timings if t.node == node)


# --- the accept path ----------------------------------------------------------


async def test_accepting_costs_exactly_one_critic_and_one_synthesis(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap path must actually be cheap. == 1, not >= 1."""
    _with_verdicts(monkeypatch, [_accept()])
    result = await _run("critic-accept")

    assert _calls(result, nodes.CRITIC) == 1
    assert _calls(result, nodes.SYNTHESIZER) == 1
    assert _calls(result, nodes.RESEARCHER) == BRANCHES
    assert result["iteration"] == 1

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [(1, "accept")]
    assert metrics.degraded == []


def test_accept_routes_straight_to_end() -> None:
    state = initial_state()
    state["iteration"] = 1
    state["critic_feedback"] = [_accept()]

    assert nodes.route_after_critic(state) == END


# --- the reject path ----------------------------------------------------------


async def test_reject_then_accept_takes_exactly_two_iterations(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_verdicts(monkeypatch, [_revise("s1"), _accept()])
    result = await _run("critic-reject-accept")

    assert result["iteration"] == 2
    assert _calls(result, nodes.CRITIC) == 2
    assert _calls(result, nodes.SYNTHESIZER) == 2
    # Three on the first pass plus exactly one retry — not six.
    assert _calls(result, nodes.RESEARCHER) == BRANCHES + 1

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [
        (1, "revise"),
        (2, "accept"),
    ]
    assert metrics.degraded == []


async def test_persistent_rejection_terminates_at_the_cap(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never fails closed, never loops forever.

    A critic that rejects everything is a real failure mode, and the run must
    still hand the user the best answer it has rather than an error or a
    recursion-limit traceback.
    """
    _with_verdicts(monkeypatch, [_revise("s1"), _revise("s1"), _revise("s1")])
    result = await _run("critic-persistent")

    assert result["iteration"] == 2
    assert _calls(result, nodes.CRITIC) == 2

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert metrics.degraded == [DEGRADED_CRITIC_MAX_ITERATIONS]

    answer = result["answer"]
    assert isinstance(answer, str)
    assert answer.startswith("the synthesized answer")
    # Exhausted with concerns outstanding: the user is told what was not verified
    # rather than handed a silently unreviewed answer.
    assert "**Unverified:**" in answer
    assert "the claim is not in any listed source" in answer


async def test_no_extra_synthesis_after_exhaustion(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two iterations means two syntheses, not three.

    Re-synthesising on exhaustion would add a full output pass to a run that has
    already spent its budget, to produce an answer nothing would review.
    """
    _with_verdicts(monkeypatch, [_revise("s1"), _revise("s1"), _revise("s1")])
    result = await _run("critic-no-extra-synthesis")

    assert _calls(result, nodes.SYNTHESIZER) == 2


# --- grounding ----------------------------------------------------------------


async def test_the_critic_sees_the_retrieved_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grounded review, not self-reflection.

    The discriminating comparison is against the synthesizer's prompt, which gets
    findings and bare URLs. If the critic's prompt carried the same thing it would
    be judging findings against its own priors — which degrades answers about as
    often as it improves them.
    """
    prompts: dict[str, str] = {}

    async def capture_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        assert isinstance(messages, list)
        prompts[node] = "\n".join(str(m.content) for m in messages)
        return AIMessage(content="the synthesized answer")

    async def capture_structured(
        node: str,
        messages: object,
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        assert isinstance(messages, list)
        prompts[node] = "\n".join(str(m.content) for m in messages)
        if schema is CriticVerdict:
            return _accept()
        return make_plan(BRANCHES)

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet=SNIPPET)]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", capture_model)
    monkeypatch.setattr(nodes, "call_structured", capture_structured)

    await _run("critic-grounding")

    assert SNIPPET in prompts[nodes.CRITIC]
    assert SNIPPET not in prompts[nodes.SYNTHESIZER]
    # It also sees what it is reviewing, not just the evidence.
    assert "the synthesized answer" in prompts[nodes.CRITIC]


def test_render_for_critic_pairs_each_subtask_with_its_question() -> None:
    state = initial_state("what is RAG?")
    state["subtasks"] = [Subtask(id="s0", question="how does retrieval work?", rationale="r")]
    state["research_results"] = [
        ResearchResult(
            subtask_id="s0",
            findings="it retrieves first",
            sources=[Source(url="https://a.example", title="A", snippet=SNIPPET)],
        )
    ]
    rendered = nodes._render_for_critic(state)

    assert "how does retrieval work?" in rendered
    assert SNIPPET in rendered
    assert "https://a.example" in rendered


def test_render_for_critic_marks_a_failed_branch() -> None:
    state = initial_state()
    state["subtasks"] = [Subtask(id="s0", question="q0", rationale="r")]
    state["research_results"] = [
        ResearchResult(subtask_id="s0", findings="", sources=[], error="SearchError: down")
    ]

    assert "FAILED — SearchError: down" in nodes._render_for_critic(state)


# --- degradation --------------------------------------------------------------


async def test_a_broken_critic_accepts_rather_than_blocking(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer already exists. Failing the query over a broken reviewer is worse."""

    async def failing_structured(
        node: str,
        messages: object,
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        if schema is CriticVerdict:
            raise ProviderError(provider="anthropic", message="critic is down", node=node)
        return make_plan(BRANCHES)

    monkeypatch.setattr(nodes, "call_structured", failing_structured)
    result = await _run("critic-broken")

    assert result["answer"] == "the synthesized answer"
    assert result["iteration"] == 1

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [(1, "accept")]
    # The round above is byte-for-byte a genuine approval. This flag is the only
    # thing separating "reviewed and approved" from "never reviewed at all", and
    # it was missing until a live run produced exactly this and looked clean.
    assert metrics.degraded == [DEGRADED_CRITIC_UNAVAILABLE]


async def test_revise_with_nothing_actionable_exits_and_says_so(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Revise" naming no subtask and no synthesis problem is not an instruction.

    Looping on it would burn a full iteration — a re-synthesis at minimum — to
    produce the same answer from the same research.
    """
    _with_verdicts(monkeypatch, [CriticVerdict(overall="revise")])
    result = await _run("critic-not-actionable")

    assert result["iteration"] == 1
    assert _calls(result, nodes.SYNTHESIZER) == 1

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert metrics.degraded == [DEGRADED_CRITIC_NOT_ACTIONABLE]


async def test_a_revise_naming_an_unknown_subtask_is_reported(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The critic faulted "subtask-1". The plan's ids are s0..s2.

    This used to count as actionable — the verdict named *a* subtask — so the
    critic recorded nothing, routing found nothing to send, and the run ended on
    an unaddressed "revise" with ``degraded == []``: indistinguishable, from the
    client, from a run whose concerns had been dealt with.
    """
    _with_verdicts(monkeypatch, [_revise("subtask-1")])
    result = await _run("critic-unknown-id")

    assert result["iteration"] == 1
    assert _calls(result, nodes.RESEARCHER) == BRANCHES

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    # The round records what routing acted on, which is nothing.
    assert [(r.iteration, r.overall, r.faulted_subtask_ids) for r in metrics.critic_rounds] == [
        (1, "revise", [])
    ]
    assert metrics.degraded == [DEGRADED_CRITIC_UNKNOWN_SUBTASK, DEGRADED_CRITIC_NOT_ACTIONABLE]


async def test_an_unknown_id_does_not_stop_the_known_one_being_retried(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    mixed = CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(subtask_id="s1", verdict="revise", reason="r", guidance="g"),
            SubtaskVerdict(subtask_id="nope", verdict="revise", reason="r", guidance="g"),
        ],
    )
    _with_verdicts(monkeypatch, [mixed, _accept()])
    result = await _run("critic-mixed-ids")

    assert _calls(result, nodes.RESEARCHER) == BRANCHES + 1

    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall, r.faulted_subtask_ids) for r in metrics.critic_rounds] == [
        (1, "revise", ["s1"]),
        (2, "accept", []),
    ]
    assert metrics.degraded == [DEGRADED_CRITIC_UNKNOWN_SUBTASK]


async def test_synthesis_only_revision_re_writes_without_re_searching(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c): the cheapest revision there is — one synthesis call, zero searches."""
    _with_verdicts(
        monkeypatch,
        [
            CriticVerdict(overall="revise", synthesis_feedback="subtask 2 is missing"),
            _accept(),
        ],
    )
    result = await _run("critic-synthesis-only")

    assert _calls(result, nodes.SYNTHESIZER) == 2
    assert _calls(result, nodes.RESEARCHER) == BRANCHES  # not re-run
    assert result["iteration"] == 2


async def test_synthesis_feedback_reaches_the_second_write_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the second attempt is identical to the first, at full cost."""
    feedback = "You omitted the third subtask entirely."
    synthesis_prompts: list[str] = []

    async def capture_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        assert isinstance(messages, list)
        if node == nodes.SYNTHESIZER:
            synthesis_prompts.append("\n".join(str(m.content) for m in messages))
        return AIMessage(content="the synthesized answer")

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet=SNIPPET)]

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", capture_model)
    _with_verdicts(
        monkeypatch,
        [CriticVerdict(overall="revise", synthesis_feedback=feedback), _accept()],
    )

    await _run("critic-synthesis-feedback")

    assert len(synthesis_prompts) == 2
    assert feedback not in synthesis_prompts[0]
    assert feedback in synthesis_prompts[1]


# --- verdict helpers ----------------------------------------------------------


def test_faulted_ids_lists_only_revisions() -> None:
    verdict = CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(subtask_id="s0", verdict="accept", reason="fine", guidance=""),
            SubtaskVerdict(subtask_id="s1", verdict="revise", reason="thin", guidance="g1"),
            SubtaskVerdict(subtask_id="s2", verdict="revise", reason="thin", guidance="g2"),
        ],
    )

    assert verdict.faulted_ids() == ["s1", "s2"]
    assert verdict.guidance_for("s1") == "g1"
    # An accepted subtask has no instruction to give, even if the field is filled.
    assert verdict.guidance_for("s0") is None
    assert verdict.guidance_for("nonexistent") is None


async def test_the_synthesizer_output_cap_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop behind the concision prompt. Latency is output-driven."""
    seen: dict[str, int | None] = {}

    async def capture_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        seen[node] = max_tokens
        return AIMessage(content="answer")

    monkeypatch.setattr(nodes, "call_model", capture_model)

    state: AtlasState = initial_state()
    state["research_results"] = [
        ResearchResult(
            subtask_id="s0",
            findings="f",
            sources=[Source(url="https://a.example", title="A", snippet="s")],
        )
    ]

    await nodes.synthesizer(state)

    assert seen[nodes.SYNTHESIZER] == 6000


def _one_result() -> list[ResearchResult]:
    return [
        ResearchResult(
            subtask_id="s0",
            findings="f",
            sources=[Source(url="https://a.example", title="A", snippet="s")],
        )
    ]


@pytest.mark.parametrize(
    ("metadata", "truncated"),
    [
        ({"stop_reason": "max_tokens"}, True),  # anthropic
        ({"finish_reason": "length"}, True),  # openai, via the fallback
        ({"stop_reason": "end_turn"}, False),
        ({"finish_reason": "stop"}, False),
        ({}, False),
    ],
)
async def test_hitting_the_output_cap_is_reported(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, str], truncated: bool
) -> None:
    """A truncated answer is otherwise silent — it just ends.

    Both providers are checked because either can serve this call. The flag is
    also how a badly-set cap announces itself: a backstop should never trip, so a
    run that reports this is evidence the ceiling sits below normal output.
    """

    async def capped_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        return AIMessage(content="a cut-off answer", response_metadata=metadata)

    monkeypatch.setattr(nodes, "call_model", capped_model)

    state = initial_state()
    state["research_results"] = _one_result()
    result = await nodes.synthesizer(state)

    degraded = result["run_metrics"].degraded
    assert (DEGRADED_SYNTHESIS_TRUNCATED in degraded) is truncated


# --- the comparison variant ---------------------------------------------------------


async def test_the_no_critic_variant_ends_at_the_synthesizer(
    graph_fakes: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For measuring what the critic buys: same graph, minus the review.

    A verdict that would reject is installed, so a critic that still ran would
    show up as a second synthesis — not just as a timing.
    """
    _with_verdicts(monkeypatch, [_revise("s1"), _accept()])
    graph = build_state_graph(critic_enabled=False).compile(checkpointer=MemorySaver())
    result = dict(await graph.ainvoke(initial_state(), config=run_config("no-critic")))

    assert nodes.CRITIC not in graph.get_graph().nodes
    assert _calls(result, nodes.CRITIC) == 0
    assert _calls(result, nodes.SYNTHESIZER) == 1
    assert _calls(result, nodes.RESEARCHER) == BRANCHES
    assert result["iteration"] == 0
    assert result["answer"] == "the synthesized answer"


def test_the_default_graph_keeps_its_critic() -> None:
    """The control: production's graph is unchanged by the switch existing."""
    assert nodes.CRITIC in build_state_graph().compile().get_graph().nodes
