"""A follow-up on the same thread is a new run, not a continuation of the last one.

The thread is the conversation; the run is one question. Only the conversation
should carry over. Everything else — the research, the critic's feedback, the
metrics, the token budget — belongs to the run that produced it.

That separation did not hold. ``initial_state`` passes ``research_results: []``,
but LangGraph applies graph input *through each channel's reducer*, and merging
an empty list into the previous run's results keeps them. A follow-up planned two
subtasks and was synthesised from five: three stale branches from the previous
question rode along, as did the previous critic's feedback (which the first
synthesis of the follow-up then read as a rejection of an answer it had never
written) and the previous run's critic rounds and timings. The budget, keyed by
thread, kept counting across questions until a perfectly normal follow-up was
declared over budget and went out unreviewed.

Nothing raised. Each follow-up returned a plausible answer. The tests below run
two questions on one thread, which no test did before.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from atlas.budget import record_spend
from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde
from atlas.graph.state import (
    CriticVerdict,
    PlannerOutput,
    ResearchResult,
    Source,
    Subtask,
    SubtaskVerdict,
)
from atlas.metrics import (
    DEGRADED_BUDGET_EXCEEDED,
    CriticRound,
    LLMCall,
    NodeTiming,
    RunMetrics,
)
from atlas.runner import execute_run
from tests.conftest import initial_state, make_call_structured, make_plan, run_config

THREAD = "conversation"
FIRST_FEEDBACK = "the write-up omits the 2024 benchmark"


@pytest.fixture
def synth_prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Search and model fakes; returns the synthesizer's final prompt per call."""
    prompts: list[str] = []

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node == nodes.SYNTHESIZER:
            prompts.append(str(messages[-1].content))
        return AIMessage(content=f"{node} output")

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    return prompts


async def _two_runs(monkeypatch: pytest.MonkeyPatch, graph: Any = None) -> dict[str, Any]:
    """Five subtasks and one write-up rejection, then a two-subtask follow-up."""
    graph = graph or build_state_graph().compile(checkpointer=MemorySaver())

    monkeypatch.setattr(
        nodes,
        "call_structured",
        make_call_structured(
            make_plan(5),
            [
                CriticVerdict(overall="revise", synthesis_feedback=FIRST_FEEDBACK),
                CriticVerdict(overall="accept"),
            ],
        ),
    )
    await graph.ainvoke(initial_state("first question"), config=run_config(THREAD, "run-1"))

    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))
    return dict(await graph.ainvoke(initial_state("follow-up"), config=run_config(THREAD, "run-2")))


async def test_a_follow_up_is_synthesised_from_its_own_research_only(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    second = await _two_runs(monkeypatch)

    assert [s.id for s in second["subtasks"]] == ["s0", "s1"]
    # The whole collection, sorted: s2..s4 belong to the first question.
    assert sorted(r.subtask_id for r in second["research_results"]) == ["s0", "s1"]


async def test_a_follow_up_does_not_read_the_previous_critics_feedback(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first run's rejection must not reach the second run's first draft."""
    await _two_runs(monkeypatch)

    # First run: draft, then the revision carrying the feedback. Second run: one
    # draft, which has never been reviewed and so has nothing to address.
    assert [FIRST_FEEDBACK in prompt for prompt in synth_prompts] == [False, True, False]


async def test_a_follow_ups_metrics_describe_that_run_alone(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    second = await _two_runs(monkeypatch)
    metrics = second["run_metrics"]

    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [(1, "accept")]
    assert [t.node for t in metrics.node_timings].count(nodes.SYNTHESIZER) == 1
    assert [t.node for t in metrics.node_timings].count(nodes.RESEARCHER) == 2
    assert [t.node for t in metrics.node_timings].count(nodes.PLANNER) == 1
    assert second["iteration"] == 1


async def test_the_conversation_itself_still_carries_over(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reset must stop at the run. ``messages`` is the thread, and keeps growing.

    The control for the other three tests: a fix that wiped the whole state
    would pass them all and silently turn every follow-up into a cold start.
    """
    planner_inputs: list[list[str]] = []
    real_structured = make_call_structured(make_plan(2))

    async def recording_structured(
        node: str,
        messages: list[BaseMessage],
        schema: type[PlannerOutput],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> object:
        if schema is PlannerOutput:
            planner_inputs.append(
                [str(m.content) for m in messages if isinstance(m, HumanMessage | AIMessage)]
            )
        return await real_structured(node, messages, schema, metrics, temperature)

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    monkeypatch.setattr(nodes, "call_structured", recording_structured)
    await graph.ainvoke(initial_state("first question"), config=run_config(THREAD, "run-1"))
    await graph.ainvoke(initial_state("follow-up"), config=run_config(THREAD, "run-2"))

    assert planner_inputs == [
        ["first question"],
        ["first question", "synthesizer output", "follow-up"],
    ]


# --- the budget -----------------------------------------------------------------


async def test_a_follow_up_starts_with_a_fresh_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each run spends 80% of the limit. Neither is over; the thread total is.

    ``max_tokens_per_query`` is a per-question limit, so the second question must
    be reviewed exactly like the first. Keyed by thread, the counter reached 160%
    at the second critic, which then skipped the review and stamped
    ``budget_exceeded`` on a run that had spent 80%.
    """
    per_call = get_settings().max_tokens_per_query // 5  # 4 calls per run -> 80%

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    async def spending_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        # The real call_model charges the budget after every call; so does this.
        await record_spend(per_call)
        return AIMessage(content="answer")

    critic_calls = 0
    plan_or_accept = make_call_structured(make_plan(3))

    async def counting_structured(
        node: str,
        messages: object,
        schema: type[Any],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> object:
        nonlocal critic_calls
        if schema is CriticVerdict:
            critic_calls += 1
        return await plan_or_accept(node, messages, schema, metrics, temperature)

    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", spending_model)
    monkeypatch.setattr(nodes, "call_structured", counting_structured)

    graph = build_state_graph().compile(checkpointer=MemorySaver())
    first = await execute_run("run-1", "first question", THREAD, graph)
    second = await execute_run("run-2", "follow-up", THREAD, graph)

    assert first["metrics"]["degraded"] == []
    assert second["metrics"]["degraded"] == []
    assert DEGRADED_BUDGET_EXCEEDED not in second["metrics"]["degraded"]
    # Both runs were actually reviewed, not waved through.
    assert critic_calls == 2


# --- the checkpoint serializer ------------------------------------------------------
#
# Every follow-up starts by loading the thread's checkpoint, which rebuilds Atlas's
# own types from their import paths. LangGraph only rebuilds types it is allowed
# to. An unlisted one does not raise: it comes back as a plain dict with a log
# line. Today everything is allowed with a deprecation warning; the allowlist in
# atlas.graph.builder keeps that true when strict becomes LangGraph's default.


def test_every_state_type_round_trips_through_the_serializer() -> None:
    serde = checkpoint_serde()
    values: list[object] = [
        Subtask(id="s0", question="q", rationale="r"),
        ResearchResult(
            subtask_id="s0", findings="f", sources=[Source(url="https://a", title="t", snippet="s")]
        ),
        CriticVerdict(
            overall="revise",
            per_subtask=[
                SubtaskVerdict(subtask_id="s0", verdict="revise", reason="r", guidance="g")
            ],
            synthesis_feedback="x",
        ),
        RunMetrics(
            llm_calls=[LLMCall(node="planner", provider="anthropic", model="m", duration_s=1.0)],
            node_timings=[NodeTiming(node="researcher", duration_s=1.0, subtask_id="s0")],
            degraded=["d"],
            critic_rounds=[CriticRound(iteration=1, overall="accept")],
        ),
    ]

    assert [serde.loads_typed(serde.dumps_typed(v)) for v in values] == values


async def test_a_follow_up_restores_atlas_types_not_dicts(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=checkpoint_serde()))
    await _two_runs(monkeypatch, graph)

    restored = (await graph.aget_state(run_config(THREAD))).values

    assert [type(s) for s in restored["subtasks"]] == [Subtask, Subtask]
    assert [type(r) for r in restored["research_results"]] == [ResearchResult, ResearchResult]
    assert [type(v) for v in restored["critic_feedback"]] == [CriticVerdict]
    assert type(restored["run_metrics"]) is RunMetrics


async def test_an_unlisted_type_breaks_the_follow_up(
    synth_prompts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: what the allowlist prevents.

    With ResearchResult left off, the first run's results come back as dicts, and
    the follow-up dies in the very first reducer to touch them — the input step
    merging ``research_results: []`` into the restored list.
    """
    partial = JsonPlusSerializer(allowed_msgpack_modules=[Subtask])
    graph = build_state_graph().compile(checkpointer=MemorySaver(serde=partial))

    with pytest.raises(AttributeError, match="subtask_id"):
        await _two_runs(monkeypatch, graph)
