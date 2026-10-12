"""A retried subtask replaces its result. It does not appear twice.

**This is the load-bearing test of Phase 2.**

Phase 1's reducer was ``operator.add``, which was correct exactly as long as every
subtask ran once. The critic loop breaks that assumption: a faulted branch runs
again and returns a result for a ``subtask_id`` that is already present. Appending
gives the synthesizer the same subtask twice with contradictory findings, and
``_dedupe_sources`` then keeps whichever URL it met first — possibly the stale one.
Nothing raises. The answer is just quietly built on rejected research.

So the reducer is tested against the thing it replaced. ``test_old_reducer_...``
runs ``operator.add`` and ``merge_research_results`` over the *same* retry input
and asserts **6 against 5**. That is the same discipline as ``test_concurrency``,
which earned its threshold by being measured against a forced serial loop: an
assertion that has not been shown to fail for the right reason has not been shown
to do anything.
"""

from __future__ import annotations

import operator
from collections import Counter

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from atlas.graph import nodes
from atlas.graph.builder import build_state_graph
from atlas.graph.state import (
    AtlasState,
    CriticVerdict,
    ResearchResult,
    Source,
    Subtask,
    SubtaskVerdict,
    merge_research_results,
)
from atlas.metrics import RunMetrics
from tests.conftest import initial_state, make_call_structured, make_plan, run_config

BRANCHES = 5
FAULTED = "s3"
GUIDANCE = "Cite the 2024 benchmark table, not the vendor blog post."
ORIGINAL_FINDINGS = "first-pass findings"
REVISED_FINDINGS = "second-pass findings, addressing the reviewer"


def _result(subtask_id: str, findings: str) -> ResearchResult:
    return ResearchResult(
        subtask_id=subtask_id,
        findings=findings,
        sources=[Source(url=f"https://ex.example/{subtask_id}", title="T", snippet="s")],
    )


# --- the reducer, in both regimes ---------------------------------------------


def test_initial_fan_out_appends_every_branch() -> None:
    """Distinct ids: behaviourally identical to operator.add.

    This is what keeps the Phase 1 merge and concurrency tests meaningful — the
    new reducer must not change what happens on a first pass.
    """
    left: list[ResearchResult] = []
    for i in range(BRANCHES):
        left = merge_research_results(left, [_result(f"s{i}", ORIGINAL_FINDINGS)])

    assert [r.subtask_id for r in left] == ["s0", "s1", "s2", "s3", "s4"]


def test_a_retry_replaces_in_place() -> None:
    existing = [_result(f"s{i}", ORIGINAL_FINDINGS) for i in range(BRANCHES)]
    merged = merge_research_results(existing, [_result(FAULTED, REVISED_FINDINGS)])

    assert len(merged) == BRANCHES
    assert [r.subtask_id for r in merged] == ["s0", "s1", "s2", "s3", "s4"]

    by_id = {r.subtask_id: r for r in merged}
    assert by_id[FAULTED].findings == REVISED_FINDINGS
    # The untouched branches are untouched — a retry must not disturb them.
    assert [by_id[f"s{i}"].findings for i in (0, 1, 2, 4)] == [ORIGINAL_FINDINGS] * 4


def test_position_is_preserved_across_a_retry() -> None:
    """Order matters: the answer's subtask ordering must not shuffle mid-loop."""
    existing = [_result(f"s{i}", ORIGINAL_FINDINGS) for i in range(BRANCHES)]
    merged = merge_research_results(existing, [_result("s0", REVISED_FINDINGS)])

    assert merged[0].subtask_id == "s0"
    assert merged[0].findings == REVISED_FINDINGS


def test_old_reducer_yields_six_where_the_new_one_yields_five() -> None:
    """The negative control. Both reducers, one input, side by side.

    Without this, ``len(...) == 5`` is a number that happens to pass. With it, the
    test states exactly what breaks and why if someone puts ``operator.add`` back.
    """
    existing = [_result(f"s{i}", ORIGINAL_FINDINGS) for i in range(BRANCHES)]
    retry = [_result(FAULTED, REVISED_FINDINGS)]

    appended = operator.add(existing, retry)
    upserted = merge_research_results(existing, retry)

    assert len(appended) == 6
    assert len(upserted) == 5

    # And the duplicate is the specific harm: one subtask, two contradictory
    # findings, with the stale one first — which is the one _dedupe_sources keeps.
    duplicated = [r for r in appended if r.subtask_id == FAULTED]
    assert [r.findings for r in duplicated] == [ORIGINAL_FINDINGS, REVISED_FINDINGS]
    assert [r.findings for r in upserted if r.subtask_id == FAULTED] == [REVISED_FINDINGS]


# --- the Send payload that carries feedback back ------------------------------


def _state_after_verdict(verdict: CriticVerdict, iteration: int = 1) -> AtlasState:
    state = initial_state("what is RAG?")
    state["subtasks"] = [Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(3)]
    state["research_results"] = [_result(f"s{i}", ORIGINAL_FINDINGS) for i in range(3)]
    state["answer"] = "a drafted answer"
    state["iteration"] = iteration
    state["critic_feedback"] = [verdict]
    return state


def test_guidance_reaches_the_retried_researcher_verbatim() -> None:
    """Compared by equality, not by "guidance is present".

    A truthy check passed when a credential was ten asterisks. It would pass just
    as happily on a truncated or misattributed instruction here.
    """
    verdict = CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(subtask_id="s0", verdict="accept", reason="fine", guidance=""),
            SubtaskVerdict(
                subtask_id="s1", verdict="revise", reason="unsupported", guidance=GUIDANCE
            ),
        ],
    )
    sends = nodes.route_after_critic(_state_after_verdict(verdict))

    assert isinstance(sends, list)
    assert len(sends) == 1
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == nodes.RESEARCHER
    assert sends[0].arg["subtask"].id == "s1"
    assert sends[0].arg["guidance"] == GUIDANCE
    assert sends[0].arg["query"] == "what is RAG?"


def test_only_faulted_subtasks_are_sent_back() -> None:
    verdict = CriticVerdict(
        overall="revise",
        per_subtask=[
            SubtaskVerdict(subtask_id="s0", verdict="revise", reason="thin", guidance="g0"),
            SubtaskVerdict(subtask_id="s1", verdict="accept", reason="fine", guidance=""),
            SubtaskVerdict(subtask_id="s2", verdict="revise", reason="thin", guidance="g2"),
        ],
    )
    sends = nodes.route_after_critic(_state_after_verdict(verdict))

    assert isinstance(sends, list)
    assert [s.arg["subtask"].id for s in sends] == ["s0", "s2"]
    assert [s.arg["guidance"] for s in sends] == ["g0", "g2"]


def test_synthesis_only_feedback_routes_to_the_synthesizer() -> None:
    """(c) as a distinct verdict: sound research, faulty write-up.

    Folding this into the branch retry would re-run searches to fix a writing
    problem — the expensive half of the run, for nothing.
    """
    verdict = CriticVerdict(overall="revise", synthesis_feedback="You omitted subtask 2 entirely.")
    assert nodes.route_after_critic(_state_after_verdict(verdict)) == nodes.SYNTHESIZER


# --- end to end through the graph ---------------------------------------------


@pytest.fixture
def retry_graph(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """One critic pass faults s3, the next accepts. Records every research call.

    ``_research`` is spied on rather than wrapped further out because it is the
    exact seam where "which subtask, with what guidance" is decided — so the
    recording is of the real routing, not of the test's own assumptions.
    """
    seen: list[tuple[str, str | None]] = []
    original_research = nodes._research

    async def spy_research(
        subtask: Subtask, metrics: RunMetrics, guidance: str | None = None
    ) -> ResearchResult:
        seen.append((subtask.id, guidance))
        return await original_research(subtask, metrics, guidance)

    async def fake_search(query: str, max_results: int | None = None) -> list[Source]:
        return [Source(url=f"https://ex.example/{query}", title="T", snippet="s")]

    async def fake_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        if node != nodes.RESEARCHER:
            return AIMessage(content="the synthesized answer")
        assert isinstance(messages, list)
        prompt = "\n".join(str(m.content) for m in messages)
        # The retry prompt carries the critic's guidance; the first pass does not.
        # Different text per pass is what makes replacement observable.
        return AIMessage(content=REVISED_FINDINGS if GUIDANCE in prompt else ORIGINAL_FINDINGS)

    verdicts = [
        CriticVerdict(
            overall="revise",
            per_subtask=[
                SubtaskVerdict(
                    subtask_id=FAULTED,
                    verdict="revise",
                    reason="claims are not in the sources",
                    guidance=GUIDANCE,
                )
            ],
        ),
        CriticVerdict(overall="accept"),
    ]

    monkeypatch.setattr(nodes, "_research", spy_research)
    monkeypatch.setattr(nodes, "search", fake_search)
    monkeypatch.setattr(nodes, "call_model", fake_model)
    monkeypatch.setattr(
        nodes, "call_structured", make_call_structured(make_plan(BRANCHES), verdicts)
    )
    return seen


async def _run(thread: str) -> dict[str, object]:
    graph = build_state_graph().compile(checkpointer=MemorySaver())
    config = run_config(thread)
    return dict(await graph.ainvoke(initial_state(), config=config))


async def test_retry_leaves_five_results_not_six(
    retry_graph: list[tuple[str, str | None]],
) -> None:
    result = await _run("retry-count")
    results = result["research_results"]

    assert isinstance(results, list)
    assert len(results) == BRANCHES
    assert sorted(r.subtask_id for r in results) == [f"s{i}" for i in range(BRANCHES)]


async def test_the_faulted_branch_carries_its_revised_findings(
    retry_graph: list[tuple[str, str | None]],
) -> None:
    result = await _run("retry-content")
    results = result["research_results"]
    assert isinstance(results, list)

    # Asserted as the whole collection rather than by dict lookup. Indexing by
    # subtask_id would silently take the *last* duplicate, so it passes just as
    # happily against operator.add — the exact defect this file is about. Sorted
    # rather than positional so the assertion does not depend on the order five
    # concurrent branches happen to merge in.
    assert sorted((r.subtask_id, r.findings) for r in results) == [
        ("s0", ORIGINAL_FINDINGS),
        ("s1", ORIGINAL_FINDINGS),
        ("s2", ORIGINAL_FINDINGS),
        (FAULTED, REVISED_FINDINGS),
        ("s4", ORIGINAL_FINDINGS),
    ]


async def test_unfaulted_branches_are_not_re_run(
    retry_graph: list[tuple[str, str | None]],
) -> None:
    """The whole cost argument for targeted retry rests on this number."""
    await _run("retry-collateral")

    counts = Counter(subtask_id for subtask_id, _ in retry_graph)
    assert counts == Counter({FAULTED: 2, "s0": 1, "s1": 1, "s2": 1, "s4": 1})


async def test_the_second_pass_receives_the_critics_guidance(
    retry_graph: list[tuple[str, str | None]],
) -> None:
    """First pass has no guidance; the retry has exactly the critic's words."""
    await _run("retry-guidance")

    assert [g for subtask_id, g in retry_graph if subtask_id == FAULTED] == [None, GUIDANCE]
    assert [g for subtask_id, g in retry_graph if subtask_id != FAULTED] == [None] * 4


async def test_the_loop_terminates_on_the_second_verdict(
    retry_graph: list[tuple[str, str | None]],
) -> None:
    result = await _run("retry-iterations")

    assert result["iteration"] == 2
    metrics = result["run_metrics"]
    assert isinstance(metrics, RunMetrics)
    assert [(r.iteration, r.overall) for r in metrics.critic_rounds] == [
        (1, "revise"),
        (2, "accept"),
    ]
    assert metrics.critic_rounds[0].faulted_subtask_ids == [FAULTED]
    assert metrics.degraded == []
