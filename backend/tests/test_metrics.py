"""Run metrics: the reducer and the node timer."""

from __future__ import annotations

import asyncio

from atlas.metrics import (
    LLMCall,
    NodeTiming,
    RunMetrics,
    merge_run_metrics,
    track_node,
)


def _call(node: str, tokens: int = 10) -> LLMCall:
    return LLMCall(
        node=node,
        provider="anthropic",
        model="claude-sonnet-5",
        input_tokens=tokens,
        output_tokens=tokens,
        duration_s=0.1,
    )


def test_merge_concatenates_every_list() -> None:
    """The reducer that stops parallel branches raising InvalidUpdateError."""
    left = RunMetrics(
        llm_calls=[_call("planner")],
        node_timings=[NodeTiming(node="planner", duration_s=0.2)],
        degraded=["a"],
    )
    right = RunMetrics(
        llm_calls=[_call("researcher")],
        node_timings=[NodeTiming(node="researcher", duration_s=0.3)],
        degraded=["b"],
    )

    merged = merge_run_metrics(left, right)

    assert len(merged.llm_calls) == 2
    assert len(merged.node_timings) == 2
    assert merged.degraded == ["a", "b"]


def test_merge_does_not_mutate_its_inputs() -> None:
    """A reducer that mutates would corrupt state across branches."""
    left = RunMetrics(llm_calls=[_call("planner")])
    right = RunMetrics(llm_calls=[_call("researcher")])

    merge_run_metrics(left, right)

    assert len(left.llm_calls) == 1
    assert len(right.llm_calls) == 1


def test_token_totals_sum_across_calls() -> None:
    metrics = RunMetrics(llm_calls=[_call("a", 10), _call("b", 5)])
    assert metrics.total_input_tokens == 15
    assert metrics.total_output_tokens == 15
    assert metrics.total_tokens == 30


def test_missing_token_counts_do_not_break_totals() -> None:
    """Structured-output responses report no usage; None must not become an error."""
    metrics = RunMetrics(
        llm_calls=[
            LLMCall(node="a", provider="openai", model="gpt-4o", duration_s=0.1),
            _call("b", 7),
        ]
    )
    assert metrics.total_input_tokens == 7


async def test_track_node_records_a_timing() -> None:
    async with track_node("planner") as metrics:
        await asyncio.sleep(0.01)

    assert len(metrics.node_timings) == 1
    assert metrics.node_timings[0].node == "planner"
    assert metrics.node_timings[0].duration_s >= 0.01


async def test_track_node_records_timing_even_when_the_body_raises() -> None:
    """A node that fails should still report how long it ran before failing."""
    metrics: RunMetrics | None = None
    try:
        async with track_node("researcher") as m:
            metrics = m
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert metrics is not None
    assert len(metrics.node_timings) == 1
    assert metrics.node_timings[0].node == "researcher"
