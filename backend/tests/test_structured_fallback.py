"""A primary that answers in the wrong shape must still hand off to the fallback.

``with_fallbacks`` reacts to exceptions. ``include_raw=True`` — needed so metrics
can say which provider answered — turns a parse failure into *data*: the
envelope comes back with ``parsing_error`` set and nothing is raised. So a
primary that answered malformed JSON was final. The planner then degraded to a
single subtask, losing the whole fan-out, while a healthy fallback sat idle.

``call_structured`` now retries a primary's parse failure on the fallback,
exactly once, and records that it did.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from atlas import models
from atlas.errors import ProviderError
from atlas.graph.state import PlannerOutput, Subtask
from atlas.metrics import DEGRADED_STRUCTURED_OUTPUT_FALLBACK, RunMetrics

PLAN = {"subtasks": [{"id": "s0", "question": "q", "rationale": "r"}]}


def _envelope(model: str, parsed: object | None, error: Exception | None) -> dict[str, Any]:
    return {
        "raw": AIMessage(content="...", response_metadata={"model_name": model}),
        "parsed": parsed,
        "parsing_error": error,
    }


def _install(
    monkeypatch: pytest.MonkeyPatch, composed: dict[str, Any], fallback: dict[str, Any]
) -> list[str]:
    """Replace both structured composers. Returns a log of which one ran."""
    calls: list[str] = []

    async def run_composed(_: object) -> dict[str, Any]:
        calls.append("composed")
        return composed

    async def run_fallback(_: object) -> dict[str, Any]:
        calls.append("fallback")
        return fallback

    monkeypatch.setattr(
        models, "get_structured_llm", lambda schema, temperature=None: RunnableLambda(run_composed)
    )
    # raising=False: before the fix this seam did not exist, and the test must
    # fail on behaviour (no handoff), not on a missing attribute.
    monkeypatch.setattr(
        models,
        "get_structured_fallback",
        lambda schema, temperature=None: RunnableLambda(run_fallback),
        raising=False,
    )
    return calls


async def test_a_primary_parse_failure_is_retried_on_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        monkeypatch,
        composed=_envelope("claude-sonnet-5", None, ValueError("subtasks: not a list")),
        fallback=_envelope("gpt-4o", PLAN, None),
    )
    metrics = RunMetrics()

    plan = await models.call_structured("planner", "q", PlannerOutput, metrics)

    assert plan == PlannerOutput(subtasks=[Subtask(id="s0", question="q", rationale="r")])
    assert calls == ["composed", "fallback"]
    # Both calls are on the record, attributed to who actually answered.
    assert [c.provider for c in metrics.llm_calls] == ["anthropic", "openai"]
    assert metrics.degraded == [DEGRADED_STRUCTURED_OUTPUT_FALLBACK]


async def test_a_well_formed_primary_never_touches_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the retry must not fire on the ordinary path."""
    calls = _install(
        monkeypatch,
        composed=_envelope("claude-sonnet-5", PLAN, None),
        fallback=_envelope("gpt-4o", PLAN, None),
    )
    metrics = RunMetrics()

    await models.call_structured("planner", "q", PlannerOutput, metrics)

    assert calls == ["composed"]
    assert metrics.degraded == []


async def test_both_malformed_raises_after_exactly_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install(
        monkeypatch,
        composed=_envelope("claude-sonnet-5", None, ValueError("bad")),
        fallback=_envelope("gpt-4o", None, ValueError("also bad")),
    )
    metrics = RunMetrics()

    with pytest.raises(ProviderError, match="structured output did not parse"):
        await models.call_structured("planner", "q", PlannerOutput, metrics)

    assert calls == ["composed", "fallback"]
    assert metrics.degraded == [DEGRADED_STRUCTURED_OUTPUT_FALLBACK]


async def test_no_second_try_when_the_fallback_already_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary raised, so the composed chain was served by GPT-4o already.

    Its parse failure is final: retrying it would ask the same provider the same
    question and call it a fallback.
    """
    calls = _install(
        monkeypatch,
        composed=_envelope("gpt-4o", None, ValueError("bad")),
        fallback=_envelope("gpt-4o", PLAN, None),
    )
    metrics = RunMetrics()

    with pytest.raises(ProviderError):
        await models.call_structured("planner", "q", PlannerOutput, metrics)

    assert calls == ["composed"]
    assert metrics.degraded == []
