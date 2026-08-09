"""Synthesizer: citation grounding, partial failure, and the empty case."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from atlas.graph import nodes
from atlas.graph.state import AtlasState, ResearchResult, Source
from atlas.metrics import DEGRADED_UNGROUNDED_CITATION, RunMetrics
from tests.conftest import initial_state

URL_A = "https://a.example/rag"
URL_B = "https://b.example/vectors"


def _state_with(results: list[ResearchResult]) -> AtlasState:
    state = initial_state("what is RAG?")
    state["research_results"] = results
    return state


def _good_results() -> list[ResearchResult]:
    return [
        ResearchResult(
            subtask_id="s0",
            findings="RAG retrieves documents first.",
            sources=[Source(url=URL_A, title="A", snippet="s")],
        ),
        ResearchResult(
            subtask_id="s1",
            findings="Vectors index the corpus.",
            sources=[Source(url=URL_B, title="B", snippet="s")],
        ),
    ]


def _patch_answer(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    async def fake_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> AIMessage:
        return AIMessage(content=answer)

    monkeypatch.setattr(nodes, "call_model", fake_model)


# --- citation grounding -------------------------------------------------------


def test_allowed_urls_collects_every_source() -> None:
    assert nodes.allowed_urls(_good_results()) == {URL_A, URL_B}


def test_every_answer_url_must_come_from_the_research() -> None:
    """Given two known URLs, no third may appear in the answer."""
    answer = f"RAG retrieves first ({URL_A}) and indexes with vectors ({URL_B})."
    assert nodes.ungrounded_urls(answer, _good_results()) == set()


def test_an_invented_url_is_detected() -> None:
    answer = f"See {URL_A} and also https://totally-made-up.example/page for more."
    assert nodes.ungrounded_urls(answer, _good_results()) == {
        "https://totally-made-up.example/page"
    }


def test_trailing_punctuation_is_not_part_of_the_url() -> None:
    """Otherwise "see https://a.example/rag." reads as a fabricated citation."""
    answer = f"As described at {URL_A}."
    assert nodes.ungrounded_urls(answer, _good_results()) == set()


async def test_ungrounded_citation_is_recorded_in_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompting is an instruction, not a guarantee. This is the check that knows."""
    _patch_answer(monkeypatch, f"See {URL_A} and https://invented.example/x.")
    result = await nodes.synthesizer(_state_with(_good_results()))

    assert DEGRADED_UNGROUNDED_CITATION in result["run_metrics"].degraded


async def test_clean_citations_are_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_answer(monkeypatch, f"Grounded in {URL_A} and {URL_B}.")
    result = await nodes.synthesizer(_state_with(_good_results()))

    assert DEGRADED_UNGROUNDED_CITATION not in result["run_metrics"].degraded


# --- partial failure ----------------------------------------------------------


async def test_failed_subtasks_are_marked_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthesizer must be told which branches died so it can say so."""
    captured: dict[str, str] = {}

    async def capture_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> AIMessage:
        assert isinstance(messages, list)
        captured["prompt"] = "\n".join(str(m.content) for m in messages)
        return AIMessage(content="answer")

    monkeypatch.setattr(nodes, "call_model", capture_model)

    results = [
        *_good_results(),
        ResearchResult(subtask_id="s2", findings="", sources=[], error="SearchError: down"),
    ]
    await nodes.synthesizer(_state_with(results))

    assert "FAILED" in captured["prompt"]
    assert "s2" in captured["prompt"]


def test_render_marks_failures_and_lists_urls() -> None:
    results = [
        *_good_results(),
        ResearchResult(subtask_id="s2", findings="", sources=[], error="boom"),
    ]
    rendered = nodes._render_findings(results)

    assert "FAILED" in rendered
    assert URL_A in rendered
    assert URL_B in rendered


# --- the empty case -----------------------------------------------------------


async def test_no_research_says_so_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model call at all.

    Answering from priors here would be the worst outcome: a confident answer
    with nothing behind it, indistinguishable from a researched one.
    """
    called = False

    async def should_not_run(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> AIMessage:
        nonlocal called
        called = True
        return AIMessage(content="should never be used")

    monkeypatch.setattr(nodes, "call_model", should_not_run)
    result = await nodes.synthesizer(_state_with([]))

    assert not called
    assert "No research was performed" in result["answer"]
