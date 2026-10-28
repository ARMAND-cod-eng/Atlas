"""Rewriting a question into the acts' own terms before searching the library.

Off by default: it is adopted only if it measures better on the held-out half
(atlas_eval retrieval --rewrite --split test). What is tested here is that the
rewrite only ever adds search terms, never article numbers a model invented,
and that it is asked for only when switched on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.state import Subtask
from atlas.library.rewrite import MAX_TERMS_CHARS, statutory_terms
from atlas.metrics import LLMCall, RunMetrics
from atlas.prompts import COMPLIANCE_QUERY_REWRITE
from atlas_eval import cli


class Model:
    """A model that answers with a fixed line, and keeps what it was sent."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.sent: list[list[BaseMessage]] = []

    async def __call__(
        self,
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        self.sent.append(messages)
        return AIMessage(content=self.reply)


async def test_the_model_is_asked_for_the_acts_terms_for_the_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model("recruitment selection of natural persons, high-risk AI system")

    terms = await statutory_terms("Is our CV screener regulated?", RunMetrics(), model)

    assert terms == "recruitment selection of natural persons, high-risk AI system"
    [messages] = model.sent
    assert [(m.type, m.content) for m in messages] == [
        ("system", COMPLIANCE_QUERY_REWRITE),
        ("human", "Is our CV screener regulated?"),
    ]


async def test_article_numbers_and_act_names_a_model_writes_are_dropped() -> None:
    # A rewrite that named "Article 52" would steer the lookup to Article 52,
    # the proposal's numbering, not the act's; one that named "GDPR" would widen
    # an AI Act question to both acts. Which provision and which act come from
    # the question; the rewrite adds words.
    model = Model(
        "transparency obligations, Article 52 AI Act, Art. 50(1), Annex III, Recital 132, GDPR"
    )

    terms = await statutory_terms("Do chatbots need a label?", RunMetrics(), model)

    assert terms == "transparency obligations"


async def test_a_long_reply_is_cut_to_a_search_line() -> None:
    terms = await statutory_terms("q", RunMetrics(), Model("word " * 200))

    assert len(terms) <= MAX_TERMS_CHARS


async def _searched_with(monkeypatch: pytest.MonkeyPatch, rewrite: bool) -> tuple[list[str], Model]:
    model = Model("third country transfer of personal data")
    queries: list[str] = []

    def law(question: str, limit: int) -> list[object]:
        queries.append(question)
        return []

    monkeypatch.setattr(nodes, "call_model", model)
    monkeypatch.setattr(nodes, "law_sources", law)
    monkeypatch.setenv("QUERY_REWRITE", "true" if rewrite else "false")
    monkeypatch.setenv("COMPLIANCE_WEB_RESULTS", "0")
    get_settings.cache_clear()
    try:
        await nodes._law_then_guidance(
            Subtask(id="s0", question="Can we move data to India?", rationale="r"), RunMetrics()
        )
    finally:
        get_settings.cache_clear()
    return queries, model


async def test_switched_on_the_library_is_searched_with_the_question_and_its_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries, model = await _searched_with(monkeypatch, rewrite=True)

    assert queries == ["Can we move data to India? third country transfer of personal data"]
    assert len(model.sent) == 1


async def test_switched_off_no_model_is_asked_and_the_question_is_searched_as_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries, model = await _searched_with(monkeypatch, rewrite=False)

    assert (queries, model.sent) == (["Can we move data to India?"], [])


# --- measuring it ----------------------------------------------------------------------


def _fake_call(reply: str, provider: str = "anthropic") -> object:
    async def call(
        node: str, messages: list[BaseMessage], metrics: RunMetrics, **_: object
    ) -> AIMessage:
        metrics.llm_calls.append(
            LLMCall(node=node, provider=provider, model="claude-sonnet-5", duration_s=0.1)
        )
        return AIMessage(content=reply)

    return call


def test_the_measurement_compares_with_and_without_and_keeps_the_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "rewrite_call", lambda: _fake_call("third country"))

    code = cli.main(["retrieval", "--rewrite", "--split", "test"])

    out = capsys.readouterr().out
    written = json.loads((tmp_path / "retrieval-rewrite-test.json").read_text(encoding="utf-8"))
    assert code == 0
    assert "20 questions rewritten" in out
    assert "recall@10 (test) as asked:" in out and "rewritten:" in out
    assert [r["item_id"] for r in written["items"]][:2] == ["c02", "c04"]
    assert {r["rewrite"] for r in written["items"]} == {"third country"}
    assert set(written) >= {"as_asked", "rewritten", "commit", "models"}


def test_a_measurement_the_fallback_answered_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "rewrite_call", lambda: _fake_call("third country", "openai"))

    code = cli.main(["retrieval", "--rewrite", "--split", "dev"])

    assert code == 1
    assert "FAIL" in capsys.readouterr().out
