"""The single-call baseline: what Atlas has to beat to justify its machinery.

One search on the question as asked, answered by Atlas's own synthesizer — the
same prompt, the same output cap, the same citation check. No planner, no
parallel researchers, no critic. If the full graph cannot beat this on the
same frozen corpus, the fan-out and the review are cost, not quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from atlas.config import get_settings
from atlas.errors import ProviderError
from atlas.graph import nodes
from atlas.metrics import RunMetrics
from atlas_eval.baselines import SINGLE_CALL_SOURCES, run_single_call
from atlas_eval.corpus import CorpusDocument, MissingCorpusError, QueryCorpus, save_corpus
from atlas_eval.dataset import load_dataset


@pytest.fixture
def synthesis(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int | None]]:
    """Every model call: (node, final prompt, max_tokens)."""
    calls: list[tuple[str, str, int | None]] = []

    async def fake_model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        calls.append((node, str(messages[-1].content), max_tokens))
        return AIMessage(content="the answer")

    monkeypatch.setattr(nodes, "call_model", fake_model)
    return calls


def _corpus(item_id: str, count: int) -> QueryCorpus:
    # Every document matches equally, so the URL tiebreak fixes the order.
    return QueryCorpus(
        item_id=item_id,
        recorded_at="2026-10-11",
        tavily_calls=1,
        documents=[
            CorpusDocument(
                url=f"https://ex.example/{i:02d}", title="retrieval", snippet=f"snippet-{i:02d}"
            )
            for i in range(count)
        ],
    )


async def test_one_search_and_one_synthesis(
    synthesis: list[tuple[str, str, int | None]], tmp_path: Path
) -> None:
    item = load_dataset().items[0]
    save_corpus(_corpus(item.id, 12), tmp_path)

    record = await run_single_call(item, corpus_root=tmp_path)

    assert record.error is None
    assert record.variant == "single-call"
    assert record.answer == "the answer"
    assert record.subtasks == []
    assert record.search_calls == [item.query]
    assert [s.url for s in record.sources] == [f"https://ex.example/{i:02d}" for i in range(10)]
    # Exactly one model call, and it is the synthesizer under its usual cap.
    assert [(node, cap) for node, _, cap in synthesis] == [
        (nodes.SYNTHESIZER, get_settings().synthesis_max_tokens)
    ]
    # The ten retrieved snippets reach the prompt; the two below the cut do not.
    prompt = synthesis[0][1]
    assert [f"snippet-{i:02d}" in prompt for i in range(12)] == [True] * 10 + [False] * 2
    # Framed like one of the graph's branches, asking the question itself.
    assert f"Subtask s0\nQuestion: {item.query}\n\n" in prompt


def test_the_baseline_reads_as_many_sources_as_documented() -> None:
    assert SINGLE_CALL_SOURCES == 10


async def test_a_missing_corpus_stops_the_run_like_the_graph_does(
    synthesis: list[tuple[str, str, int | None]], tmp_path: Path
) -> None:
    """A missing corpus is a setup error, not a finding — loud, as in run_item."""
    item = load_dataset().items[0]

    with pytest.raises(MissingCorpusError):
        await run_single_call(item, corpus_root=tmp_path)

    assert synthesis == []


async def test_a_failed_synthesis_is_a_recorded_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider failure, unlike a missing corpus, is scored — as zero."""

    async def failing_model(*args: object, **kwargs: object) -> AIMessage:
        raise ProviderError(provider="openai", message="all providers failed", node="synthesizer")

    monkeypatch.setattr(nodes, "call_model", failing_model)
    item = load_dataset().items[0]
    save_corpus(_corpus(item.id, 3), tmp_path)

    record = await run_single_call(item, corpus_root=tmp_path)

    assert (record.answer, record.variant) == ("", "single-call")
    assert record.error == "ProviderError: all providers failed"
    assert record.search_calls == [item.query]
