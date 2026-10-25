"""The domain eval's three systems, on one compliance question.

* **plain**: one call, no retrieval, citing from memory. What a team gets by
  asking a model directly, and the baseline the citation check exists to beat.
* **single-call**: one library search on the question as asked, answered by
  Atlas's own synthesizer.
* **atlas**: the full graph.

Web search is off for all three: the comparison is about citing the law, the
law comes from the committed library, and a live search would make the runs
unrepeatable. The models are faked; what is tested is what each system is
given and what it records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from atlas import models
from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.state import CriticVerdict, PlannerOutput, Subtask
from atlas.library.sources import law_sources
from atlas.metrics import RunMetrics
from atlas_eval.baselines import PLAIN_COMPLIANCE, SINGLE_CALL_SOURCES, run_plain, run_single_call
from atlas_eval.dataset import DATASETS, load_dataset
from atlas_eval.runner import run_item

ITEM = load_dataset(*DATASETS["compliance"]).items[0]


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[BaseMessage], int | None]]:
    """Every model call (node, messages, max_tokens); the critic accepts."""
    calls: list[tuple[str, list[BaseMessage], int | None]] = []

    async def model(
        node: str,
        messages: list[BaseMessage],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        calls.append((node, messages, max_tokens))
        return AIMessage(content="High-risk [AI Act Art. 6(2)].")

    async def structured(
        node: str, messages: Any, schema: type[BaseModel], metrics: RunMetrics, **_: Any
    ) -> BaseModel:
        if schema is PlannerOutput:
            return PlannerOutput(subtasks=[Subtask(id="s0", question=ITEM.query, rationale="r")])
        return CriticVerdict(overall="accept")

    # The graph calls the model through the nodes' names; plain calls the layer.
    monkeypatch.setattr(nodes, "call_model", model)
    monkeypatch.setattr(models, "call_model", model)
    monkeypatch.setattr(nodes, "call_structured", structured)
    return calls


async def test_plain_is_one_call_with_nothing_to_read(
    fake_models: list[tuple[str, list[BaseMessage], int | None]],
) -> None:
    record = await run_plain(ITEM)

    [(node, messages, max_tokens)] = fake_models
    assert [(m.type, m.content) for m in messages] == [
        ("system", PLAIN_COMPLIANCE),
        ("human", ITEM.query),
    ]
    assert (node, max_tokens) == (nodes.SYNTHESIZER, get_settings().synthesis_max_tokens)
    assert (record.variant, record.answer, record.sources, record.search_calls) == (
        "plain",
        "High-risk [AI Act Art. 6(2)].",
        [],
        [],
    )


async def test_single_call_reads_the_law_from_the_library_and_nothing_else(
    fake_models: list[tuple[str, list[BaseMessage], int | None]], tmp_path: Path
) -> None:
    # An empty corpus directory: a compliance run must never need one.
    record = await run_single_call(ITEM, corpus_root=tmp_path, domain="compliance")

    expected = law_sources(ITEM.query, SINGLE_CALL_SOURCES)
    assert record.error is None
    assert record.sources == expected
    assert {s.authority for s in record.sources} == {"primary_law"}
    [(node, messages, _)] = fake_models
    assert node == nodes.SYNTHESIZER
    assert all(f"[{s.label}]" in str(messages[-1].content) for s in expected)
    assert record.search_calls == []


async def test_atlas_reads_the_law_with_the_web_off(
    fake_models: list[tuple[str, list[BaseMessage], int | None]], tmp_path: Path
) -> None:
    record = await run_item(ITEM, corpus_root=tmp_path, domain="compliance")

    assert record.error is None
    assert record.sources != []
    assert {s.authority for s in record.sources} == {"primary_law"}
    # The researcher still asked for the web; with it off, nothing came back.
    assert record.search_calls == [ITEM.query]
