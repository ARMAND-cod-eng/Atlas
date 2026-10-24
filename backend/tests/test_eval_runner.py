"""The eval runs the production graph with no services — including the budget.

The token budget lives in Redis, and the eval has none: not in CI, often not on
a laptop. Against an absent Redis the budget fails open on every call, which is
right for production and wrong here. Every eval run reported
``budget_check_unavailable``, and a run that blew its budget was never stopped:
the critic kept reviewing and re-researching. On exactly the most expensive
runs, the eval was measuring a different system from the one that ships.

So the eval keeps the counter in its own process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from atlas import budget
from atlas.budget import record_spend
from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.state import CriticVerdict
from atlas.metrics import DEGRADED_BUDGET_EXCEEDED, RunMetrics
from atlas_eval.corpus import CorpusDocument, QueryCorpus, save_corpus
from atlas_eval.dataset import load_dataset
from atlas_eval.runner import run_item
from tests.conftest import FakeRedis, make_call_structured, make_plan


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis, tmp_path: Path) -> Path:
    """Redis is down, as in CI. Both seams the eval installs are restored after."""
    fake_redis.fail = True
    monkeypatch.setattr(nodes, "search", nodes.search)
    monkeypatch.setattr(budget, "get_client", budget.get_client)
    item = load_dataset().items[0]
    save_corpus(
        QueryCorpus(
            item_id=item.id,
            recorded_at="2026-10-11",
            tavily_calls=1,
            documents=[CorpusDocument(url="https://a.example", title="RAG", snippet="s")],
        ),
        tmp_path,
    )
    return tmp_path


def _spending_model(per_call: int) -> Any:
    async def fake(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        await record_spend(per_call)
        return AIMessage(content="answer")

    return fake


async def test_an_eval_run_has_a_working_budget_without_redis(
    offline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nodes, "call_model", _spending_model(10))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(2)))

    record = await run_item(load_dataset().items[0], corpus_root=offline)

    assert record.error is None
    assert record.metrics.degraded == []


async def test_an_eval_run_over_budget_is_stopped_as_in_production(
    offline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two researchers and a synthesis at 40% each: 120% by the first review.

    A critic that would reject forever is installed. Production stops at one
    round because the budget is exhausted; so must the eval.
    """
    per_call = get_settings().max_tokens_per_query * 2 // 5
    critic_calls = 0
    plan_or_reject = make_call_structured(
        make_plan(2), [CriticVerdict(overall="revise", synthesis_feedback="again")]
    )

    async def counting(
        node: str, messages: object, schema: type[Any], metrics: RunMetrics, temperature: Any = None
    ) -> object:
        nonlocal critic_calls
        if schema is CriticVerdict:
            critic_calls += 1
        return await plan_or_reject(node, messages, schema, metrics, temperature)

    monkeypatch.setattr(nodes, "call_model", _spending_model(per_call))
    monkeypatch.setattr(nodes, "call_structured", counting)

    record = await run_item(load_dataset().items[0], corpus_root=offline)

    assert critic_calls == 0
    assert record.iterations == 1
    assert DEGRADED_BUDGET_EXCEEDED in record.metrics.degraded


async def test_the_frozen_corpus_is_installed_on_every_run(
    offline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed once per process, a restored ``nodes.search`` stayed restored.

    Found while writing the tests above: the second test's run reached live
    Tavily (a 401 on the fake key, so nothing was spent). In a real session the
    key is real, and the eval would quietly score live search.
    """
    live_calls: list[str] = []

    async def live(
        query: str, max_results: int | None = None, include_content: bool = False
    ) -> list[Any]:
        live_calls.append(query)
        return []

    monkeypatch.setattr(nodes, "call_model", _spending_model(1))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))
    item = load_dataset().items[0]

    await run_item(item, corpus_root=offline)
    # What a fixture's teardown, or a debugging session, does between runs.
    monkeypatch.setattr(nodes, "search", live)
    record = await run_item(item, corpus_root=offline)

    assert live_calls == []
    assert record.search_calls == ["q0"]


async def test_recording_keeps_its_budget_in_process(
    monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis, tmp_path: Path
) -> None:
    """Recording runs the live agent, so it needs the budget as much as replay —
    without depending on a Redis being up to get it."""
    from atlas_eval import record
    from atlas_eval import runner as eval_runner
    from atlas_eval.corpus import corpus_path
    from atlas_eval.corpus import save_corpus as real_save

    fake_redis.fail = True
    monkeypatch.setattr(nodes, "search", nodes.search)
    monkeypatch.setattr(budget, "get_client", budget.get_client)

    async def fake_live(
        query: str, max_results: int | None = None, include_content: bool = False
    ) -> list[Any]:
        from atlas.graph.state import Source

        return [Source(url="https://a.example", title="A", snippet="s", content="page")]

    monkeypatch.setattr(record, "live_search", fake_live)
    monkeypatch.setattr(record, "save_corpus", lambda corpus: real_save(corpus, tmp_path))
    monkeypatch.setattr(record, "corpus_path", lambda item_id: corpus_path(item_id, tmp_path))
    monkeypatch.setattr(nodes, "call_model", _spending_model(50))
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))
    before = sum(eval_runner._BUDGET.spent.values())

    await record.record_corpus([load_dataset().items[0]], passes=1)

    # One researcher and one synthesis at 50 tokens each, counted in memory.
    assert sum(eval_runner._BUDGET.spent.values()) - before == 100


# --- the domain -------------------------------------------------------------------------
#
# Production defaults to the compliance domain. The general dataset was baselined
# on the general domain, and a run that took its domain from the environment
# would score the compliance prompts against that baseline: a "regression" made
# entirely of a missing ATLAS_DOMAIN.


@pytest.fixture
def compliance_environment(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """ATLAS_DOMAIN=compliance, as a production .env sets it; returns system prompts."""
    monkeypatch.setenv("ATLAS_DOMAIN", "compliance")
    get_settings.cache_clear()
    systems: list[str] = []

    async def recording(
        node: str,
        messages: list[Any],
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        systems.append(str(messages[0].content))
        return AIMessage(content="answer")

    monkeypatch.setattr(nodes, "call_model", recording)
    monkeypatch.setattr(nodes, "call_structured", make_call_structured(make_plan(1)))
    return systems


async def test_a_general_eval_run_is_the_general_domain_whatever_the_environment(
    offline: Path, compliance_environment: list[str]
) -> None:
    from atlas.domains import GENERAL

    await run_item(load_dataset().items[0], corpus_root=offline)

    assert compliance_environment == [GENERAL.researcher, GENERAL.synthesizer]


async def test_so_is_a_general_single_call_run(
    offline: Path, compliance_environment: list[str]
) -> None:
    from atlas.domains import GENERAL
    from atlas_eval.baselines import run_single_call

    await run_single_call(load_dataset().items[0], corpus_root=offline)

    assert compliance_environment == [GENERAL.synthesizer]
