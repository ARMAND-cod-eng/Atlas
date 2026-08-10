"""Shared test fixtures.

Two rules hold across this suite: no test touches the network, and no test spends
money. Every model is a fake, every credential below is fictional.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator, Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from atlas import budget
from atlas.config import get_settings
from atlas.graph.state import AtlasState, CriticVerdict, PlannerOutput, Subtask
from atlas.metrics import RunMetrics

FAKE_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "TAVILY_API_KEY": "tvly-test-not-a-real-key",
    "DATABASE_URL": "postgresql://atlas:atlas@localhost:5432/atlas",
    "REDIS_URL": "redis://localhost:6379/0",
}


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Populate a complete fake environment and isolate the settings cache.

    Environment variables outrank the .env file in pydantic-settings, so these win
    even on a machine with a real .env present. Clearing the lru_cache on both
    sides is what stops one test's settings leaking into the next.
    """
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeRedis:
    """An in-memory stand-in for the budget counter's Redis client.

    Only the three calls ``atlas.budget`` makes. ``fail`` flips every operation to
    raising, which is how the fail-open path is tested without taking Redis down.
    """

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self.fail = False

    def _guard(self) -> None:
        if self.fail:
            raise ConnectionError("redis is down")

    async def incrby(self, key: str, amount: int) -> int:
        self._guard()
        self.values[key] = self.values.get(key, 0) + amount
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self._guard()
        self.expiries[key] = seconds
        return True

    async def get(self, key: str) -> str | None:
        self._guard()
        value = self.values.get(key)
        return None if value is None else str(value)


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Point the budget counter at memory instead of a socket.

    Autouse rather than opt-in: the critic node checks the budget on every graph
    run, so without this any graph test would try to reach localhost:6379. An
    accidental network call is exactly what this suite is not allowed to make.
    """
    client = FakeRedis()
    monkeypatch.setattr(budget, "get_client", lambda: client)
    return client


# The two fake signatures mirror atlas.models exactly. They are spelled out rather
# than **kwargs on purpose: when a real signature gains a parameter — as
# call_model did with max_tokens — every fake should fail loudly, not swallow it.
FakeModel = Callable[..., Coroutine[Any, Any, AIMessage]]
FakeStructured = Callable[..., Coroutine[Any, Any, BaseModel]]


def make_plan(count: int) -> PlannerOutput:
    """A plan of ``count`` subtasks with ids ``s0..s{count-1}``."""
    return PlannerOutput(
        subtasks=[Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(count)]
    )


def make_call_model(content: str = "findings") -> FakeModel:
    """A stand-in for ``atlas.models.call_model``."""

    async def fake_model(
        node: str,
        messages: object,
        metrics: RunMetrics,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AIMessage:
        return AIMessage(content=content)

    return fake_model


def make_call_structured(
    plan: PlannerOutput,
    verdicts: Sequence[CriticVerdict] | None = None,
) -> FakeStructured:
    """A stand-in for ``atlas.models.call_structured``, dispatching on schema.

    One fake serves both structured callers. Verdicts are consumed in order and
    the last one repeats, so a graph test states the critic's decisions as a list
    and does not have to count calls. Defaults to accepting once — most tests care
    about something other than the critic and should not loop.
    """
    remaining = list(verdicts if verdicts is not None else [CriticVerdict(overall="accept")])

    async def fake_structured(
        node: str,
        messages: object,
        schema: type[BaseModel],
        metrics: RunMetrics,
        temperature: float | None = None,
    ) -> BaseModel:
        if schema is CriticVerdict:
            return remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return plan

    return fake_structured


def initial_state(query: str = "what is retrieval-augmented generation?") -> AtlasState:
    """A fresh graph input, matching what the /query route builds."""
    return {
        "query": query,
        "subtasks": [],
        "research_results": [],
        "answer": "",
        "messages": [],
        "run_metrics": RunMetrics(),
        "iteration": 0,
        "critic_feedback": [],
    }
