"""Shared test fixtures.

Two rules hold across this suite: no test touches the network, and no test spends
money. Every model is a fake, every credential below is fictional.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterator, Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from atlas import budget, eventlog
from atlas.config import get_settings
from atlas.graph.state import AtlasState, CriticVerdict, PlannerOutput, Subtask
from atlas.metrics import RunMetrics
from atlas.runtime import loop_factory


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    """Run every async test on the event loop production runs on.

    On Windows that is not asyncio's default: psycopg's async driver refuses the
    ProactorEventLoop, so the API and worker build a SelectorEventLoop (see
    atlas.runtime.loop_factory). Tests that reach a real Postgres need the same,
    and the rest are better for matching production than for not.
    """
    return {"selector": loop_factory}


FAKE_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "TAVILY_API_KEY": "tvly-test-not-a-real-key",
    "DATABASE_URL": "postgresql://atlas:atlas@localhost:5432/atlas",
    "REDIS_URL": "redis://localhost:6379/0",
    # The general domain, as every test written before the compliance domain
    # assumes; tests of the compliance domain set it per run, in the config.
    "ATLAS_DOMAIN": "general",
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


StreamEntry = tuple[str, dict[str, str]]


class FakeRedis:
    """An in-memory stand-in for the Redis client, covering what Atlas calls.

    The stream half is a real implementation rather than a stub, and that is
    deliberate: ``XREAD BLOCK`` has to actually block and actually wake, or the
    heartbeat and live-tail tests assert nothing at all. A stub that returned
    immediately would make an idle stream indistinguishable from a busy one —
    which is the specific thing those tests exist to tell apart.

    ``fail`` flips every operation to raising, which is how the fail-open paths
    are tested without taking Redis down.
    """

    def __init__(self) -> None:
        self.values: dict[str, int | str] = {}
        self.expiries: dict[str, int] = {}
        self.streams: dict[str, list[StreamEntry]] = {}
        self.fail = False
        # Signalled on every append so a blocked reader wakes immediately rather
        # than sitting out its full timeout.
        self._appended = asyncio.Event()

    def _guard(self) -> None:
        if self.fail:
            raise ConnectionError("redis is down")

    # --- strings and counters -------------------------------------------------

    async def incrby(self, key: str, amount: int) -> int:
        self._guard()
        current = self.values.get(key, 0)
        total = int(current) + amount
        self.values[key] = total
        return total

    async def expire(self, key: str, seconds: int) -> bool:
        self._guard()
        self.expiries[key] = seconds
        return True

    async def get(self, key: str) -> str | None:
        self._guard()
        value = self.values.get(key)
        return None if value is None else str(value)

    async def set(
        self, key: str, value: str, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        self._guard()
        if nx and key in self.values:
            # Redis answers a refused SET NX with nil, not False.
            return None
        self.values[key] = value
        if ex is not None:
            self.expiries[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        self._guard()
        removed = 0
        for key in keys:
            # DEL removes a key of any type, streams included, as Redis does.
            if self.values.pop(key, None) is not None or self.streams.pop(key, None) is not None:
                removed += 1
            self.expiries.pop(key, None)
        return removed

    async def exists(self, key: str) -> int:
        self._guard()
        return int(key in self.values or key in self.streams)

    # --- streams --------------------------------------------------------------

    async def xadd(
        self,
        name: str,
        fields: dict[Any, Any],
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        self._guard()
        entries = self.streams.setdefault(name, [])
        entry_id = f"{len(entries) + 1}-0"
        entries.append((entry_id, {str(k): str(v) for k, v in fields.items()}))
        if maxlen is not None and len(entries) > maxlen:
            del entries[: len(entries) - maxlen]
        self._appended.set()
        self._appended = asyncio.Event()
        return entry_id

    async def xrange(self, name: str, min: str = "-", max: str = "+") -> list[StreamEntry]:
        self._guard()
        return list(self.streams.get(name, []))

    async def xread(
        self,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
    ) -> list[tuple[str, list[StreamEntry]]]:
        self._guard()
        name, last_id = next(iter(streams.items()))
        # "$" means "only what arrives after this call", so it resolves against
        # the stream as it stands right now.
        if last_id == "$":
            existing = self.streams.get(name, [])
            last_id = existing[-1][0] if existing else "0-0"

        new = self._after(name, last_id)
        if new:
            return [(name, new[:count] if count else new)]
        if not block:
            return []

        waiter = self._appended
        try:
            await asyncio.wait_for(waiter.wait(), timeout=block / 1000)
        except TimeoutError:
            return []
        self._guard()
        new = self._after(name, last_id)
        return [(name, new[:count] if count else new)] if new else []

    def _after(self, name: str, last_id: str) -> list[StreamEntry]:
        def order(entry_id: str) -> tuple[int, int]:
            ms, _, seq = entry_id.partition("-")
            return int(ms), int(seq or 0)

        cutoff = order(last_id)
        return [entry for entry in self.streams.get(name, []) if order(entry[0]) > cutoff]


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Point every Redis consumer at one in-memory client.

    Autouse rather than opt-in: the critic checks the budget on every graph run
    and the runner appends an event for every step, so without this any graph
    test would reach for localhost:6379. An accidental network call is exactly
    what this suite is not allowed to make.

    One client for both modules, as in production — the budget counter and the
    event log share a Redis.
    """
    client = FakeRedis()
    monkeypatch.setattr(budget, "get_client", lambda: client)
    monkeypatch.setattr(eventlog, "get_client", lambda: client)
    return client


class FakeJob:
    """An arq job whose result is already known."""

    def __init__(self, job_id: str, payload: dict[str, Any] | BaseException) -> None:
        self.job_id = job_id
        self._payload = payload

    # ASYNC109 wants a timeout handled by asyncio.timeout rather than a
    # parameter, but the name and shape here are arq's, not ours — this is a
    # double for Job.result and has to match what routes.py calls.
    async def result(self, timeout: float | None = None) -> dict[str, Any]:  # noqa: ASYNC109
        if isinstance(self._payload, BaseException):
            # arq re-raises a failed job's exception in the awaiting request.
            raise self._payload
        return self._payload


class FakeArq:
    """Stands in for the arq pool by running the job inline.

    Inline rather than deferred keeps the API tests deterministic: by the time
    ``enqueue_job`` returns, the run has happened and its events are in the log.
    The queue itself is arq's to test, not ours.

    ``handler`` runs research jobs; ``handlers`` maps any other job name to its
    own. Each call is recorded in ``calls`` as ``(function, *args)``.
    """

    def __init__(
        self,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] | None = None,
    ) -> None:
        self.handler = handler
        self.handlers = handlers or {}
        self.calls: list[tuple[object, ...]] = []

    async def enqueue_job(
        self, function: str, *args: object, _job_id: str | None = None
    ) -> FakeJob:
        self.calls.append((function, *args))
        handler = self.handlers.get(function, self.handler)
        try:
            payload = await handler(*args)
        except BaseException as exc:
            # Mirrors arq, which stores a failed job's exception rather than
            # letting it escape the enqueue call.
            return FakeJob(str(_job_id), exc)
        return FakeJob(str(_job_id), payload)


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


def run_config(thread_id: str, run_id: str | None = None) -> RunnableConfig:
    """Graph config for one run, shaped like the one ``execute_run`` builds.

    ``run_id`` scopes the token budget. Defaulted from the thread so a test that
    makes one run per thread needs to name nothing; a test of two runs on one
    thread must pass distinct ids, exactly as production does.
    """
    return {"configurable": {"thread_id": thread_id, "run_id": run_id or f"run-{thread_id}"}}


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
        "revision": None,
    }
