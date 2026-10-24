"""Runs dataset items through the real graph, in-process, against frozen search.

In-process rather than through ``POST /query``: the eval scores the *agent*, and
the queue, SSE relay and HTTP layer are Phase 3's concerns, with tests of their
own. Driving the endpoint would mean booting four processes to exercise code this
suite does not score, and paying ~1.0s of queue pickup fifty times over. With
``MemorySaver`` and a frozen corpus the suite needs no services at all, which is
what makes the CI job simple enough to trust.

The graph itself is untouched — same nodes, same prompts, same critic loop. Only
``search`` is substituted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from atlas import budget
from atlas.budget import CounterStore
from atlas.domains import DomainName
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, checkpoint_serde, recursion_limit
from atlas.graph.state import ResearchResult, Source, Subtask
from atlas.metrics import RunMetrics
from atlas.result import build_response, validate_list, validate_metrics
from atlas.runner import initial_state
from atlas_eval.corpus import FrozenSearch, load_corpus
from atlas_eval.dataset import EvalItem

logger = logging.getLogger(__name__)

# The frozen search for the item currently executing. A ContextVar rather than a
# module global because items run concurrently: asyncio tasks inherit the context
# they were created in, so each run sees its own corpus. A plain global would let
# five parallel items answer each other's searches — the kind of bug that does not
# crash, it just quietly scores the wrong thing.
# Which system produced a run. "atlas" is production; the other two exist only to
# measure it against: the same graph without the critic, and one search answered
# by the same synthesizer (atlas_eval.baselines).
Variant = Literal["atlas", "no-critic", "single-call"]

_ACTIVE: ContextVar[FrozenSearch | None] = ContextVar("atlas_eval_search", default=None)


async def _dispatch(
    query: str, max_results: int | None = None, include_content: bool = False
) -> list[Source]:
    """Stands in for ``atlas.graph.nodes.search`` for the whole eval process."""
    active = _ACTIVE.get()
    if active is None:
        raise RuntimeError(
            "search was called outside an eval run — the frozen corpus is not "
            "installed, and the eval must never reach the network"
        )
    return await active.search(query, max_results, include_content)


class InMemoryBudget:
    """The token-budget counter, kept in this process instead of Redis.

    The eval has no Redis — not in CI, often not on a laptop — and against an
    absent Redis the budget fails open on every call. Right for production,
    wrong here: every eval run reported ``budget_check_unavailable``, and a run
    that blew its budget was never stopped. On exactly the most expensive runs
    the eval measured a different system from the one that ships.

    Keyed by run id like the real one, so concurrent items never share a count.
    No expiry: nothing outlives the process.
    """

    def __init__(self) -> None:
        self.spent: dict[str, int] = {}

    async def incrby(self, name: str, amount: int = 1) -> int:
        self.spent[name] = self.spent.get(name, 0) + amount
        return self.spent[name]

    async def expire(self, name: str, time: int) -> bool:
        return True

    async def get(self, name: str) -> str | None:
        value = self.spent.get(name)
        return None if value is None else str(value)


_BUDGET = InMemoryBudget()


def _offline_budget() -> CounterStore:
    return _BUDGET


def install_offline_runtime() -> None:
    """Point the graph's two external services at in-process stand-ins.

    Search goes to the frozen corpus, the budget counter to memory. Installed on
    every run, not once per process: a once-only flag meant that anything which
    restored ``nodes.search`` — a test fixture, a debugging session — left every
    later run in the process calling *live* Tavily, with no error to say so.
    """
    nodes.search = _dispatch
    install_offline_budget()


def install_offline_budget() -> None:
    """Back the token budget with this process's memory instead of Redis.

    Used alone by recording, which runs the live agent against live search but
    needs the budget enforced as much as replay does — and should not depend on
    a Redis being up to get it.
    """
    budget.get_client = _offline_budget


class RunRecord(BaseModel):
    """Everything one scored run produced.

    Persisted so scoring, calibration and the gate all read the same artefact
    rather than each re-running the agent.
    """

    item_id: str
    query: str
    answer: str
    variant: Variant = "atlas"
    subtasks: list[Subtask] = Field(default_factory=list)
    research_results: list[ResearchResult] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    iterations: int = 0
    # Every search string the agent issued, in order. Tool-call validity reads
    # this to find redundant duplicate searches.
    search_calls: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    # Set when the run raised. A failed run is scored as zero rather than dropped
    # — dropping it would quietly change the denominator.
    error: str | None = None

    @property
    def retrieved_context(self) -> str:
        """Every retrieved snippet, for the grounding scorer to verify against."""
        blocks = []
        for result in self.research_results:
            for source in result.sources:
                blocks.append(f"[{source.url}] {source.title}\n{source.snippet}")
        return "\n\n".join(blocks)


async def run_item(
    item: EvalItem,
    corpus_root: Path | None = None,
    critic_enabled: bool = True,
    domain: DomainName = "general",
) -> RunRecord:
    """Execute one dataset item and capture what it produced.

    The domain is the dataset's, never the environment's: production defaults to
    compliance, and the general dataset was baselined on the general domain.
    """
    variant: Variant = "atlas" if critic_enabled else "no-critic"
    install_offline_runtime()
    corpus = load_corpus(item.id) if corpus_root is None else load_corpus(item.id, corpus_root)
    frozen = FrozenSearch(corpus)
    token = _ACTIVE.set(frozen)

    graph = build_state_graph(critic_enabled=critic_enabled).compile(
        checkpointer=MemorySaver(serde=checkpoint_serde())
    )
    config: RunnableConfig = {
        # A fresh run id per execution, so re-running the suite within the budget
        # key's TTL does not start an item at the previous run's spend.
        "configurable": {
            "thread_id": f"eval-{item.id}",
            "run_id": str(uuid4()),
            "domain": domain,
        },
        "recursion_limit": recursion_limit(),
    }

    started = time.perf_counter()
    try:
        final = await graph.ainvoke(initial_state(item.query), config=config)
        elapsed = time.perf_counter() - started
        response = build_response(dict(final), thread_id=f"eval-{item.id}")
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer=response.answer,
            variant=variant,
            subtasks=response.subtasks,
            research_results=validate_list(final.get("research_results"), ResearchResult),
            sources=response.sources,
            metrics=validate_metrics(final.get("run_metrics")),
            iterations=response.iterations,
            search_calls=list(frozen.calls),
            duration_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.exception("eval run failed", extra={"item_id": item.id})
        # Recorded, not raised. One item blowing up must not abort a 50-item run,
        # and it must not silently shrink the denominator either.
        return RunRecord(
            item_id=item.id,
            query=item.query,
            answer="",
            variant=variant,
            search_calls=list(frozen.calls),
            duration_s=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _ACTIVE.reset(token)


async def run_items(
    items: list[EvalItem],
    concurrency: int = 5,
    run_one: Callable[[EvalItem], Awaitable[RunRecord]] = run_item,
) -> list[RunRecord]:
    """Run items concurrently, preserving dataset order in the output.

    Concurrency is bounded because each item is itself a five-branch fan-out; the
    real limit is the provider's rate limit, not ours.
    """
    limiter = asyncio.Semaphore(concurrency)

    async def guarded(item: EvalItem) -> RunRecord:
        async with limiter:
            return await run_one(item)

    return list(await asyncio.gather(*(guarded(item) for item in items)))
