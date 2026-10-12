"""Graph assembly and compilation.

Two entry points, deliberately:

* :func:`build_state_graph` is pure assembly — no I/O, no database. It exists so
  the graph's shape can be tested without standing up Postgres.
* :func:`build_graph` is the async factory the application uses. It owns the
  checkpointer's connection pool, which is why it must be awaited and why the
  caller has to close what it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from atlas.config import get_settings
from atlas.graph.nodes import (
    CRITIC,
    PLANNER,
    RESEARCHER,
    SYNTHESIZER,
    critic,
    fan_out,
    planner,
    researcher,
    route_after_critic,
    synthesizer,
)
from atlas.graph.state import (
    AtlasState,
    CriticVerdict,
    ResearchResult,
    Source,
    Subtask,
    SubtaskVerdict,
)
from atlas.metrics import CriticRound, LLMCall, NodeTiming, RunMetrics

# Every Atlas type that can sit in graph state, and so in a checkpoint. On load
# the serializer rebuilds only types it is allowed to, and an unlisted one does
# not raise — it comes back as a plain dict with a log line, and the next run on
# the thread dies in the first reducer that reads ``.subtask_id`` off it. LangGraph
# allows everything today with a deprecation warning and has said strict is
# coming; listing them makes the behaviour explicit and keeps it when it does.
# Types, not import-path strings, so a rename breaks the import rather than the
# allowlist. Nested models are listed too, in case one is ever stored alone.
CHECKPOINT_TYPES: tuple[type, ...] = (
    Subtask,
    ResearchResult,
    Source,
    CriticVerdict,
    SubtaskVerdict,
    RunMetrics,
    LLMCall,
    NodeTiming,
    CriticRound,
)


def checkpoint_serde() -> JsonPlusSerializer:
    """The serializer every checkpointer in Atlas uses — production, eval, tests."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


# These three are not tunables — the checkpointer misbehaves without them.
#
# autocommit + prepare_threshold=0 keep psycopg out of pipeline mode; on a pooled
# connection the saver otherwise fails with "cannot send pipeline when not in
# pipeline mode" once requests overlap (langchain-ai/langgraph#3193).
# dict_row is required outright: the saver reads result rows by column name.
_CONNECTION_KWARGS: dict[str, Any] = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 10


@dataclass(frozen=True, slots=True)
class AtlasGraph:
    """A compiled graph together with the pool that backs its checkpointer.

    The two travel together because the graph is unusable once the pool closes,
    so tying their lifetimes to one object makes it hard to leak the pool.
    """

    graph: CompiledStateGraph[AtlasState, Any, AtlasState, AtlasState]
    pool: AsyncConnectionPool[AsyncConnection[DictRow]]

    async def aclose(self) -> None:
        """Close the checkpointer's connection pool."""
        await self.pool.close()


def build_state_graph(
    *, critic_enabled: bool = True
) -> StateGraph[AtlasState, Any, AtlasState, AtlasState]:
    """Assemble the (uncompiled) research graph.

    ``START -> planner -> [fan_out] -> researcher (parallel) -> synthesizer ->
    critic -> END``, with the critic able to send work back to specific
    researchers or to the synthesizer.

    Kept separate from :func:`build_graph` so tests can compile this against an
    in-memory checkpointer and assert on the graph's shape without a database.

    ``critic_enabled=False`` ends at the synthesizer. It exists for the eval's
    comparison runs — measuring what the review loop buys — and production never
    sets it.
    """
    builder: StateGraph[AtlasState, Any, AtlasState, AtlasState] = StateGraph(AtlasState)
    builder.add_node(PLANNER, planner)
    builder.add_node(RESEARCHER, researcher)
    builder.add_node(SYNTHESIZER, synthesizer)
    if critic_enabled:
        builder.add_node(CRITIC, critic)

    builder.add_edge(START, PLANNER)
    # The conditional edge is where parallelism comes from: fan_out returns one
    # Send per subtask and LangGraph runs them concurrently. path_map lists both
    # possible destinations, because fan_out short-circuits to the synthesizer
    # when there is nothing to research.
    builder.add_conditional_edges(PLANNER, fan_out, [RESEARCHER, SYNTHESIZER])
    # Every researcher branch converges here. The reducers on research_results and
    # run_metrics are what let concurrent branches merge instead of colliding —
    # and on a retry, merge_research_results replaces rather than duplicates.
    builder.add_edge(RESEARCHER, SYNTHESIZER)
    if not critic_enabled:
        builder.add_edge(SYNTHESIZER, END)
        return builder
    builder.add_edge(SYNTHESIZER, CRITIC)
    # This closes the loop. Three destinations: back to specific researchers for a
    # targeted retry, back to the synthesizer when only the write-up was faulted,
    # or END on acceptance or exhaustion.
    builder.add_conditional_edges(CRITIC, route_after_critic, [RESEARCHER, SYNTHESIZER, END])
    return builder


def recursion_limit() -> int:
    """A ceiling sized from config, not left at LangGraph's default 25.

    Worst case per iteration is one superstep each for the fan-out, synthesis and
    critique, plus the planner once. Doubling gives slack for the synthesis-only
    path without letting a routing bug spin quietly — an explicit limit turns that
    into a clear error instead of 25 silent supersteps.
    """
    settings = get_settings()
    per_iteration = 3
    return 2 * (1 + per_iteration * (settings.max_critic_iterations + 1))


async def build_graph() -> AtlasGraph:
    """Open a checkpointer-backed pool and compile the graph against it.

    Async because the Postgres checkpointer needs to open connections and create
    its tables before the graph can run — none of which can happen at import time.
    The returned object owns a live pool; call ``aclose()`` on shutdown.
    """
    settings = get_settings()

    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=settings.database_url,
        kwargs=_CONNECTION_KWARGS,
        min_size=_POOL_MIN_SIZE,
        max_size=_POOL_MAX_SIZE,
        open=False,  # opening is async; do it explicitly below
    )
    await pool.open(wait=True)

    checkpointer = AsyncPostgresSaver(pool, serde=checkpoint_serde())
    # Idempotent DDL — creates the checkpoint tables on first run, no-ops after.
    await checkpointer.setup()

    compiled = build_state_graph().compile(checkpointer=checkpointer)
    return AtlasGraph(graph=compiled, pool=pool)
