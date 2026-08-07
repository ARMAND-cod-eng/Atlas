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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from atlas.config import get_settings
from atlas.graph.nodes import answer_query
from atlas.graph.state import AtlasState

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


def build_state_graph() -> StateGraph[AtlasState, Any, AtlasState, AtlasState]:
    """Assemble the (uncompiled) graph: ``START -> answer_query -> END``.

    Kept separate from :func:`build_graph` so tests can compile this against an
    in-memory checkpointer and assert on the graph's shape without a database.
    """
    builder: StateGraph[AtlasState, Any, AtlasState, AtlasState] = StateGraph(AtlasState)
    builder.add_node("answer_query", answer_query)
    builder.add_edge(START, "answer_query")
    builder.add_edge("answer_query", END)
    return builder


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

    checkpointer = AsyncPostgresSaver(pool)
    # Idempotent DDL — creates the checkpoint tables on first run, no-ops after.
    await checkpointer.setup()

    compiled = build_state_graph().compile(checkpointer=checkpointer)
    return AtlasGraph(graph=compiled, pool=pool)
