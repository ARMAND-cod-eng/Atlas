"""HTTP routes.

The health probes are module-level functions rather than inline logic so tests
can substitute them without standing up Postgres or Redis.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from langchain_core.runnables import RunnableConfig
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from atlas.graph.builder import AtlasGraph
from atlas.graph.state import AtlasState

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Inbound research request."""

    query: str = Field(min_length=1, description="The question to answer.")
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread to continue. Omit to start a new one; the "
            "generated id is returned so the next call can resume this thread."
        ),
    )


class QueryResponse(BaseModel):
    """Answer plus the thread it belongs to."""

    answer: str
    thread_id: str


class HealthResponse(BaseModel):
    """Liveness of Atlas and each backing service."""

    status: str
    db: bool
    redis: bool


async def check_db(request: Request) -> bool:
    """Round-trip a trivial query through the checkpointer's pool."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = request.app.state.graph.pool
    try:
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception:
        # A failed probe is a reportable state, not a server error — the endpoint
        # must still answer so a load balancer learns *which* dependency is down.
        logger.exception("database health check failed")
        return False
    return True


async def check_redis(request: Request) -> bool:
    """Ping Redis."""
    client: Redis = request.app.state.redis
    try:
        await client.ping()
    except Exception:
        logger.exception("redis health check failed")
        return False
    return True


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report real connectivity to each backing service.

    Always 200, even when a dependency is down — the body carries the detail.
    Returning 503 would collapse "Atlas is unreachable" and "Redis is unreachable"
    into the same signal, which is exactly the distinction this endpoint exists
    to make.
    """
    db_ok = await check_db(request)
    redis_ok = await check_redis(request)
    return HealthResponse(
        status="ok" if db_ok and redis_ok else "degraded",
        db=db_ok,
        redis=redis_ok,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Run a query through the graph and return the answer.

    A thread id is minted when the caller omits one, and always returned — the
    checkpointer keys persisted state by it, so it is how a follow-up call picks
    up where this one left off.
    """
    atlas_graph: AtlasGraph = request.app.state.graph
    thread_id = payload.thread_id or str(uuid4())

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # Typed as AtlasState rather than a bare dict so the graph's input contract is
    # checked here, at the boundary, instead of failing inside a node.
    initial: AtlasState = {"query": payload.query, "answer": "", "messages": []}

    logger.info("query received", extra={"thread_id": thread_id})
    result = await atlas_graph.graph.ainvoke(initial, config=config)

    return QueryResponse(answer=str(result["answer"]), thread_id=thread_id)
