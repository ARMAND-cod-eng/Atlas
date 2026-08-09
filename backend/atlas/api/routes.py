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
from atlas.graph.state import AtlasState, ResearchResult, Source, Subtask
from atlas.metrics import RunMetrics

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
    """Answer, the thread it belongs to, and how it was produced.

    Additive over Phase 0 — anything reading ``.answer`` keeps working. The rest
    exists because a research platform's citations are the product, and because a
    fan-out you cannot see is a fan-out you cannot debug. ``metrics.degraded``
    in particular means a partial run is visible from the client rather than only
    in server logs.
    """

    answer: str
    thread_id: str
    subtasks: list[Subtask] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)


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


def _validate_list[ModelT: BaseModel](raw: object, model: type[ModelT]) -> list[ModelT]:
    """Coerce a state value into typed models.

    Graph state round-trips through the checkpointer, which may hand back plain
    dicts rather than model instances. ``model_validate`` accepts either, so this
    is correct on a fresh run and on a resumed thread alike.
    """
    if not isinstance(raw, list):
        return []
    return [model.model_validate(item) for item in raw]


def _validate_metrics(raw: object) -> RunMetrics:
    """Coerce state's run_metrics, tolerating an absent or resumed value."""
    if raw is None:
        return RunMetrics()
    return RunMetrics.model_validate(raw)


def _dedupe_sources(results: list[ResearchResult]) -> list[Source]:
    """Flatten every researcher's sources, first occurrence of each URL winning.

    Parallel researchers frequently land on the same page, and a citation list
    that repeats one URL five times is noise.
    """
    seen: set[str] = set()
    sources: list[Source] = []
    for result in results:
        for source in result.sources:
            if source.url in seen:
                continue
            seen.add(source.url)
            sources.append(source)
    return sources


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
    initial: AtlasState = {
        "query": payload.query,
        "subtasks": [],
        "research_results": [],
        "answer": "",
        "messages": [],
        "run_metrics": RunMetrics(),
    }

    logger.info("query received", extra={"thread_id": thread_id})
    result = await atlas_graph.graph.ainvoke(initial, config=config)

    subtasks = _validate_list(result.get("subtasks"), Subtask)
    results = _validate_list(result.get("research_results"), ResearchResult)
    metrics = _validate_metrics(result.get("run_metrics"))

    logger.info(
        "query complete",
        extra={
            "thread_id": thread_id,
            "subtask_count": len(subtasks),
            "degraded": metrics.degraded,
            "total_tokens": metrics.total_tokens,
        },
    )

    return QueryResponse(
        answer=str(result["answer"]),
        thread_id=thread_id,
        subtasks=subtasks,
        sources=_dedupe_sources(results),
        metrics=metrics,
    )
