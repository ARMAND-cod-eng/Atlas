"""HTTP routes.

The API no longer executes runs — it enqueues them and reads their results. That
is the point of Phase 3: a 67-second run must not be owned by a connection that
can drop at second 40.

The health probes are module-level functions rather than inline logic so tests
can substitute them without standing up Postgres or Redis.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import uuid4

from arq.connections import ArqRedis
from arq.jobs import Job
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import text

from atlas import eventlog
from atlas.db.session import get_engine
from atlas.events import RUN_FAILED
from atlas.result import QueryResponse
from atlas.worker import QUERY_WAIT_SECONDS, RUN_RESEARCH

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


class RunAccepted(BaseModel):
    """What ``POST /runs`` returns immediately, before any work happens.

    ``run_id`` identifies this execution and keys its event stream; ``thread_id``
    identifies the conversation and keys the checkpointer. Different lifetimes —
    one thread accumulates many runs — which is why both are returned.
    """

    run_id: str
    thread_id: str
    events_url: str


class RunStatus(BaseModel):
    """Where a run has got to, and its result once there is one."""

    run_id: str
    status: Literal["running", "completed", "failed"]
    result: QueryResponse | None = None
    # The error the run ended on, when status is "failed".
    error: str | None = None


class HealthResponse(BaseModel):
    """Liveness of Atlas and each backing service."""

    status: str
    db: bool
    redis: bool


async def check_db(request: Request) -> bool:
    """Round-trip a trivial query against Postgres.

    Goes through the SQLAlchemy engine rather than the checkpointer's pool,
    because the API no longer owns that pool — the worker does. Same database,
    same answer to the question the probe is asking.
    """
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
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


async def enqueue_run(request: Request, payload: QueryRequest) -> tuple[Job, str, str]:
    """Put one run on the queue. Returns ``(job, run_id, thread_id)``."""
    arq: ArqRedis = request.app.state.arq
    run_id = str(uuid4())
    thread_id = payload.thread_id or str(uuid4())
    # Register before enqueueing. A client can subscribe the instant this call
    # returns, which is sooner than the worker can write its first event.
    await eventlog.register(run_id)
    # _job_id ties the arq job to the run, so the job's result and the run's
    # event stream are addressable by the same identifier.
    job = await arq.enqueue_job(RUN_RESEARCH, run_id, payload.query, thread_id, _job_id=run_id)
    if job is None:
        # arq returns None when a job with this id already exists. run_id is a
        # fresh uuid, so this means Redis is not in the state we believe it is —
        # better to say so than to hand back a run_id nothing will execute.
        logger.error("enqueue returned no job", extra={"run_id": run_id})
        raise HTTPException(status_code=503, detail="could not enqueue the run")
    logger.info("run enqueued", extra={"run_id": run_id, "thread_id": thread_id})
    return job, run_id, thread_id


@router.post("/runs", response_model=RunAccepted, status_code=202)
async def start_run(request: Request, payload: QueryRequest) -> RunAccepted:
    """Start a run and return immediately.

    202 rather than 200: nothing has been researched yet. The caller subscribes
    to ``events_url`` to watch it happen, and may drop and reconnect freely —
    the run is not tied to that connection.
    """
    _job, run_id, thread_id = await enqueue_run(request, payload)
    return RunAccepted(run_id=run_id, thread_id=thread_id, events_url=f"/runs/{run_id}/events")


@router.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    """The full result of a run.

    ``run_completed`` carries a summary only, so this is where a client fetches
    the complete sources list and per-call metrics — once, rather than having
    them replayed on every reconnect.
    """
    result = await eventlog.load_result(run_id)
    if result is not None:
        return RunStatus(run_id=run_id, status="completed", result=QueryResponse(**result))

    # No stored result does not mean "still running". A result is stored only on
    # success, so a failed run used to land in the branch below and report
    # "running" for as long as its event log lived — contradicting the
    # run_failed its own stream had already delivered. The terminal event is
    # written on every path, so it decides.
    terminal = await eventlog.terminal_event(run_id)
    if terminal is not None and terminal.type == RUN_FAILED:
        error = terminal.payload.get("error")
        return RunStatus(
            run_id=run_id, status="failed", error=error if isinstance(error, str) else None
        )
    if terminal is not None:
        # Completed, but store_result failed open. Say so rather than "running".
        return RunStatus(run_id=run_id, status="completed")

    if await eventlog.exists(run_id):
        return RunStatus(run_id=run_id, status="running")
    raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Run a query and wait for the answer.

    Unchanged in shape since Phase 0, for clients that want one request and one
    answer rather than a stream. (The eval suite does not use it — it runs the
    graph in-process against a frozen corpus; see ``atlas_eval.runner``.) It is
    now a thin wrapper over the same queue the streaming path uses, so there is
    one execution path rather than two.

    The cost, stated plainly: this call's latency now includes queue pickup, and
    its throughput is bounded by the worker's ``max_jobs``.

    A run that fails re-raises the worker's exception here, so a ProviderError
    still reaches the 502 handler exactly as it did when the graph ran in-process.
    """
    job, run_id, _thread_id = await enqueue_run(request, payload)
    try:
        result: dict[str, Any] = await job.result(timeout=QUERY_WAIT_SECONDS)
    except TimeoutError:
        logger.error("query timed out waiting for the worker", extra={"run_id": run_id})
        raise HTTPException(status_code=504, detail="the run did not complete in time") from None
    return QueryResponse(**result)
