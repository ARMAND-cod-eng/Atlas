"""The arq worker that executes runs.

Why a worker at all: a run takes ~67 seconds. Executing it inside the HTTP
request means a client that disconnects at 40s destroys the run and the tokens
already spent. Moving execution here makes the API stateless — the run survives a
dropped connection, a backgrounded mobile tab, and an API deploy.

Why arq rather than Celery: Atlas is async end to end — ``AsyncPostgresSaver``,
``redis.asyncio``, async nodes, async provider clients. Celery is sync-first, so
each task would spin up an event loop to call into an async graph, and it would
add a second broker beside a Redis we already run. arq's tasks *are* coroutines
and its only dependency is that same Redis. It is also the difference between
working and not working on Windows, where Celery's prefork pool is unsupported.

The graph is built once in :func:`startup` and reused for every job. Building it
per job would open a Postgres pool per run.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from arq.connections import RedisSettings
from arq.cron import cron
from arq.typing import WorkerSettingsBase
from arq.worker import Worker, create_worker, func

from atlas import eventlog
from atlas.config import get_settings
from atlas.db.session import dispose_engine, get_engine
from atlas.domains import DomainName
from atlas.graph.builder import build_graph
from atlas.offline import install_offline
from atlas.retention import forget_thread, purge_expired
from atlas.runner import execute_revision, execute_run
from atlas.runtime import configure_logging
from atlas.store import PostgresRunStore

logger = logging.getLogger(__name__)

# The job name the API enqueues. A constant so a rename cannot silently
# disconnect the producer from the consumer.
RUN_RESEARCH = "run_research"
RUN_REVISION = "run_revision"
FORGET_THREAD = "forget_thread"

# Ceiling on a single run. A run is bounded by MAX_CRITIC_ITERATIONS and the token
# budget, but a hung provider connection is not. Generous against the 67s
# baseline: this is a stuck-run guard, not a deadline.
JOB_TIMEOUT_SECONDS = 600

# How long POST /query waits for a result. The job's own ceiling plus slack for
# queue pickup, so a 504 here means the queue is wedged rather than that the run
# was merely slow.
QUERY_WAIT_SECONDS = JOB_TIMEOUT_SECONDS + 60


def redis_settings() -> RedisSettings:
    """arq's connection config, derived from the one REDIS_URL everything uses."""
    return RedisSettings.from_dsn(get_settings().redis_url)


async def run_research(
    ctx: dict[str, Any],
    run_id: str,
    query: str,
    thread_id: str,
    parent_run_id: str | None = None,
    *,
    domain: DomainName | None = None,
) -> dict[str, Any]:
    """Execute one research run, streaming its events to the log."""
    return await execute_run(
        run_id,
        query,
        thread_id,
        ctx["graph"].graph,
        parent_run_id=parent_run_id,
        store=ctx["store"],
        domain=domain,
    )


async def run_revision(
    ctx: dict[str, Any],
    run_id: str,
    thread_id: str,
    parent_run_id: str,
    subtask_id: str,
    question: str,
    *,
    domain: DomainName | None = None,
) -> dict[str, Any]:
    """Re-run one edited branch of a finished run. See atlas.revision."""
    return await execute_revision(
        run_id,
        thread_id,
        ctx["graph"].graph,
        parent_run_id=parent_run_id,
        subtask_id=subtask_id,
        question=question,
        store=ctx["store"],
        domain=domain,
    )


async def forget_thread_job(ctx: dict[str, Any], thread_id: str) -> list[str]:
    """Delete a thread everywhere it lives. See atlas.retention."""
    return await forget_thread(thread_id, ctx["store"], ctx["graph"].checkpointer)


async def purge_expired_job(ctx: dict[str, Any]) -> list[str]:
    """Nightly: forget threads idle past RETENTION_DAYS."""
    purged = await purge_expired(
        ctx["store"],
        ctx["graph"].checkpointer,
        get_settings().retention_days,
        datetime.now(UTC),
    )
    logger.info("retention purge", extra={"threads": len(purged)})
    return purged


async def startup(ctx: dict[str, Any]) -> None:
    """Build the shared graph and its Postgres pool, once per worker process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("worker starting", extra={"primary_model": settings.primary_model})
    if settings.atlas_offline:
        install_offline()
        logger.warning("offline mode: every model and the web are scripted, and answers say so")
    ctx["graph"] = await build_graph()
    # The worker owns the schema, as it owns the checkpointer's: the API only
    # reads and writes rows, and two processes racing CREATE TABLE at boot can
    # collide on Postgres's own catalog.
    store = PostgresRunStore(get_engine())
    await store.setup()
    ctx["store"] = store
    logger.info("worker ready")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Release everything the worker owns."""
    graph = ctx.get("graph")
    if graph is not None:
        await graph.aclose()
    await eventlog.close_client()
    await dispose_engine()
    logger.info("worker shutdown complete")


class WorkerSettings(WorkerSettingsBase):
    """arq's static configuration. Referenced by ``atlas.worker_main``.

    Inherits arq's protocol so ``create_worker`` type-checks structurally.
    Inheritance is safe despite arq reading ``settings_cls.__dict__``: inherited
    attributes do not appear there, so only what is defined below is passed on.

    Only genuinely static values live here. arq reads this class through
    ``settings_cls.__dict__``, so a ``property`` would be handed to ``Worker()``
    as the property object itself rather than its value — anything derived from
    settings is passed as a keyword argument at construction instead, which also
    means it is read at process start rather than at import.

    ``handle_signals`` is left at its default ``True``, which matters on Windows
    for a non-obvious reason. arq's ``_add_signal_handler`` already catches the
    ``NotImplementedError`` that ``loop.add_signal_handler`` raises there, so the
    default degrades gracefully. Setting it to ``False`` looks like the
    Windows-friendly choice and is exactly the opposite: ``Worker.close()`` then
    runs ``handle_sig(signal.SIGUSR1)``, and Windows has no ``SIGUSR1``, so
    shutdown dies with an ``AttributeError``.
    """

    # A tuple, not a list: arq only reads it, and a mutable class attribute is a
    # shared-state footgun the linter is right to flag.
    functions = (run_research, run_revision, func(forget_thread_job, name=FORGET_THREAD))
    # 03:00 every night, when a deployment is least likely to be busy.
    cron_jobs = (cron(purge_expired_job, hour={3}, minute={0}),)
    on_startup = staticmethod(startup)
    on_shutdown = staticmethod(shutdown)

    job_timeout = JOB_TIMEOUT_SECONDS

    # Retrying is wrong here. A failed run has already spent tokens, and its
    # failure is reported to the client as run_failed; a silent retry would
    # double the cost of a genuinely broken query and confuse a client that has
    # already seen a terminal event.
    max_tries = 1

    # Long enough for a client to fetch a result after a slow reconnect.
    keep_result = 3600


def build_worker() -> Worker:
    """Construct the worker with the settings-derived values filled in."""
    return create_worker(
        WorkerSettings,
        redis_settings=redis_settings(),
        max_jobs=get_settings().worker_max_jobs,
    )
