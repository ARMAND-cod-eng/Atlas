"""FastAPI application: lifespan, exception handlers, CORS.

Since Phase 3 this process does not execute runs. It enqueues them and relays
their events, so the clients it holds are the Redis one and the arq pool — both
built once at startup, because building either per-request would open a
connection pool per request.

``loop_factory``, ``JsonFormatter`` and ``configure_logging`` live in
:mod:`atlas.runtime` and are re-exported here: the worker needs them too, and the
API imports the worker, so the worker cannot import back. The re-export keeps the
documented ``--loop atlas.api.main:loop_factory`` argument resolving.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import ArqRedis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from atlas import eventlog
from atlas.api.routes import router
from atlas.api.streams import router as streams_router
from atlas.api.topology import router as topology_router
from atlas.config import get_settings
from atlas.db.session import dispose_engine, get_engine
from atlas.errors import ProviderError, SearchError
from atlas.runtime import JsonFormatter, configure_logging, loop_factory
from atlas.store import PostgresRunStore
from atlas.worker import redis_settings

# Re-exported for `--loop atlas.api.main:loop_factory` and for tests that import
# the formatter from here. Listed explicitly so the linter does not read them as
# unused imports.
__all__ = ["JsonFormatter", "app", "configure_logging", "create_app", "loop_factory"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources on startup, release them on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = logging.getLogger(__name__)

    log.info("starting Atlas", extra={"primary_model": settings.primary_model})

    # No graph here since Phase 3. The API enqueues runs and relays their events;
    # the worker owns the graph and its Postgres pool. That is what makes this
    # process stateless enough to restart without killing a 67-second run.
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    arq: ArqRedis = await create_pool(redis_settings())

    app.state.redis = redis
    app.state.arq = arq
    # Reads and writes rows only; the worker creates the table. See atlas.store.
    app.state.store = PostgresRunStore(get_engine())
    log.info("startup complete")

    try:
        yield
    finally:
        # Unconditional teardown: a failure closing one resource must not strand
        # the others.
        await arq.aclose()
        await redis.aclose()
        await eventlog.close_client()
        await dispose_engine()
        log.info("shutdown complete")


# Starlette types every handler as Callable[[Request, Exception], Response], so
# these take Exception and narrow with isinstance. That is what keeps the module
# free of `type: ignore` — a cast would silence the checker without proving
# anything, whereas the isinstance is a real guarantee at runtime.


async def provider_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """502, naming the provider and the node that was running.

    A provider being down is an upstream failure, not a bug in Atlas — 502 is the
    honest status. The body is what matters: Phase 0's bare 500 with an empty body
    turned every one-step diagnosis into a ten-step one.
    """
    if not isinstance(exc, ProviderError):
        return await unhandled_error_handler(request, exc)
    logging.getLogger(__name__).error(
        "provider error",
        extra={"provider": exc.provider, "node": exc.node, "detail": str(exc)},
    )
    return JSONResponse(
        status_code=502,
        content={
            "error": "provider_error",
            "provider": exc.provider,
            "node": exc.node,
            "detail": str(exc),
        },
    )


async def search_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """502 for a search backend failure.

    Reached only when a search fails outside a researcher — inside one it is
    caught and recorded on the ResearchResult, because a single failed branch
    must never fail the run.
    """
    if not isinstance(exc, SearchError):
        return await unhandled_error_handler(request, exc)
    logging.getLogger(__name__).error(
        "search error", extra={"query": exc.query, "detail": str(exc)}
    )
    return JSONResponse(
        status_code=502,
        content={
            "error": "search_error",
            "provider": "tavily",
            "detail": str(exc),
            "query": exc.query,
        },
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """500, but with a body and a logged traceback.

    An unhandled exception is a bug rather than a known failure mode, so the
    response says so plainly instead of leaking internals to the client. The
    traceback goes to the log, where it belongs.
    """
    logging.getLogger(__name__).exception("unhandled error serving %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": f"{type(exc).__name__} — see server logs",
        },
    )


def create_app() -> FastAPI:
    """Build the application."""
    app = FastAPI(
        title="Atlas",
        description="Multi-agent research platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(ProviderError, provider_error_handler)
    app.add_exception_handler(SearchError, search_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.add_middleware(
        CORSMiddleware,
        # Both spellings of the dev origin. They are distinct origins to the
        # browser, and which one the user has in the address bar is not something
        # the API should care about.
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(streams_router)
    app.include_router(topology_router)
    return app


app = create_app()
