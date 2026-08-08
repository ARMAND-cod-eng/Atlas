"""FastAPI application: lifespan, logging, CORS.

The graph and the Redis client are built once at startup and shared for the
process lifetime. Building either per-request would open a connection pool per
request, which is the kind of thing that looks fine until it is under load.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from atlas.api.routes import router
from atlas.config import get_settings
from atlas.db.session import dispose_engine
from atlas.errors import ProviderError, SearchError
from atlas.graph.builder import build_graph


def loop_factory() -> asyncio.AbstractEventLoop:
    """Build a SelectorEventLoop. Referenced by import path, for Windows only.

    psycopg's async driver refuses to run on ProactorEventLoop::

        InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in
        async mode.

    and uvicorn picks exactly that loop on Windows for any run without
    subprocesses (``uvicorn/loops/asyncio.py``: ``win32 and not use_subprocess ->
    ProactorEventLoop``). The checkpointer pool then never opens a connection and
    the failure surfaces as ``PoolTimeout: pool initialization incomplete after
    30.0 sec`` — which points at the database rather than at the event loop, and
    costs an afternoon to track down.

    Setting the event loop *policy* does not help: uvicorn 0.36+ passes an
    explicit ``loop_factory`` to ``asyncio.run``, bypassing the policy entirely.
    The supported override is ``--loop <module>:<attr>``, which is what
    ``atlas/__main__.py`` wires up. It also explains why ``--reload`` appeared to
    work: the reloader uses subprocesses, so uvicorn already chose Selector there.

    Not used off Windows — see ``atlas/__main__.py``, which keeps ``loop="auto"``
    there so uvloop is still picked up in the container.
    """
    return asyncio.SelectorEventLoop()


# Attributes the stdlib puts on every LogRecord. Anything outside this set was
# passed by the caller via `extra=` and belongs in the structured output.
_RESERVED_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object on one line.

    One object per line is what log shippers expect, and it keeps `extra=` fields
    queryable instead of buried in an interpolated string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        # default=str so an unserializable extra degrades to its repr rather than
        # raising inside the logging call and masking the original event.
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Install the JSON formatter on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared resources on startup, release them on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = logging.getLogger(__name__)

    log.info("starting Atlas", extra={"primary_model": settings.primary_model})

    atlas_graph = await build_graph()
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)

    app.state.graph = atlas_graph
    app.state.redis = redis
    log.info("startup complete")

    try:
        yield
    finally:
        # Unconditional teardown: a failure closing one resource must not strand
        # the others.
        await atlas_graph.aclose()
        await redis.aclose()
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
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
