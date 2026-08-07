"""FastAPI application: lifespan, logging, CORS.

The graph and the Redis client are built once at startup and shared for the
process lifetime. Building either per-request would open a connection pool per
request, which is the kind of thing that looks fine until it is under load.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from atlas.api.routes import router
from atlas.config import get_settings
from atlas.db.session import dispose_engine
from atlas.graph.builder import build_graph

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


def create_app() -> FastAPI:
    """Build the application."""
    app = FastAPI(
        title="Atlas",
        description="Multi-agent research platform",
        version="0.1.0",
        lifespan=lifespan,
    )
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
