"""Process-level concerns shared by the API and the worker: logging and the loop.

Extracted here in Phase 3 for a concrete reason rather than tidiness. The worker
needs the same JSON logging and the same Windows event-loop workaround as the
API, but the API imports the worker (to enqueue jobs), so the worker importing
``atlas.api.main`` back would be a cycle. Neither process owns these; they belong
to the runtime.

``atlas.api.main`` re-exports both names, so the documented
``--loop atlas.api.main:loop_factory`` argument keeps resolving.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any


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

    The worker hits the identical trap through arq's CLI, which is why
    ``atlas/worker_main.py`` exists and uses this too.

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
