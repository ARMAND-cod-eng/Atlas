"""Run the Atlas worker::

    uv run python -m atlas.worker_main

Not ``arq atlas.worker.WorkerSettings``, and not for style. arq's CLI calls
``asyncio.run``, which on Windows selects ``ProactorEventLoop``, which psycopg
refuses::

    InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode.

The worker builds the Postgres checkpointer, so it hits this immediately — and
the symptom is ``PoolTimeout: pool initialization incomplete``, which points at
the database rather than at the event loop. That is the same trap
``atlas/__main__.py`` documents for the API server, so this entrypoint reuses the
same escape hatch: build the loop ourselves and await ``Worker.async_run()``,
which arq exposes precisely so the caller can own the loop.

Off Windows this is a plain ``asyncio.run`` and nothing special happens.
"""

from __future__ import annotations

import asyncio
import sys

from atlas.runtime import loop_factory
from atlas.worker import build_worker


async def _run() -> None:
    worker = build_worker()
    try:
        await worker.async_run()
    finally:
        # async_run deliberately leaves connections open (it is arq's testing
        # seam), so closing is ours to do. This is also what runs on_shutdown,
        # which releases the graph's Postgres pool.
        await worker.close()


def main() -> None:
    """Start the worker on an event loop psycopg can actually use."""
    if sys.platform == "win32":
        loop = loop_factory()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
    else:
        asyncio.run(_run())


if __name__ == "__main__":
    main()
