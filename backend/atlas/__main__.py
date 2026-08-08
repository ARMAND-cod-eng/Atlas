"""Run the Atlas API server::

    uv run python -m atlas
    uv run python -m atlas --reload
    uv run python -m atlas --host 0.0.0.0 --port 8080

This entrypoint exists because plain ``uvicorn atlas.api.main:app`` is broken on
Windows: uvicorn selects ProactorEventLoop for non-subprocess runs, and psycopg's
async driver cannot run on it, so the checkpointer pool times out at startup. See
:func:`atlas.api.main.loop_factory` for the full explanation.

Running uvicorn directly still works if you pass the loop factory yourself::

    uvicorn atlas.api.main:app --loop atlas.api.main:loop_factory

Note that ``--reload`` watches the ``backend`` package only. The ``.env`` lives at
the repository root and is read once at startup, so **editing .env requires a
manual restart** — a running server keeps serving with the old values, and a bad
key surfaces later as a provider error rather than as anything pointing at config.
(Adding it to uvicorn's watch set is not practical: ``reload_includes`` only
extends the watched directories for directory matches, so covering the root .env
would mean watching the whole repository root, including .git and .venv.)
"""

from __future__ import annotations

import argparse
import sys

import uvicorn


def loop_option() -> str:
    """The value passed to uvicorn's ``--loop``.

    Windows gets our Selector factory by import path, because uvicorn would
    otherwise hand psycopg a ProactorEventLoop it cannot use. Everywhere else
    stays on ``"auto"`` so uvloop is still picked up in the container.
    """
    return "atlas.api.main:loop_factory" if sys.platform == "win32" else "auto"


def main() -> None:
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        prog="python -m atlas",
        description="Run the Atlas API server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload on code changes. Does not cover .env — restart for those.",
    )
    args = parser.parse_args()

    uvicorn.run(
        "atlas.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop=loop_option(),
    )


if __name__ == "__main__":
    main()
