"""Event loop selection.

Regression cover for a Windows-only startup failure: uvicorn hands psycopg a
ProactorEventLoop, psycopg refuses it, and the checkpointer pool dies with
``PoolTimeout: pool initialization incomplete`` — an error that points at the
database rather than the event loop. Nothing else in the suite can catch it,
because the tests never start a server.
"""

from __future__ import annotations

import asyncio
import sys

from atlas.__main__ import loop_option
from atlas.api.main import loop_factory


def test_loop_factory_builds_a_selector_loop() -> None:
    loop = loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_loop_factory_never_returns_proactor() -> None:
    """The specific loop psycopg cannot run on."""
    if sys.platform != "win32":
        return  # ProactorEventLoop does not exist off Windows
    loop = loop_factory()
    try:
        assert not isinstance(loop, asyncio.ProactorEventLoop)
    finally:
        loop.close()


def test_loop_option_overrides_only_on_windows() -> None:
    """Windows needs the custom factory; elsewhere "auto" keeps uvloop available."""
    if sys.platform == "win32":
        assert loop_option() == "atlas.api.main:loop_factory"
    else:
        assert loop_option() == "auto"


def test_loop_option_points_at_something_importable() -> None:
    """Guard the import string: uvicorn resolves it at startup, not at import."""
    if sys.platform != "win32":
        return
    module_path, _, attribute = loop_option().partition(":")
    module = __import__(module_path, fromlist=[attribute])
    assert callable(getattr(module, attribute))
