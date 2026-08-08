"""Error responses and the fallback warning.

Phase 0's bare 500 with an empty body is what these guard against. A failure has
to say what broke and where.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas.api.main import (
    provider_error_handler,
    search_error_handler,
    unhandled_error_handler,
)
from atlas.errors import ProviderError, SearchError
from atlas.models import FallbackWatcher


def build_error_app() -> FastAPI:
    """A minimal app with the real handlers and routes that raise on demand."""
    app = FastAPI()
    app.add_exception_handler(ProviderError, provider_error_handler)
    app.add_exception_handler(SearchError, search_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    @app.get("/boom/provider")
    async def boom_provider() -> None:
        raise ProviderError("anthropic", "everything failed", node="planner")

    @app.get("/boom/search")
    async def boom_search() -> None:
        raise SearchError("tavily unreachable", query="what is RAG")

    @app.get("/boom/unknown")
    async def boom_unknown() -> None:
        raise ValueError("something unexpected")

    return app


def test_provider_error_returns_502_naming_the_provider() -> None:
    client = TestClient(build_error_app(), raise_server_exceptions=False)
    response = client.get("/boom/provider")

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "provider_error"
    assert body["provider"] == "anthropic"
    assert body["node"] == "planner"
    assert "everything failed" in body["detail"]


def test_search_error_returns_502_naming_tavily() -> None:
    client = TestClient(build_error_app(), raise_server_exceptions=False)
    response = client.get("/boom/search")

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "search_error"
    assert body["provider"] == "tavily"
    assert body["query"] == "what is RAG"


def test_unhandled_error_returns_500_with_a_body() -> None:
    """Still 500 — but never again an empty body."""
    client = TestClient(build_error_app(), raise_server_exceptions=False)
    response = client.get("/boom/unknown")

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "ValueError" in body["detail"]
    # The message must not leak the exception's text to the client.
    assert "something unexpected" not in json.dumps(body)


def test_fallback_logs_a_warning_naming_the_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The signal missing in Phase 0.

    A dead primary behind a working fallback produced correct responses all
    session and said nothing. This is what says something.
    """
    watcher = FallbackWatcher()
    with caplog.at_level(logging.WARNING, logger="atlas.models"):
        watcher.on_llm_error(RuntimeError("400 temperature is deprecated"))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.error_type == "RuntimeError"
    assert "temperature is deprecated" in record.error
