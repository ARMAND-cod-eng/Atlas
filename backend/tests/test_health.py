"""The /health endpoint.

TestClient is constructed *without* a ``with`` block on purpose: entering the
context manager would run the lifespan, which opens a real Postgres pool and a
real Redis client. Skipping it keeps these tests hermetic, and the two probes are
substituted directly.
"""

from __future__ import annotations

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from atlas.api import routes
from atlas.api.main import create_app


def _patch_probes(monkeypatch: pytest.MonkeyPatch, *, db: bool, redis: bool) -> None:
    async def fake_db(request: Request) -> bool:
        return db

    async def fake_redis(request: Request) -> bool:
        return redis

    monkeypatch.setattr(routes, "check_db", fake_db)
    monkeypatch.setattr(routes, "check_redis", fake_redis)


def test_health_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probes(monkeypatch, db=True, redis=True)
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": True, "redis": True}


def test_health_reports_which_dependency_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Still 200 — the body carries the detail. That distinction is the point."""
    _patch_probes(monkeypatch, db=True, redis=False)
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "db": True, "redis": False}


def test_health_reports_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probes(monkeypatch, db=False, redis=True)
    body = TestClient(create_app()).get("/health").json()
    assert body["status"] == "degraded"
    assert body["db"] is False
