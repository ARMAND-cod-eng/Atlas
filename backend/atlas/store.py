"""The durable record of every run: what was asked, how it ended, what it revised.

Redis holds a run's live event stream and its result for an hour, which is right
for watching a run and wrong for anything that looks back. A revision names its
parent run long after that hour, and a thread's versions list needs every run
on it. So each run is also recorded here, in Postgres, for as long as the thread
is kept.

One Protocol, two stores, held to one contract by ``tests/test_store.py``. The
in-memory store serves the tests and the eval, which run without a database.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine

from atlas.graph.state import Revision

RunKind = Literal["question", "revision"]
RunStatusName = Literal["running", "completed", "failed"]


class RunRecord(BaseModel):
    """One run, as the record keeps it."""

    run_id: str
    thread_id: str
    # "question" covers a first question and a follow-up alike: both start at the
    # planner. A revision starts from another run's finished state.
    kind: RunKind
    query: str
    parent_run_id: str | None = None
    revision: Revision | None = None
    status: RunStatusName = "running"
    # The QueryResponse payload, exactly as GET /runs/{id} serves it.
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class RunStore(Protocol):
    """Where runs are recorded. Writes happen in the worker, reads in the API."""

    async def started(self, record: RunRecord) -> None: ...

    async def finished(
        self,
        run_id: str,
        *,
        status: RunStatusName,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        at: datetime | None = None,
    ) -> None: ...

    async def get(self, run_id: str) -> RunRecord | None: ...

    async def thread(self, thread_id: str) -> list[RunRecord]:
        """Every run on the thread, oldest first."""
        ...


class InMemoryRunStore:
    """The store for tests and the eval: a dict, gone with the process."""

    def __init__(self) -> None:
        self.records: dict[str, RunRecord] = {}

    async def started(self, record: RunRecord) -> None:
        self.records[record.run_id] = record

    async def finished(
        self,
        run_id: str,
        *,
        status: RunStatusName,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        at: datetime | None = None,
    ) -> None:
        record = self.records[run_id]
        self.records[run_id] = record.model_copy(
            update={
                "status": status,
                "result": dict(result) if result is not None else None,
                "error": error,
                "finished_at": at or datetime.now(UTC),
            }
        )

    async def get(self, run_id: str) -> RunRecord | None:
        return self.records.get(run_id)

    async def thread(self, thread_id: str) -> list[RunRecord]:
        runs = [r for r in self.records.values() if r.thread_id == thread_id]
        return sorted(runs, key=lambda r: r.created_at)


# Prefixed so it can never collide with the checkpointer's own tables, which
# share the database.
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS atlas_runs (
        run_id        TEXT PRIMARY KEY,
        thread_id     TEXT NOT NULL,
        kind          TEXT NOT NULL,
        query         TEXT NOT NULL,
        parent_run_id TEXT,
        revision      JSONB,
        status        TEXT NOT NULL,
        result        JSONB,
        error         TEXT,
        created_at    TIMESTAMPTZ NOT NULL,
        finished_at   TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS atlas_runs_thread ON atlas_runs (thread_id, created_at)",
)

# Taken around the DDL. Two processes running CREATE TABLE IF NOT EXISTS at once
# can still collide on Postgres's own catalog (a unique violation on pg_type),
# and the worker may be scaled to more than one process.
_SCHEMA_LOCK = 0x41544C53  # "ATLS"

_COLUMNS = (
    "run_id, thread_id, kind, query, parent_run_id, revision, status, result, error, "
    "created_at, finished_at"
)


def _json(value: object) -> str | None:
    return None if value is None else json.dumps(value)


def _record_from(row: RowMapping) -> RunRecord:
    return RunRecord.model_validate(dict(row))


class PostgresRunStore:
    """The production store, on the API's async SQLAlchemy engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def setup(self) -> None:
        """Create the table if needed. Idempotent; the worker runs it at start."""
        async with self._engine.begin() as connection:
            await connection.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEMA_LOCK}
            )
            for statement in _SCHEMA:
                await connection.execute(text(statement))

    async def started(self, record: RunRecord) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    f"INSERT INTO atlas_runs ({_COLUMNS}) VALUES (:run_id, :thread_id, :kind, "
                    ":query, :parent_run_id, CAST(:revision AS JSONB), :status, "
                    "CAST(:result AS JSONB), :error, :created_at, :finished_at)"
                ),
                {
                    **record.model_dump(exclude={"revision", "result"}),
                    "revision": _json(
                        record.revision.model_dump() if record.revision is not None else None
                    ),
                    "result": _json(record.result),
                },
            )

    async def finished(
        self,
        run_id: str,
        *,
        status: RunStatusName,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        at: datetime | None = None,
    ) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "UPDATE atlas_runs SET status = :status, result = CAST(:result AS JSONB), "
                    "error = :error, finished_at = :at WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "status": status,
                    "result": _json(dict(result) if result is not None else None),
                    "error": error,
                    "at": at or datetime.now(UTC),
                },
            )

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(f"SELECT {_COLUMNS} FROM atlas_runs WHERE run_id = :run_id"),
                        {"run_id": run_id},
                    )
                )
                .mappings()
                .first()
            )
        return _record_from(row) if row is not None else None

    async def thread(self, thread_id: str) -> list[RunRecord]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            f"SELECT {_COLUMNS} FROM atlas_runs WHERE thread_id = :thread_id "
                            "ORDER BY created_at, run_id"
                        ),
                        {"thread_id": thread_id},
                    )
                )
                .mappings()
                .all()
            )
        return [_record_from(row) for row in rows]
