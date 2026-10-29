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

from atlas.domains import DomainName
from atlas.events import AgentEvent
from atlas.graph.state import Revision
from atlas.trace import AgentCall

RunKind = Literal["question", "revision", "review"]
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
    # The domain the run researched in: chosen when its conversation started, and
    # kept by every run on it. None on rows written before it was recorded.
    domain: DomainName | None = None


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

    async def add_call(self, run_id: str, call: AgentCall) -> None: ...

    async def calls(self, run_id: str) -> list[AgentCall]:
        """Every call the run's agents made, in the order they started."""
        ...

    async def call(self, run_id: str, call_id: str) -> AgentCall | None: ...

    async def archive_events(self, run_id: str, events: list[AgentEvent]) -> None:
        """Keep a finished run's event stream past Redis's hour."""
        ...

    async def events(self, run_id: str) -> list[AgentEvent]:
        """An archived run's events, in seq order; empty if none were archived."""
        ...

    async def delete_thread(self, thread_id: str) -> list[str]:
        """Delete the thread's runs, calls and events. Returns the run ids, oldest first."""
        ...

    async def expired_threads(self, before: datetime) -> list[str]:
        """Threads whose most recent run started before ``before``, sorted."""
        ...


class InMemoryRunStore:
    """The store for tests and the eval: a dict, gone with the process."""

    def __init__(self) -> None:
        self.records: dict[str, RunRecord] = {}
        self._calls: dict[str, list[AgentCall]] = {}
        self._events: dict[str, list[AgentEvent]] = {}

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

    async def add_call(self, run_id: str, call: AgentCall) -> None:
        self._calls.setdefault(run_id, []).append(call)

    async def calls(self, run_id: str) -> list[AgentCall]:
        return sorted(self._calls.get(run_id, []), key=lambda c: (c.started_at, c.call_id))

    async def call(self, run_id: str, call_id: str) -> AgentCall | None:
        return next((c for c in self._calls.get(run_id, []) if c.call_id == call_id), None)

    async def archive_events(self, run_id: str, events: list[AgentEvent]) -> None:
        self._events[run_id] = list(events)

    async def events(self, run_id: str) -> list[AgentEvent]:
        return sorted(self._events.get(run_id, []), key=lambda e: e.seq)

    async def delete_thread(self, thread_id: str) -> list[str]:
        run_ids = [run.run_id for run in await self.thread(thread_id)]
        for run_id in run_ids:
            del self.records[run_id]
            self._calls.pop(run_id, None)
            self._events.pop(run_id, None)
        return run_ids

    async def expired_threads(self, before: datetime) -> list[str]:
        latest: dict[str, datetime] = {}
        for run in self.records.values():
            if run.thread_id not in latest or run.created_at > latest[run.thread_id]:
                latest[run.thread_id] = run.created_at
        return sorted(thread for thread, at in latest.items() if at < before)


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
    # Added after the table: a database created before keeps its rows, which read as None.
    "ALTER TABLE atlas_runs ADD COLUMN IF NOT EXISTS domain TEXT",
    """
    CREATE TABLE IF NOT EXISTS atlas_calls (
        call_id       TEXT PRIMARY KEY,
        run_id        TEXT NOT NULL,
        span_id       TEXT,
        node          TEXT NOT NULL,
        subtask_id    TEXT,
        kind          TEXT NOT NULL,
        name          TEXT NOT NULL,
        started_at    TIMESTAMPTZ NOT NULL,
        duration_s    DOUBLE PRECISION NOT NULL,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        error         TEXT,
        request       JSONB,
        response      JSONB
    )
    """,
    "CREATE INDEX IF NOT EXISTS atlas_calls_run ON atlas_calls (run_id, started_at)",
    """
    CREATE TABLE IF NOT EXISTS atlas_events (
        run_id  TEXT NOT NULL,
        seq     INTEGER NOT NULL,
        ts      TIMESTAMPTZ NOT NULL,
        type    TEXT NOT NULL,
        node    TEXT,
        payload JSONB NOT NULL,
        PRIMARY KEY (run_id, seq)
    )
    """,
)

_CALL_COLUMNS = (
    "call_id, span_id, node, subtask_id, kind, name, started_at, duration_s, "
    "input_tokens, output_tokens, error, request, response"
)

# Taken around the DDL. Two processes running CREATE TABLE IF NOT EXISTS at once
# can still collide on Postgres's own catalog (a unique violation on pg_type),
# and the worker may be scaled to more than one process.
_SCHEMA_LOCK = 0x41544C53  # "ATLS"

_COLUMNS = (
    "run_id, thread_id, kind, query, parent_run_id, revision, status, result, error, "
    "created_at, finished_at, domain"
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
                    "CAST(:result AS JSONB), :error, :created_at, :finished_at, :domain)"
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

    async def add_call(self, run_id: str, call: AgentCall) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    f"INSERT INTO atlas_calls (run_id, {_CALL_COLUMNS}) VALUES (:run_id, "
                    ":call_id, :span_id, :node, :subtask_id, :kind, :name, :started_at, "
                    ":duration_s, :input_tokens, :output_tokens, :error, "
                    "CAST(:request AS JSONB), CAST(:response AS JSONB))"
                ),
                {
                    **call.model_dump(exclude={"request", "response"}),
                    "run_id": run_id,
                    "request": json.dumps(call.request),
                    "response": json.dumps(call.response),
                },
            )

    async def calls(self, run_id: str) -> list[AgentCall]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            f"SELECT {_CALL_COLUMNS} FROM atlas_calls WHERE run_id = :run_id "
                            "ORDER BY started_at, call_id"
                        ),
                        {"run_id": run_id},
                    )
                )
                .mappings()
                .all()
            )
        return [AgentCall.model_validate(dict(row)) for row in rows]

    async def call(self, run_id: str, call_id: str) -> AgentCall | None:
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            f"SELECT {_CALL_COLUMNS} FROM atlas_calls "
                            "WHERE run_id = :run_id AND call_id = :call_id"
                        ),
                        {"run_id": run_id, "call_id": call_id},
                    )
                )
                .mappings()
                .first()
            )
        return AgentCall.model_validate(dict(row)) if row is not None else None

    async def archive_events(self, run_id: str, events: list[AgentEvent]) -> None:
        if not events:
            return
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO atlas_events (run_id, seq, ts, type, node, payload) "
                    "VALUES (:run_id, :seq, :ts, :type, :node, CAST(:payload AS JSONB)) "
                    "ON CONFLICT (run_id, seq) DO NOTHING"
                ),
                [
                    {
                        "run_id": run_id,
                        "seq": event.seq,
                        "ts": event.ts,
                        "type": event.type,
                        "node": event.node,
                        "payload": json.dumps(event.payload, default=str),
                    }
                    for event in events
                ],
            )

    async def events(self, run_id: str) -> list[AgentEvent]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            "SELECT run_id, seq, ts, type, node, payload FROM atlas_events "
                            "WHERE run_id = :run_id ORDER BY seq"
                        ),
                        {"run_id": run_id},
                    )
                )
                .mappings()
                .all()
            )
        return [AgentEvent.model_validate(dict(row)) for row in rows]

    async def delete_thread(self, thread_id: str) -> list[str]:
        async with self._engine.begin() as connection:
            run_ids = [
                str(row[0])
                for row in await connection.execute(
                    text(
                        "SELECT run_id FROM atlas_runs WHERE thread_id = :thread_id "
                        "ORDER BY created_at, run_id"
                    ),
                    {"thread_id": thread_id},
                )
            ]
            for table in ("atlas_calls", "atlas_events"):
                await connection.execute(
                    text(f"DELETE FROM {table} WHERE run_id = ANY(:run_ids)"),
                    {"run_ids": run_ids},
                )
            await connection.execute(
                text("DELETE FROM atlas_runs WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
        return run_ids

    async def expired_threads(self, before: datetime) -> list[str]:
        async with self._engine.connect() as connection:
            rows = await connection.execute(
                text(
                    "SELECT thread_id FROM atlas_runs GROUP BY thread_id "
                    "HAVING max(created_at) < :before ORDER BY thread_id"
                ),
                {"before": before},
            )
            return [str(row[0]) for row in rows]
