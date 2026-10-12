"""Async SQLAlchemy engine and session factory.

The checkpointer owns its own psycopg pool for graph state and Atlas has no
tables of its own yet, so the only query through this engine today is the API's
``/health`` probe. It exists so that a retrieval layer over pgvector, if one is
built, has a typed, correctly-configured entry point to build on.

Engine and factory are created lazily rather than at import so that importing
``atlas.db.session`` never requires a reachable database — which keeps the test
suite free of accidental connection attempts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from atlas.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_settings().sqlalchemy_url,
            # Postgres drops idle connections; without this a pooled connection
            # that died while idle surfaces as a failed query rather than a retry.
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        # expire_on_commit=False keeps ORM objects usable after commit, which
        # matters when a response model is serialized from them post-transaction.
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session that is always closed."""
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine and reset module state. Call on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
    _session_factory = None
