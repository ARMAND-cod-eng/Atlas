"""Per-run token budget, shared across parallel branches.

The counter lives in Redis rather than graph state for a specific reason: every
parallel branch reads the same pre-fan-out state snapshot, so a counter kept in
state cannot see sibling spend while the fan-out is in flight. Five researchers
would each believe they were the only one spending. Redis is already in the
stack and gives one shared counter per run.

Per **run**, not per thread. ``MAX_TOKENS_PER_QUERY`` limits one question, and a
thread holds a whole conversation of them. Keyed by thread, the counter kept
climbing across follow-ups until an ordinary one was declared over budget at its
first review, and went out unreviewed. The run id arrives in ``configurable``
beside the thread id; :mod:`atlas.runner` puts it there.

**Fails open.** Any Redis problem logs a warning and lets the run continue. A
dead cache must not break queries, and the run stays bounded by
``MAX_CRITIC_ITERATIONS`` and ``MAX_SUBTASKS`` regardless. The caller records
``budget_check_unavailable`` so the client knows the guard was not running.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Protocol

from langgraph.config import get_config
from redis.asyncio import Redis

from atlas.config import get_settings

logger = logging.getLogger(__name__)

# Long enough to outlive any single run, short enough that abandoned runs do
# not accumulate keys forever.
_KEY_TTL_SECONDS = 3600


class CounterStore(Protocol):
    """The three Redis calls the budget makes, and nothing else.

    Typed this narrowly so a process without Redis — the eval — can keep the
    counter in memory and still be checked, rather than the budget being
    silently off there. The return types are ``Awaitable`` because that is what
    redis-py's client declares.
    """

    def incrby(self, name: str, amount: int = 1) -> Awaitable[int]: ...

    def expire(self, name: str, time: int) -> Awaitable[bool]: ...

    def get(self, name: str) -> Awaitable[str | None]: ...


_client: Redis | None = None


def _key(run_id: str) -> str:
    return f"atlas:budget:{run_id}"


def get_client() -> CounterStore:
    """Process-wide Redis client, created on first use, as a :class:`CounterStore`.

    Separate from the API's client (which lives on ``app.state``) because nodes
    have no access to the FastAPI app. Lazy so that importing this module never
    requires a reachable Redis — which keeps the test suite offline.
    """
    global _client
    if _client is None:
        _client = Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


async def close_client() -> None:
    """Release the client. Called on application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def current_run_id() -> str | None:
    """The running graph's run id, or None outside a graph.

    Read from LangGraph's ambient config rather than threaded through every node
    signature. Returns None when called outside a graph run (in a unit test, for
    instance) or when the caller did not supply a run id. Callers treat that as
    "no budget tracking" and the critic records ``budget_check_unavailable`` —
    deliberately not falling back to the thread id, which is the bug this
    replaced.
    """
    try:
        config = get_config()
    except Exception:
        return None
    configurable = config.get("configurable") or {}
    run_id = configurable.get("run_id")
    return run_id if isinstance(run_id, str) else None


async def record_spend(tokens: int) -> None:
    """Add tokens to the current run's total. Never raises."""
    if tokens <= 0:
        return
    run_id = current_run_id()
    if run_id is None:
        return
    try:
        client = get_client()
        total = await client.incrby(_key(run_id), tokens)
        await client.expire(_key(run_id), _KEY_TTL_SECONDS)
        logger.debug("recorded spend", extra={"tokens": tokens, "total": total})
    except Exception as exc:
        logger.warning(
            "could not record token spend; budget tracking degraded",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:200]},
        )


async def spent() -> int | None:
    """Tokens spent by the current run, or None if unknown.

    None means "could not check" — Redis down, or running outside a graph. It is
    deliberately distinct from 0, so callers can record that the guard was not
    running rather than silently treating an unknown as within budget.
    """
    run_id = current_run_id()
    if run_id is None:
        return None
    try:
        raw = await get_client().get(_key(run_id))
    except Exception as exc:
        logger.warning(
            "could not read token spend; failing open",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:200]},
        )
        return None
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def is_exceeded() -> tuple[bool, int | None]:
    """Whether the current run has blown its budget.

    Returns ``(exceeded, spent_tokens)``. ``spent_tokens`` is None when the check
    could not run, in which case ``exceeded`` is False — failing open, as agreed.
    """
    total = await spent()
    if total is None:
        return False, None
    return total > get_settings().max_tokens_per_query, total
