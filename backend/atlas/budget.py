"""Per-run token budget, shared across parallel branches.

The counter lives in Redis rather than graph state for a specific reason: every
parallel branch reads the same pre-fan-out state snapshot, so a counter kept in
state cannot see sibling spend while the fan-out is in flight. Five researchers
would each believe they were the only one spending. Redis is already in the
stack and gives one shared counter per ``thread_id``.

**Fails open.** Any Redis problem logs a warning and lets the run continue. A
dead cache must not break queries, and the run stays bounded by
``MAX_CRITIC_ITERATIONS`` and ``MAX_SUBTASKS`` regardless. The caller records
``budget_check_unavailable`` so the client knows the guard was not running.
"""

from __future__ import annotations

import logging

from langgraph.config import get_config
from redis.asyncio import Redis

from atlas.config import get_settings

logger = logging.getLogger(__name__)

# Long enough to outlive any single run, short enough that abandoned threads do
# not accumulate keys forever.
_KEY_TTL_SECONDS = 3600

_client: Redis | None = None


def _key(thread_id: str) -> str:
    return f"atlas:budget:{thread_id}"


def get_client() -> Redis:
    """Process-wide Redis client, created on first use.

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


def current_thread_id() -> str | None:
    """The running graph's thread id, or None outside a graph.

    Read from LangGraph's ambient config rather than threaded through every node
    signature. Returns None when called outside a graph run (in a unit test, for
    instance), which callers treat as "no budget tracking".
    """
    try:
        config = get_config()
    except Exception:
        return None
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return thread_id if isinstance(thread_id, str) else None


async def record_spend(tokens: int) -> None:
    """Add tokens to the running thread's total. Never raises."""
    if tokens <= 0:
        return
    thread_id = current_thread_id()
    if thread_id is None:
        return
    try:
        client = get_client()
        total = await client.incrby(_key(thread_id), tokens)
        await client.expire(_key(thread_id), _KEY_TTL_SECONDS)
        logger.debug("recorded spend", extra={"tokens": tokens, "total": total})
    except Exception as exc:
        logger.warning(
            "could not record token spend; budget tracking degraded",
            extra={"error_type": type(exc).__name__, "error": str(exc)[:200]},
        )


async def spent() -> int | None:
    """Tokens spent by the running thread, or None if unknown.

    None means "could not check" — Redis down, or running outside a graph. It is
    deliberately distinct from 0, so callers can record that the guard was not
    running rather than silently treating an unknown as within budget.
    """
    thread_id = current_thread_id()
    if thread_id is None:
        return None
    try:
        raw = await get_client().get(_key(thread_id))
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
    """Whether the running thread has blown its budget.

    Returns ``(exceeded, spent_tokens)``. ``spent_tokens`` is None when the check
    could not run, in which case ``exceeded`` is False — failing open, as agreed.
    """
    total = await spent()
    if total is None:
        return False, None
    return total > get_settings().max_tokens_per_query, total
