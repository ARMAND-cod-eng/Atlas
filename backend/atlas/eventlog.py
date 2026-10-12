"""The durable event log behind the SSE endpoint.

A Redis Stream per run, which is the data structure this problem actually wants:
append-only, ordered, range-readable for replay, and blockingly tailable for live
delivery. A pub/sub channel would be simpler and wrong — pub/sub drops anything
published while nobody is listening, and "nobody is listening" is precisely the
window reconnection exists to cover.

Keeping events here rather than in the producing process is what decouples the
run from the connection. The worker writes; any API instance can read; a client
that drops at 40s of a 67s run reconnects and replays what it missed.

Follows the lazy-client pattern already in :mod:`atlas.budget`, and shares its
TTL: an abandoned run's events should not outlive its budget counter.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.typing import EncodableT, FieldT

from atlas.config import get_settings
from atlas.events import TERMINAL_TYPES, AgentEvent

logger = logging.getLogger(__name__)

# Bounds the memory one run can consume. A run emits tens of events, so this is
# ~30x headroom; the approximate form ("~") lets Redis trim on node boundaries,
# which is cheaper and is why it is preferred over an exact MAXLEN.
_MAX_EVENTS_PER_RUN = 2000

_client: Redis | None = None


def events_key(run_id: str) -> str:
    return f"atlas:events:{run_id}"


def result_key(run_id: str) -> str:
    return f"atlas:result:{run_id}"


def get_client() -> Redis:
    """Process-wide Redis client, created on first use.

    Lazy so importing this module never requires a reachable Redis, which is what
    keeps the test suite offline.
    """
    global _client
    if _client is None:
        _client = Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


async def close_client() -> None:
    """Release the client. Called on application and worker shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# redis-py types xadd's fields as an invariant Dict[FieldT, EncodableT], so the
# annotation matches it exactly rather than being narrowed to dict[str, str] and
# then cast at the call site.
def _encode(event: AgentEvent) -> dict[FieldT, EncodableT]:
    """Flatten an event into stream fields.

    ``payload`` is nested and Redis stream fields are flat strings, so it travels
    as JSON. The scalar fields stay top-level so a human reading ``XRANGE`` output
    can see what happened without decoding anything.
    """
    return {
        "seq": str(event.seq),
        "ts": event.ts.isoformat(),
        "type": event.type,
        "node": event.node or "",
        "payload": json.dumps(event.payload, default=str),
    }


def _decode(run_id: str, fields: dict[str, str]) -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        seq=int(fields["seq"]),
        ts=fields["ts"],
        type=fields["type"],
        node=fields["node"] or None,
        payload=json.loads(fields["payload"]),
    )


async def append(event: AgentEvent) -> None:
    """Append one event. Never raises.

    A telemetry write must not be able to kill a run that is otherwise fine. The
    cost of failing open here is a gap in the stream, which the client can *see*
    because ``seq`` is dense — a dropped event is detectable rather than silent.
    """
    key = events_key(event.run_id)
    try:
        client = get_client()
        await client.xadd(key, _encode(event), maxlen=_MAX_EVENTS_PER_RUN, approximate=True)
        await client.expire(key, get_settings().event_stream_ttl_seconds)
    except Exception as exc:
        logger.warning(
            "could not append event; the stream will show a gap",
            extra={
                "run_id": event.run_id,
                "seq": event.seq,
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            },
        )


async def replay(run_id: str, after_seq: int) -> list[AgentEvent]:
    """Every event with ``seq > after_seq``, in order.

    Reads the whole stream and filters rather than translating a seq into a stream
    id. Deliberate: a run is tens of events, the filter is exact, and the
    alternative needs a seq→id index that can disagree with the stream itself.
    """
    entries = await get_client().xrange(events_key(run_id), min="-", max="+")
    events = [_decode(run_id, fields) for _, fields in entries]
    return [event for event in events if event.seq > after_seq]


async def terminal_event(run_id: str) -> AgentEvent | None:
    """The event the run ended on, or None if it has not ended.

    The runner writes exactly one, success or failure, so this is the
    authoritative answer to "how did it end" — unlike the stored result, which
    exists only on success and only if storing it worked.
    """
    for event in reversed(await replay(run_id, 0)):
        if event.type in TERMINAL_TYPES:
            return event
    return None


async def tail(run_id: str, last_id: str, block_ms: int) -> AsyncIterator[tuple[str, AgentEvent]]:
    """Yield ``(stream_id, event)`` as events arrive, one blocking read at a time.

    Yields nothing and returns when the read times out, which the caller uses as
    its heartbeat tick — the idle path is a natural consequence of the blocking
    read rather than a second timer racing it.
    """
    response = await get_client().xread({events_key(run_id): last_id}, count=100, block=block_ms)
    for _key, entries in response:
        for stream_id, fields in entries:
            yield stream_id, _decode(run_id, fields)


def run_key(run_id: str) -> str:
    return f"atlas:run:{run_id}"


async def register(run_id: str) -> None:
    """Mark a run as known, before it has produced anything.

    Closes a race that live verification found and the tests did not: ``POST
    /runs`` returns as soon as the job is queued, and a client that subscribes
    immediately arrives before the worker has written ``run_started``. Without
    this marker :func:`exists` says no and the subscriber gets a 404 for a run
    that is about to start — the exact usage the endpoint exists for.

    The unit tests missed it because their arq double runs the job inline, so
    events were always present by the time anything subscribed. Registering here
    keeps 404 meaning "no such run" rather than "not yet".
    """
    try:
        await get_client().set(
            run_key(run_id), "queued", ex=get_settings().event_stream_ttl_seconds
        )
    except Exception as exc:
        logger.warning(
            "could not register run",
            extra={"run_id": run_id, "error_type": type(exc).__name__},
        )


async def exists(run_id: str) -> bool:
    """Whether this run is known — queued, running, or finished."""
    client = get_client()
    return bool(await client.exists(events_key(run_id))) or bool(
        await client.exists(run_key(run_id))
    )


async def store_result(run_id: str, result: dict[str, Any]) -> None:
    """Persist the final result for ``GET /runs/{run_id}``.

    ``run_completed`` carries a summary only, so the full sources list and
    per-call metrics are fetched once here rather than replayed on every
    reconnect.
    """
    try:
        await get_client().set(
            result_key(run_id),
            json.dumps(result, default=str),
            ex=get_settings().event_stream_ttl_seconds,
        )
    except Exception as exc:
        logger.warning(
            "could not store run result",
            extra={"run_id": run_id, "error_type": type(exc).__name__},
        )


async def load_result(run_id: str) -> dict[str, Any] | None:
    """The stored result, or None if the run is unfinished or expired."""
    raw = await get_client().get(result_key(run_id))
    if raw is None:
        return None
    decoded: dict[str, Any] = json.loads(raw)
    return decoded
