"""The SSE relay.

Deliberately thin: it reads a Redis Stream and writes SSE frames. It does not
execute anything, which is the property that lets a client drop at 40s of a 67s
run without the run noticing.

:func:`event_stream` is a plain async generator so the tests can drive it
directly. Everything worth asserting in this module — replay boundaries,
heartbeat cadence, terminating on a terminal event — lives in that generator
rather than in the route, so none of it needs an HTTP client or a live proxy to
verify.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.event import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from atlas import eventlog
from atlas.config import get_settings
from atlas.events import TERMINAL_TYPES

logger = logging.getLogger(__name__)

router = APIRouter()


def parse_last_event_id(raw: str | None) -> int:
    """The seq a reconnecting client already has, or 0.

    Anything unparseable is treated as 0 — replay everything — because the cost
    of re-sending a short run is trivial next to silently skipping events a
    client never received.
    """
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("ignoring unparseable Last-Event-ID", extra={"last_event_id": raw[:50]})
        return 0


async def event_stream(
    run_id: str,
    after_seq: int = 0,
    heartbeat_s: float | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Replay what the client missed, then follow the run to its end.

    The ``id`` on each frame is the event's ``seq``, which is what the browser
    sends back as ``Last-Event-ID`` on reconnect. Heartbeats deliberately carry
    **no** id: a comment frame that set one would move the client's resume point
    without delivering an event, so a reconnect straight after a heartbeat would
    skip whatever arrived in between.
    """
    interval = heartbeat_s if heartbeat_s is not None else get_settings().sse_heartbeat_seconds
    block_ms = int(interval * 1000)

    last_seq = after_seq
    last_id = "0-0"

    # Replay first. A run that already finished is fully served from here, and
    # the terminal event ends the stream without ever entering the tail loop.
    for event in await eventlog.replay(run_id, after_seq):
        last_seq = event.seq
        yield _frame(event.seq, event.type, event.model_dump(mode="json"))
        if event.type in TERMINAL_TYPES:
            return

    # XRANGE gave us events but not their stream ids, so resume the tail from the
    # end of what exists now. Anything between those two points would be a
    # duplicate, which the seq filter below removes.
    last_id = "$" if last_seq else "0-0"

    while True:
        got_any = False
        async for stream_id, event in eventlog.tail(run_id, last_id, block_ms):
            last_id = stream_id
            got_any = True
            if event.seq <= last_seq:
                continue  # already replayed
            last_seq = event.seq
            yield _frame(event.seq, event.type, event.model_dump(mode="json"))
            if event.type in TERMINAL_TYPES:
                return
        if not got_any:
            # The blocking read timed out, so the run is alive but quiet. This is
            # the heartbeat tick: no extra timer, no race with the reader.
            yield ServerSentEvent(comment="heartbeat")


def _frame(seq: int, event_type: str, body: dict[str, object]) -> ServerSentEvent:
    return ServerSentEvent(id=str(seq), event=event_type, data=json.dumps(body, default=str))


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request) -> EventSourceResponse:
    """Server-sent events for one run.

    ``Last-Event-ID`` is the standard header an ``EventSource`` resends on
    reconnect. The query parameter is the same thing for every other client —
    curl, httpx, the terminal client used in verification — none of which can set
    that header on an automatic retry because they have no automatic retry.
    """
    if not await eventlog.exists(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")

    after_seq = parse_last_event_id(
        request.headers.get("last-event-id") or request.query_params.get("last_event_id")
    )
    logger.info("sse subscriber attached", extra={"run_id": run_id, "after_seq": after_seq})

    return EventSourceResponse(
        event_stream(run_id, after_seq),
        headers={
            # Without this nginx buffers the response and delivers every event at
            # once when the run ends, which is indistinguishable from not having
            # built any of this.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
        # Our own heartbeat rides the blocking read's timeout, so sse-starlette's
        # independent ping timer would only add a second, racing source of
        # comment frames.
        ping=1_000_000,
    )
