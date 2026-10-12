"""Typed events describing a run as it happens.

The split that matters: **nodes emit payloads, the runner assigns identity.** A
node calls :func:`emit` with a type and some fields and knows nothing about
``run_id``, ``seq``, or Redis. :mod:`atlas.runner` turns each payload into a full
:class:`AgentEvent` as it consumes the stream. That keeps the graph ignorant of
the transport, and it is what makes :func:`emit` safe to call from a unit test
that invokes a node directly.

Where ``seq`` comes from is worth stating, because the obvious reading is wrong.
The fan-out is concurrent, so it looks like the counter needs to be atomic — a
Redis ``INCR`` per event, or a lock. It does not. LangGraph serialises every
chunk from every branch into one consuming loop, so the runner is the only writer
and can assign ``seq`` with ``+= 1``. No round trip, no lock, and no way to
produce a gap.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from atlas.metrics import RunMetrics, track_node

logger = logging.getLogger(__name__)

# Event types. Strings rather than an enum for the same reason the degraded
# markers are strings: they cross a process boundary and end up as JSON, and a
# consumer should not need our enum to read them.
RUN_STARTED = "run_started"
NODE_STARTED = "node_started"
NODE_COMPLETED = "node_completed"
NODE_FAILED = "node_failed"
FAN_OUT = "fan_out"
ITERATION_STARTED = "iteration_started"
CRITIC_VERDICT = "critic_verdict"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"

# A stream must end deliberately. The relay stops after one of these, and the
# runner guarantees exactly one is always written — see atlas.runner.
TERMINAL_TYPES = frozenset({RUN_COMPLETED, RUN_FAILED})


class AgentEvent(BaseModel):
    """One thing that happened during a run.

    ``seq`` is per-run, starts at 1, and is dense. A client that receives 7 then 9
    knows it missed one, which is the entire basis for reconnection: without a
    gapless counter, "I have everything up to N" is not a statement you can make.
    """

    run_id: str
    seq: int
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: str
    node: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def emit(event_type: str, node: str | None = None, **payload: object) -> None:
    """Emit an event from inside a node. Never raises.

    ``get_stream_writer`` raises ``RuntimeError`` outside a runnable context, and
    Phase 2's tests call node functions directly with no graph running at all —
    so an unguarded call here would break 126 existing tests. It is also a no-op
    under ``ainvoke``, which is what keeps ``POST /query`` working unchanged.

    Swallowing the error is right on its own terms too: telemetry that can fail a
    run is worse than no telemetry.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        # No graph running — a direct unit-test call, or a plain function call.
        return
    writer({"type": event_type, "node": node, **payload})


@dataclass(slots=True)
class NodeSpan:
    """What :func:`node_span` hands to a node body.

    Carries the metrics the node already filled in before this phase, plus a way
    to report failure *without raising*. That second part is not decoration: a
    researcher whose search dies deliberately does not raise — Phase 1 turns a
    failed branch into a ``ResearchResult`` with ``error`` set so the other four
    branches survive. Such a branch has still failed as far as a watching client
    is concerned, and must report ``node_failed`` rather than ``node_completed``.
    """

    metrics: RunMetrics = field(default_factory=RunMetrics)
    error: str | None = None
    # Extra fields merged into the terminal event for this node. Added in Phase 4
    # so a researcher can report the sources it retrieved: by the time synthesis
    # starts, every branch has finished and its sources exist, and the stream was
    # throwing them away until the very end. Surfacing them is what gives a viewer
    # something real to read during the 20-45s the synthesizer spends silent.
    detail: dict[str, Any] = field(default_factory=dict)

    def failed(self, reason: str) -> None:
        """Mark this node as failed even though nothing was raised."""
        self.error = reason

    def report(self, **fields: object) -> None:
        """Attach fields to this node's ``node_completed`` / ``node_failed``."""
        self.detail.update(fields)


@asynccontextmanager
async def node_span(node: str, subtask_id: str | None = None) -> AsyncIterator[NodeSpan]:
    """Time a node, and emit its lifecycle.

    Wraps :func:`atlas.metrics.track_node` rather than duplicating it, so timing
    and events cannot drift apart. ``subtask_id`` is what makes concurrent
    researcher events attributable — five branches emit interleaved
    ``node_started``/``node_completed`` pairs, and the id is the only thing that
    says which branch each belongs to.
    """
    emit(NODE_STARTED, node, subtask_id=subtask_id)
    span = NodeSpan()
    try:
        async with track_node(node, subtask_id) as metrics:
            span.metrics = metrics
            yield span
    except Exception as exc:
        emit(
            NODE_FAILED,
            node,
            subtask_id=subtask_id,
            error=f"{type(exc).__name__}: {exc}",
            **span.detail,
        )
        raise
    if span.error is None:
        emit(NODE_COMPLETED, node, subtask_id=subtask_id, **span.detail)
    else:
        emit(NODE_FAILED, node, subtask_id=subtask_id, error=span.error, **span.detail)
