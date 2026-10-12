"""Drives one graph run and turns it into a stream of events.

This is the only place that calls ``astream``. Both entry points go through it:
the worker for a streamed run, and ``POST /query`` for a synchronous one — one
execution path, so there is one place for a bug rather than two.

Two guarantees live here, and both are load-bearing for the client:

``seq`` is dense.
    Assigned by this loop and nowhere else. Branches run concurrently but
    LangGraph serialises their chunks into this single consumer, so a plain
    ``+= 1`` is correct. See :mod:`atlas.events`.

A terminal event is always written, exactly once.
    In ``finally``, including on cancellation. A client must be able to tell "the
    run ended" from "the stream broke", and that distinction only exists if the
    terminal event is unconditional.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from atlas import eventlog
from atlas.events import (
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    AgentEvent,
)
from atlas.graph.builder import recursion_limit
from atlas.graph.state import AtlasState
from atlas.metrics import RunMetrics
from atlas.result import build_response, response_payload

logger = logging.getLogger(__name__)


def initial_state(query: str) -> AtlasState:
    """A fresh graph input.

    Typed as ``AtlasState`` so the graph's input contract is checked here, at the
    boundary, instead of failing inside a node.
    """
    return {
        "query": query,
        "subtasks": [],
        "research_results": [],
        "answer": "",
        "messages": [],
        "run_metrics": RunMetrics(),
        "iteration": 0,
        "critic_feedback": [],
    }


class _Sequencer:
    """Assigns ``seq`` and writes to the log. One instance per run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._seq = 0

    async def write(self, event_type: str, node: str | None = None, **payload: object) -> None:
        self._seq += 1
        await eventlog.append(
            AgentEvent(
                run_id=self.run_id,
                seq=self._seq,
                ts=datetime.now(UTC),
                type=event_type,
                node=node,
                payload=payload,
            )
        )


async def execute_run(
    run_id: str,
    query: str,
    thread_id: str,
    graph: CompiledStateGraph[AtlasState, Any, AtlasState, AtlasState],
) -> dict[str, Any]:
    """Run the graph, streaming events to the log, and return the final result.

    ``stream_mode=["custom", "values"]`` is what lets one execution serve both
    purposes: ``custom`` carries the events the nodes emit, and ``values`` carries
    the full state after each superstep, the last of which is the final answer.
    Without ``values`` we would need a second call or a checkpoint read to
    discover what the run actually produced.
    """
    config: RunnableConfig = {
        # thread_id keys the checkpointer (the conversation); run_id keys the
        # token budget (this one question). See atlas.budget.
        "configurable": {"thread_id": thread_id, "run_id": run_id},
        "recursion_limit": recursion_limit(),
    }
    events = _Sequencer(run_id)
    final_state: dict[str, Any] | None = None
    terminal_written = False

    # The API claimed the thread when it queued this run; restart the claim's
    # clock now that the run is actually executing. See eventlog.claim_thread.
    await eventlog.renew_thread(thread_id, run_id)
    await events.write(RUN_STARTED, query=query, thread_id=thread_id)
    logger.info("run started", extra={"run_id": run_id, "thread_id": thread_id})

    try:
        async for item in graph.astream(
            initial_state(query), config=config, stream_mode=["custom", "values"]
        ):
            # With a list of stream modes LangGraph yields (mode, chunk), but its
            # signature is a union across all modes. Checking the shape rather
            # than casting means a LangGraph change surfaces as a skipped chunk
            # here instead of an AttributeError three frames down.
            if not (isinstance(item, tuple) and len(item) == 2):
                logger.warning("unexpected stream chunk shape", extra={"run_id": run_id})
                continue
            mode, chunk = item
            if mode == "custom":
                # A payload from a node: {"type": ..., "node": ..., **fields}.
                payload = dict(chunk)
                event_type = str(payload.pop("type"))
                node = payload.pop("node", None)
                await events.write(event_type, node, **payload)
            elif mode == "values":
                final_state = chunk

        if final_state is None:
            raise RuntimeError("the graph produced no state")

        response = build_response(final_state, thread_id)
        result = response_payload(response)
        await eventlog.store_result(run_id, result)

        # A summary, not the whole response. The full sources list and per-call
        # metrics are fetched once from GET /runs/{run_id} rather than being
        # replayed on every reconnect.
        await events.write(
            RUN_COMPLETED,
            answer=response.answer,
            thread_id=thread_id,
            iterations=response.iterations,
            subtask_count=len(response.subtasks),
            source_count=len(response.sources),
            degraded=response.metrics.degraded,
            total_tokens=response.metrics.total_tokens,
        )
        terminal_written = True
        logger.info("run completed", extra={"run_id": run_id, "iterations": response.iterations})
        return result

    except BaseException as exc:
        # BaseException so a cancelled run still terminates its stream. Without
        # this a client would sit on an open connection forever waiting for an
        # ending that is never coming.
        await events.write(
            RUN_FAILED,
            error=f"{type(exc).__name__}: {exc}"[:500],
            error_type=type(exc).__name__,
        )
        terminal_written = True
        logger.exception("run failed", extra={"run_id": run_id})
        raise

    finally:
        if not terminal_written:
            # Belt and braces. Reachable only if the success path raised after
            # the try block was left, which should be impossible — but "should be
            # impossible" is not a guarantee, and a client hanging forever is a
            # bad way to discover that.
            await events.write(RUN_FAILED, error="run ended without a terminal event")
        # Last, after the terminal event: the thread is free once the run is over,
        # whichever way it ended.
        await eventlog.release_thread(thread_id, run_id)
