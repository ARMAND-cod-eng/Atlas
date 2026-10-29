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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from atlas import eventlog
from atlas.config import get_settings
from atlas.domains import DomainName
from atlas.events import (
    AGENT_CALL,
    REVISION_STARTED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_STARTED,
    AgentEvent,
)
from atlas.graph.builder import recursion_limit
from atlas.graph.state import AtlasState
from atlas.metrics import DEGRADED_AUDIT_INCOMPLETE, RunMetrics
from atlas.result import build_response
from atlas.review.graph import review_result
from atlas.review.requirements import GDPR_ART_28
from atlas.revision import final_checkpoint, fork_for_revision, revision_started_payload
from atlas.store import RunKind, RunRecord, RunStore
from atlas.trace import AgentCall

logger = logging.getLogger(__name__)


def initial_state(query: str, *, check_scope: bool = False) -> AtlasState:
    """A fresh graph input.

    Typed as ``AtlasState`` so the graph's input contract is checked here, at the
    boundary, instead of failing inside a node. ``check_scope`` acts on the
    planner's judgment of whether the question is one for the domain at all;
    ``execute_run`` sets it for a conversation's first question.
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
        "revision": None,
        "check_scope": check_scope,
        "declined": None,
    }


async def _has_history(graph: Graph, thread_id: str) -> bool:
    """Whether the thread already holds a conversation: a question asked before."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return bool(snapshot.values.get("messages"))


class _Sequencer:
    """Assigns ``seq`` and writes to the log. One instance per run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._seq = 0
        # Every event written, for the archive once the run ends.
        self.written: list[AgentEvent] = []

    async def write(self, event_type: str, node: str | None = None, **payload: object) -> None:
        self._seq += 1
        event = AgentEvent(
            run_id=self.run_id,
            seq=self._seq,
            ts=datetime.now(UTC),
            type=event_type,
            node=node,
            payload=payload,
        )
        self.written.append(event)
        await eventlog.append(event)


Graph = CompiledStateGraph[AtlasState, Any, AtlasState, AtlasState]
# Any graph the runner can run: research, or a contract review (atlas.review.graph).
AnyGraph = CompiledStateGraph[Any, Any, Any, Any]


@dataclass(frozen=True, slots=True)
class _Response:
    """What a run produced: the result kept and served, and the summary its stream ends on."""

    result: BaseModel
    # The result's own metrics, so a marker added here is in what is served.
    metrics: RunMetrics
    # run_completed's payload, besides the degraded markers and total tokens.
    summary: dict[str, Any]


def _research_response(final_state: dict[str, Any], thread_id: str) -> _Response:
    response = build_response(final_state, thread_id)
    return _Response(
        result=response,
        metrics=response.metrics,
        summary={
            "answer": response.answer,
            "thread_id": thread_id,
            "iterations": response.iterations,
            "subtask_count": len(response.subtasks),
            "source_count": len(response.sources),
        },
    )


def _review_response(final_state: dict[str, Any], thread_id: str) -> _Response:
    review = review_result(final_state, thread_id)
    return _Response(
        result=review,
        metrics=review.metrics,
        summary={"thread_id": thread_id, "framework": review.framework, "counts": review.counts},
    )


@dataclass(frozen=True, slots=True)
class _Start:
    """How a run begins: what goes into the graph, and what the record says."""

    graph_input: Mapping[str, Any] | None
    config: RunnableConfig
    record: RunRecord
    # Events written after run_started and before the graph's first.
    opening: tuple[tuple[str, dict[str, Any]], ...] = ()


async def execute_run(
    run_id: str,
    query: str,
    thread_id: str,
    graph: Graph,
    *,
    parent_run_id: str | None = None,
    store: RunStore | None = None,
    domain: DomainName | None = None,
) -> dict[str, Any]:
    """A question on the thread — the first one, or a follow-up.

    A follow-up continues the thread's latest state, unless ``parent_run_id``
    names a run to continue from instead. After a revision the latest state is
    the revision's, and a person reading the original and asking a follow-up
    there means the original.

    Only a conversation's first question is judged for scope. A follow-up reads
    in the light of the conversation ("and from when?"), and asking a declined
    question again in its conversation is how a person researches it anyway.
    """

    async def start() -> _Start:
        configurable: dict[str, Any] = {"thread_id": thread_id, "run_id": run_id}
        if domain is not None:
            # Every node reads it through atlas.domains.current_domain.
            configurable["domain"] = domain
        if parent_run_id is not None:
            parent = await final_checkpoint(graph, thread_id, parent_run_id)
            configurable["checkpoint_id"] = parent["configurable"]["checkpoint_id"]
        first = parent_run_id is None and not await _has_history(graph, thread_id)
        return _Start(
            graph_input=initial_state(query, check_scope=first),
            # thread_id keys the checkpointer (the conversation); run_id keys the
            # token budget (this one question). See atlas.budget.
            config={"configurable": configurable, "recursion_limit": recursion_limit()},
            record=_record(run_id, thread_id, "question", query, parent_run_id, domain),
        )

    fallback = _record(run_id, thread_id, "question", query, parent_run_id, domain)
    return await _execute(run_id, thread_id, graph, start, fallback, store, _research_response)


async def execute_revision(
    run_id: str,
    thread_id: str,
    graph: Graph,
    *,
    parent_run_id: str,
    subtask_id: str,
    question: str,
    store: RunStore | None = None,
    domain: DomainName | None = None,
) -> dict[str, Any]:
    """Re-run one edited branch of a finished run. See :mod:`atlas.revision`."""

    async def start() -> _Start:
        fork = await fork_for_revision(
            graph,
            thread_id=thread_id,
            parent_run_id=parent_run_id,
            run_id=run_id,
            subtask_id=subtask_id,
            question=question,
        )
        values = (await graph.aget_state(fork)).values
        record = _record(run_id, thread_id, "revision", values["query"], parent_run_id, domain)
        configurable = {**fork["configurable"], **({"domain": domain} if domain else {})}
        return _Start(
            graph_input=None,
            config={"configurable": configurable, "recursion_limit": recursion_limit()},
            record=record.model_copy(update={"revision": values["revision"]}),
            opening=((REVISION_STARTED, revision_started_payload(values)),),
        )

    # The question is the parent's, and is not known until its state is read. A
    # revision refused before then is recorded without one.
    fallback = _record(run_id, thread_id, "revision", "", parent_run_id, domain)
    return await _execute(run_id, thread_id, graph, start, fallback, store, _research_response)


async def execute_review(
    run_id: str,
    thread_id: str,
    graph: AnyGraph,
    *,
    filename: str,
    text: str,
    framework: str = GDPR_ART_28.id,
    store: RunStore | None = None,
) -> dict[str, Any]:
    """Review a contract against a framework's requirements. See :mod:`atlas.review`.

    Against the law, so in the compliance domain whatever the server's default.
    """

    async def start() -> _Start:
        return _Start(
            graph_input={
                "filename": filename,
                "text": text,
                "framework": framework,
                "clauses": [],
                "findings": [],
                "run_metrics": RunMetrics(),
            },
            config={
                "configurable": {"thread_id": thread_id, "run_id": run_id, "domain": "compliance"},
                "recursion_limit": recursion_limit(),
            },
            record=_record(run_id, thread_id, "review", filename, None, "compliance"),
        )

    fallback = _record(run_id, thread_id, "review", filename, None, "compliance")
    return await _execute(run_id, thread_id, graph, start, fallback, store, _review_response)


def _record(
    run_id: str,
    thread_id: str,
    kind: RunKind,
    query: str,
    parent_run_id: str | None,
    domain: DomainName | None,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        thread_id=thread_id,
        kind=kind,
        query=query,
        parent_run_id=parent_run_id,
        created_at=datetime.now(UTC),
        # Recorded as it ran: the configured domain when the job named none.
        domain=domain or get_settings().atlas_domain,
    )


async def _audited(write: Awaitable[None], run_id: str) -> bool:
    """Write to the run store; on failure, log and report it rather than raise.

    The record is the audit trail, not the answer. A store outage must not cost
    a user an answer the run already produced — but the run says the trail is
    incomplete, because an absent record is otherwise indistinguishable from a
    run that never happened.
    """
    try:
        await write
    except Exception:
        logger.exception("could not write the run record", extra={"run_id": run_id})
        return False
    return True


async def _execute(
    run_id: str,
    thread_id: str,
    graph: AnyGraph,
    begin: Callable[[], Awaitable[_Start]],
    fallback: RunRecord,
    store: RunStore | None,
    respond: Callable[[dict[str, Any], str], _Response],
) -> dict[str, Any]:
    """Run the graph, streaming events to the log, and return the final result.

    ``stream_mode=["custom", "values"]`` is what lets one execution serve both
    purposes: ``custom`` carries the events the nodes emit, and ``values`` carries
    the full state after each superstep, the last of which is the final answer.
    Without ``values`` we would need a second call or a checkpoint read to
    discover what the run actually produced.

    ``begin`` runs inside the ``try``: a revision's fork can be refused (an
    unknown branch, a parent that never finished), and a refused run still owes
    its client a terminal event and its thread a release.
    """
    events = _Sequencer(run_id)
    final_state: dict[str, Any] | None = None
    terminal_written = False
    recorded = False
    audit_complete = True

    # The API claimed the thread when it queued this run; restart the claim's
    # clock now that the run is actually executing. See eventlog.claim_thread.
    await eventlog.renew_thread(thread_id, run_id)

    try:
        start = await begin()
        if store is not None:
            recorded = audit_complete = await _audited(store.started(start.record), run_id)
        await events.write(
            RUN_STARTED,
            query=start.record.query,
            thread_id=thread_id,
            # Which kind of run this is: the law, checked, or general research.
            domain=start.record.domain,
        )
        for event_type, payload in start.opening:
            await events.write(event_type, **payload)
        logger.info("run started", extra={"run_id": run_id, "thread_id": thread_id})

        async for item in graph.astream(
            start.graph_input, config=start.config, stream_mode=["custom", "values"]
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
                if event_type == AGENT_CALL:
                    # The whole call to the store, a summary to the stream: replay
                    # re-sends the stream to every reconnecting client, and a
                    # synthesizer's prompt holds every branch's findings.
                    call = AgentCall.model_validate({**payload, "node": node})
                    if store is not None:
                        written = await _audited(store.add_call(run_id, call), run_id)
                        audit_complete = audit_complete and written
                    payload = call.summary()
                await events.write(event_type, node, **payload)
            elif mode == "values":
                final_state = chunk

        if final_state is None:
            raise RuntimeError("the graph produced no state")

        response = respond(final_state, thread_id)
        if store is not None:
            finished = store.finished(
                run_id, status="completed", result=response.result.model_dump(mode="json")
            )
            audit_complete = await _audited(finished, run_id) and audit_complete
        if not audit_complete:
            response.metrics.degraded.append(DEGRADED_AUDIT_INCOMPLETE)
        result = response.result.model_dump(mode="json")
        await eventlog.store_result(run_id, result)

        # A summary, not the whole response. The full sources list and per-call
        # metrics are fetched once from GET /runs/{run_id} rather than being
        # replayed on every reconnect.
        await events.write(
            RUN_COMPLETED,
            **response.summary,
            degraded=response.metrics.degraded,
            total_tokens=response.metrics.total_tokens,
        )
        terminal_written = True
        logger.info("run completed", extra={"run_id": run_id})
        return result

    except BaseException as exc:
        # BaseException so a cancelled run still terminates its stream. Without
        # this a client would sit on an open connection forever waiting for an
        # ending that is never coming.
        error = f"{type(exc).__name__}: {exc}"[:500]
        await events.write(RUN_FAILED, error=error, error_type=type(exc).__name__)
        terminal_written = True
        if store is not None:
            if not recorded:
                recorded = await _audited(store.started(fallback), run_id)
            if recorded:
                await _audited(store.finished(run_id, status="failed", error=error), run_id)
        logger.exception("run failed", extra={"run_id": run_id})
        raise

    finally:
        if not terminal_written:
            # Belt and braces. Reachable only if the success path raised after
            # the try block was left, which should be impossible — but "should be
            # impossible" is not a guarantee, and a client hanging forever is a
            # bad way to discover that.
            await events.write(RUN_FAILED, error="run ended without a terminal event")
        # Redis keeps the stream for an hour; a version opened tomorrow from the
        # versions list still has a timeline. After the terminal event, so the
        # archive is the whole run.
        if store is not None:
            await _audited(store.archive_events(run_id, events.written), run_id)
        # Last: the thread is free once the run is over, whichever way it ended.
        await eventlog.release_thread(thread_id, run_id)
