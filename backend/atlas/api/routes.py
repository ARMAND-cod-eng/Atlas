"""HTTP routes.

The API no longer executes runs — it enqueues them and reads their results. That
is the point of Phase 3: a 67-second run must not be owned by a connection that
can drop at second 40.

The health probes are module-level functions rather than inline logic so tests
can substitute them without standing up Postgres or Redis.
"""

from __future__ import annotations

import base64
import binascii
import logging
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from arq.connections import ArqRedis
from arq.jobs import Job
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import text

from atlas import eventlog
from atlas.config import get_settings
from atlas.db.session import get_engine
from atlas.domains import DomainName
from atlas.events import RUN_FAILED
from atlas.graph.state import Revision
from atlas.result import QueryResponse
from atlas.review.contract import MAX_BYTES, UnreadableContract, text_of
from atlas.review.graph import ReviewResult
from atlas.review.requirements import FRAMEWORKS
from atlas.store import ReviewDecision, RunKind, RunRecord, RunStatusName, RunStore
from atlas.trace import AgentCall, now
from atlas.worker import FORGET_THREAD, QUERY_WAIT_SECONDS, RUN_RESEARCH, RUN_REVIEW, RUN_REVISION

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Inbound research request."""

    # Capped: every character is multiplied across the fan-out and every prompt
    # after it, so an unbounded question is an unbounded bill.
    query: str = Field(min_length=1, max_length=2000, description="The question to answer.")
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread to continue. Omit to start a new one; the "
            "generated id is returned so the next call can resume this thread."
        ),
    )
    parent_run_id: str | None = Field(
        default=None,
        description=(
            "Continue from this run's answer rather than the thread's latest. "
            "After a revision the latest is the revision's; a follow-up asked while "
            "reading the original means the original. Implies its thread."
        ),
    )
    domain: DomainName | None = Field(
        default=None,
        description=(
            "compliance: the AI Act and the GDPR, every citation checked against them. "
            "general: research from the web. Chosen by a conversation's first question "
            "(the configured domain when omitted) and kept by every run on it."
        ),
    )


class RevisionRequest(BaseModel):
    """Rewrite one branch's question in a finished run, and re-run that branch."""

    subtask_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)


class RunSummary(BaseModel):
    """One run on a thread, as the versions list shows it."""

    run_id: str
    kind: RunKind
    query: str
    parent_run_id: str | None
    revision: Revision | None
    status: RunStatusName
    created_at: datetime
    finished_at: datetime | None
    domain: DomainName | None = None


class CallSummary(BaseModel):
    """One agent call, as a list of them shows it: who, when, cost, a preview."""

    call_id: str
    span_id: str | None
    node: str
    subtask_id: str | None
    kind: str
    name: str
    started_at: datetime
    duration_s: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None
    preview: str


class RunCalls(BaseModel):
    run_id: str
    calls: list[CallSummary]


class ThreadForgetting(BaseModel):
    """A thread queued to be forgotten, and the runs that go with it."""

    thread_id: str
    runs: list[str]


class ThreadVersions(BaseModel):
    """Every run on a thread, oldest first: questions, follow-ups and revisions."""

    thread_id: str
    runs: list[RunSummary]


class RunAccepted(BaseModel):
    """What ``POST /runs`` returns immediately, before any work happens.

    ``run_id`` identifies this execution and keys its event stream; ``thread_id``
    identifies the conversation and keys the checkpointer. Different lifetimes —
    one thread accumulates many runs — which is why both are returned.
    """

    run_id: str
    thread_id: str
    events_url: str


class RunStatus(BaseModel):
    """Where a run has got to, and its result once there is one."""

    run_id: str
    status: Literal["running", "completed", "failed"]
    result: QueryResponse | None = None
    # The error the run ended on, when status is "failed".
    error: str | None = None
    # Where a contract review's findings are; a review has no research result.
    review_url: str | None = None


class ReviewRequest(BaseModel):
    """A contract to review: its text, or its file as base64. One of the two."""

    filename: str = Field(min_length=1, max_length=255, description="Its name, with its extension.")
    text: str | None = Field(default=None, description="The contract's text, pasted.")
    content_base64: str | None = Field(
        default=None,
        # Base64 of 5 MB, and a little: the file's own limit is checked once decoded.
        max_length=4 * (MAX_BYTES // 3) + 8,
        description="The file, base64: .txt, .md, .docx or a PDF with text in it.",
    )
    framework: Literal["gdpr-art-28"] = "gdpr-art-28"


class ReviewStatus(BaseModel):
    """Where a review has got to, and its findings once there are some."""

    run_id: str
    status: Literal["running", "completed", "failed"]
    result: ReviewResult | None = None
    error: str | None = None


class DecisionRequest(BaseModel):
    """A reviewer's decision on one finding."""

    decision: Literal["agree", "disagree"]
    note: str = Field(default="", max_length=2000)
    reviewer: str = Field(min_length=1, max_length=100, description="Who decides, as they give it.")


class ReviewDecisions(BaseModel):
    """A review's decisions: the current one per finding, and every one made."""

    current: list[ReviewDecision]
    history: list[ReviewDecision]


class HealthResponse(BaseModel):
    """Liveness of Atlas and each backing service."""

    status: str
    db: bool
    redis: bool


async def check_db(request: Request) -> bool:
    """Round-trip a trivial query against Postgres.

    Goes through the SQLAlchemy engine rather than the checkpointer's pool,
    because the API no longer owns that pool — the worker does. Same database,
    same answer to the question the probe is asking.
    """
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        # A failed probe is a reportable state, not a server error — the endpoint
        # must still answer so a load balancer learns *which* dependency is down.
        logger.exception("database health check failed")
        return False
    return True


async def check_redis(request: Request) -> bool:
    """Ping Redis."""
    client: Redis = request.app.state.redis
    try:
        await client.ping()
    except Exception:
        logger.exception("redis health check failed")
        return False
    return True


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report real connectivity to each backing service.

    Always 200, even when a dependency is down — the body carries the detail.
    Returning 503 would collapse "Atlas is unreachable" and "Redis is unreachable"
    into the same signal, which is exactly the distinction this endpoint exists
    to make.
    """
    db_ok = await check_db(request)
    redis_ok = await check_redis(request)
    return HealthResponse(
        status="ok" if db_ok and redis_ok else "degraded",
        db=db_ok,
        redis=redis_ok,
    )


def _store(request: Request) -> RunStore:
    """The durable run record. Set at startup (atlas.api.main), or by a test."""
    store: RunStore | None = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="run history is unavailable")
    return store


async def _finished_run(request: Request, run_id: str, purpose: str) -> RunRecord:
    """A run that completed, for a revision or a follow-up to build on."""
    run = await _store(request).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    if run.status == "failed":
        raise HTTPException(
            status_code=409, detail=f"run {run_id} failed, so it has no answer to {purpose}"
        )
    if run.status != "completed":
        raise HTTPException(status_code=409, detail=f"run {run_id} is still running")
    return run


async def _enqueue(
    request: Request, thread_id: str, function: str, *args: str, domain: DomainName
) -> tuple[Job, str]:
    """Claim the thread, then queue one job for a fresh run. Returns ``(job, run_id)``."""
    arq: ArqRedis = request.app.state.arq
    run_id = str(uuid4())

    # Claimed before anything is queued or spent; released by the runner when the
    # run ends. See eventlog.claim_thread for why two runs on one thread is a bug.
    holder = await eventlog.claim_thread(thread_id, run_id)
    if holder is not None:
        raise HTTPException(
            status_code=409,
            detail=f"thread {thread_id} already has a run in progress: {holder}",
        )

    try:
        # Register before enqueueing. A client can subscribe the instant this call
        # returns, which is sooner than the worker can write its first event.
        await eventlog.register(run_id)
        # _job_id ties the arq job to the run, so the job's result and the run's
        # event stream are addressable by the same identifier.
        job = await arq.enqueue_job(function, run_id, *args, _job_id=run_id, domain=domain)
    except BaseException:
        await eventlog.release_thread(thread_id, run_id)
        raise
    if job is None:
        # arq returns None when a job with this id already exists. run_id is a
        # fresh uuid, so this means Redis is not in the state we believe it is —
        # better to say so than to hand back a run_id nothing will execute. The
        # claim goes too, or the thread stays locked by a run that never existed.
        await eventlog.release_thread(thread_id, run_id)
        logger.error("enqueue returned no job", extra={"run_id": run_id})
        raise HTTPException(status_code=503, detail="could not enqueue the run")
    logger.info("run enqueued", extra={"run_id": run_id, "thread_id": thread_id, "job": function})
    return job, run_id


async def _thread_domain(request: Request, thread_id: str) -> DomainName | None:
    """The domain a conversation researches in: its first run's. None for a new thread."""
    store: RunStore | None = getattr(request.app.state, "store", None)
    if store is None:
        return None
    runs = await store.thread(thread_id)
    return runs[0].domain if runs else None


def _domain_for(thread_id: str, kept: DomainName | None, asked: DomainName | None) -> DomainName:
    """The domain a run on this thread runs in: the conversation's, which a run cannot switch."""
    if kept is None:
        return asked or get_settings().atlas_domain
    if asked is not None and asked != kept:
        raise HTTPException(
            status_code=422,
            detail=(
                f"thread {thread_id} researches in the {kept} domain; "
                f"start a new conversation to ask in the {asked} one"
            ),
        )
    return kept


async def enqueue_run(request: Request, payload: QueryRequest) -> tuple[Job, str, str]:
    """Put one question on the queue. Returns ``(job, run_id, thread_id)``."""
    thread_id = payload.thread_id
    parent_run_id: tuple[str, ...] = ()
    kept: DomainName | None = None
    if payload.parent_run_id is not None:
        parent = await _finished_run(request, payload.parent_run_id, "continue from")
        if thread_id is not None and thread_id != parent.thread_id:
            raise HTTPException(
                status_code=422,
                detail=f"run {parent.run_id} belongs to thread {parent.thread_id}, not {thread_id}",
            )
        thread_id = parent.thread_id
        parent_run_id = (parent.run_id,)
        kept = parent.domain
    elif thread_id is not None:
        kept = await _thread_domain(request, thread_id)
    thread_id = thread_id or str(uuid4())
    domain = _domain_for(thread_id, kept, payload.domain)
    job, run_id = await _enqueue(
        request, thread_id, RUN_RESEARCH, payload.query, thread_id, *parent_run_id, domain=domain
    )
    return job, run_id, thread_id


@router.post("/runs", response_model=RunAccepted, status_code=202)
async def start_run(request: Request, payload: QueryRequest) -> RunAccepted:
    """Start a run and return immediately.

    202 rather than 200: nothing has been researched yet. The caller subscribes
    to ``events_url`` to watch it happen, and may drop and reconnect freely —
    the run is not tied to that connection.
    """
    _job, run_id, thread_id = await enqueue_run(request, payload)
    return RunAccepted(run_id=run_id, thread_id=thread_id, events_url=f"/runs/{run_id}/events")


@router.post("/runs/{run_id}/revisions", response_model=RunAccepted, status_code=202)
async def revise_run(request: Request, run_id: str, payload: RevisionRequest) -> RunAccepted:
    """Rewrite one branch's question in a finished run, and re-run that branch.

    The revision is a new run on the same thread. Every other branch's research
    is carried over from the parent, so only the edited branch searches again;
    the answer is then re-written and reviewed. See atlas.revision.
    """
    parent = await _finished_run(request, run_id, "revise")
    questions = {
        str(subtask["id"]): str(subtask["question"])
        for subtask in (parent.result or {}).get("subtasks", [])
    }
    if payload.subtask_id not in questions:
        known = ", ".join(questions) or "none"
        raise HTTPException(
            status_code=422,
            detail=f"run {run_id} has no subtask {payload.subtask_id} (it has: {known})",
        )
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="the edited question is empty")
    if question == questions[payload.subtask_id].strip():
        # A re-run on the same question is a retry: a search and three model
        # calls to learn nothing anyone asked for.
        raise HTTPException(
            status_code=422, detail=f"the question for {payload.subtask_id} is unchanged"
        )

    _job, revision_id = await _enqueue(
        request,
        parent.thread_id,
        RUN_REVISION,
        parent.thread_id,
        parent.run_id,
        payload.subtask_id,
        question,
        # A revision re-researches under the rules its parent was researched by.
        domain=parent.domain or get_settings().atlas_domain,
    )
    return RunAccepted(
        run_id=revision_id, thread_id=parent.thread_id, events_url=f"/runs/{revision_id}/events"
    )


@router.get("/runs/{run_id}/calls", response_model=RunCalls)
async def run_calls(request: Request, run_id: str) -> RunCalls:
    """Every call the run's agents made, without the prompts and outputs.

    Those run to tens of kilobytes each and most are never opened, so they
    are fetched one call at a time from ``/runs/{run_id}/calls/{call_id}``.
    """
    store = _store(request)
    if await store.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return RunCalls(
        run_id=run_id,
        calls=[
            CallSummary(**call.summary(), node=call.node, started_at=call.started_at)
            for call in await store.calls(run_id)
        ],
    )


@router.get("/runs/{run_id}/calls/{call_id}", response_model=AgentCall)
async def run_call(request: Request, run_id: str, call_id: str) -> AgentCall:
    """One call in full: the exact messages sent, and what came back."""
    call = await _store(request).call(run_id, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail=f"unknown call: {call_id}")
    return call


@router.get("/threads/{thread_id}/versions", response_model=ThreadVersions)
async def thread_versions(request: Request, thread_id: str) -> ThreadVersions:
    """Every run on a thread, with what each revision changed."""
    runs = await _store(request).thread(thread_id)
    if not runs:
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    return ThreadVersions(
        thread_id=thread_id,
        runs=[RunSummary.model_validate(run.model_dump()) for run in runs],
    )


@router.delete("/threads/{thread_id}", response_model=ThreadForgetting, status_code=202)
async def forget(request: Request, thread_id: str) -> ThreadForgetting:
    """Delete a thread everywhere it lives: records, calls, events, checkpoints.

    Queued, because the checkpoints belong to the worker's checkpointer. A
    thread with a run in progress is refused: deleting under a running run
    would leave it writing to a conversation that no longer exists.
    """
    runs = await _store(request).thread(thread_id)
    if not runs:
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    holder = await eventlog.thread_holder(thread_id)
    if holder is not None:
        raise HTTPException(
            status_code=409, detail=f"thread {thread_id} has a run in progress: {holder}"
        )
    arq: ArqRedis = request.app.state.arq
    await arq.enqueue_job(FORGET_THREAD, thread_id, _job_id=f"forget:{thread_id}")
    logger.info("thread queued to be forgotten", extra={"thread_id": thread_id})
    return ThreadForgetting(thread_id=thread_id, runs=[run.run_id for run in runs])


@router.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(request: Request, run_id: str) -> RunStatus:
    """The full result of a run.

    ``run_completed`` carries a summary only, so this is where a client fetches
    the complete sources list and per-call metrics — once, rather than having
    them replayed on every reconnect.
    """
    result = await eventlog.load_result(run_id)
    if result is not None and "findings" in result:
        return RunStatus(run_id=run_id, status="completed", review_url=f"/reviews/{run_id}")
    if result is not None:
        return RunStatus(run_id=run_id, status="completed", result=QueryResponse(**result))

    # No stored result does not mean "still running". A result is stored only on
    # success, so a failed run used to land in the branch below and report
    # "running" for as long as its event log lived — contradicting the
    # run_failed its own stream had already delivered. The terminal event is
    # written on every path, so it decides.
    terminal = await eventlog.terminal_event(run_id)
    if terminal is not None and terminal.type == RUN_FAILED:
        error = terminal.payload.get("error")
        return RunStatus(
            run_id=run_id, status="failed", error=error if isinstance(error, str) else None
        )
    if terminal is not None:
        # Completed, but store_result failed open. Say so rather than "running".
        return RunStatus(run_id=run_id, status="completed")

    if await eventlog.exists(run_id):
        return RunStatus(run_id=run_id, status="running")

    # Redis keeps a run for an hour. The record keeps it for as long as its
    # thread: a revision's parent, opened from the versions list a day later,
    # is still a run.
    store: RunStore | None = getattr(request.app.state, "store", None)
    record = await store.get(run_id) if store is not None else None
    if record is not None and record.kind == "review":
        return RunStatus(
            run_id=run_id, status=record.status, error=record.error, review_url=f"/reviews/{run_id}"
        )
    if record is not None:
        return RunStatus(
            run_id=run_id,
            status=record.status,
            result=QueryResponse(**record.result) if record.result is not None else None,
            error=record.error,
        )
    raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")


def _contract_text(payload: ReviewRequest) -> str:
    """The contract's text, or 422 saying why it cannot be read."""
    if (payload.text is None) == (payload.content_base64 is None):
        raise HTTPException(
            status_code=422, detail="Send the contract's text or its file, one of the two."
        )
    if payload.text is not None:
        data = payload.text.encode("utf-8")
    else:
        try:
            data = base64.b64decode(payload.content_base64 or "", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"{payload.filename} could not be decoded."
            ) from exc
    # Pasted text is text, whatever the name it came with.
    name = payload.filename if payload.content_base64 is not None else f"{payload.filename}.txt"
    try:
        return text_of(name, data)
    except UnreadableContract as exc:
        raise HTTPException(
            status_code=422, detail=str(exc).replace(name, payload.filename)
        ) from exc


@router.post("/reviews", response_model=RunAccepted, status_code=202)
async def start_review(request: Request, payload: ReviewRequest) -> RunAccepted:
    """Review a contract against GDPR Art. 28, and return at once.

    The contract is read here, so a file that cannot be read is refused before
    anything is queued or spent. Its findings stream like any run's events, and
    ``GET /reviews/{run_id}`` serves them once finished.
    """
    text = _contract_text(payload)
    thread_id = str(uuid4())
    _job, run_id = await _enqueue(
        request, thread_id, RUN_REVIEW, thread_id, payload.filename, text, domain="compliance"
    )
    return RunAccepted(run_id=run_id, thread_id=thread_id, events_url=f"/runs/{run_id}/events")


@router.get("/reviews/{run_id}", response_model=ReviewStatus)
async def get_review(request: Request, run_id: str) -> ReviewStatus:
    """A review's findings, from the run's stored result, or its record once Redis forgets it."""
    result = await eventlog.load_result(run_id)
    if result is not None and "findings" in result:
        return ReviewStatus(run_id=run_id, status="completed", result=ReviewResult(**result))
    store: RunStore | None = getattr(request.app.state, "store", None)
    record = await store.get(run_id) if store is not None else None
    if record is None or record.kind != "review":
        raise HTTPException(status_code=404, detail=f"unknown review: {run_id}")
    return ReviewStatus(
        run_id=run_id,
        status=record.status,
        result=ReviewResult(**record.result) if record.result is not None else None,
        error=record.error,
    )


async def _review_record(request: Request, run_id: str) -> RunRecord:
    record = await _store(request).get(run_id)
    if record is None or record.kind != "review":
        raise HTTPException(status_code=404, detail=f"unknown review: {run_id}")
    return record


@router.put("/reviews/{run_id}/decisions/{requirement_id}", response_model=ReviewDecision)
async def decide(
    request: Request, run_id: str, requirement_id: str, payload: DecisionRequest
) -> ReviewDecision:
    """Agree or disagree with one finding of a finished review. Kept after any it replaces."""
    record = await _review_record(request, run_id)
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"review {run_id} is not finished")
    framework = FRAMEWORKS[str((record.result or {}).get("framework", "gdpr-art-28"))]
    if requirement_id not in {requirement.id for requirement in framework.requirements}:
        raise HTTPException(
            status_code=404, detail=f"this review has no requirement {requirement_id}"
        )
    reviewer = payload.reviewer.strip()
    if not reviewer:
        raise HTTPException(status_code=422, detail="a decision needs the reviewer's name")
    decision = ReviewDecision(
        run_id=run_id,
        requirement_id=requirement_id,
        decision=payload.decision,
        note=payload.note.strip(),
        reviewer=reviewer,
        # Strictly increasing within the process, so two decisions made back to
        # back keep their order on a clock that ticks coarsely.
        decided_at=now(),
    )
    await _store(request).add_decision(decision)
    return decision


@router.get("/reviews/{run_id}/decisions", response_model=ReviewDecisions)
async def review_decisions(request: Request, run_id: str) -> ReviewDecisions:
    """The current decision on each finding, and every decision made, oldest first."""
    await _review_record(request, run_id)
    history = await _store(request).decisions(run_id)
    # The latest per finding wins.
    current = {decision.requirement_id: decision for decision in history}
    return ReviewDecisions(current=list(current.values()), history=history)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Run a query and wait for the answer.

    Unchanged in shape since Phase 0, for clients that want one request and one
    answer rather than a stream. (The eval suite does not use it — it runs the
    graph in-process against a frozen corpus; see ``atlas_eval.runner``.) It is
    now a thin wrapper over the same queue the streaming path uses, so there is
    one execution path rather than two.

    The cost, stated plainly: this call's latency now includes queue pickup, and
    its throughput is bounded by the worker's ``max_jobs``.

    A run that fails re-raises the worker's exception here, so a ProviderError
    still reaches the 502 handler exactly as it did when the graph ran in-process.
    """
    job, run_id, _thread_id = await enqueue_run(request, payload)
    try:
        result: dict[str, Any] = await job.result(timeout=QUERY_WAIT_SECONDS)
    except TimeoutError:
        logger.error("query timed out waiting for the worker", extra={"run_id": run_id})
        raise HTTPException(status_code=504, detail="the run did not complete in time") from None
    return QueryResponse(**result)
