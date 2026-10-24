"""Forgetting a conversation: on request, and once it has been idle too long.

A thread lives in four places, and deleting it means all of them: its run
records, the agent calls and events recorded for those runs (atlas.store), the
graph's checkpoints, and whatever Redis still holds. People's questions are
personal data, and a tool that answers questions about the GDPR should keep
them no longer than it has reason to.

Nothing is deleted from under a running run: forgetting a thread claims it
first, exactly as a run does, and a thread that is busy is refused (on request)
or left for the next night (by the purge).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from atlas import eventlog
from atlas.store import RunStore

logger = logging.getLogger(__name__)


class ThreadBusyError(RuntimeError):
    """The thread has a run in progress, so it cannot be deleted now."""


async def forget_thread(
    thread_id: str, store: RunStore, checkpointer: BaseCheckpointSaver[str]
) -> list[str]:
    """Delete a thread everywhere it lives. Returns its run ids, oldest first."""
    claim = f"forget:{uuid4().hex}"
    holder = await eventlog.claim_thread(thread_id, claim)
    if holder is not None:
        raise ThreadBusyError(f"thread {thread_id} has a run in progress: {holder}")
    try:
        run_ids = await store.delete_thread(thread_id)
        await checkpointer.adelete_thread(thread_id)
        await eventlog.forget_runs(run_ids)
    finally:
        await eventlog.release_thread(thread_id, claim)
    logger.info("thread forgotten", extra={"thread_id": thread_id, "runs": len(run_ids)})
    return run_ids


async def purge_expired(
    store: RunStore,
    checkpointer: BaseCheckpointSaver[str],
    retention_days: int,
    now: datetime,
) -> list[str]:
    """Forget every thread idle for longer than ``retention_days``. 0 keeps all.

    A busy thread is skipped, not failed: it is in use, so it is not idle, and
    the next night's purge will look again.
    """
    if retention_days <= 0:
        return []
    purged: list[str] = []
    for thread_id in await store.expired_threads(now - timedelta(days=retention_days)):
        try:
            await forget_thread(thread_id, store, checkpointer)
        except ThreadBusyError:
            logger.info("skipping a busy thread", extra={"thread_id": thread_id})
            continue
        purged.append(thread_id)
    return purged
