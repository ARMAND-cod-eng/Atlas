"""Node functions for the Atlas graph.

Each node takes the current state and returns a *partial* state — only the keys
it actually changed. LangGraph merges the partial into the running state using
each key's reducer.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from atlas.graph.state import AtlasState
from atlas.models import get_llm

logger = logging.getLogger(__name__)


async def answer_query(state: AtlasState) -> dict[str, Any]:
    """Answer the query with a single model call.

    This is the whole graph in Phase 0. It exists to prove the wiring end to end —
    state schema, provider layer with fallback, async execution, and checkpoint
    persistence — before any of the multi-agent machinery is built on top of it.
    """
    question = HumanMessage(content=state["query"])
    logger.info(
        "answering query",
        extra={
            "query_length": len(state["query"]),
            "history_length": len(state["messages"]),
        },
    )

    # Send the accumulated history along with the new question. The checkpointer
    # restores `messages` for this thread_id, so a follow-up call continues the
    # conversation instead of starting over. Sending only the new question would
    # leave the persisted history write-only — stored every turn, read never.
    response = await get_llm().ainvoke([*state["messages"], question])

    # .text is a property in langchain-core 1.x; calling it as a method is deprecated.
    return {"answer": response.text, "messages": [question, response]}
