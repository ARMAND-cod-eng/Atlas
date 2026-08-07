"""The LangGraph state schema.

The state is the contract between nodes: everything one node needs from another
travels through here, which is what lets nodes stay independently testable.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AtlasState(TypedDict):
    """State threaded through every node in the graph.

    Phase 0 carries only the query, the answer, and the message history.

    This grows in later phases to hold the planner's decomposed subtasks, the
    per-researcher findings that the fan-out produces, and the critic's structured
    feedback — the last of which is what lets the critic route back to an earlier
    node instead of merely annotating a finished answer.
    """

    query: str
    answer: str
    # add_messages appends rather than overwrites, and reconciles message IDs, so
    # parallel branches can each contribute to one history without clobbering.
    messages: Annotated[list[AnyMessage], add_messages]
