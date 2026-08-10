"""Graph assembly.

Compiles against an in-memory checkpointer so the graph's shape can be verified
without Postgres — which is why builder.py keeps assembly and compilation in
separate functions.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from atlas.config import get_settings
from atlas.graph import nodes
from atlas.graph.builder import build_state_graph, recursion_limit
from atlas.graph.state import AtlasState


def test_graph_compiles() -> None:
    compiled = build_state_graph().compile(checkpointer=MemorySaver())
    assert compiled is not None


def test_graph_has_the_research_nodes() -> None:
    compiled = build_state_graph().compile(checkpointer=MemorySaver())
    names = set(compiled.nodes)

    assert {nodes.PLANNER, nodes.RESEARCHER, nodes.SYNTHESIZER, nodes.CRITIC} <= names


def test_state_schema_keys() -> None:
    """The state contract nodes rely on. Grows in later phases; must not shrink."""
    assert set(AtlasState.__annotations__) == {
        "query",
        "subtasks",
        "research_results",
        "answer",
        "messages",
        "run_metrics",
        "iteration",
        "critic_feedback",
    }


def test_recursion_limit_is_sized_from_the_iteration_cap() -> None:
    """Explicit, and derived from config rather than left at LangGraph's 25.

    Asserted against the arithmetic rather than a bare constant, so raising
    MAX_CRITIC_ITERATIONS cannot leave the ceiling behind and turn a legitimate
    third pass into a recursion error.
    """
    settings = get_settings()
    assert settings.max_critic_iterations == 2
    assert recursion_limit() == 2 * (1 + 3 * (settings.max_critic_iterations + 1))
