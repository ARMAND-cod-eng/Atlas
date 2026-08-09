"""The conditional edge that creates the parallelism."""

from __future__ import annotations

from langgraph.types import Send

from atlas.graph import nodes
from atlas.graph.state import AtlasState, Subtask
from tests.conftest import initial_state


def _state_with(count: int) -> AtlasState:
    state = initial_state("original question")
    state["subtasks"] = [Subtask(id=f"s{i}", question=f"q{i}", rationale="r") for i in range(count)]
    return state


def test_one_send_per_subtask() -> None:
    sends = nodes.fan_out(_state_with(4))

    assert isinstance(sends, list)
    assert len(sends) == 4
    assert all(isinstance(s, Send) for s in sends)


def test_every_send_targets_the_researcher() -> None:
    sends = nodes.fan_out(_state_with(3))
    assert isinstance(sends, list)
    assert {s.node for s in sends} == {nodes.RESEARCHER}


def test_each_send_carries_its_subtask_and_the_original_query() -> None:
    """The researcher needs its own subtask plus the query for context."""
    sends = nodes.fan_out(_state_with(3))
    assert isinstance(sends, list)

    carried = [s.arg["subtask"].id for s in sends]
    assert carried == ["s0", "s1", "s2"]
    assert all(s.arg["query"] == "original question" for s in sends)


def test_empty_subtasks_routes_to_the_synthesizer() -> None:
    """Not an empty Send list.

    Returning [] would mean no researcher runs and the synthesizer receives an
    empty research_results with no indication why — a silent no-op. Routing
    explicitly lets the synthesizer say that no research was performed.
    """
    destination = nodes.fan_out(_state_with(0))

    assert destination == nodes.SYNTHESIZER
    assert not isinstance(destination, list)
