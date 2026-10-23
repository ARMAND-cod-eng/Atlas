"""The graph's shape, as the frontend receives it.

**This is the desync guard.** The frontend draws whatever `/graph/topology`
returns, so the risk is not that the endpoint breaks — it is that the graph
changes in a later phase and nothing notices until a demo shows a picture that
no longer matches what executes. Pinning the node and edge sets to explicit
values turns that into a failing test at the moment the graph changes.

Which means: **if a later phase legitimately changes the graph, this test is
supposed to fail.** Update the expected sets deliberately; do not relax the
assertion into a subset check.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from atlas.api.main import create_app
from atlas.api.topology import build_topology
from atlas.graph import nodes

EXPECTED_NODES = {
    "__start__",
    nodes.PLANNER,
    nodes.RESEARCHER,
    nodes.SYNTHESIZER,
    nodes.CRITIC,
    nodes.HUMAN_EDIT,
    "__end__",
}

EXPECTED_EDGES = {
    ("__start__", nodes.PLANNER, False),
    (nodes.PLANNER, nodes.RESEARCHER, True),
    (nodes.PLANNER, nodes.SYNTHESIZER, True),
    (nodes.RESEARCHER, nodes.SYNTHESIZER, False),
    (nodes.SYNTHESIZER, nodes.CRITIC, False),
    (nodes.CRITIC, nodes.RESEARCHER, True),
    (nodes.CRITIC, nodes.SYNTHESIZER, True),
    (nodes.CRITIC, "__end__", True),
    # A person's edit to a finished run enters here, never from START.
    (nodes.HUMAN_EDIT, nodes.RESEARCHER, True),
}


def test_the_topology_is_exactly_this() -> None:
    topology = build_topology()

    assert {node.id for node in topology.nodes} == EXPECTED_NODES
    assert {(e.source, e.target, e.conditional) for e in topology.edges} == EXPECTED_EDGES


def test_the_retry_edge_is_present_and_conditional() -> None:
    """The single most important edge for the UI.

    Because it is declared here rather than inferred from a run, the client can
    draw the retry loop dormant from the first frame. Inferring topology from
    events would never surface it on a run the critic accepts — the most
    interesting structure in the system, invisible exactly when things go well.
    """
    retry = [
        edge
        for edge in build_topology().edges
        if edge.source == nodes.CRITIC and edge.target == nodes.RESEARCHER
    ]

    assert len(retry) == 1
    assert retry[0].conditional is True
    assert retry[0].id == f"{nodes.CRITIC}->{nodes.RESEARCHER}"


def test_only_the_researcher_fans_out() -> None:
    """The count is unknowable here; that a fan-out happens is not."""
    topology = build_topology()

    assert {node.id for node in topology.nodes if node.fan_out} == {nodes.RESEARCHER}


def test_start_and_end_are_marked_terminal_not_hidden() -> None:
    """Reported, so the endpoint describes the graph as it is.

    What to draw is the client's decision — it filters these out because they are
    LangGraph plumbing, not agents.
    """
    topology = build_topology()

    assert {node.id for node in topology.nodes if node.terminal} == {"__start__", "__end__"}


def test_the_endpoint_serves_it() -> None:
    response = TestClient(create_app()).get("/graph/topology")
    body = response.json()

    assert response.status_code == 200
    assert {node["id"] for node in body["nodes"]} == EXPECTED_NODES
    assert {(e["source"], e["target"], e["conditional"]) for e in body["edges"]} == EXPECTED_EDGES
