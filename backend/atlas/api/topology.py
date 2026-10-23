"""The graph's shape, derived from the compiled graph itself.

The frontend draws whatever this returns. That is the whole point: a hardcoded
topology in React does not become *wrong* when the graph changes, it becomes
wrong **plausibly** — it keeps rendering a sensible picture that no longer
matches what runs, and a renamed node simply never lights up because events
arrive for a node the UI does not have. One quiet dark box in an otherwise
working display. Deriving the shape from ``compiled.get_graph()`` makes that
class of drift impossible.

``test_topology`` pins the node and edge sets to explicit expected values, so a
graph change in a later phase fails a test rather than a demo.
"""

from __future__ import annotations

from fastapi import APIRouter
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from atlas.graph.builder import build_state_graph
from atlas.graph.nodes import RESEARCHER

router = APIRouter()

# LangGraph's entry and exit markers. Reported rather than hidden — this endpoint
# describes the graph as it is, and what to draw is the client's decision.
START_NODE = "__start__"
END_NODE = "__end__"


class TopologyNode(BaseModel):
    """One node in the compiled graph."""

    id: str
    # True for a node that becomes N concurrent invocations at runtime. The count
    # is genuinely unknowable here — Send decides it when the planner returns — so
    # this is a flag, not a number, and the client expands the node when fan_out
    # arrives. The backend owns the graph, so it is the right place to know which
    # node fans out.
    fan_out: bool = False
    terminal: bool = False


class TopologyEdge(BaseModel):
    """One edge. ``conditional`` is what makes the retry edge drawable."""

    id: str
    source: str
    target: str
    conditional: bool


class Topology(BaseModel):
    """The graph, as the client should draw it.

    The critic's ``critic -> researcher`` edge is in here from the first frame,
    marked conditional. That lets the UI show the retry loop as *possible* before
    it ever fires — which is the difference between a viewer understanding the
    architecture and watching an edge appear from nowhere at second 90.
    """

    nodes: list[TopologyNode] = Field(default_factory=list)
    edges: list[TopologyEdge] = Field(default_factory=list)


def build_topology() -> Topology:
    """Read the compiled graph's shape.

    Compiles against an in-memory checkpointer: the shape is a property of the
    assembly, not of the store behind it, so this needs no database.
    """
    builder = build_state_graph()
    drawable = builder.compile(checkpointer=MemorySaver()).get_graph()

    nodes = [
        TopologyNode(
            id=node_id,
            fan_out=node_id == RESEARCHER,
            terminal=node_id in {START_NODE, END_NODE},
        )
        for node_id in drawable.nodes
    ]
    edges = [
        TopologyEdge(
            id=f"{edge.source}->{edge.target}",
            source=edge.source,
            target=edge.target,
            conditional=edge.conditional,
        )
        for edge in drawable.edges
    ]
    # get_graph() finds edges by simulating a run from START, so a node entered
    # some other way is drawn with none. human_edit is entered only through
    # aupdate_state, and its edge to the researcher went missing — the one edge
    # that shows a person can redirect a branch. The builder's declared routing
    # is the same graph's own record of it, so read the missing edges from there.
    drawn = {(edge.source, edge.target) for edge in edges}
    edges += [
        TopologyEdge(id=f"{source}->{target}", source=source, target=target, conditional=True)
        for source, branches in builder.branches.items()
        for branch in branches.values()
        for target in (branch.ends or {}).values()
        if (source, target) not in drawn
    ]
    return Topology(nodes=nodes, edges=edges)


@router.get("/graph/topology", response_model=Topology)
async def graph_topology() -> Topology:
    """The graph's nodes and edges.

    Static for the process lifetime — the graph is assembled at import — so a
    client may fetch it once and cache it.
    """
    return build_topology()
