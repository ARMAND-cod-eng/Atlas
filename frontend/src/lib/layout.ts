/**
 * Topology → node positions.
 *
 * The requirement this module exists to satisfy: **positions are computed once
 * per topology change, never per event.** A graph that reflows every time a node
 * lights up is unusable — text jumps mid-read and the eye loses the branch it
 * was following.
 *
 * That is enforced structurally rather than by discipline. Layout is keyed by a
 * *topology signature* built from node and edge ids only. Statuses, timings,
 * sources and verdicts are all absent from the signature, so no amount of event
 * traffic can change it, and the memo cannot be invalidated by a repaint.
 *
 * In a normal run the signature changes exactly twice: once on mount, and once
 * at `fan_out` when the single researcher placeholder becomes N lanes. A retry
 * changes no topology at all — the same lanes re-enter `running` — so it costs
 * no relayout.
 */

import dagre from 'dagre';

import type { Topology } from './events';
import { RESEARCHER, type RunState } from './reducer';

export const NODE_WIDTH = 210;
export const NODE_HEIGHT = 72;

export interface Positioned {
  readonly id: string;
  readonly x: number;
  readonly y: number;
}

export interface ShapeEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  /**
   * A back edge — one that points against the flow, like the critic's retry.
   *
   * Kept out of dagre's ranking. Feeding a cycle to a layered layout makes it
   * reverse edges internally to break the loop, which shoved the researcher
   * lanes into two columns and destroyed the one visual that matters most: five
   * branches side by side. Laying out the acyclic graph and drawing the loop
   * back over it is both the standard treatment and the honest picture — a
   * feedback edge should look like feedback.
   */
  readonly feedback: boolean;
}

export interface GraphShape {
  /** Node ids to draw, in layout order. */
  readonly nodeIds: readonly string[];
  readonly edges: readonly ShapeEdge[];
}

/**
 * The drawable graph for the current run state.
 *
 * LangGraph's `__start__` / `__end__` are dropped: they are plumbing, not
 * agents, and drawing them spends the viewer's attention on nothing. They are
 * still *reported* by the backend, because that endpoint describes the graph as
 * it is — what to draw is this layer's decision.
 */
export function shapeFor(topology: Topology, state: RunState): GraphShape {
  const drawable = topology.nodes.filter((node) => !node.terminal).map((node) => node.id);
  const branchIds = state.branches.map((branch) => branch.id);

  // Before fan_out the count is genuinely unknown — the planner has not decided
  // yet — so one placeholder stands in. No ghost lanes: rendering five and
  // collapsing to three would imply knowledge we do not have.
  const nodeIds = state.fannedOut
    ? drawable.flatMap((id) => (id === RESEARCHER ? branchIds : [id]))
    : drawable;

  const structural = topology.edges.filter(
    (edge) => !isTerminalId(edge.source) && !isTerminalId(edge.target),
  );
  // Searched over the whole graph, `__start__` included: the loop is whatever
  // closes a cycle as seen from where a run begins.
  const feedback = findFeedbackEdges(topology.edges);

  const present = new Set(nodeIds);
  const edges = structural
    .flatMap((edge) =>
      expand(edge.source, edge.target, branchIds, state.fannedOut, feedback.has(edge.id)),
    )
    .filter((edge) => present.has(edge.source) && present.has(edge.target));

  return { nodeIds, edges };
}

function isTerminalId(id: string): boolean {
  return id === '__start__' || id === '__end__';
}

/**
 * Which edges point backwards, found rather than hardcoded.
 *
 * A depth-first search over the topology: an edge into a node already on the
 * current stack closes a cycle. Derived so that a graph change in a later phase
 * — a second loop, a different retry target — is handled without anyone
 * remembering to update a list of edge names here.
 *
 * The search starts where runs start: at the nodes nothing points into
 * (`__start__`, and the human edit). Started from whichever edge the API listed
 * first, a cycle is "closed" by a different edge: from the critic, the forward
 * edge synthesizer -> critic looked like the loop, and the critic was drawn
 * before the planner.
 */
function findFeedbackEdges(
  edges: readonly { id: string; source: string; target: string }[],
): ReadonlySet<string> {
  const outgoing = new Map<string, { id: string; target: string }[]>();
  for (const edge of edges) {
    const list = outgoing.get(edge.source) ?? [];
    list.push({ id: edge.id, target: edge.target });
    outgoing.set(edge.source, list);
  }

  const back = new Set<string>();
  const visited = new Set<string>();
  const onStack = new Set<string>();

  const visit = (node: string): void => {
    visited.add(node);
    onStack.add(node);
    for (const edge of outgoing.get(node) ?? []) {
      if (onStack.has(edge.target)) {
        back.add(edge.id);
      } else if (!visited.has(edge.target)) {
        visit(edge.target);
      }
    }
    onStack.delete(node);
  };

  const targets = new Set(edges.map((edge) => edge.target));
  const entries = edges.map((edge) => edge.source).filter((source) => !targets.has(source));
  // Then anything only reachable around a cycle, so every edge is classified.
  for (const node of [...entries, ...edges.map((edge) => edge.source)]) {
    if (!visited.has(node)) visit(node);
  }
  return back;
}

/** One topology edge becomes one edge per lane wherever it touches the fan-out. */
function expand(
  source: string,
  target: string,
  branchIds: readonly string[],
  fannedOut: boolean,
  feedback: boolean,
): ShapeEdge[] {
  if (!fannedOut) return [{ id: `${source}->${target}`, source, target, feedback }];

  const sources = source === RESEARCHER ? branchIds : [source];
  const targets = target === RESEARCHER ? branchIds : [target];
  return sources.flatMap((from) =>
    targets.map((to) => ({ id: `${from}->${to}`, source: from, target: to, feedback })),
  );
}

/**
 * A stable identity for a shape. Two shapes with the same signature must lay out
 * identically, so this is what the memo is keyed on.
 *
 * Sorted, because iteration order is not part of the topology — five lanes that
 * arrive in a different order are the same graph and must not trigger a reflow.
 */
export function signatureOf(shape: GraphShape): string {
  const nodes = [...shape.nodeIds].sort().join(',');
  const edges = [...shape.edges.map((edge) => edge.id)].sort().join(',');
  return `${nodes}|${edges}`;
}

const cache = new Map<string, readonly Positioned[]>();

/**
 * Positions for a shape, computed at most once per distinct topology.
 *
 * Left-to-right, so the pipeline reads the way it runs and the fan-out spreads
 * vertically into a band the eye can scan.
 */
export function layout(shape: GraphShape): readonly Positioned[] {
  const key = signatureOf(shape);
  const cached = cache.get(key);
  if (cached !== undefined) return cached;

  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: 'LR', nodesep: 26, ranksep: 96, marginx: 16, marginy: 16 });
  graph.setDefaultEdgeLabel(() => ({}));

  for (const id of shape.nodeIds) {
    graph.setNode(id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  // Feedback edges are drawn but not ranked — see ShapeEdge.feedback.
  for (const edge of shape.edges) {
    if (!edge.feedback) graph.setEdge(edge.source, edge.target);
  }
  dagre.layout(graph);

  const positioned = shape.nodeIds.map((id) => {
    const node = graph.node(id);
    // dagre reports centres; xyflow wants top-left corners.
    return {
      id,
      x: (node?.x ?? 0) - NODE_WIDTH / 2,
      y: (node?.y ?? 0) - NODE_HEIGHT / 2,
    };
  });

  cache.set(key, positioned);
  return positioned;
}

/** Test seam — the cache is process-wide and would leak between cases. */
export function clearLayoutCache(): void {
  cache.clear();
}
