'use client';

import {
  Background,
  ReactFlow,
  type Edge,
  type NodeChange,
  type NodeTypes,
  useReactFlow,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import '@xyflow/react/dist/style.css';

import { AgentNode, type AgentNodeData, type AgentNodeType } from './AgentNode';
import type { Topology } from '@/lib/events';
import { layout, shapeFor } from '@/lib/layout';
import { type Measurements, recordMeasurements } from '@/lib/measure';
import { RESEARCHER, type RunState, type NodeStatus } from '@/lib/reducer';

const NODE_TYPES: NodeTypes = { agent: AgentNode };

const LABELS: Record<string, string> = {
  planner: 'Planner',
  researcher: 'Researchers',
  synthesizer: 'Synthesizer',
  critic: 'Critic',
  human_edit: 'Your edit',
};

interface Props {
  readonly topology: Topology;
  readonly state: RunState;
  readonly now: number;
}

export function GraphCanvas({ topology, state, now }: Props): React.JSX.Element {
  const shape = useMemo(() => shapeFor(topology, state), [topology, state]);

  // React Flow hides a node until it knows its size, and the nodes below are
  // rebuilt from the run state on every event. Their sizes are kept here and put
  // back on each rebuild (see lib/measure.ts); without them, a run that stopped
  // changing left the whole graph hidden.
  const [measured, setMeasured] = useState<Measurements>(() => new Map());
  const onNodesChange = useCallback((changes: NodeChange<AgentNodeType>[]) => {
    const sized = changes.flatMap((change) =>
      change.type === 'dimensions' ? [{ type: change.type, id: change.id, dimensions: change.dimensions }] : [],
    );
    if (sized.length > 0) setMeasured((current) => recordMeasurements(current, sized));
  }, []);

  // No useMemo here on purpose. `layout` is memoised internally on the topology
  // signature — which contains no status, timing or verdict — so calling it every
  // render is a Map lookup, and the positions are identical object references
  // until the topology genuinely changes. Wrapping it in useMemo would mean
  // either lying about the dependency array or disabling a lint rule to hide
  // that; keeping the cache in one place needs neither.
  const positions = layout(shape);

  const nodes: AgentNodeType[] = useMemo(() => {
    const byId = new Map(positions.map((position) => [position.id, position]));
    // Lanes are numbered by position, not by subtask id. The planner usually
    // emits short ids ("1".."5"), but its degraded path mints a uuid — and a
    // lane captioned "Researcher bd6a8d39-fb49-4448-a465-cd52233d3725" is
    // unreadable. The id is still on the node via its React Flow id; the label
    // is for humans.
    const laneNumber = new Map(state.branches.map((branch, index) => [branch.id, index + 1]));

    return shape.nodeIds.map((id) => {
      const position = byId.get(id) ?? { x: 0, y: 0 };
      const branch = state.branches.find((candidate) => candidate.id === id);

      const data: AgentNodeData =
        branch !== undefined
          ? {
              label: `Researcher ${laneNumber.get(branch.id) ?? '?'}`,
              caption: branch.question,
              status: branch.status,
              startedAt: branch.startedAt,
              finishedAt: branch.finishedAt,
              error: branch.error,
              runs: branch.runs,
              sourceCount: branch.sources.length,
              now,
            }
          : {
              label: LABELS[id] ?? id,
              caption: id === RESEARCHER && !state.fannedOut ? 'awaiting the plan' : null,
              status: state.nodes[id]?.status ?? ('idle' satisfies NodeStatus),
              startedAt: state.nodes[id]?.startedAt ?? null,
              finishedAt: state.nodes[id]?.finishedAt ?? null,
              error: state.nodes[id]?.error ?? null,
              runs: state.nodes[id]?.runs ?? 0,
              sourceCount: null,
              now,
            };

      return {
        id,
        type: 'agent' as const,
        position: { x: position.x, y: position.y },
        data,
        measured: measured.get(id),
      };
    });
  }, [shape, positions, state, now, measured]);

  const edges: Edge[] = useMemo(
    () =>
      shape.edges.map((edge) => {
        const retryTarget = state.branches.find(
          (branch) => branch.id === edge.target && state.retryTargets.includes(branch.subtaskId),
        );
        const firing = edge.feedback && retryTarget !== undefined;
        // Latched, so the retry stays readable after the branches finish
        // re-running rather than blinking out at the moment of interest.
        const fired = edge.feedback && state.retryFired;

        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          animated: firing || isCarryingFlow(edge.source, edge.target, state),
          style: {
            stroke: firing ? 'var(--retry)' : fired ? 'var(--retry)' : edge.feedback ? 'var(--line-strong)' : 'var(--ink-faint)',
            strokeWidth: firing ? 2 : 1,
            strokeDasharray: edge.feedback && !firing ? '4 4' : undefined,
            opacity: edge.feedback && !fired ? 0.35 : 1,
          },
        };
      }),
    [shape, state],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      nodeTypes={NODE_TYPES}
      fitView
      proOptions={{ hideAttribution: true }}
      nodesDraggable={false}
      nodesConnectable={false}
      edgesFocusable={false}
      className="bg-canvas"
    >
      <Background color="var(--line)" gap={18} />
      <FitWhenTheGraphChanges
        signature={shape.nodeIds.join(',')}
        ready={shape.nodeIds.every((id) => measured.has(id))}
      />
    </ReactFlow>
  );
}

/**
 * Fit the graph to its pane whenever its nodes change (fan-out, a revision),
 * once every node has a size. ReactFlow's own `fitView` fits at mount, before
 * the nodes have sizes, and its "initialised" signal never fires for nodes
 * supplied with their measurements as these are: on a narrow screen the graph
 * stayed cropped at both edges. `ready` comes from the measurements GraphCanvas
 * keeps itself.
 */
function FitWhenTheGraphChanges({
  signature,
  ready,
}: {
  readonly signature: string;
  readonly ready: boolean;
}): null {
  const { fitView } = useReactFlow();
  useEffect(() => {
    if (!ready) return;
    // After this render commits the sizes; a timer, not a frame, so a tab in the
    // background (where frames are paused) still fits.
    const timer = window.setTimeout(() => void fitView({ padding: 0.12 }), 0);
    return () => window.clearTimeout(timer);
  }, [signature, ready, fitView]);
  return null;
}

/** An edge is carrying flow when its source is done and its target is running. */
function isCarryingFlow(source: string, target: string, state: RunState): boolean {
  const statusOf = (id: string): NodeStatus => {
    const branch = state.branches.find((candidate) => candidate.id === id);
    return branch?.status ?? state.nodes[id]?.status ?? 'idle';
  };
  return statusOf(source) === 'done' && statusOf(target) === 'running';
}
