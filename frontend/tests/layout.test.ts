/**
 * Layout stability.
 *
 * The requirement is "positions computed once per topology change, never per
 * event", and the only way to test that as a property rather than a hope is to
 * assert on the topology signature: it must be invariant under event traffic and
 * must change exactly when the drawable graph does.
 */

import { describe, expect, it } from 'vitest';

import accepted from './fixtures/accepted.json';
import retry from './fixtures/retry.json';
import revision from './fixtures/revision.json';
import demoTopology from '../src/demo/topology.json';

import type { AgentEvent, Topology } from '../src/lib/events';
import { clearLayoutCache, layout, shapeFor, signatureOf } from '../src/lib/layout';
import {
  CRITIC,
  HUMAN_EDIT,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  acceptAll,
  initialState,
} from '../src/lib/reducer';

const ACCEPTED = accepted as readonly AgentEvent[];
const RETRY = retry as readonly AgentEvent[];
const REVISION = revision as readonly AgentEvent[];

// The nodes and edges GET /graph/topology returns (pinned by backend/tests/test_topology.py),
// listed in flow order. The API's own order is in src/demo/topology.json.
const TOPOLOGY: Topology = {
  nodes: [
    { id: '__start__', fan_out: false, terminal: true },
    { id: PLANNER, fan_out: false, terminal: false },
    { id: RESEARCHER, fan_out: true, terminal: false },
    { id: SYNTHESIZER, fan_out: false, terminal: false },
    { id: CRITIC, fan_out: false, terminal: false },
    { id: HUMAN_EDIT, fan_out: false, terminal: false },
    { id: '__end__', fan_out: false, terminal: true },
  ],
  edges: [
    { id: '__start__->planner', source: '__start__', target: PLANNER, conditional: false },
    { id: 'planner->researcher', source: PLANNER, target: RESEARCHER, conditional: true },
    { id: 'planner->synthesizer', source: PLANNER, target: SYNTHESIZER, conditional: true },
    { id: 'researcher->synthesizer', source: RESEARCHER, target: SYNTHESIZER, conditional: false },
    { id: 'synthesizer->critic', source: SYNTHESIZER, target: CRITIC, conditional: false },
    { id: 'critic->researcher', source: CRITIC, target: RESEARCHER, conditional: true },
    { id: 'critic->synthesizer', source: CRITIC, target: SYNTHESIZER, conditional: true },
    { id: 'critic->__end__', source: CRITIC, target: '__end__', conditional: true },
    { id: 'human_edit->researcher', source: HUMAN_EDIT, target: RESEARCHER, conditional: true },
  ],
};

const SKELETON = [PLANNER, RESEARCHER, SYNTHESIZER, CRITIC];

function shapeAfter(events: readonly AgentEvent[]) {
  return shapeFor(TOPOLOGY, acceptAll(initialState(SKELETON), events));
}

describe('the drawable shape', () => {
  it('drops LangGraph plumbing', () => {
    const shape = shapeAfter([]);

    expect(shape.nodeIds).toEqual([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]);
    expect(shape.edges.map((e) => e.id)).not.toContain('__start__->planner');
    expect(shape.edges.map((e) => e.id)).not.toContain('critic->__end__');
  });

  it('shows one researcher placeholder before fan_out', () => {
    const shape = shapeAfter(ACCEPTED.slice(0, 3));

    expect(shape.nodeIds).toEqual([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]);
  });

  it('expands into one lane per subtask after fan_out', () => {
    const shape = shapeAfter(ACCEPTED);

    expect(shape.nodeIds).toEqual([
      PLANNER,
      'researcher:s0',
      'researcher:s1',
      'researcher:s2',
      'researcher:s3',
      'researcher:s4',
      SYNTHESIZER,
      CRITIC,
    ]);
  });

  it('keeps the retry edge, fanned out to every lane', () => {
    const shape = shapeAfter(ACCEPTED);
    const retryEdges = shape.edges.filter(
      (e) => e.source === CRITIC && e.target.startsWith('researcher:'),
    );

    expect(retryEdges.map((e) => e.id)).toEqual([
      'critic->researcher:s0',
      'critic->researcher:s1',
      'critic->researcher:s2',
      'critic->researcher:s3',
      'critic->researcher:s4',
    ]);
  });

  it('draws the edit only on a revision, an idle box on every other run', () => {
    expect(shapeAfter(ACCEPTED).edges.some((e) => e.source === HUMAN_EDIT)).toBe(false);

    const shape = shapeAfter(REVISION);
    const editEdges = shape.edges.filter((e) => e.source === HUMAN_EDIT);

    expect(shape.nodeIds).toContain(HUMAN_EDIT);
    // Fanned out to every lane, since any branch can be edited.
    expect(editEdges.map((e) => [e.id, e.feedback])).toEqual([
      ['human_edit->researcher:s0', false],
      ['human_edit->researcher:s1', false],
      ['human_edit->researcher:s2', false],
      ['human_edit->researcher:s3', false],
      ['human_edit->researcher:s4', false],
    ]);
  });

  it('identifies the loop-closing edges without being told which they are', () => {
    // Derived by cycle detection, so a different graph in a later phase is
    // handled without anyone remembering to update a hardcoded list.
    const shape = shapeAfter(ACCEPTED.slice(0, 3));
    const feedback = shape.edges.filter((e) => e.feedback).map((e) => e.id);

    expect(feedback.sort()).toEqual(['critic->researcher', 'critic->synthesizer']);
  });

  it('finds the same loop whatever order the API lists the edges in', () => {
    // The API's real order lists the critic's edges first. Searched from the
    // first edge's source, the critic looked like the start of the graph and
    // synthesizer->critic like the loop: the critic was drawn before the
    // planner.
    const state = acceptAll(initialState(SKELETON), ACCEPTED.slice(0, 3));
    const orders = [
      demoTopology.edges,
      TOPOLOGY.edges,
      [...TOPOLOGY.edges].reverse(),
    ];

    for (const edges of orders) {
      const shape = shapeFor({ ...TOPOLOGY, edges }, state);
      expect(shape.edges.filter((e) => e.feedback).map((e) => e.id).sort()).toEqual([
        'critic->researcher',
        'critic->synthesizer',
      ]);
    }
  });
});

describe('signature stability', () => {
  it('does not change across any event after fan_out', () => {
    // The property that matters. Replay the run one event at a time and assert
    // the signature is constant from fan_out onward — if any status, timing or
    // verdict leaked into it, this fails.
    const fanOutIndex = ACCEPTED.findIndex((e) => e.type === 'fan_out');
    const signatures = new Set<string>();

    for (let i = fanOutIndex + 1; i <= ACCEPTED.length; i += 1) {
      signatures.add(signatureOf(shapeAfter(ACCEPTED.slice(0, i))));
    }

    expect(signatures.size).toBe(1);
  });

  it('changes exactly twice across a whole run', () => {
    const signatures: string[] = [];
    for (let i = 0; i <= ACCEPTED.length; i += 1) {
      const signature = signatureOf(shapeAfter(ACCEPTED.slice(0, i)));
      if (signature !== signatures[signatures.length - 1]) signatures.push(signature);
    }

    // Skeleton, then fanned out. Nothing else.
    expect(signatures).toHaveLength(2);
  });

  it('does not change when the critic sends work back', () => {
    // A retry re-runs lanes; it does not alter the graph. Relayout here would
    // reflow the canvas at the single most interesting moment of the run.
    const verdict = RETRY.findIndex((e) => e.type === 'critic_verdict');
    const before = signatureOf(shapeAfter(RETRY.slice(0, verdict + 1)));
    const after = signatureOf(shapeAfter(RETRY));

    expect(after).toBe(before);
  });

  it('is order-independent', () => {
    const shape = shapeAfter(ACCEPTED);
    const reversed = {
      nodeIds: [...shape.nodeIds].reverse(),
      edges: [...shape.edges].reverse(),
    };

    expect(signatureOf(reversed)).toBe(signatureOf(shape));
  });
});

describe('positions', () => {
  it('are identical for the same topology', () => {
    clearLayoutCache();
    const first = layout(shapeAfter(ACCEPTED));
    clearLayoutCache();
    const second = layout(shapeAfter(ACCEPTED));

    expect(second).toEqual(first);
  });

  it('place every node exactly once', () => {
    clearLayoutCache();
    const shape = shapeAfter(ACCEPTED);
    const positions = layout(shape);

    expect(positions.map((p) => p.id)).toEqual(shape.nodeIds);
  });

  it('lay the pipeline out top to bottom with the lanes side by side', () => {
    clearLayoutCache();
    const positions = layout(shapeAfter(ACCEPTED));
    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));

    const planner = byId[PLANNER];
    const synthesizer = byId[SYNTHESIZER];
    const critic = byId[CRITIC];
    const lanes = positions.filter((p) => p.id.startsWith('researcher:'));

    expect(planner).toBeDefined();
    expect(synthesizer).toBeDefined();
    expect(critic).toBeDefined();
    // Read the way the timeline beside it reads, down the pane.
    expect(planner!.y).toBeLessThan(lanes[0]!.y);
    expect(lanes[0]!.y).toBeLessThan(synthesizer!.y);
    expect(synthesizer!.y).toBeLessThan(critic!.y);
    // Lanes share a row and differ across it — that shape is what makes
    // five concurrent branches readable at a glance.
    expect(new Set(lanes.map((l) => l.y)).size).toBe(1);
    expect(new Set(lanes.map((l) => l.x)).size).toBe(lanes.length);
  });
});
