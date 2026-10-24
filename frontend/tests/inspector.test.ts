/**
 * What the inspector says about an agent, read off recorded runs.
 */

import { describe, expect, it } from 'vitest';

import compliance from './fixtures/compliance.json';
import retry from './fixtures/retry.json';
import revision from './fixtures/revision.json';

import type { AgentEvent } from '../src/lib/events';
import { agentOf, byAuthority, reviewsOf } from '../src/lib/inspector';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../src/lib/reducer';

function stateOf(events: unknown) {
  return acceptAll(initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]), events as readonly AgentEvent[]);
}

describe('a branch the critic sent back', () => {
  const state = stateOf(retry);

  it('has one attempt per run, each with the search it made', () => {
    const agent = agentOf(state, null, 'researcher:s2');

    expect(agent?.kind).toBe('branch');
    expect(agent?.attempts.map((a) => [a.number, a.calls.map((c) => [c.kind, c.name])])).toEqual([
      [1, [['tool', 'web']]],
      [2, [['tool', 'web']]],
    ]);
    expect(agent?.attempts.map((a) => a.spanId)).toEqual(
      state.branches.find((b) => b.subtaskId === 's2')?.spans,
    );
  });

  it("carries the critic's reason for sending it back", () => {
    expect(reviewsOf(state.criticRounds, 's2')).toEqual([
      {
        iteration: 1,
        verdict: 'revise',
        reason: 'the claims are not supported by the listed sources',
        guidance: 'Search for primary benchmark data rather than vendor posts.',
      },
      { iteration: 2, verdict: 'accept', reason: null, guidance: null },
    ]);
  });
});

describe('a revision', () => {
  const state = stateOf(revision);

  it('says what the edited branch asked before', () => {
    const agent = agentOf(state, null, 'researcher:s2');

    expect(agent?.kind === 'branch' ? [agent.branch.question, agent.before] : null).toEqual([
      'What are the per-token costs of long-context models?',
      'What are the cost characteristics of each?',
    ]);
  });

  it('shows a carried branch with no attempts in this run', () => {
    const agent = agentOf(state, null, 'researcher:s0');

    expect(agent?.kind === 'branch' ? [agent.branch.origin, agent.attempts.length, agent.before] : null).toEqual([
      'carried',
      0,
      null,
    ]);
  });
});

describe('a compliance branch', () => {
  it('lists the law before the guidance and the web', () => {
    const state = stateOf(compliance);
    const branch = state.branches[0];
    const grouped = byAuthority(branch?.sources ?? []);

    expect(grouped.law.every((s) => s.label?.startsWith('AI Act'))).toBe(true);
    expect(grouped.law.length + grouped.guidance.length + grouped.other.length).toBe(
      branch?.sources.length,
    );
    expect(grouped.law.length).toBeGreaterThan(0);
  });
});

describe('the stages', () => {
  it('describe what each is for', () => {
    const state = stateOf(retry);

    expect(agentOf(state, null, PLANNER)).toMatchObject({ kind: 'stage', title: 'Planner' });
    expect(agentOf(state, null, CRITIC)).toMatchObject({ kind: 'stage', title: 'Critic' });
  });

  it('a node the run never reached is nothing to inspect', () => {
    expect(agentOf(stateOf(retry), null, 'researcher:s9')).toBeNull();
  });
});
