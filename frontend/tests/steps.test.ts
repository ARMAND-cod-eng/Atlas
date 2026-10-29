/**
 * The run as four steps (plan, research, write, review), and its timeline cut
 * into those steps in the order they happened, internal calls folded away.
 */

import { describe, expect, it } from 'vitest';

import accepted from './fixtures/accepted.json';
import partial from './fixtures/partial-failure.json';
import retry from './fixtures/retry.json';
import revision from './fixtures/revision.json';
import failed from './fixtures/run-failed.json';

import type { AgentEvent } from '../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../src/lib/reducer';
import { groupedTimeline, runSteps } from '../src/lib/timeline';

const run = (events: unknown, upTo?: number) =>
  acceptAll(
    initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
    (events as readonly AgentEvent[]).slice(0, upTo),
  );

describe('the steps', () => {
  it('of a finished run are all done, and say what each came to', () => {
    expect(runSteps(run(retry)).map((s) => [s.label, s.status, s.detail])).toEqual([
      ['Plan', 'done', '5 branches'],
      ['Research', 'done', '10 sources read'],
      ['Write', 'done', '2 drafts'],
      ['Review', 'done', 'accepted in round 2'],
    ]);
  });

  it('of a run in progress show where it is', () => {
    const events = retry as readonly AgentEvent[];
    const fannedOut = events.findIndex((e) => e.type === 'fan_out') + 1;

    expect(runSteps(run(retry, fannedOut + 1)).map((s) => s.status)).toEqual([
      'done',
      'running',
      'idle',
      'idle',
    ]);
  });

  it('say when a branch failed, and when the run did', () => {
    expect(runSteps(run(partial))[1]).toEqual({
      key: 'research',
      label: 'Research',
      status: 'done',
      detail: '1 of 5 failed',
    });
    expect(runSteps(run(failed)).map((s) => s.status)).toContain('failed');
  });

  it('of a revision say the plan was carried over', () => {
    expect(runSteps(run(revision))[0]).toEqual({
      key: 'plan',
      label: 'Plan',
      status: 'done',
      detail: 'carried over',
    });
  });
});

describe('the timeline, by step', () => {
  it('follows the run in order, a retry starting a second round', () => {
    expect(groupedTimeline(run(retry), { showCalls: false }).map((g) => g.label)).toEqual([
      'Plan',
      'Research',
      'Write',
      'Review',
      'Research · round 2',
      'Write · round 2',
      'Review · round 2',
      'Finished',
    ]);
  });

  it('folds internal calls away until asked for', () => {
    const state = run(accepted);
    const count = (showCalls: boolean) =>
      groupedTimeline(state, { showCalls }).flatMap((g) => g.entries).length;
    const calls = state.timeline.filter((e) => e.type === 'agent_call').length;

    expect(calls).toBeGreaterThan(0);
    expect(count(true) - count(false)).toBe(calls);
    expect(
      groupedTimeline(state, { showCalls: false })
        .flatMap((g) => g.entries)
        .some((e) => e.type === 'agent_call'),
    ).toBe(false);
  });

  it('starts a revision with the edit', () => {
    expect(groupedTimeline(run(revision), { showCalls: false })[0]?.label).toBe('Your edit');
  });
});
