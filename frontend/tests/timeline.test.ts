/**
 * The timeline, read off recorded runs: what a person sees of how a run went.
 */

import { describe, expect, it } from 'vitest';

import compliance from './fixtures/compliance.json';
import retry from './fixtures/retry.json';
import revision from './fixtures/revision.json';

import type { AgentEvent } from '../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../src/lib/reducer';
import { timelineEntries } from '../src/lib/timeline';

function entries(events: unknown) {
  const state = acceptAll(initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]), events as readonly AgentEvent[]);
  return timelineEntries(state);
}

describe('the timeline of a compliance run', () => {
  it('shows the citation check catching the invented article, then passing', () => {
    const checks = entries(compliance).filter((e) => e.text.toLowerCase().includes('citation'));

    expect(checks.map((e) => [e.tone, e.text, e.detail])).toEqual([
      ['failed', 'Citation check: 1 problem', 'AI Act Art. 114 does not exist'],
      ['law', '3 citations checked against the law', null],
    ]);
  });

  it('says each branch searched the law', () => {
    const searches = entries(compliance).filter((e) => e.text.endsWith('searched the law'));

    expect(searches.map((e) => [e.text, e.nodeId])).toEqual([
      ['s0 searched the law', 'researcher:s0'],
      ['s1 searched the law', 'researcher:s1'],
    ]);
  });

  it('runs forward in time', () => {
    const times = entries(compliance).map((e) => e.at);

    expect(times).toEqual([...times].sort((a, b) => a - b));
    expect(times[0]).toBe(0);
  });
});

describe('the timeline of a retry', () => {
  it("names the branch the critic sent back, and why", () => {
    const [review] = entries(retry).filter((e) => e.tone === 'retry');

    expect(review?.text).toBe('Review 1: sent s2 back');
    expect(review?.detail).toBe('s2: the claims are not supported by the listed sources');
  });
});

describe('the timeline of a revision', () => {
  it('opens with what the person changed', () => {
    const [received, rewritten] = entries(revision);

    expect(received?.text).toBe('Question received');
    expect([rewritten?.tone, rewritten?.text, rewritten?.detail]).toEqual([
      'edited',
      'Branch s2 rewritten; 4 branches carried over',
      'What are the cost characteristics of each? → What are the per-token costs of long-context models?',
    ]);
  });
});
