/**
 * The run's status, as the header's pill says it.
 */

import { describe, expect, it } from 'vitest';

import failed from './fixtures/run-failed.json';
import retry from './fixtures/retry.json';

import type { AgentEvent } from '../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../src/lib/reducer';
import { runStatus } from '../src/lib/runStatus';

const fresh = () => initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]);
const run = (events: unknown) => acceptAll(fresh(), events as readonly AgentEvent[]);

describe("the header's status", () => {
  it('is "Starting" before the first event', () => {
    expect(runStatus(fresh(), 0)).toEqual({ tone: 'neutral', text: 'Starting' });
  });

  it('counts the time while the run is going', () => {
    const events = retry as readonly AgentEvent[];
    const started = acceptAll(fresh(), events.slice(0, 3));
    const now = Date.parse(events[0]?.ts ?? '') + 23_000;

    expect(runStatus(started, now)).toEqual({ tone: 'running', text: 'Running · 23s' });
  });

  it("says how long a finished run took, from the run's own clock", () => {
    const status = runStatus(run(retry), Number.MAX_SAFE_INTEGER);

    expect(status.tone).toBe('done');
    expect(status.text).toMatch(/^Complete · \d+(\.\d)?s$/);
  });

  it('says a failed run failed', () => {
    expect(runStatus(run(failed), 0)).toEqual({ tone: 'failed', text: 'Failed' });
  });
});
