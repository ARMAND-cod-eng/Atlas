// @vitest-environment jsdom
/**
 * The timeline, rendered: internal calls folded away until asked for, and the
 * choice remembered for the next visit.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import accepted from '../fixtures/accepted.json';

import { Timeline } from '../../src/components/Timeline';
import type { AgentEvent } from '../../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../../src/lib/reducer';

const state = acceptAll(
  initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
  accepted as readonly AgentEvent[],
);

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe('the timeline', () => {
  it('shows the steps, and each call only when asked', () => {
    render(<Timeline state={state} />);

    expect(screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)).toEqual([
      'Plan',
      'Research',
      'Write',
      'Review',
      'Finished',
    ]);
    expect(screen.queryAllByText(/searched the web$/)).toEqual([]);

    fireEvent.click(screen.getByLabelText('Show every call'));

    expect(screen.getAllByText(/searched the web$/).length).toBeGreaterThan(0);
    expect(window.localStorage.getItem('atlas:timeline-calls')).toBe('true');
  });

  it('remembers the choice', () => {
    window.localStorage.setItem('atlas:timeline-calls', 'true');
    render(<Timeline state={state} />);

    expect(screen.getAllByText(/searched the web$/).length).toBeGreaterThan(0);
  });
});
