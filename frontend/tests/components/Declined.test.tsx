// @vitest-environment jsdom
/**
 * A question judged to be about neither the AI Act nor the GDPR, as a client
 * sees it: the run ends at the plan, the answer says why, and the one thing to
 * do next is offered. Driven by a stream recorded from the real graph.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import declined from '../fixtures/declined.json';

import { AnswerCard } from '../../src/components/Answer';
import type { AgentEvent } from '../../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../../src/lib/reducer';
import { groupedTimeline, runSteps } from '../../src/lib/timeline';

const REASON = 'The question is about the history of Cameroon.';
const state = acceptAll(
  initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
  declined as readonly AgentEvent[],
);

afterEach(cleanup);

describe('a declined run', () => {
  it('keeps the reason, and finishes', () => {
    expect([state.declined, state.phase, state.fannedOut]).toEqual([REASON, 'completed', false]);
  });

  it('says so in its steps: planned, and nothing after', () => {
    expect(runSteps(state).map((s) => [s.label, s.status, s.detail])).toEqual([
      ['Plan', 'done', 'about neither act'],
      ['Research', 'idle', 'not run'],
      ['Write', 'idle', 'not run'],
      ['Review', 'idle', 'not run'],
    ]);
  });

  it('says so in its timeline, under the plan', () => {
    const plan = groupedTimeline(state, { showCalls: false }).find((g) => g.label === 'Plan');

    expect(plan?.entries.map((e) => [e.text, e.detail])).toContainEqual([
      'Not about the AI Act or the GDPR',
      REASON,
    ]);
  });
});

describe('its answer card', () => {
  it('says why it was not researched, in place of an answer', () => {
    render(<AnswerCard state={state} result={null} rows={[]} />);

    expect(screen.getByRole('heading', { level: 2 }).textContent).toBe('Not about the AI Act or the GDPR');
    expect(screen.getByText(REASON)).toBeDefined();
    // Not an answer: no article, no verdict, nothing to copy.
    expect(screen.queryByRole('article')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Copy' })).toBeNull();
    // A replay has nothing to research it with.
    expect(screen.queryByRole('button', { name: 'Research anyway' })).toBeNull();
  });

  it('researches it anyway when asked', async () => {
    const onResearchAnyway = vi.fn(async () => null);
    render(<AnswerCard state={state} result={null} rows={[]} onResearchAnyway={onResearchAnyway} />);

    fireEvent.click(screen.getByRole('button', { name: 'Research anyway' }));

    expect(onResearchAnyway).toHaveBeenCalledTimes(1);
    expect(await screen.findByRole('button', { name: 'Starting…' })).toHaveProperty('disabled', true);
  });

  it('says why when it could not start', async () => {
    const onResearchAnyway = vi.fn(async () => 'This conversation is still answering.');
    render(<AnswerCard state={state} result={null} rows={[]} onResearchAnyway={onResearchAnyway} />);

    fireEvent.click(screen.getByRole('button', { name: 'Research anyway' }));

    expect((await screen.findByRole('alert')).textContent).toBe('This conversation is still answering.');
    expect(screen.getByRole('button', { name: 'Research anyway' })).toHaveProperty('disabled', false);
  });
});
