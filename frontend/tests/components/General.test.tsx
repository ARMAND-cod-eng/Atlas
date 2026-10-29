// @vitest-environment jsdom
/**
 * Asking a general question: chosen when a conversation starts, said on every
 * screen of the run, and offered where Atlas turns a question away.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import declined from '../fixtures/declined.json';

import Home from '../../src/app/page';
import { AnswerCard } from '../../src/components/Answer';
import { QuestionForm } from '../../src/components/QuestionForm';
import { RunHeader } from '../../src/components/RunHeader';
import { startRun } from '../../src/lib/api';
import type { AgentEvent } from '../../src/lib/events';
import {
  CRITIC,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  type RunState,
  acceptAll,
  initialState,
} from '../../src/lib/reducer';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

const empty = () => initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]);
const started = (domain?: string): RunState =>
  acceptAll(empty(), [
    {
      run_id: 'r',
      seq: 1,
      ts: '2026-10-29T10:00:00Z',
      type: 'run_started',
      node: null,
      payload: { query: 'What is RAG?', thread_id: 't', ...(domain === undefined ? {} : { domain }) },
    } as AgentEvent,
  ]);

/** Every request body sent to the API, parsed. */
function capture(): unknown[] {
  const bodies: unknown[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body ?? '{}')));
      return new Response(JSON.stringify({ run_id: 'run-2', thread_id: 't', events_url: '/e' }), { status: 202 });
    }),
  );
  return bodies;
}

beforeEach(() => {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: false,
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('a run’s domain', () => {
  it('is kept from run_started, and unknown before', () => {
    expect([started('general').domain, started('compliance').domain, started().domain]).toEqual([
      'general',
      'compliance',
      null,
    ]);
  });

  it('is asked for when a conversation starts, and never on a follow-up', async () => {
    const bodies = capture();

    await startRun('What is RAG?', undefined, undefined, 'general');
    await startRun('And then?', 't', 'r');

    expect(bodies).toEqual([
      { query: 'What is RAG?', domain: 'general' },
      { query: 'And then?', thread_id: 't', parent_run_id: 'r' },
    ]);
  });
});

describe('the question box', () => {
  it('sends the domain it was given', async () => {
    const bodies = capture();
    render(
      <QuestionForm placeholder="Ask" hint="" submitLabel="Research" domain="general" onStarted={() => undefined} />,
    );

    fireEvent.change(screen.getByLabelText('Your question'), { target: { value: 'What is RAG?' } });
    fireEvent.click(screen.getByRole('button', { name: 'Research' }));

    await screen.findByRole('button', { name: 'Starting…' });
    expect(bodies).toEqual([{ query: 'What is RAG?', domain: 'general' }]);
  });
});

describe('the home page', () => {
  it('asks about the law by default, and anything once switched', async () => {
    const bodies = capture();
    render(<Home />);
    const law = screen.getByRole('button', { name: 'Ask about EU law' });
    const anything = screen.getByRole('button', { name: 'Ask anything' });
    expect([law.getAttribute('aria-pressed'), anything.getAttribute('aria-pressed')]).toEqual(['true', 'false']);

    fireEvent.click(anything);
    fireEvent.change(screen.getByLabelText('Your question'), { target: { value: 'History of Cameroon' } });
    fireEvent.click(screen.getByRole('button', { name: 'Research' }));

    await screen.findByRole('button', { name: 'Starting…' });
    expect(anything.getAttribute('aria-pressed')).toBe('true');
    expect(bodies).toEqual([{ query: 'History of Cameroon', domain: 'general' }]);
    // And says what that means for the answer.
    expect(screen.getByText(/not checked against the law/)).toBeDefined();
  });
});

describe('a general run', () => {
  it('says so in its header', () => {
    render(<RunHeader state={started('general')} now={0} />);

    expect(screen.getByText('General research')).toBeDefined();
  });

  it('is not marked as general research when it researched the law', () => {
    render(<RunHeader state={started('compliance')} now={0} />);

    expect(screen.queryByText('General research')).toBeNull();
  });

  it('says its sources are the web’s, not checked against the law', () => {
    const state = { ...started('general'), answer: 'RAG retrieves, then generates.', phase: 'completed' as const };
    render(<AnswerCard state={state} result={null} rows={[]} />);

    expect(screen.getByText('Sources from the web, not checked against the law')).toBeDefined();
  });
});

describe('a question turned away', () => {
  const state = acceptAll(empty(), declined as readonly AgentEvent[]);

  it('can be asked as a general question', () => {
    const onAskGeneral = vi.fn(async () => null);
    render(<AnswerCard state={state} result={null} rows={[]} onAskGeneral={onAskGeneral} />);

    fireEvent.click(screen.getByRole('button', { name: 'Ask as a general question' }));

    expect(onAskGeneral).toHaveBeenCalledTimes(1);
  });
});
