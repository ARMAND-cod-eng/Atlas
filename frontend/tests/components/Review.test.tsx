// @vitest-environment jsdom
/**
 * A contract review, as a reviewer sees it: what each of GDPR Art. 28's terms
 * came to, the contract's own words for it, and a decision on each. Driven by
 * a review recorded from the real graph (capture_fixtures.py, "review"): five
 * met, two partial, one missing, and one whose quote the contract does not have.
 */

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import recorded from '../fixtures/review.json';

import Home from '../../src/app/page';
import { ReviewWorkspace } from '../../src/components/Review';
import { LAW_AS_OF } from '../../src/components/Disclaimer';
import type { AgentEvent } from '../../src/lib/events';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, acceptAll, initialState } from '../../src/lib/reducer';
import { reviewMarkdown, reviewOf } from '../../src/lib/review';

vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }));

const state = acceptAll(initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]), recorded as readonly AgentEvent[]);
const view = reviewOf(state.timeline);

interface Sent {
  readonly method: string;
  readonly url: string;
  readonly body: unknown;
}

/** The API: no decisions yet, and every write echoed back as made. */
function serve(): Sent[] {
  const sent: Sent[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      const body = init?.body === undefined ? null : JSON.parse(String(init.body));
      sent.push({ method, url, body });
      if (method === 'PUT') {
        const requirement = url.split('/').at(-1);
        return new Response(
          JSON.stringify({ run_id: 'review', requirement_id: requirement, decided_at: '2026-10-29T10:00:00Z', ...body }),
          { status: 200 },
        );
      }
      if (method === 'POST') return new Response(JSON.stringify({ run_id: 'review-2', thread_id: 't' }), { status: 202 });
      return new Response(JSON.stringify({ current: [], history: [] }), { status: 200 });
    }),
  );
  return sent;
}

beforeEach(() => {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: true,
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  Reflect.deleteProperty(navigator, 'clipboard');
});

describe('a review, read from its stream', () => {
  it('has the contract as read, and a finding for every term, in the law’s order', () => {
    expect([view.filename, view.title, view.clauses.length]).toEqual([
      'dpa-01.md',
      'GDPR Art. 28: processor contracts',
      12,
    ]);
    expect(view.requirements.map((r) => [r.label, view.findings[r.id]?.status])).toEqual([
      ['GDPR Art. 28(3)', 'met'],
      ['GDPR Art. 28(3)(a)', 'met'],
      ['GDPR Art. 28(3)(b)', 'met'],
      ['GDPR Art. 28(3)(c)', 'met'],
      ['GDPR Art. 28(3)(d)', 'partial'],
      ['GDPR Art. 28(3)(e)', 'met'],
      ['GDPR Art. 28(3)(f)', 'unverified'],
      ['GDPR Art. 28(3)(g)', 'partial'],
      ['GDPR Art. 28(3)(h)', 'missing'],
    ]);
    expect(view.counts).toEqual({ met: 5, partial: 2, missing: 1, unverified: 1 });
  });
});

describe('the review page', () => {
  it('shows each term’s status, and the contract’s words for it in place', () => {
    serve();
    render(<ReviewWorkspace runId="review" state={state} />);
    const end = screen.getByRole('article', { name: 'GDPR Art. 28(3)(g)' });
    expect(within(end).getByText('Partial')).toBeDefined();

    fireEvent.click(within(end).getByRole('button', { name: 'Show GDPR Art. 28(3)(g) in the contract' }));

    const contract = screen.getByRole('region', { name: 'The contract' });
    expect([...contract.querySelectorAll('mark')].map((mark) => mark.textContent)).toEqual([
      'the Processor shall delete all personal data',
    ]);
  });

  it('says a quote the contract does not have is not in it', () => {
    serve();
    render(<ReviewWorkspace runId="review" state={state} />);
    const assistance = screen.getByRole('article', { name: 'GDPR Art. 28(3)(f)' });

    expect(within(assistance).getByText('Unverified')).toBeDefined();
    expect(within(assistance).getByText('Not found in the contract')).toBeDefined();
    expect(within(assistance).queryByRole('button', { name: /in the contract$/ })).toBeNull();
  });

  it('signs a finding off in the reviewer’s name, and keeps the name for the next', async () => {
    const sent = serve();
    render(<ReviewWorkspace runId="review" state={state} />);
    const audits = screen.getByRole('article', { name: 'GDPR Art. 28(3)(h)' });
    expect(within(audits).getByRole('button', { name: 'Agree with GDPR Art. 28(3)(h)' })).toHaveProperty('disabled', true);

    fireEvent.change(screen.getByLabelText('Your name'), { target: { value: 'A. Reviewer' } });
    fireEvent.change(within(audits).getByLabelText('Note on GDPR Art. 28(3)(h)'), {
      target: { value: 'Clause 9 covers audits.' },
    });
    fireEvent.click(within(audits).getByRole('button', { name: 'Disagree with GDPR Art. 28(3)(h)' }));

    await waitFor(() =>
      expect(within(audits).getByRole('button', { name: 'Disagree with GDPR Art. 28(3)(h)' }).getAttribute('aria-pressed')).toBe(
        'true',
      ),
    );
    expect(sent.filter((s) => s.method === 'PUT')).toEqual([
      {
        method: 'PUT',
        url: 'http://127.0.0.1:8000/reviews/review/decisions/audits',
        body: { decision: 'disagree', note: 'Clause 9 covers audits.', reviewer: 'A. Reviewer' },
      },
    ]);
    expect(window.localStorage.getItem('atlas:reviewer')).toBe('A. Reviewer');
  });

  it('copies the review as a report, decisions and caveat included', async () => {
    serve();
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    render(<ReviewWorkspace runId="review" state={state} />);

    fireEvent.click(screen.getByRole('button', { name: 'Copy report' }));

    await screen.findByRole('button', { name: 'Copied' });
    expect(writeText.mock.calls).toEqual([[reviewMarkdown(view, {}, LAW_AS_OF)]]);
  });
});

describe('the report', () => {
  it('says each term’s status, where, in the contract’s words, and who agreed', () => {
    const report = reviewMarkdown(
      view,
      {
        end: {
          run_id: 'review',
          requirement_id: 'end',
          decision: 'agree',
          note: 'Return must be added.',
          reviewer: 'A. Reviewer',
          decided_at: '2026-10-29T10:00:00Z',
        },
      },
      '23 October 2026',
    );
    const lines = report.split('\n');

    expect(lines.slice(0, 3)).toEqual([
      '# Review of dpa-01.md',
      '',
      'GDPR Art. 28: processor contracts · 5 met · 2 partial · 1 missing · 1 unverified',
    ]);
    const end = lines.indexOf('## GDPR Art. 28(3)(g): Delete or return the data at the end');
    expect(lines.slice(end, end + 6)).toEqual([
      '## GDPR Art. 28(3)(g): Delete or return the data at the end',
      '',
      '**Partial**, Clause 7: “the Processor shall delete all personal data”',
      '',
      "Deletion only: the Controller's choice to have the data returned is missing.",
      '',
    ]);
    expect(lines.slice(end + 6, end + 7)).toEqual(['Reviewer: agrees (A. Reviewer). Return must be added.']);
    const assistance = lines.indexOf(
      '## GDPR Art. 28(3)(f): Help with security, breaches and impact assessments (Articles 32 to 36)',
    );
    expect(lines[assistance + 2]).toBe('**Unverified**: its quote is not in the contract');
    expect(lines.at(-2)).toBe(
      '_A research aid, not legal advice: the law as published in the Official Journal, retrieved 23 October 2026._',
    );
  });
});

describe('starting a review from the home page', () => {
  it('sends pasted text', async () => {
    const sent = serve();
    render(<Home />);

    fireEvent.click(screen.getByRole('button', { name: 'Review a contract' }));
    fireEvent.change(screen.getByLabelText('Or paste the contract'), { target: { value: '1. Instructions\nOnly on instructions.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Review' }));

    await waitFor(() => expect(sent.filter((s) => s.method === 'POST')).toHaveLength(1));
    expect(sent.find((s) => s.method === 'POST')).toEqual({
      method: 'POST',
      url: 'http://127.0.0.1:8000/reviews',
      body: { filename: 'pasted contract', text: '1. Instructions\nOnly on instructions.' },
    });
  });

  it('sends a file as base64', async () => {
    const sent = serve();
    render(<Home />);

    fireEvent.click(screen.getByRole('button', { name: 'Review a contract' }));
    fireEvent.change(screen.getByLabelText('Contract file'), {
      target: { files: [new File(['1. X'], 'dpa.txt', { type: 'text/plain' })] },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Review' }));

    await waitFor(() => expect(sent.filter((s) => s.method === 'POST')).toHaveLength(1));
    expect(sent.find((s) => s.method === 'POST')?.body).toEqual({ filename: 'dpa.txt', content_base64: 'MS4gWA==' });
  });
});
