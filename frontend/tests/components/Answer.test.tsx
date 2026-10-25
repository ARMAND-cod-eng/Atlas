// @vitest-environment jsdom
/**
 * The answer, rendered: each citation of the law a chip with the citation
 * check's verdict, and the provision's own text on hover.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import compliance from '../fixtures/compliance.json';

import { Answer } from '../../src/components/Answer';
import { API_BASE } from '../../src/lib/api';
import type { AgentEvent, RunResult } from '../../src/lib/events';
import {
  CRITIC,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  type RunState,
  acceptAll,
  initialState,
} from '../../src/lib/reducer';

const base = acceptAll(
  initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
  compliance as readonly AgentEvent[],
);

function withAnswer(answer: string, check: RunState['citationChecks'][number] | null): RunState {
  return { ...base, answer, citationChecks: check === null ? [] : [check] };
}

const CHECK = {
  ok: false,
  cited: ['AI Act Art. 6(2)', 'AI Act Art. 99(9)', 'GDPR Art. 35(1)'],
  unresolved: ['AI Act Art. 99(9)'],
  unretrieved: ['GDPR Art. 35(1)'],
  misquoted: [],
  ungroundedUrls: [],
};

const ANSWER = 'High-risk [AI Act Art. 6(2)]; fines [AI Act Art. 99(9)]; a DPIA [GDPR Art. 35(1)].';

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('citation chips', () => {
  it('carry the verdict of the last citation check', () => {
    render(<Answer state={withAnswer(ANSWER, CHECK)} result={null} />);

    expect(
      screen.getAllByRole('button').map((chip) => [chip.textContent, chip.className.includes('text-law')]),
    ).toEqual([
      ['AI Act Art. 6(2)✓', true],
      ['AI Act Art. 99(9)✗', false],
      ['GDPR Art. 35(1)?', false],
    ]);
    expect(screen.getByText(/2 of 3 citations failed the check/)).toBeDefined();
  });

  it("show the provision's text from the library on hover", async () => {
    const asked: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        asked.push(url);
        return new Response(
          JSON.stringify({
            label: 'AI Act Art. 6(2)',
            act: 'AI Act',
            title: 'Classification rules for high-risk AI systems',
            text: 'In addition to the high-risk AI systems referred to in paragraph 1…',
            url: 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_6',
          }),
          { status: 200 },
        );
      }),
    );
    render(<Answer state={withAnswer(ANSWER, CHECK)} result={null} />);

    fireEvent.mouseEnter(screen.getByRole('button', { name: /AI Act Art\. 6\(2\)/ }));

    const card = await screen.findByRole('tooltip');
    await within(card).findByText(/In addition to the high-risk AI systems/);
    expect(within(card).getByRole('link').getAttribute('href')).toBe(
      'https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_6',
    );
    expect(asked).toEqual([`${API_BASE}/library/provision?label=AI%20Act%20Art.%206(2)`]);
  });

  it('say a provision does not exist rather than show a near one', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{"detail":"no such provision"}', { status: 404 })));
    render(<Answer state={withAnswer(ANSWER, CHECK)} result={null} />);

    fireEvent.mouseEnter(screen.getByRole('button', { name: /AI Act Art\. 99\(9\)/ }));

    const card = await screen.findByRole('tooltip');
    expect(card.textContent).toBe('No such provision: the act has nothing by this number.');
  });

  it('take the verdict on the answer shown, not on an earlier draft', () => {
    const draft = { ...CHECK, cited: ['AI Act Art. 6(2)'], unresolved: ['AI Act Art. 6(2)'], unretrieved: [] };
    const final = { ...CHECK, ok: true, cited: ['AI Act Art. 6(2)'], unresolved: [], unretrieved: [] };
    render(<Answer state={{ ...withAnswer('[AI Act Art. 6(2)]', null), citationChecks: [draft, final] }} result={null} />);

    expect(screen.getByRole('button').textContent).toBe('AI Act Art. 6(2)✓');
  });

  it('are plain where no check ran', () => {
    render(<Answer state={withAnswer('See [GDPR Art. 5].', null)} result={null} />);

    expect(screen.getByRole('button').textContent).toBe('GDPR Art. 5');
    expect(screen.queryByText(/citations/)).toBeNull();
  });
});

describe('sources', () => {
  it('list two paragraphs of one article, which share its URL, without a key clash', () => {
    const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const url = 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_6';
    const result = {
      sources: [
        { url, title: 'Classification rules', snippet: '', label: 'AI Act Art. 6(1)', authority: 'primary_law' },
        { url, title: 'Classification rules', snippet: '', label: 'AI Act Art. 6(2)', authority: 'primary_law' },
        { url: 'https://example.com/a', title: 'A guide', snippet: '', label: null, authority: 'secondary' },
      ],
    } as unknown as RunResult;

    render(<Answer state={withAnswer('Yes.', null)} result={result} />);

    expect(screen.getByText('Primary law · 2')).toBeDefined();
    expect(screen.getByText('Other sources · 1')).toBeDefined();
    expect(errors).not.toHaveBeenCalled();
  });
});
