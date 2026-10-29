// @vitest-environment jsdom
/**
 * The versions list, and the comparison of two answers: asked for from the
 * answer's own header, shown in its place.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AnswerCard } from '../../src/components/Answer';
import { Versions } from '../../src/components/Versions';
import { API_BASE, type RunVersion } from '../../src/lib/api';
import { CRITIC, PLANNER, RESEARCHER, SYNTHESIZER, type RunState, initialState } from '../../src/lib/reducer';
import { versionRows } from '../../src/lib/versions';

const RUNS: readonly RunVersion[] = [
  {
    run_id: 'v1-run',
    kind: 'question',
    query: 'Is a CV-screening tool high-risk?',
    parent_run_id: null,
    revision: null,
    status: 'completed',
    created_at: '2026-10-24T10:00:00Z',
    finished_at: '2026-10-24T10:01:00Z',
  },
  {
    run_id: 'v2-run',
    kind: 'revision',
    query: 'Is a CV-screening tool high-risk?',
    parent_run_id: 'v1-run',
    revision: { parent_run_id: 'v1-run', subtask_id: 's1', before: 'What must a provider do?', after: 'What must a deployer do?' },
    status: 'completed',
    created_at: '2026-10-24T10:05:00Z',
    finished_at: '2026-10-24T10:06:00Z',
  },
];

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('the versions list', () => {
  it('links every version but the one on screen', () => {
    render(<Versions rows={versionRows(RUNS, 'v2-run')} />);

    const list = within(screen.getByRole('region', { name: 'Versions' }));
    expect(list.getAllByRole('link').map((link) => link.getAttribute('href'))).toEqual(['/runs/v1-run']);
    expect(screen.getByText('Edited s1').closest('[aria-current="page"]')).not.toBeNull();
    expect(screen.getByText('revises v1')).toBeDefined();
  });
});

function answering(answer: string): RunState {
  return { ...initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]), answer, phase: 'completed' };
}

const heading = () => screen.getByRole('heading', { level: 2 }).textContent;

describe('comparing two versions', () => {
  it("fetches the other version's answer and shows what changed in the answer's place", async () => {
    const asked: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        asked.push(url);
        return new Response(JSON.stringify({ result: { answer: 'The provider must register it.' } }), { status: 200 });
      }),
    );
    render(
      <AnswerCard
        state={answering('The deployer must monitor and register it.')}
        result={null}
        rows={versionRows(RUNS, 'v2-run')}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Compare with v1' }));

    expect((await screen.findByText(/^Since v1/)).textContent).toBe('Since v1: +3 words, −1 words');
    expect(document.querySelector('del')?.textContent).toBe('provider');
    expect([...document.querySelectorAll('ins')].map((ins) => ins.textContent)).toEqual(['deployer', 'monitor and ']);
    expect(asked).toEqual([`${API_BASE}/runs/v1-run`]);
    expect(heading()).toBe('Changes since v1');
    expect(screen.getByRole('button', { name: 'Compare with v1' }).getAttribute('aria-pressed')).toBe('true');
    // In its place: the answer itself is not shown beside its changes.
    expect(screen.queryByRole('article')).toBeNull();
  });

  it('goes back to the answer', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ result: { answer: 'Old.' } }), { status: 200 })),
    );
    render(<AnswerCard state={answering('New.')} result={null} rows={versionRows(RUNS, 'v2-run')} />);

    fireEvent.click(screen.getByRole('button', { name: 'Compare with v1' }));
    await screen.findByText(/^Since v1/);
    fireEvent.click(screen.getByRole('button', { name: /Back to answer/ }));

    expect(heading()).toBe('Answer');
    expect(screen.getByRole('article').textContent).toBe('New.');
    expect(screen.queryByText(/^Since/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Compare with v1' }).getAttribute('aria-pressed')).toBe('false');
  });

  it("never shows one version's changes under another's name while it loads", async () => {
    const third: RunVersion = { ...RUNS[0]!, run_id: 'v3-run', query: 'And for GPAI?', parent_run_id: 'v2-run' };
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) =>
        url.endsWith('/v1-run')
          ? Promise.resolve(new Response(JSON.stringify({ result: { answer: 'Old.' } }), { status: 200 }))
          : new Promise<Response>(() => undefined),
      ),
    );
    render(<AnswerCard state={answering('New.')} result={null} rows={versionRows([...RUNS, third], 'v2-run')} />);

    fireEvent.click(screen.getByRole('button', { name: 'Compare with v1' }));
    await screen.findByText(/^Since v1/);
    fireEvent.click(screen.getByRole('button', { name: 'Compare with v3' }));

    expect(screen.getByText('Loading v3…')).toBeDefined();
    expect(screen.queryByText(/^Since/)).toBeNull();
    expect(heading()).toBe('Changes since v3');
  });

  it('is not offered when there is nothing finished to compare with', () => {
    render(<AnswerCard state={answering('Yes.')} result={null} rows={versionRows(RUNS.slice(0, 1), 'v1-run')} />);

    expect(screen.queryByRole('group', { name: 'Compare with' })).toBeNull();
    expect(screen.getByRole('article').textContent).toBe('Yes.');
  });
});
