// @vitest-environment jsdom
/**
 * The workspace, rendered whole on a phone: a branch is redirected from the
 * answer, where the question that came back with the wrong sources is shown.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import retry from '../fixtures/retry.json';

import { RunWorkspace } from '../../src/components/RunWorkspace';
import type { AgentEvent } from '../../src/lib/events';
import {
  CRITIC,
  PLANNER,
  RESEARCHER,
  SYNTHESIZER,
  acceptAll,
  initialState,
} from '../../src/lib/reducer';

const state = acceptAll(
  initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]),
  retry as readonly AgentEvent[],
);
const S2 = state.branches.find((branch) => branch.subtaskId === 's2');

beforeEach(() => {
  // A phone: one pane at a time, behind tabs.
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: false,
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
  // The versions list asks the API; it has nothing to say here.
  vi.stubGlobal('fetch', vi.fn(async () => new Response('[]', { status: 404 })));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState(null, '', '/');
});

const workspace = (runId: string | null) => (
  <RunWorkspace
    runId={runId}
    loadCall={null}
    state={state}
    result={null}
    topology={null}
    now={0}
    connection="closed"
  />
);

describe('editing a branch from the answer', () => {
  it("opens that branch's question in the Agents tab, ready to type in", async () => {
    render(workspace('run-1'));
    // A finished run opens on its answer.
    expect(screen.getByRole('tab', { name: 'Answer' }).getAttribute('aria-selected')).toBe('true');

    fireEvent.click(screen.getByRole('button', { name: 'Edit & re-run s2' }));

    expect(screen.getByRole('tab', { name: 'Agents' }).getAttribute('aria-selected')).toBe('true');
    const question = screen.getByLabelText("Rewrite this branch's question");
    expect(question).toHaveProperty('value', S2?.question);
    await waitFor(() => expect(document.activeElement).toBe(question));
    // And the link says so, as if it had been opened by hand.
    await waitFor(() => expect(window.location.search).toBe('?node=researcher%3As2'));
  });

  it('is not offered on a replay, which has no run to revise', () => {
    render(workspace(null));

    expect(screen.getByRole('tab', { name: 'Answer' }).getAttribute('aria-selected')).toBe('true');
    expect(screen.queryByRole('button', { name: /Edit & re-run/ })).toBeNull();
  });
});
