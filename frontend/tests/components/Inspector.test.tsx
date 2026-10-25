// @vitest-environment jsdom
/**
 * The inspector, rendered: rewriting a branch and re-running it, and reading
 * the exact prompt an agent was sent.
 *
 * `fetch` is replaced per test and every request it receives is kept, so a
 * test fails on what was sent, not only on what came back.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import failed from '../fixtures/run-failed.json';
import retry from '../fixtures/retry.json';

import { Inspector } from '../../src/components/Inspector';
import { API_BASE, type AgentCall, fetchCall } from '../../src/lib/api';
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

function stateOf(events: unknown): RunState {
  return acceptAll(initialState([PLANNER, RESEARCHER, SYNTHESIZER, CRITIC]), events as readonly AgentEvent[]);
}

interface Sent {
  readonly url: string;
  readonly method: string;
  readonly body: unknown;
}

/** Answer every request with `status` and `body`, keeping what was asked. */
function serve(status: number, body: unknown): Sent[] {
  const sent: Sent[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      sent.push({
        url,
        method: init?.method ?? 'GET',
        body: init?.body === undefined ? null : JSON.parse(String(init.body)),
      });
      return new Response(JSON.stringify(body), { status });
    }),
  );
  return sent;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const S2 = 'Which current models support context windows of 1M tokens or more?';

describe('rewriting a branch', () => {
  const state = stateOf(retry);
  const question = state.branches.find((b) => b.subtaskId === 's2')?.question;

  it('re-runs it with the new question and opens the new run', async () => {
    const sent = serve(202, { run_id: 'run-2', thread_id: 't', events_url: '/runs/run-2/events' });
    const onRevised = vi.fn();
    render(<Inspector runId="run-1" loadCall={null} state={state} result={null} selected="researcher:s2" onRevised={onRevised} />);

    fireEvent.change(screen.getByLabelText("Rewrite this branch's question"), {
      target: { value: '  Which models accept a 1M-token prompt in 2026?  ' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Re-run this branch' }));

    await waitFor(() => expect(onRevised).toHaveBeenCalledWith('run-2'));
    expect(sent).toEqual([
      {
        url: `${API_BASE}/runs/run-1/revisions`,
        method: 'POST',
        body: { subtask_id: 's2', question: 'Which models accept a 1M-token prompt in 2026?' },
      },
    ]);
  });

  it('cannot be sent unchanged', () => {
    serve(202, {});
    render(<Inspector runId="run-1" loadCall={null} state={state} result={null} selected="researcher:s2" />);

    expect(screen.getByLabelText("Rewrite this branch's question")).toHaveProperty('value', question);
    expect(screen.getByRole('button', { name: 'Re-run this branch' })).toHaveProperty('disabled', true);
  });

  it('says so when the conversation is busy, and stays on this run', async () => {
    serve(409, { detail: 'thread busy' });
    const onRevised = vi.fn();
    render(<Inspector runId="run-1" loadCall={null} state={state} result={null} selected="researcher:s2" onRevised={onRevised} />);

    fireEvent.change(screen.getByLabelText("Rewrite this branch's question"), { target: { value: S2 } });
    fireEvent.click(screen.getByRole('button', { name: 'Re-run this branch' }));

    expect((await screen.findByRole('alert')).textContent).toBe(
      'This conversation is answering something else. Try again when it finishes.',
    );
    expect(onRevised).not.toHaveBeenCalled();
  });

  it("shows the API's reason for refusing", async () => {
    serve(422, { detail: 'the question is unchanged' });
    render(<Inspector runId="run-1" loadCall={null} state={state} result={null} selected="researcher:s2" />);

    fireEvent.change(screen.getByLabelText("Rewrite this branch's question"), { target: { value: S2 } });
    fireEvent.click(screen.getByRole('button', { name: 'Re-run this branch' }));

    expect((await screen.findByRole('alert')).textContent).toBe(
      'the API refused the edit (422): the question is unchanged',
    );
  });

  it('is not offered on a replay, which has no API behind it', () => {
    render(<Inspector runId={null} loadCall={null} state={state} result={null} selected="researcher:s2" />);

    expect(screen.queryByLabelText("Rewrite this branch's question")).toBeNull();
  });

  it('is not offered on a failed run, which has no answer to revise', () => {
    const broken = stateOf(failed);
    const branch = broken.branches[0];
    expect(branch).toBeDefined();
    render(<Inspector runId="run-1" loadCall={null} state={broken} result={null} selected={branch?.id ?? ''} />);

    expect(screen.queryByText('Asked')).not.toBeNull();
    expect(screen.queryByLabelText("Rewrite this branch's question")).toBeNull();
  });
});

describe('the prompt tab', () => {
  const state = stateOf(retry);
  const latest = state.branches.find((b) => b.subtaskId === 's2')?.spans.at(-1);
  // The recorded run's own model call, in the retried branch's second attempt.
  const model = state.calls.find((c) => c.span_id === latest && c.kind === 'model');
  const record: AgentCall = {
    call_id: model?.call_id ?? '',
    span_id: model?.span_id ?? null,
    node: RESEARCHER,
    subtask_id: 's2',
    kind: 'model',
    name: 'scripted',
    started_at: '2026-10-24T00:00:00Z',
    duration_s: 0.01,
    request: [
      { role: 'system', content: 'You are a researcher.' },
      { role: 'human', content: 'Subtask s2\nQuestion: which models…' },
    ],
    response: 'findings',
    input_tokens: null,
    output_tokens: null,
    error: null,
  };

  it("fetches the latest attempt's model call from the API and shows what was sent", async () => {
    const sent = serve(200, record);
    render(
      <Inspector
        runId="run-1"
        loadCall={(id) => fetchCall('run-1', id)}
        state={state}
        result={null}
        selected="researcher:s2"
      />,
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Prompt' }));

    expect((await screen.findByText(/^Subtask s2/)).textContent).toBe('Subtask s2\nQuestion: which models…');
    expect(screen.getByText('You are a researcher.')).toBeDefined();
    expect(sent.map((s) => s.url)).toEqual([`${API_BASE}/runs/run-1/calls/${record.call_id}`]);
  });

  it('reads a recording without asking any API', async () => {
    const sent = serve(500, {});
    const recorded = new Map([[record.call_id, record]]);
    render(
      <Inspector
        runId={null}
        loadCall={(id) => Promise.resolve(recorded.get(id) ?? null)}
        state={state}
        result={null}
        selected="researcher:s2"
      />,
    );

    fireEvent.click(screen.getByRole('tab', { name: 'Prompt' }));

    expect((await screen.findByText(/^Subtask s2/)).textContent).toBe('Subtask s2\nQuestion: which models…');
    expect(sent).toEqual([]);
  });

  it('with no record of calls, says prompts are kept on live runs', () => {
    const sent = serve(200, {});
    render(<Inspector runId={null} loadCall={null} state={state} result={null} selected="researcher:s2" />);

    fireEvent.click(screen.getByRole('tab', { name: 'Prompt' }));

    expect(screen.getByText(/Prompts are recorded on live runs/)).toBeDefined();
    expect(sent).toEqual([]);
  });
});
