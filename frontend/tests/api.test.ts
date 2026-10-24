/**
 * Starting a run: what each answer from `POST /runs` means to the user.
 *
 * `fetch` is replaced per test. Outcomes are compared whole, so a 409 that
 * came back as a generic error — the bug this exists to prevent, since the
 * follow-up box is exactly where a second click lands on a busy thread — fails
 * on the `kind`, not on some looser check.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchCall, fetchProvision, reviseRun, runExists, startRun } from '../src/lib/api';

function respond(status: number, body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(body), { status })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('startRun', () => {
  it('returns the new run id on 202', async () => {
    respond(202, { run_id: 'r-1', thread_id: 't-1', events_url: '/runs/r-1/events' });

    expect(await startRun('what is RAG?')).toEqual({ kind: 'started', runId: 'r-1' });
  });

  it('sends the thread id only for a follow-up', async () => {
    const sent: unknown[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        sent.push(JSON.parse(String(init?.body)));
        return new Response(JSON.stringify({ run_id: 'r-2' }), { status: 202 });
      }),
    );

    await startRun('first question');
    await startRun('and then?', 't-1');

    expect(sent).toEqual([{ query: 'first question' }, { query: 'and then?', thread_id: 't-1' }]);
  });

  it('reports a busy thread as busy, not as a failure', async () => {
    respond(409, { detail: 'thread t-1 already has a run in progress: r-0' });

    expect(await startRun('and then?', 't-1')).toEqual({ kind: 'busy' });
  });

  it('passes the API detail through for a rejected question', async () => {
    respond(422, { detail: [{ msg: 'String should have at most 2000 characters' }] });

    expect(await startRun('x'.repeat(2001))).toEqual({
      kind: 'error',
      message: 'the API returned 422: String should have at most 2000 characters',
    });
  });

  it('names the API address when it cannot be reached', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }));

    expect(await startRun('what is RAG?')).toEqual({
      kind: 'error',
      message: 'could not reach the API at http://127.0.0.1:8000. Is it running?',
    });
  });
});

/**
 * Whether a run page has anything to show.
 *
 * The page used to open the event stream unconditionally. For an unknown id —
 * a typo, or a link older than the hour events are kept — the stream 404s,
 * the browser closes it without saying why, and the page sat on "starting"
 * forever: a run that does not exist, looking exactly like one about to begin.
 */
describe('runExists', () => {
  it('is known when the API answers', async () => {
    respond(200, { run_id: 'r-1', status: 'running', result: null, error: null });

    expect(await runExists('r-1')).toBe('known');
  });

  it('is missing on 404', async () => {
    respond(404, { detail: 'unknown run: r-9' });

    expect(await runExists('r-9')).toBe('missing');
  });

  it('is unreachable when the API is down, so the stream can still retry', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch');
    }));

    expect(await runExists('r-1')).toBe('unreachable');
  });

  it('treats any other answer as known rather than declaring the run gone', async () => {
    respond(500, { error: 'internal_error' });

    expect(await runExists('r-1')).toBe('known');
  });
});

describe('reviseRun', () => {
  it('returns the revision run on 202', async () => {
    respond(202, { run_id: 'r-2', thread_id: 't-1', events_url: '/runs/r-2/events' });

    expect(await reviseRun('r-1', 's2', 'edited')).toEqual({ kind: 'started', runId: 'r-2' });
  });

  it('says the thread is busy on 409', async () => {
    respond(409, { detail: 'thread t-1 already has a run in progress: r-9' });

    expect(await reviseRun('r-1', 's2', 'edited')).toEqual({ kind: 'busy' });
  });

  it("passes on why the API refused an edit", async () => {
    respond(422, { detail: 'the question for s2 is unchanged' });

    expect(await reviseRun('r-1', 's2', 'same')).toEqual({
      kind: 'error',
      message: 'the API refused the edit (422): the question for s2 is unchanged',
    });
  });
});

describe('startRun from a version', () => {
  it('names the version the follow-up continues', async () => {
    const sent: unknown[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_url: string, init?: RequestInit) => {
        sent.push(JSON.parse(String(init?.body)));
        return new Response(JSON.stringify({ run_id: 'r-3' }), { status: 202 });
      }),
    );

    await startRun('and then?', 't-1', 'r-1');

    expect(sent).toEqual([{ query: 'and then?', thread_id: 't-1', parent_run_id: 'r-1' }]);
  });
});

describe('fetching what never changes', () => {
  it('asks for a call once, however often it is opened', async () => {
    const fetched = vi.fn(async () => new Response(JSON.stringify({ call_id: 'c-1' }), { status: 200 }));
    vi.stubGlobal('fetch', fetched);

    await fetchCall('r-cache', 'c-1');
    await fetchCall('r-cache', 'c-1');

    expect(fetched).toHaveBeenCalledTimes(1);
  });

  it('asks again after a failure rather than remembering it', async () => {
    const fetched = vi.fn(async () => new Response('{}', { status: 503 }));
    vi.stubGlobal('fetch', fetched);

    expect(await fetchProvision('GDPR Art. 99(9)')).toBeNull();
    expect(await fetchProvision('GDPR Art. 99(9)')).toBeNull();

    expect(fetched).toHaveBeenCalledTimes(2);
  });
});
