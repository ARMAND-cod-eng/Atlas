import type { Domain } from './events';

/**
 * 127.0.0.1, not localhost — and this is worth two seconds per request.
 *
 * On Windows `localhost` resolves to `::1` before `127.0.0.1`, and uvicorn binds
 * IPv4 only. The browser therefore tries IPv6 first, waits for that attempt to
 * fail, and only then connects — measured at **2,050 ms against 22 ms** for the
 * identical request to `127.0.0.1`. Every call pays it: the topology fetch, the
 * POST, and the SSE connection, which is why the run page appeared to take many
 * seconds to show anything.
 *
 * Naming the loopback address explicitly removes the resolution step entirely.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

/** What starting a run can come to, in terms the UI can act on. */
export type StartOutcome =
  | { readonly kind: 'started'; readonly runId: string }
  // The thread already has a run in progress. Not a failure: the conversation
  // is fine, it is just answering the previous question.
  | { readonly kind: 'busy' }
  | { readonly kind: 'error'; readonly message: string };

/**
 * `POST /runs`. A `threadId` continues that conversation; omitting it starts one.
 *
 * Never throws: every outcome, including an unreachable API, comes back as a
 * value, so a caller cannot forget a branch and leave a button spinning.
 */
export async function startRun(
  query: string,
  threadId?: string,
  parentRunId?: string,
  /** A new conversation's; a follow-up runs in its conversation's. */
  domain?: Domain,
): Promise<StartOutcome> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // A follow-up names the version it continues: after a revision, the
      // thread's latest state is the revision's, not the one on screen.
      body: JSON.stringify({
        query,
        ...(threadId === undefined ? {} : { thread_id: threadId }),
        ...(parentRunId === undefined ? {} : { parent_run_id: parentRunId }),
        ...(domain === undefined ? {} : { domain }),
      }),
    });
  } catch {
    return { kind: 'error', message: `could not reach the API at ${API_BASE}. Is it running?` };
  }

  if (response.status === 409) return { kind: 'busy' };
  if (!response.ok) {
    return { kind: 'error', message: `the API returned ${response.status}${await detailOf(response)}` };
  }
  const body = (await response.json()) as { run_id: string };
  return { kind: 'started', runId: body.run_id };
}

/** FastAPI's error detail: a string, or validation errors whose first `msg` says enough. */
async function detailOf(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return `: ${body.detail}`;
    if (Array.isArray(body.detail)) {
      const first: unknown = body.detail[0];
      if (typeof first === 'object' && first !== null && 'msg' in first && typeof first.msg === 'string') {
        return `: ${first.msg}`;
      }
    }
  } catch {
    // Not JSON — the status code alone will have to do.
  }
  return '';
}

export type RunPresence = 'known' | 'missing' | 'unreachable';

/**
 * Whether the API knows this run, before the page opens its event stream.
 *
 * `POST /runs` registers a run before it returns, so a real run is always
 * known — even one still queued. A 404 therefore means a wrong id or a run
 * older than the hour its events are kept. Asked first because the stream
 * cannot say so itself: the browser closes a 404'd EventSource without
 * exposing the status, and the page would wait on "starting" forever.
 *
 * Only a 404 counts as missing. A 500 or an unreachable API says nothing about
 * the run, so the page carries on and lets the stream retry.
 */
export async function runExists(runId: string): Promise<RunPresence> {
  try {
    const response = await fetch(`${API_BASE}/runs/${runId}`);
    return response.status === 404 ? 'missing' : 'known';
  } catch {
    return 'unreachable';
  }
}

/**
 * `POST /runs/{runId}/revisions`: rewrite one branch's question and re-run it.
 *
 * A 409 means the thread is busy (the outcome a person can wait out); a 422 is
 * the API refusing the edit (an unchanged question, an unknown branch), whose
 * detail says why.
 */
export async function reviseRun(runId: string, subtaskId: string, question: string): Promise<StartOutcome> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/runs/${runId}/revisions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subtask_id: subtaskId, question }),
    });
  } catch {
    return { kind: 'error', message: `could not reach the API at ${API_BASE}. Is it running?` };
  }
  if (response.status === 409) return { kind: 'busy' };
  if (!response.ok) {
    return { kind: 'error', message: `the API refused the edit (${response.status})${await detailOf(response)}` };
  }
  const body = (await response.json()) as { run_id: string };
  return { kind: 'started', runId: body.run_id };
}

/** One run on a thread, as `GET /threads/{id}/versions` lists it. */
export interface RunVersion {
  readonly run_id: string;
  readonly kind: 'question' | 'revision';
  readonly query: string;
  readonly parent_run_id: string | null;
  readonly revision: {
    readonly parent_run_id: string;
    readonly subtask_id: string;
    readonly before: string;
    readonly after: string;
  } | null;
  readonly status: 'running' | 'completed' | 'failed';
  readonly created_at: string;
  readonly finished_at: string | null;
}

/** Every run on a thread, oldest first; null when the API cannot say. */
export async function fetchVersions(threadId: string): Promise<readonly RunVersion[] | null> {
  try {
    const response = await fetch(`${API_BASE}/threads/${threadId}/versions`);
    if (!response.ok) return null;
    const body = (await response.json()) as { runs: readonly RunVersion[] };
    return body.runs;
  } catch {
    return null;
  }
}

/** One agent call in full: the exact messages sent and what came back. */
export interface AgentCall {
  readonly call_id: string;
  readonly span_id: string | null;
  readonly node: string;
  readonly subtask_id: string | null;
  readonly kind: 'model' | 'tool';
  readonly name: string;
  readonly started_at: string;
  readonly duration_s: number;
  /** A model's messages; a tool's `{query}`. */
  readonly request: readonly { readonly role: string; readonly content: string }[] | { readonly query: string };
  /** A model's text or parsed output; a tool's results. */
  readonly response: unknown;
  readonly input_tokens: number | null;
  readonly output_tokens: number | null;
  readonly error: string | null;
}

/** A provision of the law, as a citation chip shows it. */
export interface Provision {
  readonly label: string;
  readonly act: string;
  readonly title: string;
  readonly text: string;
  readonly url: string;
}

// Calls and provisions never change once written, so a page keeps what it has
// fetched: reopening an agent, or hovering a citation twice, costs nothing.
const calls = new Map<string, Promise<AgentCall | null>>();
const provisions = new Map<string, Promise<Provision | null>>();

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url);
    return response.ok ? ((await response.json()) as T) : null;
  } catch {
    return null;
  }
}

/** `GET /runs/{runId}/calls/{callId}`, fetched once per page. */
export function fetchCall(runId: string, callId: string): Promise<AgentCall | null> {
  const key = `${runId}/${callId}`;
  let pending = calls.get(key);
  if (pending === undefined) {
    pending = getJson<AgentCall>(`${API_BASE}/runs/${runId}/calls/${callId}`);
    // A failure is not kept: the next attempt asks again.
    void pending.then((call) => call === null && calls.delete(key));
    calls.set(key, pending);
  }
  return pending;
}

/** `GET /library/provision?label=`, fetched once per page. */
export function fetchProvision(label: string): Promise<Provision | null> {
  let pending = provisions.get(label);
  if (pending === undefined) {
    pending = getJson<Provision>(`${API_BASE}/library/provision?label=${encodeURIComponent(label)}`);
    void pending.then((found) => found === null && provisions.delete(label));
    provisions.set(label, pending);
  }
  return pending;
}

/** A finished run's answer, for comparing versions. Null if not finished or unknown. */
export async function fetchAnswer(runId: string): Promise<string | null> {
  const body = await getJson<{ result: { answer: string } | null }>(`${API_BASE}/runs/${runId}`);
  return body?.result?.answer ?? null;
}
