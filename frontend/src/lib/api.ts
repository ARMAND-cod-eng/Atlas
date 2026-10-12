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
export async function startRun(query: string, threadId?: string): Promise<StartOutcome> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(threadId === undefined ? { query } : { query, thread_id: threadId }),
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
