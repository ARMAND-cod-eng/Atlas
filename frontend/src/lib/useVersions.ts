'use client';

import { useEffect, useState } from 'react';

import { type RunVersion, fetchVersions } from './api';

/**
 * The runs on a thread, asked for again whenever `refresh` changes: a run that
 * finishes changes its own status in the list. Null until known, and for a
 * replay, which has no thread behind it.
 */
export function useVersions(
  threadId: string | null,
  runId: string | null,
  refresh: unknown,
): readonly RunVersion[] | null {
  const [fetched, setFetched] = useState<{
    readonly threadId: string;
    readonly runs: readonly RunVersion[] | null;
  } | null>(null);

  useEffect(() => {
    if (threadId === null || runId === null) return;
    let live = true;
    void fetchVersions(threadId).then((runs) => {
      if (live) setFetched({ threadId, runs });
    });
    return () => {
      live = false;
    };
  }, [threadId, runId, refresh]);

  return fetched !== null && fetched.threadId === threadId ? fetched.runs : null;
}
