'use client';

import { use } from 'react';

import { ReviewWorkspace } from '@/components/Review';
import { RunHeader } from '@/components/RunHeader';
import { Badge } from '@/components/ui';
import { isFinished } from '@/lib/reducer';
import { useNow, useRunStream } from '@/lib/useRunStream';

/** One contract review, live as it runs and the same once finished. */
export default function ReviewPage({
  params,
}: {
  readonly params: Promise<{ readonly runId: string }>;
}): React.JSX.Element {
  const { runId } = use(params);
  // A review's result is read from its events; GET /runs serves research results.
  const { state, missing } = useRunStream(runId, { fetchResult: false });
  const now = useNow(!isFinished(state) && !missing);

  return (
    <main className="flex min-h-dvh flex-col bg-canvas">
      <RunHeader state={state} now={now} badges={<Badge tone="law">GDPR Art. 28 review</Badge>} />
      {missing ? (
        <p className="mx-auto max-w-xl p-8 text-sm text-ink-muted">
          No review with this id. It may have been deleted, or the link is wrong.
        </p>
      ) : (
        <ReviewWorkspace runId={runId} state={state} />
      )}
    </main>
  );
}
