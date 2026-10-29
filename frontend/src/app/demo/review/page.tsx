'use client';

import { ReviewWorkspace } from '@/components/Review';
import { RunHeader } from '@/components/RunHeader';
import { Badge, Button } from '@/components/ui';
import review from '@/demo/review.json';
import type { AgentEvent } from '@/lib/events';
import { useReplay } from '@/lib/useReplay';

// Recorded by backend/scripts/capture_fixtures.py through the real review graph,
// its judgments scripted: five terms met, two partial, one missing, and one whose
// quote the contract does not have, which the check reports as unverified.
const EVENTS = review as readonly AgentEvent[];

/** A recorded contract review against GDPR Art. 28, replayed with no backend. */
export default function ReviewDemoPage(): React.JSX.Element {
  const { state, done, restart } = useReplay(EVENTS);
  const last = state.timeline.at(-1);
  const now = last === undefined ? 0 : Date.parse(last.ts);

  return (
    <main className="flex min-h-dvh flex-col bg-canvas text-ink">
      <RunHeader
        state={state}
        now={now}
        badges={
          <>
            <Badge tone="law">GDPR Art. 28 review</Badge>
            <Badge title="A recorded review, replayed: no backend, judgments scripted">Demo replay</Badge>
          </>
        }
        actions={
          <Button size="sm" variant="ghost" onClick={restart} disabled={!done}>
            Replay
          </Button>
        }
      />
      <ReviewWorkspace runId={null} state={state} />
    </main>
  );
}
