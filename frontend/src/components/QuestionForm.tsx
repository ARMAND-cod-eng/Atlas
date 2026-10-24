'use client';

import { useState } from 'react';

import { startRun } from '@/lib/api';

interface Props {
  /** Continue this conversation. Omitted, the question starts a new one. */
  readonly threadId?: string;
  readonly placeholder: string;
  readonly hint: string;
  readonly submitLabel: string;
  readonly onStarted: (runId: string) => void;
}

/**
 * One question box, for the first question and for follow-ups alike.
 *
 * A follow-up sent while the thread is still answering gets a 409 from the API
 * — two runs writing one checkpoint would overwrite each other's turn — and is
 * shown as what it is: a conversation that is busy, not one that is broken.
 */
export function QuestionForm({
  threadId,
  placeholder,
  hint,
  submitLabel,
  onStarted,
}: Props): React.JSX.Element {
  const [query, setQuery] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<{ tone: 'busy' | 'error'; text: string } | null>(null);

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (query.trim() === '' || submitting) return;

    setSubmitting(true);
    setNotice(null);
    const outcome = await startRun(query, threadId);
    if (outcome.kind === 'started') {
      // Routing rather than holding the run in memory: the run belongs to the
      // worker, so its URL should outlive this tab.
      onStarted(outcome.runId);
      return;
    }
    setNotice(
      outcome.kind === 'busy'
        ? {
            tone: 'busy',
            text: 'This conversation is still answering its last question. Ask again once it finishes.',
          }
        : { tone: 'error', text: outcome.message },
    );
    setSubmitting(false);
  }

  return (
    <form onSubmit={submit}>
      <textarea
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit(event);
        }}
        rows={3}
        maxLength={2000}
        placeholder={placeholder}
        className="w-full resize-none rounded-md border border-line bg-canvas px-3 py-2 text-sm leading-relaxed text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none"
      />
      <div className="mt-2 flex items-center justify-between gap-3">
        <span className="text-2xs text-ink-faint">{hint}</span>
        <button
          type="submit"
          disabled={submitting || query.trim() === ''}
          className="shrink-0 rounded border border-line-strong px-3 py-1.5 text-xs text-ink hover:border-ink-faint disabled:opacity-40"
        >
          {submitting ? 'starting…' : submitLabel}
        </button>
      </div>
      {notice !== null && (
        <p
          role="status"
          className={`mt-3 text-xs ${notice.tone === 'busy' ? 'text-retry' : 'text-failed'}`}
        >
          {notice.text}
        </p>
      )}
    </form>
  );
}
