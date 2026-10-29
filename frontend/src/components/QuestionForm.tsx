'use client';

import { useRef, useState } from 'react';

import { Button } from './ui';
import { startRun } from '@/lib/api';
import type { Domain } from '@/lib/events';

interface Props {
  /** Continue this conversation. Omitted, the question starts a new one. */
  readonly threadId?: string;
  /**
   * The version a follow-up continues. After a revision the thread's latest
   * state is the revision's, which may not be the answer on screen.
   */
  readonly parentRunId?: string;
  /** A new conversation's domain; omitted, the API's configured one. */
  readonly domain?: Domain;
  readonly placeholder: string;
  /** Questions a click puts in the box, to edit or send as they are. */
  readonly examples?: readonly string[];
  readonly hint: string;
  readonly submitLabel: string;
  /** The home page's box is the page; a follow-up's sits under an answer. */
  readonly size?: 'lg' | 'md';
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
  parentRunId,
  domain,
  placeholder,
  examples = [],
  hint,
  submitLabel,
  size = 'md',
  onStarted,
}: Props): React.JSX.Element {
  const [query, setQuery] = useState('');
  const box = useRef<HTMLTextAreaElement>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<{ tone: 'busy' | 'error'; text: string } | null>(null);

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (query.trim() === '' || submitting) return;

    setSubmitting(true);
    setNotice(null);
    const outcome = await startRun(query, threadId, parentRunId, domain);
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
      {/* One box: the question and what sends it, so focus on either lights the whole. */}
      <div className="rounded-xl border border-line-strong bg-surface shadow-sm transition-colors focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
        <textarea
          ref={box}
          aria-label="Your question"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit(event);
          }}
          rows={size === 'lg' ? 3 : 2}
          maxLength={2000}
          placeholder={placeholder}
          className={`block w-full resize-none rounded-t-xl bg-transparent px-4 pt-3 leading-relaxed text-ink placeholder:text-ink-faint focus:outline-none focus-visible:outline-none ${
            size === 'lg' ? 'text-base' : 'text-sm'
          }`}
        />
        <div className="flex items-center justify-between gap-3 px-3 pb-3 pt-1">
          <span className="pl-1 text-2xs text-ink-faint">{hint}</span>
          <Button type="submit" variant="primary" size={size === 'lg' ? 'md' : 'sm'} disabled={submitting || query.trim() === ''}>
            {submitting ? 'Starting…' : submitLabel}
          </Button>
        </div>
      </div>
      {examples.length > 0 && (
        <div className="mt-4">
          <p id="examples-label" className="text-2xs font-medium text-ink-muted">
            Try one
          </p>
          <ul aria-labelledby="examples-label" className="mt-2 flex flex-wrap gap-2">
            {examples.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  onClick={() => {
                    setQuery(example);
                    box.current?.focus();
                  }}
                  className="rounded-full border border-line bg-surface px-3 py-1 text-left text-2xs text-ink-muted transition-colors hover:border-accent hover:text-ink"
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
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
