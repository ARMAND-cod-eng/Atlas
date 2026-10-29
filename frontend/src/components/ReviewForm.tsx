'use client';

import { useState } from 'react';

import { Button } from './ui';
import { startReview } from '@/lib/review';

/** A file's bytes as base64, read in the browser: the API takes the file in JSON. */
function base64Of(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] ?? '');
    reader.onerror = () => reject(reader.error ?? new Error('the file could not be read'));
    reader.readAsDataURL(file);
  });
}

/**
 * A contract to review against GDPR Art. 28: a file, or its text pasted. Read
 * by the API before anything is queued, so a file it cannot read is refused
 * here, with why.
 */
export function ReviewForm({ onStarted }: { readonly onStarted: (runId: string) => void }): React.JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const ready = file !== null || text.trim() !== '';

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (!ready || sending) return;
    setSending(true);
    setMessage(null);
    let outcome;
    try {
      outcome =
        file !== null
          ? await startReview({ filename: file.name, content_base64: await base64Of(file) })
          : await startReview({ filename: 'pasted contract', text });
    } catch {
      outcome = { kind: 'error' as const, message: 'the file could not be read in the browser' };
    }
    if (outcome.kind === 'started') {
      onStarted(outcome.runId);
      return;
    }
    setMessage(outcome.message);
    setSending(false);
  }

  return (
    <form onSubmit={(event) => void submit(event)} className="space-y-3">
      <div className="rounded-xl border border-line-strong bg-surface p-4 shadow-sm">
        <p className="text-xs font-medium text-ink">Against GDPR Art. 28: the terms a processor contract must contain</p>
        <label className="mt-3 flex cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-line-strong bg-canvas px-4 py-6 text-center transition-colors hover:border-accent">
          <span className="text-xs font-medium text-ink">{file === null ? 'Choose a file' : file.name}</span>
          <span className="text-2xs text-ink-faint">.pdf with selectable text, .docx, .txt or .md · up to 5 MB</span>
          <input
            type="file"
            aria-label="Contract file"
            accept=".pdf,.docx,.txt,.md"
            className="sr-only"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>
        <label className="mt-3 block text-2xs font-medium text-ink-muted">
          Or paste the contract
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            disabled={file !== null}
            rows={4}
            placeholder="1. Subject matter&#10;The Processor processes personal data on behalf of the Controller…"
            className="mt-1 block w-full resize-y rounded-md border border-line bg-canvas px-3 py-2 text-xs font-normal text-ink placeholder:text-ink-faint disabled:opacity-50"
          />
        </label>
        <div className="mt-3 flex items-center justify-between gap-3">
          <span className="text-2xs text-ink-faint">
            Each term judged against the clauses most like it; every quote checked against the contract.
          </span>
          <Button type="submit" variant="primary" disabled={!ready || sending}>
            {sending ? 'Starting…' : 'Review'}
          </Button>
        </div>
      </div>
      {message !== null && (
        <p role="alert" className="text-xs text-failed">
          {message}
        </p>
      )}
    </form>
  );
}
