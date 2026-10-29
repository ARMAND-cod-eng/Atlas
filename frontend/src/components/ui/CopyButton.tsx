'use client';

import { useEffect, useState } from 'react';

import { Button } from '.';

type Outcome = 'idle' | 'copied' | 'failed';

/**
 * Copies a text, and says so only once the clipboard has it: a "Copied" shown
 * before the write resolved would claim a copy that a blocked clipboard never
 * made.
 */
export function CopyButton({
  text,
  label = 'Copy',
  title,
  className = '',
}: {
  /** The text, or how to make it: built when clicked, not on every render. */
  readonly text: string | (() => string);
  readonly label?: string;
  readonly title?: string;
  readonly className?: string;
}): React.JSX.Element {
  const [outcome, setOutcome] = useState<Outcome>('idle');

  useEffect(() => {
    if (outcome === 'idle') return;
    const timer = window.setTimeout(() => setOutcome('idle'), 2000);
    return () => window.clearTimeout(timer);
  }, [outcome]);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(typeof text === 'string' ? text : text());
      setOutcome('copied');
    } catch {
      // No clipboard (an insecure origin, a denied permission): say it failed.
      setOutcome('failed');
    }
  }

  return (
    <Button variant="ghost" size="sm" onClick={() => void copy()} title={title} className={className}>
      <svg viewBox="0 0 16 16" aria-hidden="true" className="h-3 w-3">
        <rect x="5" y="5" width="8.5" height="8.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
        <path d="M3 10.5V3.5A1 1 0 0 1 4 2.5h6.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      </svg>
      <span aria-live="polite">{outcome === 'copied' ? 'Copied' : outcome === 'failed' ? 'Could not copy' : label}</span>
    </Button>
  );
}
