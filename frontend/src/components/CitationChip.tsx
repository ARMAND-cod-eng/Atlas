'use client';

import { useId, useRef, useState } from 'react';

import { type Provision, fetchProvision } from '@/lib/api';
import type { CitationStatus } from '@/lib/citations';

const STATUS: Record<CitationStatus, { readonly chip: string; readonly mark: string; readonly says: string }> = {
  verified: {
    chip: 'border-law/40 bg-law/10 text-law',
    mark: '✓',
    says: 'Checked: the provision exists and was retrieved in this run.',
  },
  unknown: {
    chip: 'border-failed/50 bg-failed/10 text-failed line-through',
    mark: '✗',
    says: 'No such provision: the act has nothing by this number.',
  },
  unretrieved: {
    chip: 'border-retry/50 bg-retry/10 text-retry',
    mark: '?',
    says: 'The provision exists, but no researcher retrieved it in this run.',
  },
  misquoted: {
    chip: 'border-retry/50 bg-retry/10 text-retry',
    mark: '“?',
    says: 'A quotation beside this citation is not in the provision’s text.',
  },
  unchecked: {
    chip: 'border-line-strong bg-raised text-ink-muted',
    mark: '',
    says: 'Not checked in this run.',
  },
};

type Loaded = { readonly state: 'idle' | 'loading' } | { readonly state: 'done'; readonly provision: Provision | null };

/**
 * One citation of the law, as a chip: what the run's citation check found, and
 * on hover, focus or tap the provision's text from the library, with a link to
 * the Official Journal.
 */
export function CitationChip({
  label,
  status,
}: {
  readonly label: string;
  readonly status: CitationStatus;
}): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const [alignRight, setAlignRight] = useState(false);
  const [loaded, setLoaded] = useState<Loaded>({ state: 'idle' });
  const button = useRef<HTMLButtonElement>(null);
  const id = useId();
  const style = STATUS[status];

  function show(): void {
    // Opened near the pane's right edge, the card would be cut off: open leftwards.
    const rect = button.current?.getBoundingClientRect();
    if (rect !== undefined) setAlignRight(rect.left > window.innerWidth - 340);
    setOpen(true);
    if (loaded.state !== 'idle') return;
    setLoaded({ state: 'loading' });
    void fetchProvision(label).then((provision) => setLoaded({ state: 'done', provision }));
  }

  return (
    <span
      className="relative inline-block"
      onMouseEnter={show}
      onMouseLeave={() => setOpen(false)}
      // Closed only when focus leaves the chip and its card: tabbing to the
      // card's link must not close the card under it.
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <button
        ref={button}
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onFocus={show}
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={(event) => event.key === 'Escape' && setOpen(false)}
        className={`mx-0.5 inline-flex items-baseline gap-1 rounded border px-1.5 py-px align-baseline text-2xs font-medium no-underline ${style.chip}`}
      >
        {label}
        {style.mark !== '' && <span aria-hidden="true">{style.mark}</span>}
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`absolute top-full z-30 mt-1 block w-80 max-w-[80vw] rounded-md border border-line-strong bg-surface p-3 text-left shadow-lg ${
            alignRight ? 'right-0' : 'left-0'
          }`}
        >
          <span className="block text-2xs text-ink-muted">{style.says}</span>
          <ProvisionText loaded={loaded} status={status} />
        </span>
      )}
    </span>
  );
}

function ProvisionText({
  loaded,
  status,
}: {
  readonly loaded: Loaded;
  readonly status: CitationStatus;
}): React.JSX.Element | null {
  if (loaded.state !== 'done') {
    return <span className="mt-2 block text-2xs text-ink-faint">Loading the provision…</span>;
  }
  const { provision } = loaded;
  if (provision === null) {
    // An unknown provision has no text to show; its status already says why.
    return status === 'unknown' ? null : (
      <span className="mt-2 block text-2xs text-ink-faint">The provision’s text could not be loaded.</span>
    );
  }
  return (
    <>
      <span className="mt-2 block text-xs font-medium text-ink">
        {provision.label}
        <span className="font-normal text-ink-muted"> · {provision.title}</span>
      </span>
      <span className="mt-1 block max-h-48 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-ink">
        {provision.text}
      </span>
      <a
        href={provision.url}
        target="_blank"
        rel="noreferrer"
        className="mt-2 inline-block text-2xs text-accent hover:underline"
      >
        Read it in the Official Journal ↗
      </a>
    </>
  );
}
