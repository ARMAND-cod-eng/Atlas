'use client';

import { type CSSProperties, useEffect, useId, useRef, useState } from 'react';

import { type Provision, fetchProvision } from '@/lib/api';
import type { CitationStatus } from '@/lib/citations';
import type { SupportEntry } from '@/lib/events';

const STATUS: Record<CitationStatus, { readonly chip: string; readonly mark: string; readonly says: string }> = {
  supported: {
    chip: 'border-law/40 bg-law/10 text-law',
    mark: '✓',
    says: 'Checked: the provision exists and was retrieved, and a model reading it found it says what it is cited for.',
  },
  unsupported: {
    chip: 'border-retry/50 bg-retry/10 text-retry',
    mark: '≠',
    says: 'The provision exists and was retrieved, but a model reading it found it does not say what it is cited for.',
  },
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

const CARD_WIDTH = 320;
const MARGIN = 8;

/**
 * Where the card goes: under the chip, or over it near the window's foot, and
 * never past either side. Fixed to the window, because the result pane scrolls
 * and would cut off a card positioned inside it.
 */
function placeCard(chip: DOMRect): CSSProperties {
  const width = Math.min(CARD_WIDTH, window.innerWidth - 2 * MARGIN);
  const left = Math.min(Math.max(MARGIN, chip.left), window.innerWidth - width - MARGIN);
  const below = window.innerHeight - chip.bottom;
  return below < 240 && chip.top > below
    ? { position: 'fixed', left, width, bottom: window.innerHeight - chip.top }
    : { position: 'fixed', left, width, top: chip.bottom };
}

/**
 * One citation of the law, as a chip: what the run's citation check found, and
 * on hover, focus or tap the provision's text from the library, with a link to
 * the Official Journal.
 */
export function CitationChip({
  label,
  status,
  judgements = [],
}: {
  readonly label: string;
  readonly status: CitationStatus;
  /** What the support check said of this citation, sentence by sentence. */
  readonly judgements?: readonly SupportEntry[];
}): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<CSSProperties>({});
  const [loaded, setLoaded] = useState<Loaded>({ state: 'idle' });
  const button = useRef<HTMLButtonElement>(null);
  const id = useId();
  const style = STATUS[status];

  // A card fixed to the window would stay put while its chip scrolled away.
  useEffect(() => {
    if (!open) return;
    const close = (): void => setOpen(false);
    window.addEventListener('scroll', close, true);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);

  function show(): void {
    const rect = button.current?.getBoundingClientRect();
    if (rect !== undefined) setPlace(placeCard(rect));
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
        // The gap between chip and card is the card's own padding, so the pointer
        // crossing it never leaves the chip's hover.
        <span id={id} role="tooltip" style={place} className="z-50 block py-1 text-left">
          <span className="block rounded-md border border-line-strong bg-surface p-3 shadow-lg">
            <span className="block text-2xs text-ink-muted">{style.says}</span>
            <Judgements judgements={judgements} />
            <ProvisionText loaded={loaded} status={status} />
          </span>
        </span>
      )}
    </span>
  );
}

function Judgements({ judgements }: { readonly judgements: readonly SupportEntry[] }): React.JSX.Element | null {
  if (judgements.length === 0) return null;
  return (
    <span className="mt-2 block space-y-1.5">
      {judgements.map((entry) => (
        <span key={entry.sentence} className="block text-2xs leading-relaxed">
          <span className="block text-ink-faint">Cited for: {entry.sentence}</span>
          {entry.reason !== '' ? (
            <span className="block text-retry">{entry.reason}</span>
          ) : (
            <span className="block text-ink">It says: “{entry.quote}”</span>
          )}
        </span>
      ))}
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
