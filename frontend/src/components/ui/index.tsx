/**
 * The workspace's few shared parts: buttons, badges, headings, disclosures,
 * empty states and keys. One look for each, so a reader learns it once and
 * every pane says "this is a button" the same way.
 */

import type { ReactNode } from 'react';

// --- Button --------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'secondary' | 'ghost';
type ButtonSize = 'sm' | 'md';

const BUTTON_VARIANT: Readonly<Record<ButtonVariant, string>> = {
  primary: 'bg-accent text-accent-ink hover:opacity-90 disabled:opacity-40',
  secondary: 'border border-line-strong bg-surface text-ink hover:border-ink-faint disabled:opacity-40',
  ghost: 'text-ink-muted hover:bg-raised hover:text-ink disabled:opacity-40',
};

const BUTTON_SIZE: Readonly<Record<ButtonSize, string>> = {
  sm: 'gap-1.5 px-2 py-1 text-2xs',
  md: 'gap-2 px-3 py-1.5 text-xs',
};

export function buttonClass(variant: ButtonVariant = 'secondary', size: ButtonSize = 'md'): string {
  return `inline-flex shrink-0 items-center justify-center rounded-md font-medium transition-colors disabled:cursor-not-allowed ${BUTTON_VARIANT[variant]} ${BUTTON_SIZE[size]}`;
}

export function Button({
  variant = 'secondary',
  size = 'md',
  className = '',
  type = 'button',
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  readonly variant?: ButtonVariant;
  readonly size?: ButtonSize;
}): React.JSX.Element {
  return <button type={type} className={`${buttonClass(variant, size)} ${className}`} {...props} />;
}

// --- Badge ---------------------------------------------------------------------------

export type Tone = 'neutral' | 'running' | 'done' | 'failed' | 'retry' | 'edited' | 'law' | 'accent';

const BADGE_TONE: Readonly<Record<Tone, string>> = {
  neutral: 'bg-raised text-ink-muted',
  running: 'bg-running/10 text-running',
  done: 'bg-done/10 text-done',
  failed: 'bg-failed/10 text-failed',
  retry: 'bg-retry/10 text-retry',
  edited: 'bg-edited/10 text-edited',
  law: 'bg-law/10 text-law',
  accent: 'bg-accent/10 text-accent',
};

export function Badge({
  tone = 'neutral',
  children,
  title,
  className = '',
}: {
  readonly tone?: Tone;
  readonly children: ReactNode;
  readonly title?: string;
  readonly className?: string;
}): React.JSX.Element {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-2xs font-medium ${BADGE_TONE[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

// --- SectionHeading ------------------------------------------------------------------

/** A section's name in sentence case, with an optional count and actions beside it. */
export function SectionHeading({
  children,
  count,
  actions,
}: {
  readonly children: ReactNode;
  readonly count?: number;
  readonly actions?: ReactNode;
}): React.JSX.Element {
  return (
    <div className="mb-2 flex items-center gap-2">
      <h3 className="text-xs font-semibold text-ink">
        {children}
        {count !== undefined && <span className="ml-1.5 font-normal text-ink-faint">{count}</span>}
      </h3>
      {actions !== undefined && <div className="ml-auto flex items-center gap-1">{actions}</div>}
    </div>
  );
}

// --- Disclosure ----------------------------------------------------------------------

/** A section that can be folded away: what is supporting detail, not the answer. */
export function Disclosure({
  title,
  count,
  defaultOpen = false,
  children,
}: {
  readonly title: ReactNode;
  readonly count?: number;
  readonly defaultOpen?: boolean;
  readonly children: ReactNode;
}): React.JSX.Element {
  return (
    <details open={defaultOpen} className="group rounded-lg border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-semibold text-ink [&::-webkit-details-marker]:hidden">
        <svg
          viewBox="0 0 16 16"
          aria-hidden="true"
          className="h-3 w-3 shrink-0 text-ink-faint transition-transform group-open:rotate-90"
        >
          <path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <span>{title}</span>
        {count !== undefined && <span className="font-normal text-ink-faint">{count}</span>}
      </summary>
      <div className="border-t border-line px-3 py-3">{children}</div>
    </details>
  );
}

// --- EmptyState ----------------------------------------------------------------------

export function EmptyState({
  title,
  children,
}: {
  readonly title: string;
  readonly children?: ReactNode;
}): React.JSX.Element {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-1.5 px-6 py-8 text-center">
      <p className="text-xs font-medium text-ink">{title}</p>
      {children !== undefined && <div className="max-w-sm text-2xs text-ink-muted">{children}</div>}
    </div>
  );
}

// --- Kbd -----------------------------------------------------------------------------

export function Kbd({ children }: { readonly children: ReactNode }): React.JSX.Element {
  return (
    <kbd className="rounded border border-line-strong bg-raised px-1 font-mono text-2xs text-ink">
      {children}
    </kbd>
  );
}
