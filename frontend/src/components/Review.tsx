'use client';

import { useEffect, useRef, useState } from 'react';

import { CitationChip } from './CitationChip';
import { Disclaimer, LAW_AS_OF } from './Disclaimer';
import { Badge, Button, type Tone } from './ui';
import { CopyButton } from './ui/CopyButton';
import { type AgentCall, fetchCall } from '@/lib/api';
import type { FindingStatus, ReviewClause, ReviewFinding, ReviewRequirement } from '@/lib/events';
import type { RunState } from '@/lib/reducer';
import {
  type Decision,
  STATUSES,
  STATUS_LABEL,
  fetchDecisions,
  putDecision,
  quoteSpan,
  reviewMarkdown,
  reviewOf,
} from '@/lib/review';

const TONE: Readonly<Record<FindingStatus, Tone>> = {
  met: 'done',
  partial: 'retry',
  missing: 'failed',
  // Claimed and not proven: the one a reviewer must read for themselves.
  unverified: 'edited',
};

const REVIEWER = 'atlas:reviewer';

/** The reviewer's name, remembered in this browser; never fatal where storage is blocked. */
function useReviewer(): readonly [string, (name: string) => void] {
  const [name, setName] = useState('');
  useEffect(() => {
    try {
      setName(window.localStorage.getItem(REVIEWER) ?? '');
    } catch {
      // Blocked storage: typed each time.
    }
  }, []);
  return [
    name,
    (next) => {
      setName(next);
      try {
        window.localStorage.setItem(REVIEWER, next);
      } catch {
        // Not remembered, still used.
      }
    },
  ];
}

/**
 * A contract review: what each term the law requires came to, the contract's
 * own words for it, and the reviewer's decision on each. Findings fill in as
 * the review runs; decisions are kept by the API, after any they replace.
 */
export function ReviewWorkspace({
  runId,
  state,
}: {
  /** Null for a replay: no API behind it to keep decisions or serve prompts. */
  readonly runId: string | null;
  readonly state: RunState;
}): React.JSX.Element {
  const view = reviewOf(state.timeline);
  const finished = state.phase === 'completed';
  const [shown, setShown] = useState<string | null>(null);
  const [reviewer, setReviewer] = useReviewer();
  const [decisions, setDecisions] = useState<Readonly<Record<string, Decision>>>({});
  const contract = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!finished || runId === null) return;
    let live = true;
    void fetchDecisions(runId).then((current) => {
      if (live) setDecisions(current);
    });
    return () => {
      live = false;
    };
  }, [runId, finished]);

  const finding = shown === null ? undefined : view.findings[shown];
  const marked = finding?.found_in ?? null;

  function show(requirementId: string, clauseId: string): void {
    setShown(requirementId);
    window.setTimeout(() => {
      contract.current?.querySelector(`#clause-${clauseId}`)?.scrollIntoView?.({ block: 'center', behavior: 'smooth' });
    }, 0);
  }

  async function decide(requirementId: string, decision: 'agree' | 'disagree', note: string): Promise<boolean> {
    if (runId === null) return false;
    const kept = await putDecision(runId, requirementId, { decision, note, reviewer: reviewer.trim() });
    if (kept !== null) setDecisions((current) => ({ ...current, [requirementId]: kept }));
    return kept !== null;
  }

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-4 sm:p-5">
      <section className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-line bg-surface p-4">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">{view.title ?? 'Reading the contract…'}</h2>
          <p className="mt-0.5 flex flex-wrap gap-1.5">
            {STATUSES.map((status) => (
              <Badge key={status} tone={TONE[status]}>
                {view.counts[status]} {status}
              </Badge>
            ))}
          </p>
        </div>
        {runId === null ? (
          <span className="ml-auto text-2xs text-ink-faint">A recording: sign-off needs a live review.</span>
        ) : (
          <label className="ml-auto flex items-center gap-2 text-2xs text-ink-muted">
            Your name
            <input
              value={reviewer}
              onChange={(event) => setReviewer(event.target.value)}
              placeholder="to sign off each finding"
              className="w-48 rounded-md border border-line-strong bg-surface px-2 py-1 text-xs text-ink"
            />
          </label>
        )}
        <CopyButton label="Copy report" text={() => reviewMarkdown(view, decisions, LAW_AS_OF)} />
      </section>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <ol className="space-y-3" aria-label="Findings">
          {view.requirements.map((requirement) => (
            <li key={requirement.id}>
              <FindingCard
                runId={runId}
                requirement={requirement}
                finding={view.findings[requirement.id]}
                clauses={view.clauses}
                callId={
                  state.calls.filter((call) => call.node === 'reviewer' && call.subtask_id === requirement.id).at(-1)
                    ?.call_id ?? null
                }
                selected={shown === requirement.id}
                onShow={show}
                decision={decisions[requirement.id]}
                canDecide={finished && reviewer.trim() !== ''}
                onDecide={decide}
              />
            </li>
          ))}
        </ol>

        <section
          ref={contract}
          aria-label="The contract"
          className="h-fit space-y-3 rounded-xl border border-line bg-surface p-4 lg:sticky lg:top-4 lg:max-h-[calc(100dvh-8rem)] lg:overflow-y-auto"
        >
          <h2 className="text-sm font-semibold text-ink">{view.filename ?? 'The contract'}</h2>
          {view.clauses.map((clause) => (
            <ClauseText
              key={clause.id}
              clause={clause}
              marked={clause.id === marked}
              quote={clause.id === marked ? (finding?.quote ?? null) : null}
            />
          ))}
        </section>
      </div>

      <Disclaimer className="border-t border-line pt-3" />
    </div>
  );
}

/**
 * A clause's text for reading: the file's hard line breaks inside a sentence
 * joined, a list item's and a paragraph's kept. Quotes are matched whatever the
 * whitespace, so the mark lands the same either way.
 */
function reflow(text: string): string {
  return text.replace(/([^\n])\n(?!\n|\s*(?:\(?[a-z0-9]{1,3}\)|[-–•*])\s)/gi, '$1 ');
}

function ClauseText({
  clause,
  marked,
  quote,
}: {
  readonly clause: ReviewClause;
  readonly marked: boolean;
  readonly quote: string | null;
}): React.JSX.Element {
  const text = reflow(clause.text);
  const span = quote === null ? null : quoteSpan(text, quote);
  return (
    <div
      id={`clause-${clause.id}`}
      className={`rounded-md px-2 py-1.5 transition-colors ${marked ? 'bg-accent/10 ring-1 ring-accent/40' : ''}`}
    >
      <h3 className="text-2xs font-semibold text-ink-muted">
        {clause.label}
        {clause.heading !== null && <span className="ml-1.5 font-normal text-ink-faint">{clause.heading}</span>}
      </h3>
      <p className="mt-0.5 whitespace-pre-line text-xs leading-relaxed text-ink">
        {span === null ? (
          text
        ) : (
          <>
            {text.slice(0, span[0])}
            <mark className="rounded-sm bg-done/20 px-0.5 text-ink">{text.slice(span[0], span[1])}</mark>
            {text.slice(span[1])}
          </>
        )}
      </p>
    </div>
  );
}

function FindingCard({
  runId,
  requirement,
  finding,
  clauses,
  callId,
  selected,
  onShow,
  decision,
  canDecide,
  onDecide,
}: {
  readonly runId: string | null;
  readonly requirement: ReviewRequirement;
  readonly finding: ReviewFinding | undefined;
  readonly clauses: readonly ReviewClause[];
  readonly callId: string | null;
  readonly selected: boolean;
  readonly onShow: (requirementId: string, clauseId: string) => void;
  readonly decision: Decision | undefined;
  readonly canDecide: boolean;
  readonly onDecide: (requirementId: string, decision: 'agree' | 'disagree', note: string) => Promise<boolean>;
}): React.JSX.Element {
  const [note, setNote] = useState(decision?.note ?? '');
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);
  const foundIn = clauses.find((clause) => clause.id === finding?.found_in);

  async function decide(choice: 'agree' | 'disagree'): Promise<void> {
    setSaving(true);
    setFailed(false);
    const kept = await onDecide(requirement.id, choice, note.trim());
    setSaving(false);
    setFailed(!kept);
  }

  return (
    <article
      aria-label={requirement.label}
      className={`rounded-xl border bg-surface p-4 transition-colors ${selected ? 'border-accent' : 'border-line'}`}
    >
      <header className="flex flex-wrap items-center gap-2">
        <CitationChip label={requirement.label} status="unchecked" />
        <span className="min-w-0 flex-1 text-xs font-medium text-ink">{requirement.title}</span>
        {finding === undefined ? (
          <Badge>Judging…</Badge>
        ) : (
          <Badge tone={TONE[finding.status]}>{STATUS_LABEL[finding.status]}</Badge>
        )}
      </header>

      {finding !== undefined && (
        <div className="mt-3 space-y-2">
          {finding.quote !== null && (
            <figure className="space-y-1">
              <blockquote className="border-l-2 border-line-strong pl-3 text-xs italic leading-relaxed text-ink">
                “{finding.quote}”
              </blockquote>
              <figcaption className="flex flex-wrap items-center gap-2 text-2xs">
                {foundIn !== undefined ? (
                  <>
                    <span className="text-done">✓ Found in {foundIn.label}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={`Show ${requirement.label} in the contract`}
                      onClick={() => onShow(requirement.id, foundIn.id)}
                    >
                      Show in the contract
                    </Button>
                  </>
                ) : (
                  <span className="text-failed">Not found in the contract</span>
                )}
              </figcaption>
            </figure>
          )}
          <p className="text-xs leading-relaxed text-ink-muted">{finding.reason}</p>

          {runId !== null && (
            <div className="flex flex-wrap items-center gap-2 border-t border-line pt-2.5">
              {(['agree', 'disagree'] as const).map((choice) => (
                <Button
                  key={choice}
                  size="sm"
                  variant={decision?.decision === choice ? 'primary' : 'secondary'}
                  aria-label={`${choice === 'agree' ? 'Agree' : 'Disagree'} with ${requirement.label}`}
                  aria-pressed={decision?.decision === choice}
                  disabled={!canDecide || saving}
                  title={canDecide ? undefined : 'Add your name above to sign off'}
                  onClick={() => void decide(choice)}
                >
                  {choice === 'agree' ? 'Agree' : 'Disagree'}
                </Button>
              ))}
              <input
                aria-label={`Note on ${requirement.label}`}
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="Note (optional)"
                maxLength={2000}
                className="min-w-0 flex-1 rounded-md border border-line bg-canvas px-2 py-1 text-2xs text-ink"
              />
              {decision !== undefined && (
                <span className="text-2xs text-ink-faint">
                  {decision.decision === 'agree' ? 'Agreed' : 'Disagreed'} by {decision.reviewer}
                </span>
              )}
              {failed && (
                <span role="alert" className="text-2xs text-failed">
                  Not saved. Try again.
                </span>
              )}
            </div>
          )}
          {runId !== null && callId !== null && <PromptPeek runId={runId} callId={callId} />}
        </div>
      )}
    </article>
  );
}

/** The exact prompt and answer behind a finding, fetched when opened. */
function PromptPeek({ runId, callId }: { readonly runId: string; readonly callId: string }): React.JSX.Element {
  const [call, setCall] = useState<AgentCall | null | undefined>(undefined);
  return (
    <details
      className="text-2xs"
      onToggle={(event) => {
        if ((event.currentTarget as HTMLDetailsElement).open && call === undefined) {
          void fetchCall(runId, callId).then(setCall);
        }
      }}
    >
      <summary className="cursor-pointer text-ink-faint hover:text-ink">What the model was asked</summary>
      {call === undefined ? (
        <p className="mt-1 text-ink-faint">Loading…</p>
      ) : call === null ? (
        <p className="mt-1 text-failed">The API could not return this call.</p>
      ) : (
        <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-canvas p-2 font-mono text-2xs text-ink">
          {'query' in call.request
            ? call.request.query
            : call.request.map((message) => `${message.role}:\n${message.content}`).join('\n\n')}
        </pre>
      )}
    </details>
  );
}
