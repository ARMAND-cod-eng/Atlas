/**
 * A contract review, read from its run's events: what it reviews against, the
 * contract as Atlas read it, and a finding per term as each is judged.
 *
 * Read from the stream rather than fetched at the end, so a review fills in as
 * it runs, and replays the same from the archive long after Redis forgets it.
 * The reviewer's decisions are the API's, fetched and written one at a time.
 */

import { API_BASE } from './api';
import {
  REQUIREMENT_JUDGED,
  REVIEW_STARTED,
  type AgentEvent,
  type FindingStatus,
  type ReviewClause,
  type ReviewFinding,
  type ReviewRequirement,
} from './events';

export const STATUSES: readonly FindingStatus[] = ['met', 'partial', 'missing', 'unverified'];

export interface ReviewView {
  readonly filename: string | null;
  readonly title: string | null;
  readonly requirements: readonly ReviewRequirement[];
  readonly clauses: readonly ReviewClause[];
  /** By requirement id; absent until that term is judged. */
  readonly findings: Readonly<Record<string, ReviewFinding>>;
  readonly counts: Readonly<Record<FindingStatus, number>>;
}

export function reviewOf(timeline: readonly AgentEvent[]): ReviewView {
  let started: Extract<AgentEvent, { type: typeof REVIEW_STARTED }>['payload'] | null = null;
  const findings: Record<string, ReviewFinding> = {};
  for (const event of timeline) {
    if (event.type === REVIEW_STARTED) started = event.payload;
    else if (event.type === REQUIREMENT_JUDGED) findings[event.payload.requirement_id] = event.payload;
  }
  const counts = { met: 0, partial: 0, missing: 0, unverified: 0 };
  for (const finding of Object.values(findings)) counts[finding.status] += 1;
  return {
    filename: started?.filename ?? null,
    title: started?.title ?? null,
    requirements: started?.requirements ?? [],
    clauses: started?.clauses ?? [],
    findings,
    counts,
  };
}

export const STATUS_LABEL: Readonly<Record<FindingStatus, string>> = {
  met: 'Met',
  partial: 'Partial',
  missing: 'Missing',
  unverified: 'Unverified',
};

/** A reviewer's decision on one finding, as the API keeps it. */
export interface Decision {
  readonly run_id: string;
  readonly requirement_id: string;
  readonly decision: 'agree' | 'disagree';
  readonly note: string;
  readonly reviewer: string;
  readonly decided_at: string;
}

/** The review as Markdown: each term, where the contract meets it in its own words, and who agreed. */
export function reviewMarkdown(
  view: ReviewView,
  decisions: Readonly<Record<string, Decision>>,
  lawAsOf: string,
): string {
  const labelOf = new Map(view.clauses.map((clause) => [clause.id, clause.label]));
  const lines = [
    `# Review of ${view.filename ?? 'a contract'}`,
    '',
    [view.title ?? 'Review', ...STATUSES.map((status) => `${view.counts[status]} ${status}`)].join(' · '),
    '',
  ];
  for (const requirement of view.requirements) {
    const finding = view.findings[requirement.id];
    lines.push(`## ${requirement.label}: ${requirement.title}`, '');
    if (finding === undefined) {
      lines.push('Not judged yet.', '');
      continue;
    }
    const status = `**${STATUS_LABEL[finding.status]}**`;
    if (finding.found_in !== null && finding.quote !== null) {
      lines.push(`${status}, ${labelOf.get(finding.found_in) ?? finding.found_in}: “${finding.quote}”`);
    } else if (finding.status === 'unverified' && finding.quote !== null) {
      lines.push(`${status}: its quote is not in the contract`);
    } else {
      lines.push(status);
    }
    lines.push('', finding.reason, '');
    const decision = decisions[requirement.id];
    if (decision !== undefined) {
      const verb = decision.decision === 'agree' ? 'agrees' : 'disagrees';
      lines.push(`Reviewer: ${verb} (${decision.reviewer}).${decision.note ? ` ${decision.note}` : ''}`, '');
    }
  }
  lines.push(
    `_A research aid, not legal advice: the law as published in the Official Journal, retrieved ${lawAsOf}._`,
    '',
  );
  return lines.join('\n');
}

export type ReviewStart =
  | { readonly kind: 'started'; readonly runId: string }
  | { readonly kind: 'error'; readonly message: string };

/** `POST /reviews`: the contract's text, or its file as base64. */
export async function startReview(
  body: { readonly filename: string } & ({ readonly text: string } | { readonly content_base64: string }),
): Promise<ReviewStart> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/reviews`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    return { kind: 'error', message: `could not reach the API at ${API_BASE}. Is it running?` };
  }
  if (!response.ok) {
    const detail = await response
      .json()
      .then((json: { detail?: unknown }) => (typeof json.detail === 'string' ? json.detail : null))
      .catch(() => null);
    return { kind: 'error', message: detail ?? `the API returned ${response.status}` };
  }
  const accepted = (await response.json()) as { run_id: string };
  return { kind: 'started', runId: accepted.run_id };
}

/** The current decision per finding; empty where there are none, or no API. */
export async function fetchDecisions(runId: string): Promise<Record<string, Decision>> {
  try {
    const response = await fetch(`${API_BASE}/reviews/${runId}/decisions`);
    if (!response.ok) return {};
    const body = (await response.json()) as { current: readonly Decision[] };
    return Object.fromEntries(body.current.map((decision) => [decision.requirement_id, decision]));
  } catch {
    return {};
  }
}

/** `PUT` one decision; the decision as kept, or null with nothing kept. */
export async function putDecision(
  runId: string,
  requirementId: string,
  body: { readonly decision: 'agree' | 'disagree'; readonly note: string; readonly reviewer: string },
): Promise<Decision | null> {
  try {
    const response = await fetch(`${API_BASE}/reviews/${runId}/decisions/${requirementId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return response.ok ? ((await response.json()) as Decision) : null;
  } catch {
    return null;
  }
}

/** The quote's place in a clause's text, whitespace and case aside: where to mark it. */
export function quoteSpan(text: string, quote: string): readonly [number, number] | null {
  const words = quote.trim().split(/\s+/).filter((word) => word !== '');
  if (words.length === 0) return null;
  const pattern = new RegExp(words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+'), 'i');
  const match = pattern.exec(text);
  return match === null ? null : [match.index, match.index + match[0].length];
}
