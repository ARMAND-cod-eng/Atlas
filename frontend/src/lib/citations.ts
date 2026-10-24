/**
 * Citations of the law in an answer: found, put in the library's spelling, and
 * marked so the answer can show each one as a chip.
 *
 * The parser is a port of the backend's (`atlas/library/citations.py`), held
 * to it by `tests/fixtures/citation-cases.json`, which the backend writes from
 * its own parser. The two must agree: a chip looks up its provision, and its
 * verdict in the run's citation check, by the label the backend would give it.
 */

import type { CitationCheck } from './reducer';

const BRACKET = /\[([^[\]]{3,200})\]/g;
const LABEL =
  /^(AI Act|GDPR)\s+(Art(?:icle)?\.?|Annex|Recital)\s*(\d{1,3}|[IVXLC]+)\s*((?:\(\s*[0-9a-z]{1,4}\s*\))*)$/i;
const ACT_START = /^\s*(AI Act|GDPR)\b/i;

/** A citation in the library's own spelling, or null if the text is not one. */
export function labelOf(text: string): string | null {
  const match = LABEL.exec(text.trim().split(/\s+/).join(' '));
  if (match === null) return null;
  const [, actName = '', kindName = '', number = '', rest = ''] = match;
  const act = actName.toLowerCase() === 'ai act' ? 'AI Act' : 'GDPR';
  const kind = kindName.toLowerCase();
  const subdivisions = rest.toLowerCase().replace(/\s+/g, '');
  if (kind.startsWith('art')) return `${act} Art. ${Number.parseInt(number, 10)}${subdivisions}`;
  if (kind === 'annex') return `${act} Annex ${number.toUpperCase()}${subdivisions}`;
  return `${act} Recital ${Number.parseInt(number, 10)}`;
}

/** The citations one bracket holds: "GDPR Art. 33(1); GDPR Art. 34(1)" is two. */
function labelsIn(bracket: string): string[] {
  const parts: string[] = [];
  for (const chunk of bracket.split(';')) {
    // A comma separates two citations only when the act is named again.
    const [first = '', ...pieces] = chunk.split(',');
    let current = first;
    for (const piece of pieces) {
      if (ACT_START.test(piece)) {
        parts.push(current);
        current = piece;
      } else {
        current = `${current},${piece}`;
      }
    }
    parts.push(current);
  }
  return parts.map(labelOf).filter((label): label is string => label !== null);
}

/** Every provision the text cites in brackets, in order, repeats included. */
export function citedLabels(text: string): string[] {
  return [...text.matchAll(BRACKET)].flatMap((match) => labelsIn(match[1] ?? ''));
}

export const CITE_PREFIX = '#cite-';

/**
 * The answer's Markdown with each citation made a link to `#cite-N`, and the
 * labels those links stand for. A bracket that is already a link (or an
 * image), or holds no citation, is left as written. Two brackets side by side
 * are two citations, as the backend counts them, not a reference link.
 */
export function markCitations(markdown: string): { readonly markdown: string; readonly labels: readonly string[] } {
  const labels: string[] = [];
  const marked = markdown.replace(BRACKET, (whole, inner: string, offset: number) => {
    if (markdown[offset + whole.length] === '(') return whole;
    const found = labelsIn(inner);
    if (found.length === 0) return whole;
    return found
      .map((label) => {
        labels.push(label);
        return `[${label}](${CITE_PREFIX}${labels.length - 1})`;
      })
      .join(' ');
  });
  return { markdown: marked, labels };
}

export type CitationStatus = 'verified' | 'unknown' | 'unretrieved' | 'misquoted' | 'unchecked';

/**
 * What the run's last citation check said of one citation. A run in a domain
 * without the check (or that never wrote an answer) has nothing to say.
 */
export function statusOf(label: string, check: CitationCheck | null): CitationStatus {
  if (check === null) return 'unchecked';
  if (check.unresolved.includes(label)) return 'unknown';
  if (check.unretrieved.includes(label)) return 'unretrieved';
  if (check.misquoted.some(([cited]) => cited === label)) return 'misquoted';
  return check.cited.includes(label) ? 'verified' : 'unchecked';
}
