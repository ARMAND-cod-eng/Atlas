/**
 * The answer as Markdown, to paste into a document or a ticket.
 *
 * The answer is kept as written. Its citations already stand in brackets, so
 * each provision gets a reference definition and the brackets become links
 * wherever the Markdown is rendered, without rewriting a word. Taken out of the
 * workspace an answer about the law loses the caveat around it, so the caveat
 * goes with it.
 */

import { citedLabels } from './citations';
import type { SourceRef } from './events';

export function answerMarkdown({
  question,
  answer,
  sources,
  lawAsOf,
}: {
  readonly question: string | null;
  readonly answer: string;
  /** Every source the run had: a citation links only to one of these. */
  readonly sources: readonly SourceRef[];
  readonly lawAsOf: string;
}): string {
  const byLabel = new Map<string, SourceRef>();
  for (const source of sources) {
    if (source.label !== null && source.label !== undefined && !byLabel.has(source.label)) {
      byLabel.set(source.label, source);
    }
  }
  const cited = [...new Set(citedLabels(answer))];

  const lines: string[] = [];
  if (question !== null) lines.push(`## ${question}`, '');
  lines.push(answer.trim(), '');

  if (cited.length > 0) {
    lines.push('Provisions cited:', '');
    for (const label of cited) {
      const source = byLabel.get(label);
      lines.push(
        source === undefined
          ? `- ${label} — not among the sources this run retrieved`
          : source.title === label
            ? `- [${label}]`
            : `- [${label}] — ${source.title}`,
      );
    }
    lines.push('');
    const linked = cited.flatMap((label) => {
      const source = byLabel.get(label);
      // Angle brackets: a destination with parentheses or spaces stays one link.
      return source === undefined ? [] : [`[${label}]: <${source.url}>`];
    });
    if (linked.length > 0) lines.push(...linked, '');
  }

  lines.push(
    `_A research aid, not legal advice: the law as published in the Official Journal, retrieved ${lawAsOf}._`,
    '',
  );
  return lines.join('\n');
}
