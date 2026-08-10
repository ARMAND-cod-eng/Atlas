'use client';

import { formatSeconds, formatTokens } from '@/lib/format';
import type { RunResult } from '@/lib/events';

interface Props {
  readonly result: RunResult | null;
}

/**
 * Per-node timings and per-call attribution.
 *
 * The summed-vs-wall-clock gap is the parallelism, stated as a number: five
 * branches costing the slowest one rather than the sum is the whole point of the
 * fan-out, and it is invisible unless both figures are on screen. Provider and
 * model are shown per call because a run silently served entirely by the
 * fallback was a real Phase 0 defect that looked like a working system.
 */
export function Metrics({ result }: Props): React.JSX.Element | null {
  if (result === null) return null;

  const timings = result.metrics.node_timings;
  const researcherTimings = timings.filter((timing) => timing.node === 'researcher');
  const summed = researcherTimings.reduce((total, timing) => total + timing.duration_s, 0);
  const slowest = researcherTimings.reduce((max, timing) => Math.max(max, timing.duration_s), 0);

  const byNode = new Map<string, number>();
  for (const timing of timings) {
    byNode.set(timing.node, (byNode.get(timing.node) ?? 0) + timing.duration_s);
  }

  return (
    <section className="rounded-md border border-neutral-800 bg-neutral-950 p-3">
      <h2 className="text-[11px] uppercase tracking-wide text-neutral-500">metrics</h2>

      <dl className="mt-2 grid grid-cols-2 gap-y-1 text-[11px]">
        {[...byNode.entries()].map(([node, seconds]) => (
          <div key={node} className="contents">
            <dt className="text-neutral-500">{node}</dt>
            <dd className="text-right tabular-nums text-neutral-300">{formatSeconds(seconds)}</dd>
          </div>
        ))}
      </dl>

      {researcherTimings.length > 1 && (
        <p className="mt-2 border-t border-neutral-800 pt-2 text-[11px] text-neutral-400">
          {researcherTimings.length} branches cost{' '}
          <span className="tabular-nums text-neutral-200">{formatSeconds(slowest)}</span> — the
          slowest — not{' '}
          <span className="tabular-nums text-neutral-200">{formatSeconds(summed)}</span>, their sum.
        </p>
      )}

      <div className="mt-3 border-t border-neutral-800 pt-2">
        <p className="text-[10px] uppercase tracking-wide text-neutral-500">model calls</p>
        <ul className="mt-1 space-y-0.5">
          {result.metrics.llm_calls.map((call, index) => (
            <li key={index} className="flex items-baseline gap-2 text-[10px] tabular-nums">
              <span className="w-20 shrink-0 truncate text-neutral-400">{call.node}</span>
              <span className="w-28 shrink-0 truncate text-neutral-500" title={call.model}>
                {call.model}
              </span>
              <span className="text-neutral-500">
                {formatTokens(call.input_tokens)}→{formatTokens(call.output_tokens)}
              </span>
              <span className="ml-auto text-neutral-400">{formatSeconds(call.duration_s)}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
