'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { API_BASE } from '@/lib/useRunStream';

export default function Home(): React.JSX.Element {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (query.trim() === '' || submitting) return;

    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) throw new Error(`the API returned ${response.status}`);
      const body = (await response.json()) as { run_id: string };
      // Routing rather than holding the run in memory: the run belongs to the
      // worker, so its URL should outlive this tab.
      router.push(`/runs/${body.run_id}`);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? `${cause.message}. Is the API running on ${API_BASE}?`
          : 'could not start the run',
      );
      setSubmitting(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-6">
      <h1 className="text-sm font-medium tracking-wide text-neutral-300">Atlas</h1>
      <p className="mt-1 text-[13px] text-neutral-500">
        Multi-agent research. A planner decomposes the question, researchers run in parallel, and a
        critic reviews the result before you see it.
      </p>

      <form onSubmit={submit} className="mt-6">
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit(event);
          }}
          rows={3}
          placeholder="Compare retrieval-augmented generation with long-context language models for enterprise document QA."
          className="w-full resize-none rounded-md border border-neutral-800 bg-neutral-950 px-3 py-2 text-[13px] leading-relaxed text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none"
        />
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-neutral-600">a run takes about a minute</span>
          <button
            type="submit"
            disabled={submitting || query.trim() === ''}
            className="rounded border border-neutral-700 px-3 py-1.5 text-[12px] text-neutral-200 hover:border-neutral-500 disabled:opacity-40"
          >
            {submitting ? 'starting…' : 'Research'}
          </button>
        </div>
      </form>

      {error !== null && <p className="mt-3 text-[12px] text-red-400">{error}</p>}
    </main>
  );
}
