'use client';

import { useRouter } from 'next/navigation';

import { QuestionForm } from '@/components/QuestionForm';

export default function Home(): React.JSX.Element {
  const router = useRouter();

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col justify-center px-6">
      <h1 className="text-sm font-medium tracking-wide text-neutral-300">Atlas</h1>
      <p className="mt-1 text-[13px] text-neutral-500">
        Multi-agent research. A planner decomposes the question, researchers run in parallel, and a
        critic reviews the result before you see it.
      </p>

      <div className="mt-6">
        <QuestionForm
          placeholder="Compare retrieval-augmented generation with long-context language models for enterprise document QA."
          hint="a run takes about a minute"
          submitLabel="Research"
          onStarted={(runId) => router.push(`/runs/${runId}`)}
        />
      </div>
    </main>
  );
}
