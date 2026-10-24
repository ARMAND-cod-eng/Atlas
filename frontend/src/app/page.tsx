'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';

import { Disclaimer } from '@/components/Disclaimer';
import { QuestionForm } from '@/components/QuestionForm';

const EXAMPLES = [
  'Is a CV-screening tool high-risk under the AI Act, and what must its provider do?',
  'Do we need a DPIA before using emotion recognition on call-centre recordings?',
  'Which AI Act obligations apply to a provider of a general-purpose AI model, and from when?',
];

export default function Home(): React.JSX.Element {
  const router = useRouter();

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col justify-center px-6 py-12">
      <h1 className="text-sm font-medium tracking-wide text-ink">Atlas</h1>
      <p className="mt-1 text-base text-ink">
        EU AI Act and GDPR research, with every citation checked against the law.
      </p>
      <p className="mt-3 text-sm leading-relaxed text-ink-muted">
        A planner splits your question into branches. Researchers read the regulations and official
        guidance in parallel. A critic checks the answer&apos;s citations: that each article exists,
        that it was actually read, and that any quotation is its own words. Every agent&apos;s
        prompt and output can be opened, and any branch redirected and re-run.
      </p>

      <div className="mt-6">
        <QuestionForm
          placeholder="Ask about the AI Act or the GDPR, from a provider's or a deployer's side."
          examples={EXAMPLES}
          hint="a run takes about a minute"
          submitLabel="Research"
          onStarted={(runId) => router.push(`/runs/${runId}`)}
        />
      </div>

      <Link href="/demo" className="mt-6 self-start text-xs text-accent hover:underline">
        Watch a recorded run, no API needed →
      </Link>

      <Disclaimer className="mt-10 border-t border-line pt-4" />
    </main>
  );
}
