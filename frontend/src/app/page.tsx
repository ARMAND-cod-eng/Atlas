'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { Disclaimer } from '@/components/Disclaimer';
import { QuestionForm } from '@/components/QuestionForm';
import { ReviewForm } from '@/components/ReviewForm';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Badge, buttonClass } from '@/components/ui';
import type { Domain } from '@/lib/events';

const EXAMPLES = [
  'Is a CV-screening tool high-risk under the AI Act, and what must its provider do?',
  'Do we need a DPIA before using emotion recognition on call-centre recordings?',
  'Which AI Act obligations apply to a provider of a general-purpose AI model, and from when?',
];

/** What to ask, and how: the law with its citations checked, or anything from the web. */
const MODES: Readonly<
  Record<
    Domain,
    {
      readonly label: string;
      readonly placeholder: string;
      readonly hint: string;
      readonly examples: readonly string[];
    }
  >
> = {
  compliance: {
    label: 'Ask about EU law',
    placeholder: "Ask about the AI Act or the GDPR, from a provider's or a deployer's side.",
    hint: 'Ctrl + Enter to send · a run takes about a minute',
    examples: EXAMPLES,
  },
  general: {
    label: 'Ask anything',
    placeholder: 'Ask anything. Atlas researches it from the web, in parallel branches.',
    hint: 'Ctrl + Enter to send · from the web, not checked against the law',
    examples: [
      'How does retrieval-augmented generation work, and where does it fail?',
      'Compare PostgreSQL and MongoDB for an analytics workload.',
    ],
  },
};

/** What sets the answers apart, each in a line: the three things a reader can check or do. */
const POINTS: readonly { readonly title: string; readonly text: string; readonly icon: React.JSX.Element }[] = [
  {
    title: 'Citations checked',
    text: 'Each article it cites is looked up in the law: that it exists, that it was read, and, by a second reading, that it says what it is cited for.',
    icon: (
      <path d="M8 1.5 2.5 3.75v4c0 3.2 2.3 5.6 5.5 6.75 3.2-1.15 5.5-3.55 5.5-6.75v-4L8 1.5Zm-2.25 6.5 1.6 1.6 3-3.1" />
    ),
  },
  {
    title: 'Every agent, open',
    text: 'Read the exact prompt each agent was sent, what it produced, and every attempt it made.',
    icon: (
      <>
        <path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8s-2.4 4.5-6.5 4.5S1.5 8 1.5 8Z" />
        <circle cx="8" cy="8" r="2" />
      </>
    ),
  },
  {
    title: 'Redirect a branch',
    text: 'Rewrite one branch’s question: only it is researched again, and the two answers can be compared word by word.',
    icon: (
      <>
        <circle cx="4" cy="3.5" r="1.5" />
        <circle cx="4" cy="12.5" r="1.5" />
        <circle cx="12" cy="6" r="1.5" />
        <path d="M4 5v6M12 7.5c0 2.5-2.5 3-6.6 4.2" />
      </>
    ),
  },
];

export default function Home(): React.JSX.Element {
  const router = useRouter();
  // Ask about the law, ask anything, or review a contract against it.
  const [mode, setMode] = useState<Domain | 'review'>('compliance');
  const shown = mode === 'review' ? null : MODES[mode];

  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-5xl items-center gap-3 px-4 py-3 sm:px-6">
        <span className="text-sm font-semibold text-ink">Atlas</span>
        <Link href="/demo" className={`${buttonClass('ghost', 'sm')} ml-auto`}>
          Recorded run
        </Link>
        <ThemeToggle />
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 py-10 sm:px-6 sm:py-16">
        <Badge tone="law" className="self-start">
          EU AI Act · GDPR
        </Badge>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Research the law, with every citation checked against it.
        </h1>
        <p className="mt-3 max-w-[60ch] text-base leading-relaxed text-ink-muted">
          Atlas splits your question into branches, researches them in parallel from the regulations
          and official guidance, and checks each article it cites before you read the answer.
        </p>

        <div className="mt-8">
          <div role="group" aria-label="What to ask" className="mb-3 inline-flex rounded-lg bg-raised p-1">
            {[...(Object.keys(MODES) as Domain[]), 'review' as const].map((id) => (
              <button
                key={id}
                type="button"
                aria-pressed={mode === id}
                onClick={() => setMode(id)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  mode === id ? 'bg-surface text-ink shadow-sm' : 'text-ink-muted hover:text-ink'
                }`}
              >
                {id === 'review' ? 'Review a contract' : MODES[id].label}
              </button>
            ))}
          </div>
          {mode === 'review' || shown === null ? (
            <ReviewForm onStarted={(runId) => router.push(`/reviews/${runId}`)} />
          ) : (
            <QuestionForm
              size="lg"
              domain={mode}
              placeholder={shown.placeholder}
              examples={shown.examples}
              hint={shown.hint}
              submitLabel="Research"
              onStarted={(runId) => router.push(`/runs/${runId}`)}
            />
          )}
        </div>

        <ul className="mt-12 grid gap-6 sm:grid-cols-3">
          {POINTS.map((point) => (
            <li key={point.title}>
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10 text-accent">
                <svg
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {point.icon}
                </svg>
              </span>
              <h2 className="mt-2.5 text-sm font-semibold text-ink">{point.title}</h2>
              <p className="mt-1 text-xs leading-relaxed text-ink-muted">{point.text}</p>
            </li>
          ))}
        </ul>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Link href="/demo" className={buttonClass('secondary', 'md')}>
            Watch a recorded run
          </Link>
          <span className="text-2xs text-ink-faint">No API key needed: a recorded run, replayed in the browser.</span>
        </div>
      </main>

      <footer className="mx-auto w-full max-w-3xl px-4 pb-8 sm:px-6">
        <Disclaimer className="border-t border-line pt-4" />
      </footer>
    </div>
  );
}
