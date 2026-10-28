'use client';

import { useEffect, useState } from 'react';

import { type AgentCall, reviseRun } from '@/lib/api';
import type { RunResult, SourceRef } from '@/lib/events';
import { type Agent, type Attempt, agentOf, byAuthority, reviewsOf } from '@/lib/inspector';
import type { InspectorTab } from '@/lib/inspectorLink';
import { type CallEntry, type RunState, isFinished } from '@/lib/reducer';

type Tab = InspectorTab;

export type CallSource = (callId: string) => Promise<AgentCall | null>;

const TABS: readonly { readonly id: Tab; readonly label: string }[] = [
  { id: 'task', label: 'Task' },
  { id: 'prompt', label: 'Prompt' },
  { id: 'output', label: 'Output' },
  { id: 'attempts', label: 'Attempts' },
];

interface Props {
  /** The run on screen; null for a replay, which has no API behind it. */
  readonly runId: string | null;
  /**
   * Where a call's full record comes from: the API on a live run, the recording
   * on the demo. Null where there is neither.
   */
  readonly loadCall: CallSource | null;
  readonly state: RunState;
  readonly result: RunResult | null;
  readonly selected: string | null;
  /** The open tab, when the page keeps it (in its URL); otherwise the inspector does. */
  readonly tab?: Tab;
  readonly onTab?: (tab: Tab) => void;
  /** A revision was started from this run: show it. */
  readonly onRevised?: (runId: string) => void;
}

/**
 * One agent, opened: what it was asked, the exact prompt it was sent, what it
 * produced, and every attempt it made. A finished branch can be rewritten here
 * and re-run on its own.
 */
export function Inspector({
  runId,
  loadCall,
  state,
  result,
  selected,
  tab: shownTab,
  onTab,
  onRevised,
}: Props): React.JSX.Element {
  const [ownTab, setOwnTab] = useState<Tab>('task');
  const tab = shownTab ?? ownTab;
  const setTab = onTab ?? setOwnTab;
  const agent = selected === null ? null : agentOf(state, result, selected);

  if (agent === null) {
    return (
      <p className="p-3 text-xs text-ink-faint">
        Select an agent, here or in the timeline, to see its task, the exact prompt it was sent and
        what it produced. Keys: <kbd className="font-mono">j</kbd> / <kbd className="font-mono">k</kbd>{' '}
        step through the agents, <kbd className="font-mono">e</kbd> edits a branch,{' '}
        <kbd className="font-mono">Esc</kbd> closes.
      </p>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-3 border-b border-line px-3">
        <h3 className="py-1.5 text-xs font-medium text-ink">{agent.title}</h3>
        <div role="tablist" aria-label={`${agent.title} details`} className="flex gap-1">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={tab === item.id}
              onClick={() => setTab(item.id)}
              className={`px-2 py-1.5 text-2xs ${
                tab === item.id ? 'border-b-2 border-accent text-ink' : 'text-ink-faint hover:text-ink'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      <div role="tabpanel" className="min-h-0 flex-1 overflow-y-auto p-3">
        {tab === 'task' && (
          <TaskTab agent={agent} state={state} runId={runId} onRevised={onRevised} />
        )}
        {/* Keyed by agent: the call picked for one agent means nothing for the next. */}
        {tab === 'prompt' && <CallTab key={agent.id} agent={agent} loadCall={loadCall} show="request" />}
        {tab === 'output' && <OutputTab agent={agent} state={state} loadCall={loadCall} />}
        {tab === 'attempts' && <AttemptsTab attempts={agent.attempts} />}
      </div>
    </div>
  );
}

// --- task ------------------------------------------------------------------------------

function TaskTab({
  agent,
  state,
  runId,
  onRevised,
}: {
  readonly agent: Agent;
  readonly state: RunState;
  readonly runId: string | null;
  readonly onRevised?: (runId: string) => void;
}): React.JSX.Element {
  if (agent.kind === 'stage') {
    return (
      <div className="space-y-3">
        <p className="text-xs text-ink-muted">{agent.role}</p>
        {agent.id === 'planner' && state.branches.length > 0 && (
          <Labelled label="Planned">
            <ol className="space-y-1">
              {state.branches.map((branch) => (
                <li key={branch.id} className="flex gap-2 text-xs">
                  <span className="shrink-0 font-mono text-2xs text-ink-faint">{branch.subtaskId}</span>
                  <span className="text-ink">{branch.question}</span>
                </li>
              ))}
            </ol>
          </Labelled>
        )}
        {agent.id === 'critic' && <Reviews reviews={reviewsOf(state.criticRounds, null)} />}
      </div>
    );
  }

  const { branch } = agent;
  return (
    <div className="space-y-3">
      <Labelled label="Asked">
        <p className="text-xs text-ink">{branch.question}</p>
      </Labelled>
      {agent.before !== null && (
        <p className="rounded border border-edited/40 bg-edited/10 px-2 py-1 text-2xs text-ink">
          Rewritten by you. It asked: <span className="text-ink-muted">{agent.before}</span>
        </p>
      )}
      {branch.origin === 'carried' && (
        <p className="rounded border border-line bg-raised px-2 py-1 text-2xs text-ink-muted">
          Carried over: this branch&apos;s research was done in the version this one revises.
        </p>
      )}
      <Reviews reviews={reviewsOf(state.criticRounds, branch.subtaskId)} />
      {runId !== null && isFinished(state) && state.phase === 'completed' && (
        <EditBranch
          key={`${branch.subtaskId}:${branch.question}`}
          runId={runId}
          subtaskId={branch.subtaskId}
          question={branch.question}
          onRevised={onRevised}
        />
      )}
    </div>
  );
}

function EditBranch({
  runId,
  subtaskId,
  question,
  onRevised,
}: {
  readonly runId: string;
  readonly subtaskId: string;
  readonly question: string;
  readonly onRevised?: (runId: string) => void;
}): React.JSX.Element {
  const [draft, setDraft] = useState(question);
  const [sending, setSending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const changed = draft.trim() !== '' && draft.trim() !== question.trim();

  async function submit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (!changed || sending) return;
    setSending(true);
    setMessage(null);
    const outcome = await reviseRun(runId, subtaskId, draft.trim());
    setSending(false);
    if (outcome.kind === 'started') {
      onRevised?.(outcome.runId);
    } else if (outcome.kind === 'busy') {
      setMessage('This conversation is answering something else. Try again when it finishes.');
    } else {
      setMessage(outcome.message);
    }
  }

  const id = `edit-${subtaskId}`;
  return (
    <form onSubmit={(event) => void submit(event)} className="space-y-2 border-t border-line pt-3">
      <label htmlFor={id} className="block text-2xs uppercase tracking-wide text-ink-faint">
        Rewrite this branch&apos;s question
      </label>
      <textarea
        id={id}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        rows={3}
        maxLength={2000}
        className="w-full resize-y rounded border border-line bg-canvas px-2 py-1.5 text-xs text-ink focus:border-accent"
      />
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={!changed || sending}
          className="rounded bg-accent px-2.5 py-1 text-2xs font-medium text-accent-ink disabled:opacity-40"
        >
          {sending ? 'Starting…' : 'Re-run this branch'}
        </button>
        <span className="text-2xs text-ink-faint">
          Only this branch searches again; the others are carried over, and the answer is rewritten.
        </span>
      </div>
      {message !== null && (
        <p role="alert" className="text-2xs text-failed">
          {message}
        </p>
      )}
    </form>
  );
}

function Reviews({
  reviews,
}: {
  readonly reviews: ReturnType<typeof reviewsOf>;
}): React.JSX.Element | null {
  if (reviews.length === 0) return null;
  return (
    <Labelled label="Reviews">
      <ul className="space-y-1">
        {reviews.map((review) => (
          <li key={review.iteration} className="text-2xs">
            <span className={review.verdict === 'accept' ? 'text-done' : 'text-retry'}>
              #{review.iteration} {review.verdict === 'accept' ? 'accepted' : 'sent back'}
            </span>
            {review.reason !== null && <span className="text-ink-muted"> — {review.reason}</span>}
            {review.guidance !== null && (
              <span className="block pl-4 text-ink-faint">asked to: {review.guidance}</span>
            )}
          </li>
        ))}
      </ul>
    </Labelled>
  );
}

// --- the calls ----------------------------------------------------------------------------

/** The latest attempt's model call (or its first call), fetched in full when shown. */
/**
 * The calls of the latest attempt worth opening: its model calls, in the order
 * they were reported, or its first search when it made none. The critic makes
 * two in a round (its review and the citation support check); a structured
 * call retried on the fallback makes two as well.
 */
function openable(attempt: Attempt | null): readonly CallEntry[] {
  if (attempt === null) return [];
  const models = attempt.calls.filter((call) => call.kind === 'model');
  return models.length > 0 ? models : attempt.calls.slice(0, 1);
}

/** One of the latest attempt's calls, fetched in full when shown. */
function useCall(
  agent: Agent,
  loadCall: CallSource | null,
  choice: number,
): {
  readonly attempt: Attempt | null;
  readonly choices: readonly CallEntry[];
  readonly call: AgentCall | null;
  readonly state: 'none' | 'replay' | 'loading' | 'ready' | 'unavailable';
} {
  const attempt = agent.attempts.at(-1) ?? null;
  const choices = openable(attempt);
  const summary = choices[Math.min(choice, choices.length - 1)] ?? null;
  const callId = summary?.call_id ?? null;
  const [fetched, setFetched] = useState<{ readonly id: string; readonly call: AgentCall | null } | null>(null);

  useEffect(() => {
    if (loadCall === null || callId === null) return;
    let live = true;
    void loadCall(callId).then((call) => {
      if (live) setFetched({ id: callId, call });
    });
    return () => {
      live = false;
    };
  }, [loadCall, callId]);

  if (callId === null) return { attempt, choices, call: null, state: 'none' };
  if (loadCall === null) return { attempt, choices, call: null, state: 'replay' };
  if (fetched === null || fetched.id !== callId) {
    return { attempt, choices, call: null, state: 'loading' };
  }
  return {
    attempt,
    choices,
    call: fetched.call,
    state: fetched.call === null ? 'unavailable' : 'ready',
  };
}

function CallPicker({
  count,
  choice,
  onChoose,
}: {
  readonly count: number;
  readonly choice: number;
  readonly onChoose: (choice: number) => void;
}): React.JSX.Element | null {
  if (count < 2) return null;
  return (
    <div className="mb-2 flex gap-1">
      {Array.from({ length: count }, (_, index) => (
        <button
          key={index}
          type="button"
          aria-label={`Call ${index + 1} of ${count}`}
          aria-pressed={choice === index}
          onClick={() => onChoose(index)}
          className={`rounded border px-1.5 py-0.5 text-2xs ${
            choice === index ? 'border-accent text-ink' : 'border-line text-ink-faint hover:text-ink'
          }`}
        >
          Call {index + 1}
        </button>
      ))}
    </div>
  );
}

function CallTab({
  agent,
  loadCall,
  show,
}: {
  readonly agent: Agent;
  readonly loadCall: CallSource | null;
  readonly show: 'request' | 'response';
}): React.JSX.Element {
  const [choice, setChoice] = useState(0);
  const { attempt, choices, call, state } = useCall(agent, loadCall, choice);
  const picker = <CallPicker count={choices.length} choice={choice} onChoose={setChoice} />;
  if (state === 'none') {
    return (
      <p className="text-xs text-ink-faint">
        {agent.kind === 'branch' && agent.branch.origin === 'carried'
          ? 'Carried over: this branch made no calls in this version.'
          : 'No calls recorded for this agent in this run.'}
      </p>
    );
  }
  if (state === 'replay') {
    return (
      <p className="text-xs text-ink-faint">
        Prompts are recorded on live runs. This replay carries what each call was and what it cost:
        see Attempts.
      </p>
    );
  }
  if (state === 'loading') {
    return (
      <div>
        {picker}
        <p className="text-xs text-ink-faint">Loading the call…</p>
      </div>
    );
  }
  if (call === null) return <p className="text-xs text-failed">The API could not return this call.</p>;

  const header = (
    <>
      {picker}
      <p className="mb-2 text-2xs text-ink-faint">
        Attempt {attempt?.number} · {call.kind === 'model' ? call.name : `${call.name} search`} ·{' '}
        {call.duration_s.toFixed(1)}s
        {call.input_tokens !== null && ` · ${call.input_tokens.toLocaleString()} in`}
        {call.output_tokens !== null && ` / ${call.output_tokens.toLocaleString()} out`}
      </p>
    </>
  );

  if (show === 'response') {
    return (
      <div>
        {header}
        {call.error !== null && <p className="mb-2 text-2xs text-failed">{call.error}</p>}
        <Pre>{typeof call.response === 'string' ? call.response : JSON.stringify(call.response, null, 2)}</Pre>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {header}
      {'query' in call.request ? (
        <p className="text-xs text-ink">
          Searched for: <span className="font-mono text-2xs">{call.request.query}</span>
        </p>
      ) : (
        call.request.map((message, index) => (
          <details
            key={index}
            open={message.role !== 'system'}
            className="rounded border border-line bg-canvas"
          >
            <summary className="cursor-pointer px-2 py-1 text-2xs uppercase tracking-wide text-ink-muted">
              {message.role === 'system' ? 'System prompt' : message.role === 'human' ? 'Sent' : message.role}
              <span className="ml-2 normal-case tracking-normal text-ink-faint">
                {message.content.length.toLocaleString()} characters
              </span>
            </summary>
            <Pre>{message.content}</Pre>
          </details>
        ))
      )}
    </div>
  );
}

function Pre({ children }: { readonly children: React.ReactNode }): React.JSX.Element {
  return (
    <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words px-2 py-1.5 font-mono text-2xs leading-relaxed text-ink">
      {children}
    </pre>
  );
}

// --- output -------------------------------------------------------------------------------

function OutputTab({
  agent,
  state,
  loadCall,
}: {
  readonly agent: Agent;
  readonly state: RunState;
  readonly loadCall: CallSource | null;
}): React.JSX.Element {
  if (agent.kind === 'branch') {
    const grouped = byAuthority(agent.branch.sources);
    return (
      <div className="space-y-3">
        <Labelled label="Findings">
          {agent.findings === null ? (
            <p className="text-xs text-ink-faint">Not written yet.</p>
          ) : (
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink">{agent.findings}</p>
          )}
        </Labelled>
        <SourceList title="Primary law" tone="text-law" sources={grouped.law} />
        <SourceList title="Official guidance" tone="text-guidance" sources={grouped.guidance} />
        <SourceList title="Other sources" tone="text-secondary" sources={grouped.other} />
      </div>
    );
  }
  if (agent.id === 'synthesizer' && state.citationChecks.length > 0) {
    return (
      <Labelled label="Citation checks, one per draft">
        <ol className="space-y-1">
          {state.citationChecks.map((check, index) => {
            // Draft n is read in critic round n; an unsupported citation is a
            // problem with the draft as much as an invented one.
            const unsupported = state.supportChecks
              .filter((support) => support.iteration === index + 1)
              .flatMap((support) => support.unsupported);
            const clean = check.ok && unsupported.length === 0;
            return (
              <li key={index} className="text-2xs">
                <span className={clean ? 'text-done' : 'text-failed'}>
                  Draft {index + 1}: {clean ? 'every citation verified' : 'problems found'}
                </span>
                {check.unresolved.map((label) => (
                  <span key={label} className="block pl-4 text-ink-muted">{label} does not exist</span>
                ))}
                {check.unretrieved.map((label) => (
                  <span key={label} className="block pl-4 text-ink-muted">{label} was never retrieved</span>
                ))}
                {check.misquoted.map(([label, quote]) => (
                  <span key={`${label}${quote}`} className="block pl-4 text-ink-muted">
                    “{quote}” is not in {label}
                  </span>
                ))}
                {unsupported.map((entry) => (
                  <span key={`${entry.label}${entry.sentence}`} className="block pl-4 text-ink-muted">
                    {entry.label} does not say what it is cited for: {entry.reason}
                  </span>
                ))}
              </li>
            );
          })}
        </ol>
      </Labelled>
    );
  }
  return <CallTab key={agent.id} agent={agent} loadCall={loadCall} show="response" />;
}

function SourceList({
  title,
  tone,
  sources,
}: {
  readonly title: string;
  readonly tone: string;
  readonly sources: readonly SourceRef[];
}): React.JSX.Element | null {
  if (sources.length === 0) return null;
  return (
    <Labelled label={`${title} · ${sources.length}`}>
      <ul className="space-y-0.5">
        {sources.map((source, index) => (
          <li key={`${source.label ?? source.url}:${index}`} className="truncate text-2xs">
            <a href={source.url} target="_blank" rel="noreferrer" className="hover:underline">
              {source.label !== null && source.label !== undefined && (
                <span className={`mr-1.5 font-medium ${tone}`}>{source.label}</span>
              )}
              <span className="text-ink-muted">{source.title}</span>
            </a>
          </li>
        ))}
      </ul>
    </Labelled>
  );
}

// --- attempts -----------------------------------------------------------------------------

function AttemptsTab({ attempts }: { readonly attempts: readonly Attempt[] }): React.JSX.Element {
  if (attempts.length === 0) {
    return <p className="text-xs text-ink-faint">No attempts recorded in this run.</p>;
  }
  return (
    <ol className="space-y-3">
      {attempts.map((attempt) => (
        <li key={attempt.spanId}>
          <p className="text-2xs uppercase tracking-wide text-ink-faint">
            Attempt {attempt.number}
            <span className="ml-2 font-mono normal-case tracking-normal">{attempt.spanId}</span>
          </p>
          <ul className="mt-1 space-y-1">
            {attempt.calls.map((call) => (
              <li key={call.call_id} className="rounded border border-line bg-canvas px-2 py-1 text-2xs">
                <span className="text-ink">
                  {call.kind === 'model' ? call.name : `searched ${call.name === 'library' ? 'the law' : 'the web'}`}
                </span>
                <span className="ml-2 text-ink-faint">
                  {call.duration_s.toFixed(1)}s
                  {call.input_tokens !== null && ` · ${(call.input_tokens + (call.output_tokens ?? 0)).toLocaleString()} tokens`}
                </span>
                {call.error !== null && <span className="block text-failed">{call.error}</span>}
                <span className="block truncate text-ink-faint">{call.preview}</span>
              </li>
            ))}
          </ul>
        </li>
      ))}
    </ol>
  );
}

function Labelled({
  label,
  children,
}: {
  readonly label: string;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <section>
      <h4 className="mb-1 text-2xs uppercase tracking-wide text-ink-faint">{label}</h4>
      {children}
    </section>
  );
}
