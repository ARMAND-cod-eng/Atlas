# Atlas

[![ci](https://github.com/ARMAND-cod-eng/Atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/ARMAND-cod-eng/Atlas/actions/workflows/ci.yml)

**An auditable research copilot for the EU AI Act and the GDPR.** Ask a compliance question, such
as *"Is our CV-screening tool high-risk, and what must we do as its provider?"* A team of agents
researches it in parallel against the official text of the law and writes an answer that cites it
article by article. Every citation is checked against the law before you see the answer. Every
agent's work is open: you can read the exact prompt each one was sent, and redirect one branch of
the research and re-run only that branch.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/workspace-dark.png">
  <img alt="The Atlas workspace: the run's timeline, step by step, on the left; the agent graph in the middle, with the synthesizer open on its citation checks (draft 1 cited an article that does not exist and one that does not say what it was cited for; draft 2 passed); and on the right the answer, its citation chips verified" src="docs/img/workspace-light.png">
</picture>

*The recorded demo, with the synthesizer's drafts open. The first draft cited an article that does
not exist ("AI Act Art. 114"; the act has 113). It also cited a real, retrieved provision for
something it does not say: Art. 6(1) makes high-risk the AI that is a safety component of a
regulated product, which a recruitment tool is not. Both checks sent the draft back. The rewrite
keeps the two citations that hold: Annex III(4)(a), and Art. 6(2), the rule that makes it high-risk.*

> A research aid, not legal advice. The law's text is EUR-Lex's, reused under Commission Decision
> 2011/833/EU; see [NOTICE](NOTICE).

### The problem

Every company shipping an AI or data product into the EU now has to answer questions like these. The
AI Act's obligations phase in between 2025 and 2027, on top of the GDPR. Each question takes a
compliance analyst or outside counsel hours. A language model answers in seconds, fluently, and cites
articles that are wrong or do not exist. Fabricated legal citations are the best-known way generative
AI fails in law, and courts have sanctioned lawyers for filing them. A better prompt does not fix
that. A check that knows the law does.

### What Atlas does about it

- **Citations are checked against the law, not trusted.** Each bracketed citation in an answer, such
  as `[GDPR Art. 33(1)]`, is held to three tests with no model involved:
  - the provision exists in the act;
  - a researcher actually retrieved it in this run;
  - any quotation beside it is the provision's own words.

  Then a model reads each citation that passed, one sentence and one provision at a time, and judges
  whether the provision says what the sentence says. A "yes" counts only if it quotes the provision's
  own words, and the quote is checked against the text.

  A draft that fails either check is sent back with the failures listed. In the UI each citation is
  a chip carrying its verdict; hover it to read the article, and why it was or was not judged to
  support its sentence.
- **It says when a question is not about the law.** Asked "history of cameroon", Atlas used to
  research it, cite GDPR Recital 35 (it mentions a medical "history") and mark the citation
  verified, because it exists and was retrieved. Now the planner judges, in the plan it writes
  anyway, whether a conversation's first question is about either act. One that is not ends at the
  plan, with the reason, and **Research anyway** asks it again in its conversation, where it is
  planned like any follow-up. A question that names an act, cites a provision or asks what a
  defined term means is always researched, whatever the planner says.
- **It reads the law, not a search engine's idea of it.**
  - Both regulations are ingested from the Publications Office's official XHTML into provisions with
    stable ids. Golden counts are tested: the AI Act's 113 articles, 13 annexes and 180 recitals; the
    GDPR's 99 articles and 173 recitals.
  - Researchers read the law first: by reference ("Article 6"), then by defined term, then by BM25.
  - Official guidance and the web come after it, kept apart and labelled by authority.
- **Every agent's work can be inspected.** Click any agent in the graph to see:
  - its task;
  - the exact messages it was sent;
  - what it produced;
  - every attempt, with each search and model call.

  These records are captured as the calls happen ([`atlas/trace.py`](backend/atlas/trace.py)), never
  reconstructed from today's prompt templates.
- **You can redirect a branch and re-run only that branch.** Rewrite one researcher's question on a
  finished run, and Atlas:
  1. forks the run's checkpoint;
  2. researches that branch again, alone;
  3. carries the other branches over;
  4. rewrites the answer.

  Each edit becomes a new version, and any two versions can be compared word by word. A person
  overseeing the AI and redirecting it is the workflow, not an afterthought.
- **It keeps an audit trail, and deletes on request.**
  - Postgres keeps every run, every agent call and every event stream past the live stream's hour.
  - `DELETE /threads/{id}` removes a conversation from every place it is stored.
  - A nightly job purges threads past their retention period.

  A GDPR tool should practise what it cites.

### Try it in a minute, with no API keys

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:3000/demo>. It shows a recorded run replayed through the real workspace:

- researchers reading the regulation library;
- the first draft caught twice: once citing an invented article, and once citing a real provision
  for what it does not say;
- the rewrite that fixes both.

Click any agent to read the prompt it was actually sent. The open agent and tab live in the URL, so
<http://127.0.0.1:3000/demo?node=researcher:s1&tab=prompt> goes straight to one researcher's prompt. The recording ran through the real graph and
prompts, with only the model scripted, and the run says so: each call reports the model `scripted`.

### Or the whole product, offline

The demo is a recording. To run the real stack yourself with no API key (API, worker, graph, library,
checks, store, workspace), set `ATLAS_OFFLINE=true` in `.env`:

```powershell
Copy-Item .env.example .env        # then set ATLAS_OFFLINE=true
docker compose --profile app up --build
```

Every model and the web search are then scripted:
- answers are built from what the library returns for each branch;
- every answer says it was scripted;
- the page marks the run "offline mode".

Ask a question, open a researcher, rewrite its question, re-run that branch, and compare the two
versions: the rewritten branch reads different law and the answer changes with it. The scripts never
pretend to be a model, so this shows the machinery, not the quality.

### Results

**Measured so far:**

| | |
|---|---|
| Library retrieval, one search per question | recall@10 **0.900** over the 40 compliance questions, every required provision found for 32 of 40 (`atlas_eval retrieval`). On the held-out half, measured once after the last change: **0.875**, up from 0.800 |
| Engine regression gate, 50 general questions | completion 0.995, grounding 0.955, tool use 0.963; run-to-run noise measured, gate thresholds set from it (5 / 4 / 3 pp) |
| Tests | **563** backend (plus 13 against Postgres) · **144** frontend, no network calls, no spend |

**Not measured yet: the headline.** The domain eval compares three systems on 40 questions:

- plain Claude, answering from memory;
- one library search and one answer;
- Atlas.

It scores the share of answers citing a provision that does not exist, plus citation precision and
recall, all deterministically. It is built and tested ([below](#the-domain-eval)) but has not been
run. Its numbers will come from committed run files, never typed in.

How often the support check judges correctly is not measured yet either. It is a model's judgement,
held to quotes it cannot fake, but nobody has counted its mistakes. `atlas_eval support` counts
them on 24 labelled cases:
- 12 citations that do support their sentence, in plain words rather than the act's;
- 12 near misses, such as Art. 34(1) cited for the 72-hour deadline that is Art. 33(1).

It needs model credit to run.

The 40 questions were drafted by a model from the regulation text, and every expected citation is
checked against the library. None has been reviewed by a person yet. `atlas_eval claim` says so, and
a test keeps that statement honest.

---

## Architecture

```mermaid
graph LR
    UI[Next.js workspace]
    API[FastAPI]
    Worker[arq worker]

    UI -->|POST /runs, /revisions| API
    API -->|enqueue| Queue[(Redis queue)]
    Queue --> Worker
    Worker -->|events| Stream[(Redis Stream)]
    Stream -->|SSE + Last-Event-ID| API
    API -.->|live events| UI

    Worker --> Graph
    Worker --> Store[(Postgres: checkpoints,<br/>runs, agent calls, events)]
    Graph --> Library[Regulation library<br/>AI Act + GDPR]
    Graph --> Web[Web search]

    subgraph Graph[LangGraph]
        direction LR
        P[planner] -->|Send x N| R[researcher]
        H[human edit] -->|one branch| R
        R --> S[synthesizer]
        S --> C[critic + citation check]
        C -->|accept| E([END])
        C -->|revise: faulted branches only| R
        C -->|revise: write-up only| S
    end
```

**The workspace has three panes, the answer given the most room.** On a phone they are tabs, and
the Answer tab opens once there is one.

| Pane | What it shows |
|---|---|
| **Timeline** | Every version of the answer on this conversation; where the run is, as four steps (plan, research, write, review); and the run's timeline cut into those steps: each branch, each critic verdict with its reason, each edit. Each search and model call is folded away until asked for. |
| **Agents** | The live graph, top to bottom, and under it an inspector for the agent you select: its task, the exact prompt it was sent, what it produced, every attempt. Keys: `j` / `k` step through the agents in the order they ran, `e` edits the selected branch's question, `Esc` closes. |
| **Answer** | The answer with its citation chips and the checks' verdict. Copy puts it on the clipboard as Markdown with its citations linked; Compare swaps it for a word-level diff against another version. Below, folded: the sources by authority, the research by branch (each with Edit & re-run), and the run's details. |

**Editing a branch forks the run; it does not restart it.** A revision writes through a dedicated
`human_edit` node with LangGraph's `aupdate_state`, from the finished run's checkpoint. Only the
edited branch researches again, and the synthesizer and critic run as usual. The tests pin each way
this could quietly go wrong:

- a stale critique from the parent leaking into the new answer;
- the edit reaching the researcher framed as a reviewer's rejection;
- old drafts replayed to the synthesizer;
- the parent's spend counted against the new run's budget;
- a second revision of the same run silently dropped.

**The run does not belong to the connection.** The API enqueues and relays; the worker executes. A
client that drops at second 40 of a 67-second run costs nothing: the run finishes, and reconnecting
replays exactly what was missed from the event log. Verified by killing the API mid-run: the run
completed with nothing listening, and the UI converged to the identical final state on restart.

**The critic sends back branches, not the whole graph.** Re-running everything would cost about 72s
and 31k tokens per iteration. Re-planning would also produce a *different decomposition*, so
per-subtask feedback would have nothing to attach to. A targeted retry costs the slowest faulted
branch plus one synthesis, and it keeps each subtask's identity, so guidance reaches the branch that
earned it.

Build notes: what was built, the defects found along the way, and how each was verified.

| | |
|---|---|
| [`docs/REPORT.md`](docs/REPORT.md) | **The compliance copilot: decisions, measurements, failures caught, limitations** |
| [`docs/PHASE-0.md`](docs/PHASE-0.md) | Foundations: config, provider fallback, checkpointing |
| [`docs/PHASE-1.md`](docs/PHASE-1.md) | Planner fan-out, parallel researchers, grounded citations |
| [`docs/PHASE-2.md`](docs/PHASE-2.md) | The critic loop, targeted retry, enforced token budget |
| [`docs/PHASE-3.md`](docs/PHASE-3.md) | Worker, Redis event log, SSE with resumable replay |
| [`docs/PHASE-4.md`](docs/PHASE-4.md) | The frontend and the live agent-graph visualisation |

### Measured


| | |
|---|---|
| Fan-out is genuinely concurrent | 5 branches: slowest **23.3s**, summed **93.4s** — the fan-out costs the slowest branch, not the sum |
| Typical run | **67–92s** wall clock, dominated by provider tok/s variance rather than by Atlas |
| Streaming overhead | SSE relay **0.18s**; queue pickup **~1.0s**; a whole run's events cost **78ms** of Redis |
| Concision prompt on the synthesizer | 4,392 → 2,752 output tokens, **40.5s → 27.1s** |
| Time to first visual feedback | **1.14s** from submit to the graph on screen |

---

## Prerequisites

| | |
|---|---|
| Python | 3.12 (pinned in `.python-version`; uv installs it automatically) |
| [uv](https://docs.astral.sh/uv/) | 0.12+ |
| Node | 20+ (developed on 24) |
| Docker Desktop | with the WSL 2 backend |
| API keys | Anthropic, OpenAI, Tavily |

## Quickstart

Docker and three API keys. From the repository root:

```powershell
Copy-Item .env.example .env   # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
docker compose --profile app up --build
```

Open <http://127.0.0.1:3000> and ask a question. That starts Postgres, Redis, the API, the worker
and the frontend in containers. To develop with fast reloads instead, run the pieces on the host as
below.

## Run the whole thing

Four terminals, for development. All commands are PowerShell, run from the repository root.

**1. Configure.** One `.env` at the root serves both the backend and Docker Compose.

```powershell
Copy-Item .env.example .env
notepad .env
```

**2. Start infrastructure.**

```powershell
docker compose up -d
```

Confirm both services are healthy and that pgvector was enabled by the init script:

```powershell
docker compose ps
docker compose exec -T postgres psql -U atlas -d atlas -c "\dx vector"
```

> Keep the `-T`. Without it, `docker compose exec` allocates a TTY and psql pipes
> its output through `less` inside the container — leaving your terminal stuck at
> `(END)` with every subsequent keystroke going to the pager instead of the shell.

**3. Install dependencies.**

```powershell
cd backend
uv sync
```

**4. Run the worker** — terminal one. Research runs execute here, not in the API.

```powershell
uv run python -m atlas.worker_main
```

**5. Run the API** — terminal two.

```powershell
uv run python -m atlas --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

**6. Run the frontend** — terminal three.

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:3000> and ask a question.

> **Both the worker and the API are required.** With no worker running, `POST /runs`
> still returns a `run_id` and the event stream sits on heartbeats forever, because
> nothing is producing events; `POST /query` waits and returns 504. If a run never
> seems to start, check that the worker terminal is alive.

> **Use `127.0.0.1`, not `localhost` — it is worth two seconds a request.** On
> Windows `localhost` resolves to `::1` before `127.0.0.1`, and uvicorn binds IPv4
> only, so every call waits for the IPv6 attempt to fail first. Measured:
> **2,050 ms via `localhost` against 22 ms via `127.0.0.1`** for the same request.
> The frontend defaults to `127.0.0.1:8000`; override with `NEXT_PUBLIC_API_BASE_URL`
> in `frontend/.env.local`.

## Verifying it works

Health check — both booleans come from live probes, not constants:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

**Watch a run from the terminal.** `POST /runs` returns immediately with a `run_id`;
the event stream shows the graph working. Branches start together and finish out of
order — that interleaving is the fan-out, live.

```powershell
$run = Invoke-RestMethod -Uri http://127.0.0.1:8000/runs -Method Post -ContentType 'application/json' -Body '{"query":"Is a CV-screening tool high-risk under the AI Act, and what must its provider do?"}'
curl.exe -N "http://127.0.0.1:8000/runs/$($run.run_id)/events"
```

Kill that `curl` with Ctrl-C partway through and the run keeps going — it belongs to
the worker, not to your connection. Check on it, then resume from where you left off
by passing the last `id:` you saw:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)"
curl.exe -N "http://127.0.0.1:8000/runs/$($run.run_id)/events?last_event_id=6"
```

`GET /runs/{run_id}` reports `running`, `completed` with the full result, or `failed` with the
error the run ended on.

A browser `EventSource` sends the `Last-Event-ID` header automatically on reconnect;
the query parameter is for clients like `curl` that do not retry. The same thing works
in the UI — refresh `/runs/{run_id}` mid-run and it rebuilds from the log.

**Inspect, revise, compare.** Once the run completes:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)/calls"            # every agent call, summarised
Invoke-RestMethod "http://127.0.0.1:8000/library/provision?label=AI%20Act%20Art.%206(2)"
$branch = (Invoke-RestMethod "http://127.0.0.1:8000/runs/$($run.run_id)").result.subtasks[1].id
$v2 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/runs/$($run.run_id)/revisions" -Method Post -ContentType 'application/json' -Body (@{subtask_id=$branch; question='What must a deployer of a high-risk AI system do?'} | ConvertTo-Json)
Invoke-RestMethod "http://127.0.0.1:8000/threads/$($run.thread_id)/versions"
```

The revision is a new run on the same thread: stream it like any other, and only the rewritten
branch researches again. `GET /runs/{id}/calls/{call_id}` returns one call whole, prompt included.

**The synchronous endpoint still exists**, for clients that want one request and one answer:

```powershell
$r = Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body '{"query":"When does a deployer need a fundamental rights impact assessment under the AI Act?"}'
$r.answer
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType 'application/json' -Body (@{query='Summarize that in one sentence.'; thread_id=$r.thread_id} | ConvertTo-Json)
```

Passing the returned `thread_id` back continues the conversation: the follow-up is planned against
the previous turn, which confirms the Postgres checkpointer is persisting state. It is still a
separate run, with its own research and its own token budget. A thread runs one question at a time:
a second request while one is in progress gets `409`, because two runs writing one checkpoint would
silently overwrite each other's turn.

**Verify the fallback actually fires.** The behaviour most worth confirming by hand: break
the primary provider and check an answer still comes back via GPT-4o. The key is read by the
**worker**, so set it there:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-deliberately-invalid"
uv run python -m atlas.worker_main
```

An answer means the fallback chain worked. Restart without the override to go back to normal.

## Development

```powershell
cd backend
uv run ruff check . ; uv run ruff format . ; uv run mypy --strict atlas atlas_eval ; uv run pytest -q
```

```powershell
cd frontend
npm run typecheck ; npm run lint ; npm test
```

`mypy --strict` with no `# type: ignore` anywhere; TypeScript strict plus
`noUncheckedIndexedAccess`, no `any`, no `eslint-disable`. Neither suite makes a network call or
spends money — every model is a fake, and the frontend's fixtures are real event streams recorded
from the graph by `backend/scripts/capture_fixtures.py`.

`ci.yml` runs both suites, plus `ruff format --check`, on every push to `main` and every pull
request. It never calls a model.

## Evaluation

Two datasets, for two questions:

- **Is the engine still sound?** The 50 general research questions. They gate every change to
  prompts, graph code or model config.
- **Is the product better than the alternatives?** The 40 compliance questions. They run the
  [domain eval](#the-domain-eval).

### The engine eval

`backend/atlas_eval` scores the agent on a 50-query dataset in three categories:

| | What it measures | How |
|---|---|---|
| **Completion** | Does the answer contain each fact on the item's rubric? | Pattern and number checks where possible, an LLM judge for the rest |
| **Grounding** | What fraction of the answer's atomic claims do the retrieved sources support? | The answer is split into claims, and each is checked against the retrieved snippets |
| **Tool use** | Subtask count in the expected range, no redundant searches, retries within the loop's bounds, subtasks genuinely independent | Deterministic checks, plus one judged check for independence |

Grounding here is claim-level. The graph's own `ungrounded_urls` check only confirms that a cited URL
was retrieved; it cannot catch a real URL cited for something it does not say. The suite runs the
real graph in-process against a **frozen corpus** of recorded Tavily results, so two runs are
comparable and CI never touches live search. The judge is GPT-4o at temperature 0: a fixed
temperature is impossible on the Claude models Atlas runs on, and a model should not grade its own
answers. The exception is a run the fallback answered, where GPT-4o would be grading itself. Each
run's metrics record which provider answered every call, and a run the fallback answered more than
5% of is refused by `run`, `noise` and `report`: it measures the fallback, not the system.

**Status: baseline committed, comparisons pending.** Every one of the 50 items was read and
approved by a person, and nine were corrected first: rubric lines that were wrong, ambiguous, out of
date, or asked for more than the question did. The corpus is recorded (1,229 documents, 4.6 MB
gzipped), and the baseline has been run twice on identical code to measure the noise:

| | Baseline | Rerun, identical code | Gate threshold (measured noise floor) |
|---|---|---|---|
| Completion | 0.995 | 0.970 | 5pp |
| Grounding | 0.955 | 0.943 | 4pp |
| Tool use | 0.963 | 0.970 | 3pp |

Completion is at the ceiling: 49 of 50 baseline items score 1.0, so this dataset cannot separate
systems on completion, and grounding is where differences can show. The comparison against simpler
systems has not been run yet. `atlas_eval claim` prints the strongest description the dataset
supports, and a test keeps it honest:

```
50 hand-labeled query traces
```

Reproducing them, from `backend/`:

```powershell
uv run python -m atlas_eval review      # approve or skip each item; resumable
uv run python -m atlas_eval record      # the only step that searches the web: live Tavily, 3 passes per item
uv run python -m atlas_eval run --name baseline --write-baseline --corpus-date <YYYY-MM-DD>
uv run python -m atlas_eval run --name rerun1          # identical code again: pure noise
uv run python -m atlas_eval noise --runs baseline rerun1
uv run python -m atlas_eval label --run baseline       # human labels for judge calibration
uv run python -m atlas_eval calibrate --run baseline   # judge-human agreement: raw, kappa, prevalence
```

Then commit `atlas_eval/data/corpus/`, `atlas_eval/data/baseline.json` and
`atlas_eval/data/runs/baseline.json`. `noise` reports, per category, how far two runs of identical
code drift apart and so the smallest regression the gate can catch at n=50. A threshold below that
is a promise the gate cannot keep, and gets raised.

The comparison and the chart come from the same run files, never typed in by hand:

```powershell
uv run python -m atlas_eval run --name no-critic --variant no-critic
uv run python -m atlas_eval run --name single-call --variant single-call
uv run python -m atlas_eval report --runs baseline no-critic single-call --out ../docs/RESULTS.md --chart ../docs/img/quality-vs-cost
```

**The CI gate fails closed.** The paid runs live in `eval.yml`, which never fires on a plain push.
Every PR from this repository runs a 12-item smoke that fails if any item crashes. The full 50-item
run, gated against the baseline, runs on PRs that touch prompts, graph code, model config or the
eval itself, on PRs labelled `run-eval`, and on demand. A category fails when it regresses by more
than its threshold *and* its 95% paired-bootstrap CI excludes zero. The thresholds are the noise floor
`noise` measured on two runs of identical code: 5pp for completion, 4pp for grounding, 3pp for tool
use. The gate began at 2pp for all three, below every one of them, so it was raised. The gate also refuses to pass when it has
no per-item baseline to pair against. **Until the files above are committed, every eval job
fails.** That is the gate reporting it has nothing to compare against, not a flaky test.

**Comparison runs.** `run --variant no-critic` runs the same graph without the review loop, and
`run --variant single-call` answers each item with one search and Atlas's own synthesizer — same
prompt, same corpus, no planner, researchers or critic. They exist to measure what the full graph
buys, and a baseline can only be written from the production variant.

### The domain eval

Three systems answer the 40 compliance questions:

| Variant | What it is |
|---|---|
| `plain` | One model call, no retrieval, asked to cite in the same bracketed form. What a team gets by asking a model directly. |
| `single-call` | One library search on the question as asked, answered by Atlas's own synthesizer. |
| `atlas` | The full graph, including the support check. |

**Web search is off for all three, and each run file says so.** The comparison is about citing the
law, and the law comes from the committed library. A live search would make the runs unrepeatable.

**Citations are scored without a judge.** Each item lists the provisions an answer *must* cite and
the ones it *may*. A citation matches an expected provision when one contains the other, which is
the citation check's own rule: citing Article 6 covers 6(2). Five measures, each with a bootstrap CI:

| Measure | What it is |
|---|---|
| **fabricated** | The share of answers citing at least one provision that does not exist. The headline. |
| **uncited** | The share of answers citing nothing. A system that never cites can never fabricate, and would otherwise look flawless. |
| **validity** | The share of citations that name real law, over the answers that cite. |
| **precision** | The share of real citations that are on point. Real but irrelevant law is how a confident answer pads itself. |
| **recall** | The share of the must-cite provisions an answer cites, over every answer. |

The judge still scores completion against each item's rubric. Grounding is scored for the two
systems that retrieve. `plain` retrieves nothing and plans nothing, so neither grounding nor tool use
can apply to it, and it is not charged for them.

From `backend/`, after reviewing the questions:

```powershell
uv run python -m atlas_eval review --dataset compliance      # a person approves each item first
uv run python -m atlas_eval run --dataset compliance --variant plain --name plain-v1
uv run python -m atlas_eval run --dataset compliance --variant single-call --name single-v1
uv run python -m atlas_eval run --dataset compliance --name atlas-v1
uv run python -m atlas_eval citations --runs atlas-v1 plain-v1 single-v1 --reference atlas-v1 --out ../docs/CITATIONS.md
```

`citations` costs nothing. It re-scores saved answers with today's scorer and pairs each system with
the reference item by item. `retrieval` measures the library search alone, also for free.

**The scope check is measured on its own.** `atlas_eval scope` runs it on 64 questions about the
law (the 40 compliance questions, and 24 casual ones such as "right to be forgotten") and 80 that
are not (the 50 general research questions, and 30 everyday ones, among them the DSA and NIS2,
which the library does not hold). Free by default: the guard and offline mode's stand-in. With
`--planner` it measures the planner's judgment, one call per question. Counted both ways: a
question about the law turned away, the worse failure, and an off-topic one caught.

**The support check is measured on its own.** Its verdicts are counted against 24 labelled cases,
one model call each:

```powershell
uv run python -m atlas_eval support --name v1
```

It reports three numbers:
- **precision:** of the "does not support" verdicts, how many were right. A wrong one costs a
  rewrite.
- **recall:** of the citations that don't support their sentence, how many were caught.
- **false-alarm rate:** of the citations that do support it, how many were sent back.

The labels check themselves. Each "supports" case names the provision's words that prove it, and a
test checks those words are in the provision. A run the fallback model answered is refused: it
would measure the fallback's judgement.

---

## Things worth knowing

**A component that never fires looks exactly like one that works.** This is the bug class the
project keeps producing, and most of the design exists to make it impossible. A `temperature=0.0`
that 400s meant every answer came from the fallback for a whole session, and everything looked fine.
A `SecretStr` stringified to `'**********'` produced a silent 401 that read as "no results". A
critic whose structured output failed fell open to `accept` and recorded a round byte-for-byte
identical to a real approval. Each was caught by a check built to *discriminate* — a wall-clock
measurement against a forced serial loop, a reducer compared against `operator.add`, a deliberately
poisoned search branch the critic had to reject — and each now reports itself in `metrics.degraded`,
which the UI shows prominently rather than tucking away.

**A thread is a conversation; a run is one question.** Only the conversation carries over between
runs on the same thread. Each run starts with fresh research, critic feedback, metrics and token
budget. That took deliberate work. LangGraph applies graph input through each key's reducer, so
passing `research_results: []` does not clear the previous run's results — it merges into them.
Before this was fixed, a follow-up was synthesised partly from the last question's research, and
the budget counted a whole conversation against a per-question limit. The planner, which runs
once at the start of every run, now resets those keys explicitly. `tests/test_run_isolation.py`
runs two questions on one thread to keep it that way.

**One engine, two domains.** `ATLAS_DOMAIN` selects the prompts and what a researcher reads.
`compliance` is the default: the regulation library first, then the web, with the citation check
enforced. `general` is the original web researcher. The eval always sets the domain from the dataset,
never from the environment: production defaults to compliance, and the general baseline was measured
on general.

**Model selection** is a single config constant, `PRIMARY_MODEL`, read only by `atlas/models.py`.
Nothing else names a model. Override it in `.env` without touching code.

**Provider fallback** sets `max_retries=0` on the primary deliberately. LangChain retries *inside* a
runnable before the exception reaches `with_fallbacks`, so a non-zero value would work an overloaded
provider through its full backoff schedule while the healthy fallback sits idle.
`tests/test_fallback.py` guards it.

**The synthesizer's output cap is a runaway guard, not a length target.** Caps of 2,000 and 3,500
both truncated ordinary answers mid-sentence — severing a URL, which then surfaced as
`ungrounded_citation` for a citation that was never fabricated. The concision *prompt* is what
actually shortens the answer; the cap sits above the largest output ever observed, and a run that
trips it reports `synthesis_truncated`.

**Windows, twice.** psycopg's async driver cannot run on `ProactorEventLoop`, which is what both
uvicorn and arq's CLI select by default — the symptom is `PoolTimeout: pool initialization
incomplete`, which points at the database while the database is fine. `python -m atlas` and
`python -m atlas.worker_main` both build a Selector loop for you. And arq's `handle_signals=False`
looks like the Windows-friendly choice but makes shutdown call `signal.SIGUSR1`, which does not
exist there; the default `True` is correct because arq already swallows the `NotImplementedError`.

**`--reload` does not cover `.env`.** It watches the `backend` package; `.env` lives at the
repository root and is read once at startup. After editing it, stop and restart the process.

## Roadmap

**Next: run the domain eval.** A person reviews the 40 compliance questions, and then the three
systems run, at an estimated $12–15 in total. The headline at the top of this file is filled from
those run files. The engine's remaining comparisons (no-critic and single-call on the general
dataset) run in the same session.

**Then, in order of what the measurements say:**

- Most remaining retrieval misses need reasoning, not vocabulary: knowing that screening
  applicants is "evaluation of personal aspects", or that a transfer needs "appropriate safeguards".
  A query rewrite is built for them (`atlas/library/rewrite.py`): one cheap model call names the
  legal concepts a question turns on, in the acts' words, and adds them to the search. It is off
  until measured:

  ```powershell
  uv run python -m atlas_eval retrieval --rewrite --split test
  ```

  That reports recall on the held-out half with and without it, item by item. Embeddings come only
  if it falls short.
- Calibrate the judge against the human labels already collected.
- Add a third act. That is an ingest command, a golden-count test and labels; the engine does not
  change.

**Not planned:** per-agent prompt editing, uploading your own documents, multi-tenant
authentication. The workspace's edit point is a branch's task, deliberately. A rewritten question is
something a reviewer can read and judge. An edited system prompt changes every later answer in ways
nobody reviews.
