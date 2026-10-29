# Atlas as a compliance copilot: technical report

*As of 25 October 2026.*

**Status.**

- **Built, tested and demonstrable:** the product.
- **Built and tested, not run:** the domain eval, which would put a number on its central claim.

Every figure below comes from a command in the repository or a commit that records it. Nothing here
is projected.

## 1. The bet

Questions about the EU AI Act and the GDPR are expensive to answer and easy to get confidently
wrong. A language model answers them fluently and cites articles that do not exist, or that say
something else. In law that is the costly failure: it looks verifiable, and a reader acts on it.

The bet is that the fix is not a better prompt but a **check that knows the law**:

- the regulation as data;
- citations tested against it with no model involved;
- research and review built around those tests;
- a workspace where a person can see each agent's work and redirect it.

Atlas already had the engine for this: a planner, parallel researchers, a critic with targeted
retry, durable streaming, and an eval with a measured noise floor. What it lacked was a problem.

## 2. What was built

| Piece | Where |
|---|---|
| The two acts as citable units | `backend/atlas/library/`, `scripts/ingest_regulation.py` |
| Library search: reference, definitions, BM25 | `atlas/library/search.py`, `atlas/tools/rank.py` |
| The citation check | `atlas/library/citations.py`, enforced in the critic (`atlas/graph/nodes.py`) |
| The compliance domain: prompts, law-first research | `atlas/domains.py`, `atlas/prompts.py` |
| Revising one branch of a finished run | `atlas/revision.py`, the `human_edit` node in `atlas/graph/builder.py` |
| Every agent call, recorded | `atlas/trace.py`, `atlas/models.py` |
| Runs, calls and event streams in Postgres; deletion and retention | `atlas/store.py`, `atlas/retention.py` |
| The three-pane workspace | `frontend/src/components/` (`RunWorkspace`, `Inspector`, `Versions`, `CitationChip`) |
| The domain eval | `atlas_eval/baselines.py`, `atlas_eval/citation_metrics.py`, `dataset.compliance.v1.jsonl` |

## 3. Decisions

### 3.1 One engine, a domain pack

**The decision.** The graph is unchanged. A *domain* is the four prompts plus whether researchers read
the library before the web. `general` is Atlas as its first baseline measured it. `compliance` is the
product and the default.

**How the general domain is held still.** Its prompts moved verbatim into `atlas/prompts.py`, and a
test pins their digest, recorded before the move. A prompt edit is a behaviour change to be
measured, not slipped in.

**A trap the new default opened.** The general eval would have taken its domain from the
environment. It would then have scored the compliance prompts against the general baseline and
reported the difference as a regression. The eval now takes its domain from the dataset. Tests with
`ATLAS_DOMAIN=compliance` failed before that fix.

### 3.2 The law as data

**Where the text comes from.** EUR-Lex's website puts scripted downloads behind a bot check. The
Publications Office's content API serves the same Official Journal XHTML by CELEX number without one.
Each act is fetched once and committed gzipped, with its checksum, ELI and retrieval date. Everything
after that works offline, from text anyone can check against the Official Journal.

**How it is parsed.** The text becomes 2,426 units: articles, paragraphs, points, sub-points,
annexes and recitals. Each has a stable id and the label a citation uses ("AI Act Art. 6(3)", "GDPR
Art. 35(3)(a)"). The parse is pinned to facts that do not come from the parser: the counts of
articles, annexes and recitals, and the opening words of the provisions the domain leans on.

**Two parse defects found by those pins:**

- **Paragraphs of other regulations.** The AI Act's markup has 509 paragraph anchors, not 500. The
  other nine belong to other regulations, quoted inside the AI Act's amending articles (105–109). A
  parse that collected them would have filed one as "AI Act Art. 8(5)", a provision that does not
  exist, and the citation check would then have accepted it.
- **Superscripts.** The systemic-risk threshold in Art. 51(2) is 10 to the power 25. Flattened, it
  read "1025", off by twenty-two orders of magnitude, in the text the check treats as ground truth.
  Exponents are now written `10^25`. Footnote markers, which broke verbatim quotation, are dropped.

### 3.3 Retrieval without embeddings, until measured

The library is searched in order:

1. a reference the question names ("Article 6") is looked up, not ranked;
2. a question asking what a term means gets the definition;
3. BM25 ranks the passages, within the act the question names.

Embeddings were deferred until a measurement said lexical search fell short. Each change was kept
only if it raised recall@10 on the 40 compliance questions' must-cite provisions (`atlas_eval
retrieval`, recorded in commit `dafb3ca`):

| Change | recall@10 |
|---|---|
| BM25 over paragraphs, both acts | 0.500 |
| Rank only within the act the question names | 0.525 |
| Rank a paragraph on its own words, not its points' | 0.662 |
| Index every point and sub-point as its own passage | 0.688 |
| Read "EU" as "Union", the acts' own word | 0.700 |
| Definitions for questions that ask what a term means | 0.750 |
| A glossary of everyday words for the acts' own (below) | **0.900** |

Down-weighting recitals was tried and left out: no change at two weights and a loss at a third.

**The glossary was measured on questions it was not shaped on.** By 0.750, the misses were mostly
vocabulary: a question says "chatbot", and the AI Act says "AI systems intended to interact directly
with natural persons". `atlas/library/glossary.py` maps eight such phrases to the acts' words and adds
them to the query, never in place of it. Before building it, the 40 questions were split by id and
the split committed:
- odd ids, to develop on;
- even ids, held out and measured once at the end.

| | Before | After |
|---|---|---|
| Dev half, the 20 it was shaped on | 0.700 | 0.925 |
| **Held-out half, measured once** | 0.800 | **0.875** |
| All 40 | 0.750 | 0.900; recall@5 0.625 → 0.713 |

The held-out gain is a third of the dev gain. That gap is the overfitting the split exists to
expose. One caveat: the list of missed labels across all 40 had been seen before the split,
though not the held-out questions themselves.

The remaining misses mostly need reasoning rather than vocabulary: that screening applicants is
"evaluation of personal aspects", or that a transfer needs "appropriate safeguards". For those, a
query rewrite is built and switched off (`QUERY_REWRITE`, `atlas/library/rewrite.py`). One model
call names the legal concepts a question turns on, in the acts' words, and they are added to the
search.

**The rewrite is held to vocabulary.** Article numbers and act names the model writes are dropped:
- which provision a question is about comes from the question, so a rewrite naming "Article 52",
  the proposal's numbering, cannot steer the lookup;
- which act it is about also comes from the question, so a rewrite naming the GDPR cannot widen an
  AI Act question.

It is adopted only if `atlas_eval retrieval --rewrite --split test` shows it does better on the
held-out half. That needs credit.

One caveat: this is one search per question. A full run searches once per planned branch, and its
retrieval has not been measured separately.

### 3.4 A citation check that is not a judgement

Each bracketed citation in an answer faces three tests, with no model involved:

- **exists**: the act has the provision. "AI Act Art. 114" does not; the act has 113 articles.
- **retrieved**: a researcher in this run read it. Citing the article whose paragraph was read is
  allowed, as is citing a point inside a paragraph that was read.
- **quoted faithfully**: words in quotation marks beside the citation appear in the provision, after
  normalising what typesetting changes (curly quotes, case, elisions).

**How it is enforced.** The critic runs the check after its own review. A failed check sends the
answer back for a rewrite with the failures spelled out, *whatever the model critic said*. A model can
approve an answer citing an article that does not exist; this check is not a judgement, so it does
not defer to one. When the rounds run out, the answer goes out marked `unverified_citation` rather
than silently.

**Then, a check of what the citation says.** The first live run through the new workspace cited
`[AI Act Art. 2(3)]` for the act reaching providers outside the Union. That paragraph is the
national-security exclusion; the rule meant is Art. 2(1)(c). The provision existed and had been
retrieved, so the three tests passed it. The model critic, asked to hold claims to their provisions,
passed it too. Telling the two apart takes reading, so a model does it, as narrowly as possible
(`atlas/library/support.py`):

- **It sees one claim at a time.** For each citation that passed the three tests, the judge gets
  one sentence and the cited provision's own text. It never sees the whole answer or its sources.
- **"Supported" must quote the provision's words, verbatim.** The quote is checked against the
  text with the same matcher as quotations in answers. A support claim whose quote is not there is
  *unverified*. That is not evidence the citation is wrong, so nothing is sent back for it; nor is
  it evidence it is right, so it is not counted as supported.
- **Only an explicit "does not support" sends a draft back.** The judge's one-line reason goes to
  the rewrite. Out of rounds, the answer goes out marked `unsupported_citation`.
- **Its cost is bounded.** One call per critic round, run alongside the critic's own review, up to
  12 citations, with the rest counted unchecked. A failed call is recorded as
  `citation_support_unavailable`, never read as a pass.

**How the UI and the API stay in step.** The UI shows each citation as a chip with the last check's
verdict. A support judgement is shown only on the draft it judged: draft *n* is read in critic
round *n*, and a reading of another draft says nothing about the answer on screen. The chip's parser is a TypeScript port of the backend's. It is held to a fixture the backend
writes from its own parser (`citation-cases.json`), so the two cannot drift apart silently.

### 3.5 Editing a branch forks a run

A person rewrites one researcher's question on a finished run. The run is not restarted:

1. LangGraph's `aupdate_state` forks the thread from the parent's final checkpoint.
2. The write goes through a dedicated `human_edit` node whose routing sends only the edited branch.
3. The other branches' research carries over through the upsert reducer.
4. The synthesizer and critic run as usual.

**Why not write the edit as a critic verdict?** That was the obvious alternative, and it was
rejected:

- the synthesizer looks for the latest critique of the write-up, and would have found the parent's;
- the researcher would have received the edit framed as a reviewer's rejection.

**What the fork must not inherit.** The parent's checkpoint holds state the revision must not
inherit. Each leak would produce a plausible answer built on someone else's work:

- earlier drafts, replayed as prior turns;
- the parent's verdicts;
- its spend, which the budget check sums;
- its iteration count, which would leave the critic no rounds;
- its answer.

Each has a test. A mutation sweep removed each reset in turn, and each removal failed a test.

### 3.6 Traces recorded, not reconstructed

The inspector shows each agent's exact prompt. It is recorded where every call passes through:
`atlas.models` for the model, the researcher's search helpers for the tools. A reconstruction from
state and today's templates would show what the prompt *would* have been, and would quietly diverge
the day a template changed.

**What travels where.**

- The runner stores each call whole.
- The live stream carries only a summary: who, tokens, timing and a 280-character preview. Replay
  resends the stream to every reconnecting client, and a synthesizer's prompt holds every branch's
  findings.
- Prompts never enter graph state.

**How calls are ordered.** Calls are listed in the order they started. On Windows the clock ticks in
milliseconds, so back-to-back calls tied, and the tie fell to a random id. Start times are now
strictly increasing within the process that runs a run.

### 3.7 Runs outlive Redis, and can be forgotten

Redis keeps a run for an hour, which is right for watching it live and wrong for a version opened
tomorrow. The following go to Postgres under one store contract, tested against a real database:

- runs;
- every call;
- every event stream, archived after the terminal event.

If the record cannot be written, the run is not failed. It is marked `audit_incomplete`: the answer
is fine, but the trail behind it is not complete.

**Deletion.** People's questions are personal data, and a tool that answers GDPR questions should
practise it.

- `DELETE /threads/{id}` deletes a conversation everywhere it lives: records, calls, archived events,
  checkpoints and Redis keys.
- A nightly job forgets threads idle past the retention period.
- A running thread is refused (409), never deleted from under its run.

### 3.8 The workspace

| Pane | Contents |
|---|---|
| **Timeline** | The versions of the answer, the run as four steps, and its timeline by step. |
| **Agents** | The graph, with an inspector under it: task, prompt, output, attempts. |
| **Answer** | The answer with its citation chips and verdict, copy as Markdown, a word-level diff against any other version; sources, research by branch and run details folded beneath. |

**Prompts load only when opened.** On a live run they come from the API. The `/demo` replay ships its
recorded calls instead, so it shows real prompts with no backend.

**The demo recording is real except for the model.** It runs through the real model layer with only
the model scripted, and each recorded call names its model `scripted`.

### 3.9 The domain eval

Three systems answer the 40 compliance questions:

| System | What it is |
|---|---|
| `plain` | One call from memory, asked to cite in the same bracketed form. |
| `single-call` | One library search and Atlas's synthesizer. |
| `atlas` | The full graph. |

**Choices made deliberately:**

- **Web search is off for all three.** The comparison is about citing the law, the law comes from
  the committed library, and a live search would make the runs unrepeatable. Each run file records
  `"web": "off"`.
- **Citations are scored deterministically**, against the library and each question's must-cite and
  may-cite provisions:
  - the share of answers citing a provision that does not exist (the headline);
  - the share citing nothing (so a system that never cites cannot look perfect);
  - validity, precision and recall.

  Each has a bootstrap CI, and systems are compared item by item against a reference run.
- **`plain` is judged on completion only.** Scored as it stood, the judge would have charged it 0.67
  for tool-use checks it cannot pass and graded its grounding against no evidence. Grounding and tool
  use are marked as not applying to it.

### 3.10 A question about neither act

Asked "history of cameroon", Atlas researched it, cited GDPR Recital 35, which mentions a medical
"history", and marked the citation verified: it exists, and it was retrieved. The citation check
answers whether a citation is real, not whether the answer is about the question.

**Word overlap cannot make the call.** Measured before anything was built, on 64 questions about
the law and 80 that are not: the best-matching off-topic question (the DSA, 27.9) outscored 56 of
the 64 in-scope ones, and the core GDPR question "we had a data breach, who do we
have to tell and how fast?" shared less of its words with the law than "How do I file my income
taxes in the United States?".

**So the planner judges, in the plan it writes anyway.** `PlannerOutput` gained one field,
`out_of_scope`, and the compliance planner one rule, biased to plan when in doubt. A question
judged out of scope ends at the plan with a `scope_declined` event and an answer that says why.
Three things hold the judgment to account:

- **The guard.** A question that names an act, cites a provision or asks what a defined term means
  is planned whatever the planner says. Measured, it holds for 37 of the 40 compliance questions
  and 6 of the 24 casual ones, and for none of the 80 off-topic ones.
- **Only a conversation's first question is judged**, found from the thread's checkpoint. A
  follow-up reads in the light of what came before ("and from when?").
- **Research anyway** asks the question again in its conversation, which makes it a follow-up. No
  flag reaches the worker; the rule above is the override.

Offline mode has no model to judge, so it scripts the judgment from the library's best match, with
a floor set below every in-scope question measured. It turns away none of the 64 in-scope questions
and catches only 9 of the 80 off-topic ones, in-sample, since the floor was chosen looking at every
set. The planner's judgment, which does the real work, is not measured yet.

The measurement also found a retrieval gap: "what is a DPIA" matched nothing at all. The glossary
now knows DPIA and FRIA. No eval question uses either abbreviation, so the measured retrieval
numbers are unchanged, and the held-out half is still measured once.

### 3.11 Reviewing a contract against the law

The same bet as the citation check, pointed at a document the user brings. GDPR Art. 28(3) lists the
terms a processor contract must stipulate; legal teams check vendor contracts against it by hand,
and DORA Art. 30 asks the same of every ICT service contract a bank signs.

**The law is the checklist.** Each of the nine requirements names its provision by label, and its
text comes from the library: what a contract is held to is the law as published, never a prompt's
paraphrase.

**A model reads; the contract's words decide.** The contract is cut into clauses at its own
numbered headings, each with a label a reviewer can find. Each requirement is judged in its own
call against the clauses ranked most like it (BM25, as the library ranks the law). A "met" or
"partial" must quote the contract, and the quote is looked for in the clause it cites with the same
test the law's quotations get. One that is not there is **unverified**, never met. A requirement no
clause shares a word with is missing, without a model call.

**A review is a run.** It is a small graph of its own, run by the same runner: its events stream
and replay, every judgment's prompt is kept under its requirement for the inspector, and the run is
kept in the record as a review.

**A person signs it off.** Each finding can be agreed or disagreed with, with a note and the
reviewer's name. A changed decision is a new row, never an overwrite, so the record says who
decided what, and when. Atlas has no sign-in, so the name is kept as given.

**Measured, free:** for 56 of 56 terms met or partly met in eight contracts with known gaps, the
clause that meets it is among those the model is shown. It was 54: "keep the data safe" and
"subcontractors" shared no word with the law, so each requirement now carries the words contracts
use for it. That fix was made on these contracts, so the number is in-sample. **Not measured:** the
review's own accuracy, which `atlas_eval contracts --model` measures in 72 calls.

Looking at the first sample in the browser found two reading bugs, both now tested: a wrapped line
starting "Article 32 of the GDPR" was taken for a clause heading, and a wrapped first sentence for a
clause's title.

### 3.12 A general question, said to be one

A question about neither act can still be asked: "Ask anything" researches it from the web, in the
general domain. The domain is chosen when a conversation starts and kept by every run on it: a
follow-up or a revision cannot switch it, since the answer it builds on was researched under the
other's rules. A general answer is marked as such, its citations not checked against the law.

## 4. Failures that looked like success

The class of bug this project keeps producing is the component that silently does nothing, or the
wrong thing, while everything looks fine. These were found during this work:

| What looked fine | What was wrong | How it was caught |
|---|---|---|
| The library parsed | Nine quoted paragraphs from other regulations would have become phantom AI Act provisions that the check accepts | Golden counts: 509 anchors where 500 were expected |
| Art. 51(2) read normally | "10^25" had become "1025" | Reading the provisions the domain leans on, pinned in tests |
| The general eval would have run | Under the new default it would have scored the compliance prompts and reported a regression | Tests with `ATLAS_DOMAIN=compliance`, failing first |
| The graph drew | The critic was drawn before the planner: cycle detection started at whichever edge the API listed first | Looking at the demo in a browser; then a test over three edge orders |
| Attempts listed their searches | Previews were Python reprs, `[{'label': ...` | Looking at the inspector |
| Calls were listed in order | Windows' clock gave back-to-back calls the same time; ties fell to a random id | The demo's exported calls put a critic before the draft it reviewed; then a test of 2,000 timestamps |
| The component tests typechecked | Two never had: on a case-insensitive disk `Inspector.test.tsx` and `inspector.test.ts` share a name, and TypeScript keeps only the `.ts` | A required prop added without a type error |
| The demo showed agents | Its recording bypassed the model layer, so it held no prompts to show | Opening the Prompt tab on the demo |
| A compliance baseline could be scored | The judge charged `plain` for tool use and grounding it cannot have | A test, failing first, at 0.67 |
| A live answer's citations all passed | One cited the national-security exclusion, Art. 2(3), for extraterritorial reach, Art. 2(1)(c); real, retrieved, and wrong | Reading the provision a chip linked to. That reading is now the support check |
| The demo's citations were "supported" | The demo's scripted judge said "supported" for any provision it had no script for. After the glossary improved retrieval, it passed Art. 6(1) for a CV tool | Re-recording after the search change and reading the new first draft. An unscripted provision now stops the recording, and offline mode judges nothing |
| A deleted run's stream was gone | The Redis test double's DEL ignored streams, so the tests could not tell | Making the double behave like Redis |
| `DELETE /threads/{id}` was queued | A renamed worker function would have left every deletion queued for nobody | A test pinning the job's registered name |
| An off-topic question got a verified answer | "history of cameroon" was researched, and GDPR Recital 35 cited and marked verified: real, retrieved, and about health data | A person asking it in the app. The planner now judges scope (§3.10) |
| A finished review served by `GET /runs/{id}` | It read every stored result as a research answer: a review's crashed it with a 500 | A test, failing first, before the run page's stream could hit it |
| A contract's clauses were read | Clause 4 wrapped onto "Article 32 of the GDPR", which became a clause of its own | Looking at the first sample's clauses before recording its review |
| "What is a DPIA" was a short question | It matched no provision: the abbreviation is practice's, not the GDPR's | Measuring the scope check, which needed every in-scope question's best match |

## 5. Measured and not measured

| | |
|---|---|
| Library retrieval, one search per question | recall@10 0.900 (held-out half 0.875, measured once), recall@5 0.713, all must-cites for 32 of 40 |
| Engine regression gate, 50 general questions | completion 0.995, grounding 0.955, tool use 0.963; thresholds 5 / 4 / 3 pp from a measured noise floor |
| Contract review, candidate search | 56 of 56 terms met or partly met have their clause among those the model is shown, in eight contracts with known gaps (in-sample) |
| Tests | 642 backend, 16 more against Postgres, 219 frontend; no network, no spend |
| **Not measured: the domain eval** | Built and tested; needs about $12–15 of model calls (estimated) |
| **Not measured: Atlas's own retrieval** | One search per branch, rather than per question |
| **Not measured: the contract review's accuracy** | `atlas_eval contracts --model`, 72 calls: gaps caught, missing terms passed as met, false alarms |
| **Not measured: the planner's scope judgment** | `atlas_eval scope --planner` on 144 labelled questions, one planner call each: about 150k tokens, roughly $1 (estimated) |
| **Not measured: the support check's accuracy** | `atlas_eval support` measures it on 24 labelled cases (12 supported, 12 near misses), each "supports" label proved by the provision's own words; needs credit |
| **Not done: dataset review** | The 40 compliance questions were drafted by a model; none has been reviewed by a person |

## 6. Limitations

- **The acts are as originally published.** They do not include later amendments, such as the
  Digital Omnibus proposals, which may move the AI Act's high-risk dates. Only the printed Official
  Journal is authentic.
- **The dataset is unreviewed.** Its expected citations are checked against the library, but the
  questions and rubrics are a model's drafts until a person reviews them, and the README says so.
  A second read by the drafting model checked every rubric line against the text of the provisions
  it cites, and 15 of its numbers and phrasings word for word, and found nothing to correct. That
  is a model checking its own work, not a review. Three lines rest on interpretation rather than
  the text, and a reviewer should look at them first:
  - c06 and c40 assume the company built its chatbot, so the disclosure duty falls on it as provider;
  - c33 says payroll is not occasional processing, which comes from regulators' guidance.
- **The eval measures citing the law, not guidance.** With the web off, it does not measure how
  well Atlas uses official guidance.
- **Whether a provision supports its sentence is a judgement.** The deterministic check stops at
  existence, retrieval and quotation. The support check goes further, but it is a model reading, held
  to quotes it cannot fake and not yet measured. A wrong "does not support" costs a rewrite; a missed
  one goes out as before.
- **One person.** The dataset, the labels and the review are one person's.

## 7. Reproducing

From `backend/`:

```powershell
uv run pytest -q                                          # 642 tests, offline
uv run pytest -q -m postgres                              # with ATLAS_TEST_DATABASE_URL set
uv run python -m atlas_eval retrieval                     # recall@10, free
uv run python -m atlas_eval claim --dataset compliance    # what the dataset supports claiming
uv run python scripts/capture_fixtures.py                 # re-record the frontend's fixtures and demo
```

From `frontend/`: `npm test`, then `npm run dev` and open `/demo`.
