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
| Definitions for questions that ask what a term means | **0.750** |

At that point every must-cite provision is found for 26 of 40 questions; recall@5 is 0.625.
Down-weighting recitals was tried and left out: no change at two weights and a loss at a third.

The remaining misses are mostly paraphrase, a "chatbot" for an AI system that interacts with
people. The next fix to try is one model call that rewrites the question into the act's vocabulary,
measured the same way.

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
| **Evolution** | The versions of the answer, and the run's timeline. |
| **Agents** | The graph, with an inspector under it: task, prompt, output, attempts. |
| **Result** | The answer with its citation chips, sources by authority, and a word-level diff against any other version. |

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
| A deleted run's stream was gone | The Redis test double's DEL ignored streams, so the tests could not tell | Making the double behave like Redis |
| `DELETE /threads/{id}` was queued | A renamed worker function would have left every deletion queued for nobody | A test pinning the job's registered name |

## 5. Measured and not measured

| | |
|---|---|
| Library retrieval, one search per question | recall@10 0.750, recall@5 0.625, all must-cites for 26 of 40 |
| Engine regression gate, 50 general questions | completion 0.995, grounding 0.955, tool use 0.963; thresholds 5 / 4 / 3 pp from a measured noise floor |
| Tests | 527 backend, 13 more against Postgres, 135 frontend; no network, no spend |
| **Not measured: the domain eval** | Built and tested; needs about $12–15 of model calls (estimated) |
| **Not measured: Atlas's own retrieval** | One search per branch, rather than per question |
| **Not measured: the support check's accuracy** | How often its "does not support" is right, and how often it misses one |
| **Not done: dataset review** | The 40 compliance questions were drafted by a model; none has been reviewed by a person |

## 6. Limitations

- **The acts are as originally published.** They do not include later amendments, such as the
  Digital Omnibus proposals, which may move the AI Act's high-risk dates. Only the printed Official
  Journal is authentic.
- **The dataset is unreviewed.** Its expected citations are checked against the library, but the
  questions and rubrics are a model's drafts until a person reviews them, and the README says so.
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
uv run pytest -q                                          # 515 tests, offline
uv run pytest -q -m postgres                              # with ATLAS_TEST_DATABASE_URL set
uv run python -m atlas_eval retrieval                     # recall@10, free
uv run python -m atlas_eval claim --dataset compliance    # what the dataset supports claiming
uv run python scripts/capture_fixtures.py                 # re-record the frontend's fixtures and demo
```

From `frontend/`: `npm test`, then `npm run dev` and open `/demo`.
