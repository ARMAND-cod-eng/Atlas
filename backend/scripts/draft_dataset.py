"""Draft the evaluation dataset.

    uv run python scripts/draft_dataset.py

**This script is the provenance record.** The 50 queries and their candidate
rubrics were drafted by a model — this file — and every item is written with
``reviewed_by: null``. Nothing here sets it. A human marks an item reviewed only
by going through ``atlas_eval review``, and ``provenance_summary`` computes what
the dataset then supports being called. If review is incomplete, the README says
"model-drafted, human-reviewed" rather than "hand-labeled", and a test enforces
that. The claim cannot outrun the work.

Once any item has been reviewed, this script refuses to overwrite the dataset:
the reviewed file is the source of truth from then on, and review may have
corrected items, so it can legitimately differ from the drafts below.

Rubric entries are deliberately short — 3 to 5 required claims each. A rubric is a
checklist of what an answer must contain, not a model answer: scoring against
prose measures phrasing, scoring against a checklist measures content.
"""

from __future__ import annotations

from atlas_eval.dataset import (
    DATASET_PATH,
    META_PATH,
    Dataset,
    DatasetMeta,
    EvalItem,
    ExpectedSubtasks,
    Provenance,
    RubricEntry,
    load_dataset,
    provenance_summary,
    write_jsonl,
)

# (id, query, min, max, recency, tags, [(claim, check, patterns|number)], wrong_but_plausible)
Draft = tuple[str, str, int, int, str, list[str], list[tuple[str, str, object]], str]

DRAFTS: list[Draft] = [
    # --- 1-2 subtasks: single-fact and narrow-scope (18) ----------------------
    (
        "q01",
        "What is retrieval-augmented generation?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("Retrieval happens before generation", "judge", None),
            ("An external knowledge source or corpus is queried", "judge", None),
            ("Retrieved text is placed in the model's context", "judge", None),
        ],
        "Describing fine-tuning as if it were RAG, or claiming the model's weights are updated.",
    ),
    (
        "q02",
        "What does the acronym BM25 stand for and what is it used for?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("BM stands for Best Matching", "deterministic", ["best match"]),
            ("It is a ranking function for information retrieval", "judge", None),
            ("It scores documents by term frequency and inverse document frequency", "judge", None),
        ],
        "Confusing BM25 with a neural embedding model or with TF-IDF exactly.",
    ),
    (
        "q03",
        "Who wrote the paper 'Attention Is All You Need' and in what year was it published?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("Published in 2017", "deterministic", None, 2017),
            ("Authors were at Google", "judge", None),
            ("Vaswani is named as an author", "deterministic", ["vaswani"]),
        ],
        "Attributing it to OpenAI, or giving 2018 (the BERT year).",
    ),
    (
        "q04",
        "What is a vector database?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("Stores high-dimensional embedding vectors", "judge", None),
            ("Supports approximate nearest-neighbour similarity search", "judge", None),
        ],
        "Describing a relational database with a vector column as the whole story.",
    ),
    (
        "q05",
        "What is the context window of GPT-4o?",
        1,
        2,
        "current",
        ["simple", "recency"],
        [("128,000 tokens", "deterministic", ["128,000", "128000", "128k"])],
        "Quoting 32k (GPT-4's older window) or 1M (a Gemini figure).",
    ),
    (
        "q07",
        "What is the difference between precision and recall?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("Precision is the fraction of retrieved items that are relevant", "judge", None),
            ("Recall is the fraction of relevant items that are retrieved", "judge", None),
            ("There is a trade-off between them", "judge", None),
        ],
        "Swapping the two definitions, which is the classic error.",
    ),
    (
        "q08",
        "What is HNSW and why is it used in vector search?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            (
                "Hierarchical Navigable Small World",
                "deterministic",
                ["hierarchical navigable small world"],
            ),
            ("It is an approximate nearest-neighbour index", "judge", None),
            ("It trades exactness for speed", "judge", None),
        ],
        "Presenting it as exact search, or confusing it with IVF.",
    ),
    (
        "q10",
        "What is the purpose of a re-ranker in a retrieval pipeline?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("It reorders an initial candidate set", "judge", None),
            ("It is more accurate but more expensive than first-stage retrieval", "judge", None),
        ],
        "Describing it as replacing first-stage retrieval rather than refining it.",
    ),
    (
        "q11",
        "What is prompt injection?",
        1,
        2,
        "evergreen",
        ["simple", "security"],
        [
            ("Untrusted input contains instructions the model follows", "judge", None),
            ("It can subvert the developer's original instructions", "judge", None),
        ],
        "Confusing it with SQL injection, or with jailbreaking specifically.",
    ),
    (
        "q13",
        "What is the current stable version of Python?",
        1,
        2,
        "current",
        ["simple", "recency"],
        [
            ("A 3.x version is named", "judge", None),
            ("The version given is 3.12 or later", "judge", None),
        ],
        "Quoting 3.9 or 3.10 from stale training data rather than from the sources.",
    ),
    (
        "q14",
        "What is an embedding model?",
        1,
        2,
        "evergreen",
        ["simple"],
        [
            ("Maps text to a fixed-length numeric vector", "judge", None),
            ("Semantically similar text maps to nearby vectors", "judge", None),
        ],
        "Describing a generative model instead.",
    ),
    (
        "q15",
        "What is the difference between a cross-encoder and a bi-encoder?",
        2,
        3,
        "evergreen",
        ["moderate"],
        [
            ("A bi-encoder encodes query and document separately", "judge", None),
            ("A cross-encoder encodes them jointly", "judge", None),
            ("Bi-encoders allow precomputed document embeddings", "judge", None),
            ("Cross-encoders are more accurate but slower", "judge", None),
        ],
        "Getting the direction backwards, or claiming cross-encoders can be precomputed.",
    ),
    (
        "q16",
        "What is Server-Sent Events and how does it differ from WebSockets?",
        2,
        3,
        "evergreen",
        ["moderate"],
        [
            ("SSE is unidirectional, server to client", "judge", None),
            ("WebSockets are bidirectional", "judge", None),
            ("SSE runs over plain HTTP", "judge", None),
            ("SSE reconnects automatically", "judge", None),
        ],
        "Claiming SSE is bidirectional, or that it needs a special protocol upgrade.",
    ),
    (
        "q17",
        "What is Cohen's kappa and why is it preferred over raw agreement?",
        2,
        3,
        "evergreen",
        ["moderate", "stats"],
        [
            ("It measures inter-rater agreement", "judge", None),
            ("It corrects for agreement expected by chance", "judge", None),
            ("It can be misleading when one class dominates", "judge", None),
        ],
        "Presenting kappa as always superior without noting the prevalence problem.",
    ),
    (
        "q18",
        "What is the CAP theorem?",
        2,
        3,
        "evergreen",
        ["moderate"],
        [
            (
                "Consistency, Availability, Partition tolerance",
                "deterministic",
                ["partition tolerance"],
            ),
            ("Only two of the three can be guaranteed during a partition", "judge", None),
        ],
        "Stating you must permanently pick two, rather than the choice arising under partition.",
    ),
    # --- 3 subtasks: comparative and mechanism questions (16) -----------------
    (
        "q19",
        "Compare BM25 and dense vector retrieval for technical documentation search.",
        3,
        4,
        "evergreen",
        ["moderate", "comparison"],
        [
            ("BM25 relies on lexical term overlap", "judge", None),
            ("Dense retrieval captures semantic similarity", "judge", None),
            ("BM25 handles rare exact terms such as error codes well", "judge", None),
            ("Hybrid approaches combine both", "judge", None),
        ],
        "Declaring dense retrieval strictly better, ignoring exact-match failure modes.",
    ),
    (
        "q20",
        "How does PostgreSQL's MVCC work and what are its trade-offs?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("Multiple row versions coexist", "judge", None),
            ("Readers do not block writers", "judge", None),
            ("Dead tuples require VACUUM", "judge", None),
            ("Table bloat is a consequence", "judge", None),
        ],
        "Omitting VACUUM and bloat, which is the actual operational cost.",
    ),
    (
        "q21",
        "What are the trade-offs between fine-tuning and RAG for domain adaptation?",
        3,
        4,
        "evergreen",
        ["moderate", "comparison"],
        [
            ("RAG updates knowledge without retraining", "judge", None),
            ("Fine-tuning changes model behaviour and style", "judge", None),
            ("RAG makes provenance traceable to sources", "judge", None),
            ("Fine-tuning has higher upfront cost", "judge", None),
        ],
        "Framing them as mutually exclusive rather than complementary.",
    ),
    (
        "q22",
        "How do Redis Streams differ from Redis Pub/Sub?",
        3,
        4,
        "evergreen",
        ["moderate", "comparison"],
        [
            ("Pub/Sub drops messages with no subscriber", "judge", None),
            ("Streams persist entries", "judge", None),
            ("Streams support consumer groups", "judge", None),
            ("Streams allow reading history by id", "judge", None),
        ],
        "Treating them as interchangeable, missing that Pub/Sub has no durability.",
    ),
    (
        "q23",
        "What causes the 'lost in the middle' effect in long-context language models?",
        3,
        4,
        "current",
        ["moderate", "recency"],
        [
            ("Accuracy is higher for information at the start or end", "judge", None),
            ("Performance follows a U-shaped curve by position", "judge", None),
            ("It persists in models marketed as long-context", "judge", None),
        ],
        "Claiming larger context windows have solved it.",
    ),
    (
        "q24",
        "How should a team decide between a monorepo and multiple repositories?",
        3,
        4,
        "evergreen",
        ["moderate", "comparison"],
        [
            ("Monorepos simplify atomic cross-project changes", "judge", None),
            ("Monorepos need more tooling investment", "judge", None),
            ("Separate repos give independent versioning", "judge", None),
        ],
        "Presenting one as universally correct without naming the conditions.",
    ),
    (
        "q25",
        "What is the difference between OAuth 2.0 and OpenID Connect?",
        3,
        4,
        "evergreen",
        ["moderate", "comparison"],
        [
            ("OAuth 2.0 is for authorisation", "judge", None),
            ("OpenID Connect adds authentication", "judge", None),
            ("OIDC is a layer on top of OAuth 2.0", "judge", None),
            ("OIDC introduces the ID token", "judge", None),
        ],
        "Calling OAuth 2.0 an authentication protocol, the single most common error here.",
    ),
    (
        "q26",
        "How does HTTP/3 differ from HTTP/2, and what problem does it solve?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("HTTP/3 runs over QUIC", "deterministic", ["quic"]),
            ("QUIC uses UDP", "deterministic", ["udp"]),
            ("It removes TCP head-of-line blocking", "judge", None),
        ],
        "Saying HTTP/3 eliminates head-of-line blocking entirely, including at the stream level.",
    ),
    (
        "q27",
        "What are the main approaches to LLM evaluation and what does each miss?",
        3,
        5,
        "evergreen",
        ["moderate", "meta"],
        [
            ("Reference-based metrics compare to a gold answer", "judge", None),
            ("LLM-as-judge scales but needs calibration", "judge", None),
            ("Human evaluation is the most reliable and least scalable", "judge", None),
            ("Benchmarks can be contaminated by training data", "judge", None),
        ],
        "Presenting LLM-as-judge as objective without mentioning calibration or bias.",
    ),
    (
        "q28",
        "How does Kubernetes decide where to schedule a pod?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("Filtering removes infeasible nodes", "judge", None),
            ("Scoring ranks the remaining nodes", "judge", None),
            ("Resource requests influence placement", "judge", None),
            ("Affinity and taints constrain placement", "judge", None),
        ],
        "Describing it as simple round-robin or purely least-loaded.",
    ),
    (
        "q29",
        "What is the difference between a database index scan and a sequential scan, and when is each faster?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("Index scans use an auxiliary structure", "judge", None),
            ("Sequential scans read the whole table", "judge", None),
            ("Sequential scans can win for high-selectivity queries", "judge", None),
        ],
        "Claiming an index is always faster, which is the classic misconception.",
    ),
    (
        "q30",
        "How do async event loops handle blocking I/O, and what goes wrong if you block one?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("The loop runs coroutines cooperatively on one thread", "judge", None),
            ("A blocking call stalls every other task", "judge", None),
            ("Blocking work belongs in a thread or process pool", "judge", None),
        ],
        "Suggesting async makes blocking calls safe automatically.",
    ),
    (
        "q31",
        "What are the trade-offs of using JWTs for session management?",
        3,
        4,
        "evergreen",
        ["moderate", "security"],
        [
            ("JWTs are stateless and need no server-side lookup", "judge", None),
            ("They cannot easily be revoked before expiry", "judge", None),
            ("Short expiry plus refresh tokens is the usual mitigation", "judge", None),
        ],
        "Recommending JWTs for sessions without mentioning revocation at all.",
    ),
    (
        "q32",
        "How does Docker's layered filesystem work and why does layer ordering matter?",
        3,
        4,
        "evergreen",
        ["moderate"],
        [
            ("Each instruction creates a layer", "judge", None),
            ("Layers are cached and reused", "judge", None),
            ("A changed layer invalidates all later layers", "judge", None),
        ],
        "Omitting cache invalidation, which is the entire practical consequence.",
    ),
    # --- 4-5 subtasks: genuinely multi-hop (16) -------------------------------
    (
        "q35",
        "Compare RAG and long-context LLMs for enterprise document QA across accuracy, cost, latency and operational complexity.",
        4,
        5,
        "current",
        ["multihop", "comparison", "recency"],
        [
            ("RAG retrieves a subset before generating", "judge", None),
            ("Long-context models place whole documents in the prompt", "judge", None),
            ("Long-context inference cost grows with input length", "judge", None),
            ("RAG adds indexing and retrieval infrastructure", "judge", None),
            ("Accuracy depends on retrieval quality for RAG", "judge", None),
        ],
        "Declaring a universal winner instead of naming the conditions under which each wins.",
    ),
    (
        "q36",
        "What should a team consider when migrating a monolith to microservices, and what commonly goes wrong?",
        4,
        5,
        "evergreen",
        ["multihop"],
        [
            ("Service boundaries should follow domain boundaries", "judge", None),
            ("Distributed transactions become difficult", "judge", None),
            ("Operational complexity increases substantially", "judge", None),
            ("A distributed monolith is a common failure mode", "judge", None),
        ],
        "Listing only benefits, presenting microservices as a straightforward upgrade.",
    ),
    (
        "q37",
        "How do modern LLM providers price their APIs, and what drives the cost of a request?",
        4,
        5,
        "current",
        ["multihop", "recency"],
        [
            ("Pricing is per token", "judge", None),
            ("Input and output tokens are priced differently", "judge", None),
            ("Output tokens cost more than input tokens", "judge", None),
            ("Prompt caching can reduce repeated input cost", "judge", None),
        ],
        "Quoting specific per-million prices from memory rather than from the retrieved sources.",
    ),
    (
        "q38",
        "What are the security considerations when building an agent that can call external tools?",
        4,
        5,
        "evergreen",
        ["multihop", "security"],
        [
            ("Untrusted content can carry injected instructions", "judge", None),
            ("Tool permissions should be least-privilege", "judge", None),
            ("Irreversible actions warrant confirmation", "judge", None),
            ("Tool outputs should be treated as data, not instructions", "judge", None),
        ],
        "Treating prompt filtering as sufficient mitigation on its own.",
    ),
    (
        "q39",
        "How has the transformer architecture changed since 2017, and which changes mattered most?",
        4,
        5,
        "evergreen",
        ["multihop"],
        [
            ("The original used absolute positional encodings", "judge", None),
            ("Rotary or relative position encodings became common", "judge", None),
            ("Attention efficiency work such as FlashAttention reduced memory cost", "judge", None),
            (
                "Mixture-of-experts increased parameter count without proportional compute",
                "judge",
                None,
            ),
        ],
        "Listing every published variant as equally important, with no sense of what stuck.",
    ),
    (
        "q40",
        "What does it take to run a Postgres database reliably in production?",
        4,
        5,
        "evergreen",
        ["multihop"],
        [
            ("Backups must be tested by restoring them", "judge", None),
            ("Replication provides failover", "judge", None),
            ("Connection pooling is needed at scale", "judge", None),
            ("VACUUM and autovacuum tuning matter", "judge", None),
            ("Monitoring of slow queries and bloat is required", "judge", None),
        ],
        "Covering backups only, omitting vacuum and connection limits.",
    ),
    (
        "q41",
        "Compare the major approaches to distributed consensus and where each is used.",
        4,
        5,
        "evergreen",
        ["multihop", "comparison"],
        [
            ("Paxos is the foundational algorithm", "judge", None),
            ("Raft was designed for understandability", "judge", None),
            ("Raft uses leader election and a replicated log", "judge", None),
            ("etcd or Consul use Raft in practice", "judge", None),
        ],
        "Conflating consensus with two-phase commit.",
    ),
    (
        "q42",
        "How do observability practices differ between traces, metrics and logs, and when do you reach for each?",
        4,
        5,
        "evergreen",
        ["multihop", "comparison"],
        [
            ("Metrics are aggregated numeric time series", "judge", None),
            ("Logs are discrete timestamped events", "judge", None),
            ("Traces follow one request across services", "judge", None),
            ("Traces are the tool for latency attribution in distributed systems", "judge", None),
        ],
        "Treating the three as interchangeable telemetry.",
    ),
    (
        "q43",
        "What are the practical differences between the major Python packaging tools in current use?",
        4,
        5,
        "current",
        ["multihop", "recency", "comparison"],
        [
            ("pip is the baseline installer", "judge", None),
            ("Poetry and uv provide lockfile-based resolution", "judge", None),
            ("uv is substantially faster", "judge", None),
            ("pyproject.toml is the shared configuration format", "judge", None),
        ],
        "Recommending setup.py-era workflows as current best practice.",
    ),
    (
        "q44",
        "How should an engineering team approach reducing cloud spend without degrading reliability?",
        4,
        5,
        "evergreen",
        ["multihop"],
        [
            ("Measure and attribute spend before cutting", "judge", None),
            ("Right-sizing over-provisioned resources is a common win", "judge", None),
            ("Reserved or committed-use pricing reduces steady-state cost", "judge", None),
            ("Reliability targets should be set before optimising", "judge", None),
        ],
        "Recommending aggressive cuts without any measurement step.",
    ),
    (
        "q45",
        "What are the failure modes of vector search at scale, and how are they mitigated?",
        4,
        5,
        "evergreen",
        ["multihop"],
        [
            ("Approximate indexes trade recall for speed", "judge", None),
            ("Index rebuild cost grows with corpus size", "judge", None),
            ("Embedding drift after a model change requires re-embedding", "judge", None),
            ("Filtering combined with ANN search can degrade recall", "judge", None),
        ],
        "Presenting vector search as a solved, drop-in component.",
    ),
    (
        "q46",
        "How do rate limiting algorithms differ, and which suits an API gateway?",
        4,
        5,
        "evergreen",
        ["multihop", "comparison"],
        [
            ("Token bucket permits bursts", "judge", None),
            ("Leaky bucket smooths output rate", "judge", None),
            ("Fixed windows suffer boundary spikes", "judge", None),
            ("Sliding windows reduce the boundary problem", "judge", None),
        ],
        "Describing only fixed-window counters, missing the boundary burst problem.",
    ),
    # --- adversarial (8) -------------------------------------------------------
    (
        "q47",
        "Why is MongoDB always faster than PostgreSQL?",
        2,
        4,
        "evergreen",
        ["adversarial", "false-premise"],
        [
            ("The premise is challenged or corrected", "judge", None),
            ("Relative performance depends on workload and access pattern", "judge", None),
            ("Postgres can outperform MongoDB for some workloads", "judge", None),
        ],
        "Accepting the premise and producing a list of reasons MongoDB is faster.",
    ),
    (
        "q48",
        "When did Python 4 introduce static typing?",
        1,
        3,
        "evergreen",
        ["adversarial", "false-premise"],
        [
            ("Python 4 has not been released", "judge", None),
            ("Type hints were introduced in Python 3.5 via PEP 484", "judge", None),
            ("Type hints are not enforced at runtime", "judge", None),
        ],
        "Inventing a Python 4 release date and a feature list for it.",
    ),
    (
        "q49",
        "What is the best programming language?",
        1,
        3,
        "evergreen",
        ["adversarial", "ambiguous"],
        [
            ("The question is identified as underspecified", "judge", None),
            ("Suitability depends on the task and constraints", "judge", None),
        ],
        "Confidently naming one language as objectively best.",
    ),
    (
        "q50",
        "Is nuclear power safe?",
        3,
        5,
        "evergreen",
        ["adversarial", "contested", "ambiguous"],
        [
            (
                "The answer notes that sources disagree or that it depends on the metric",
                "judge",
                None,
            ),
            ("Deaths per unit of energy is one framing used", "judge", None),
            ("Waste storage and accident risk are raised as concerns", "judge", None),
        ],
        "Giving a flat yes or no as though the question had a settled factual answer.",
    ),
]

# Items where the correct behaviour is to report partial failure rather than
# answer confidently. Deliberately obscure or unresolvable from public sources.
FAILURE_EXPECTED: list[Draft] = [
    (
        "q51",
        "What was the exact internal codename of the Atlas project's Phase 7 milestone?",
        1,
        3,
        "evergreen",
        ["failure-expected"],
        [
            ("The answer states the information could not be found", "judge", None),
            ("No specific codename is fabricated", "judge", None),
        ],
        "Inventing a plausible-sounding codename with a confident tone.",
    ),
    (
        "q52",
        "How many unreported production incidents did Anthropic have in March 2024?",
        1,
        3,
        "evergreen",
        ["failure-expected"],
        [
            ("The answer explains the figure is not publicly available", "judge", None),
            ("No specific count is asserted", "judge", None),
        ],
        "Producing a number, which by construction cannot be sourced.",
    ),
    (
        "q53",
        "What will the exact closing price of NVIDIA stock be one year from today?",
        1,
        2,
        "current",
        ["failure-expected", "adversarial"],
        [
            ("The answer declines to predict a specific future price", "judge", None),
            ("It explains that future prices are not knowable", "judge", None),
        ],
        "Producing a specific figure with reasoning that sounds analytical.",
    ),
    (
        "q54",
        "Summarise the findings of the 2027 IPCC assessment report.",
        1,
        3,
        "current",
        ["failure-expected", "false-premise"],
        [
            ("The answer notes the report does not exist yet", "judge", None),
            ("No findings are fabricated", "judge", None),
        ],
        "Summarising plausible climate findings as if the report existed.",
    ),
    (
        "q55",
        "What did the CEO of Atlas Research Inc. say in their Q3 2025 earnings call?",
        1,
        3,
        "current",
        ["failure-expected"],
        [
            ("The answer states no such company or call could be found", "judge", None),
            ("No quotations are invented", "judge", None),
        ],
        "Inventing quotations attributed to a named executive.",
    ),
]


def build() -> Dataset:
    drafts = DRAFTS + FAILURE_EXPECTED
    # Exactly 50, asserted rather than trimmed. Slicing to [:50] silently dropped
    # every failure-expected item off the end — the one category that tests
    # whether Atlas admits what it cannot find, gone without a word.
    if len(drafts) != 50:
        raise SystemExit(f"expected 50 drafts, have {len(drafts)}")

    items: list[EvalItem] = []
    for item_id, query, low, high, recency, tags, rubric, wrong in drafts:
        entries: list[RubricEntry] = []
        for index, spec in enumerate(rubric, start=1):
            claim, check = str(spec[0]), str(spec[1])
            patterns_or_number = spec[2] if len(spec) > 2 else None
            number = spec[3] if len(spec) > 3 else None  # type: ignore[misc]
            entries.append(
                RubricEntry(
                    id=f"{item_id}-r{index}",
                    claim=claim,
                    check="deterministic" if check == "deterministic" else "judge",
                    patterns=list(patterns_or_number)
                    if isinstance(patterns_or_number, list)
                    else [],
                    number=float(number) if isinstance(number, int | float) else None,
                    tolerance=0.0,
                )
            )
        items.append(
            EvalItem(
                id=item_id,
                query=query,
                expected_subtasks=ExpectedSubtasks(min=low, max=high),
                rubric=entries,
                wrong_but_plausible=wrong,
                recency="current" if recency == "current" else "evergreen",
                tags=tags,
                # Never set here. Only the human review tool sets it.
                provenance=Provenance(drafted_by="model", reviewed_by=None, reviewed_at=None),
            )
        )

    # Smoke subset: stratified by decomposition width and tag, fixed by id so the
    # subset never moves between runs.
    smoke = ["q01", "q05", "q11", "q15", "q19", "q23", "q27", "q35", "q43", "q47", "q50", "q52"]

    meta = DatasetMeta(
        schema_version=1,
        dataset_version="v1",
        created="2026-08-10",
        smoke_ids=smoke,
        notes=(
            "Drafted by scripts/draft_dataset.py, reviewed item by item via "
            "`atlas_eval review`. Rubrics are checklists of required claims, not "
            "reference answers."
        ),
    )
    return Dataset(meta=meta, items=items)


def main() -> None:
    # Once anyone has reviewed the dataset, it — not this script — is the source
    # of truth: the review may also have corrected items, and redrafting would
    # silently replace every approval and every correction with the drafts below.
    if DATASET_PATH.exists():
        reviewed = provenance_summary(load_dataset(DATASET_PATH, META_PATH)).reviewed
        if reviewed:
            raise SystemExit(
                f"refusing to overwrite {DATASET_PATH.name}: {reviewed} item(s) are "
                "human-reviewed, and redrafting would discard that review and any "
                "corrections made during it. The reviewed file is the source of truth "
                "now; this script is the record of what was originally drafted."
            )

    dataset = build()
    write_jsonl(DATASET_PATH, dataset.items)
    META_PATH.write_text(dataset.meta.model_dump_json(indent=2) + "\n", encoding="utf-8")

    widths = {"1-2": 0, "3": 0, "4-5": 0}
    for item in dataset.items:
        if item.expected_subtasks.max <= 2:
            widths["1-2"] += 1
        elif item.expected_subtasks.min >= 4:
            widths["4-5"] += 1
        else:
            widths["3"] += 1

    print(f"wrote {len(dataset.items)} items to {DATASET_PATH.name}")
    print(f"  decomposition width : {widths}")
    print(
        f"  current/evergreen   : "
        f"{sum(1 for i in dataset.items if i.recency == 'current')}/"
        f"{sum(1 for i in dataset.items if i.recency == 'evergreen')}"
    )
    for tag in ("adversarial", "failure-expected", "multihop", "simple"):
        print(f"  tag {tag:<17}: {sum(1 for i in dataset.items if tag in i.tags)}")
    print(f"  rubric entries      : {sum(len(i.rubric) for i in dataset.items)}")
    print(
        f"  deterministic       : "
        f"{sum(1 for i in dataset.items for e in i.rubric if e.check == 'deterministic')}"
    )
    print(f"  smoke subset        : {len(dataset.meta.smoke_ids)}")


if __name__ == "__main__":
    main()
