"""Frozen search: a recorded corpus per query, replayed by a local retriever.

Live Tavily results change daily, so uncalibrated runs are not comparable and CI
would flake on the web rather than on the code.

**Why the corpus is frozen and not the query→result mapping.** The obvious design
— key fixtures on the exact search string — cannot work here. Claude rejects an
explicit ``temperature`` (the Phase 0 defect), so the planner's subtask wording
drifts between runs and the researcher issues different search strings each time.
Exact-match fixtures would then miss, and a miss is either a flaky CI failure or,
worse, a silent empty result that reads as "nothing found".

So the *documents* are frozen instead. Each dataset item gets a pool of real
Tavily results, gathered over several recording passes, and replay ranks that pool
locally for whatever string the researcher invents. Deterministic, offline, and
it never returns an unexplained empty list.

**Stated limitation.** This measures the agent against a fixed corpus of real web
content. It does not measure Tavily's ranking, and it cannot catch a regression
that only appears against live search. That belongs in the README, not here.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from atlas.graph.state import Source
from atlas_eval.dataset import DATA_DIR

logger = logging.getLogger(__name__)

CORPUS_DIR = DATA_DIR / "corpus"

_TOKEN = re.compile(r"[a-z0-9]+")


class CorpusDocument(BaseModel):
    """One recorded search result, plus how it was found."""

    url: str
    title: str
    snippet: str
    # Which search strings surfaced this document during recording. Kept for
    # provenance — it is how you check the pool actually covers the item.
    probe_queries: list[str] = Field(default_factory=list)


class QueryCorpus(BaseModel):
    """The frozen document pool for one dataset item."""

    item_id: str
    recorded_at: str
    tavily_calls: int
    documents: list[CorpusDocument]

    def path(self, root: Path = CORPUS_DIR) -> Path:
        return root / f"{self.item_id}.json"


class MissingCorpusError(RuntimeError):
    """Raised in replay when an item has no recorded corpus.

    Loud on purpose. The alternative — falling through to live search, or
    returning an empty list — either breaks the offline guarantee or fabricates a
    "found nothing" result that the agent will faithfully report as fact.
    """


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FrozenSearch:
    """A deterministic stand-in for :func:`atlas.tools.search.search`.

    Ranking is BM25-style over the item's pool. The scoring function matters less
    than two properties it must have:

    * **Total order.** Ties are broken by URL, because a sort that leaves equal
      scores in arbitrary order makes two runs of identical code differ — which
      is exactly the variance this module exists to remove.
    * **No network, ever.** There is no fallback path to Tavily. If the corpus is
      missing, it raises.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus: QueryCorpus) -> None:
        self.corpus = corpus
        self._docs = corpus.documents
        self._tokens = [tokenize(f"{d.title} {d.snippet}") for d in self._docs]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._counts = [Counter(t) for t in self._tokens]
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._tokens:
            self._doc_freq.update(set(tokens))
        self.calls: list[str] = []

    def _score(self, index: int, terms: Iterable[str]) -> float:
        total = len(self._docs)
        counts = self._counts[index]
        length = self._lengths[index] or 1
        score = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + self.K1 * (
                1 - self.B + self.B * length / (self._avg_length or 1)
            )
            score += idf * (frequency * (self.K1 + 1)) / denominator
        return score

    async def search(self, query: str, max_results: int | None = None) -> list[Source]:
        """Rank the frozen pool. Signature matches ``atlas.tools.search.search``."""
        self.calls.append(query)
        limit = max_results if max_results is not None else 5
        terms = tokenize(query)

        # Sort by (-score, url): the URL is the tiebreak that makes the order total.
        ranked = sorted(
            range(len(self._docs)),
            key=lambda i: (-self._score(i, terms), self._docs[i].url),
        )
        chosen = [self._docs[i] for i in ranked[:limit]]
        return [Source(url=d.url, title=d.title, snippet=d.snippet) for d in chosen]


def corpus_path(item_id: str, root: Path = CORPUS_DIR) -> Path:
    return root / f"{item_id}.json"


def load_corpus(item_id: str, root: Path = CORPUS_DIR) -> QueryCorpus:
    path = corpus_path(item_id, root)
    if not path.exists():
        raise MissingCorpusError(
            f"no recorded corpus for item {item_id!r} at {path}. "
            "Run `atlas_eval record` — replay never falls through to live search."
        )
    return QueryCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def save_corpus(corpus: QueryCorpus, root: Path = CORPUS_DIR) -> Path:
    path = corpus_path(corpus.item_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(corpus.model_dump_json())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class Recorder:
    """Wraps live search, keeping every document it sees.

    Used only by ``atlas_eval record``. Deduplicates by URL and accumulates the
    probe queries that surfaced each document, so several recording passes over
    the same item merge into one pool.
    """

    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        self.documents: dict[str, CorpusDocument] = {}
        self.tavily_calls = 0

    def absorb(self, query: str, sources: Iterable[Source]) -> None:
        self.tavily_calls += 1
        for source in sources:
            existing = self.documents.get(source.url)
            if existing is None:
                self.documents[source.url] = CorpusDocument(
                    url=source.url,
                    title=source.title,
                    snippet=source.snippet,
                    probe_queries=[query],
                )
            elif query not in existing.probe_queries:
                existing.probe_queries.append(query)

    def finish(self, previous: QueryCorpus | None = None) -> QueryCorpus:
        """Merge into any existing pool and stamp the recording time."""
        merged = dict(self.documents)
        calls = self.tavily_calls
        if previous is not None:
            calls += previous.tavily_calls
            for document in previous.documents:
                if document.url not in merged:
                    merged[document.url] = document
                else:
                    for probe in document.probe_queries:
                        if probe not in merged[document.url].probe_queries:
                            merged[document.url].probe_queries.append(probe)
        # Sorted by URL so the committed file is stable across recordings.
        documents = [merged[url] for url in sorted(merged)]
        return QueryCorpus(
            item_id=self.item_id,
            recorded_at=datetime.now(UTC).date().isoformat(),
            tavily_calls=calls,
            documents=documents,
        )
