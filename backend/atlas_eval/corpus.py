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

import gzip
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from atlas.graph.state import Source
from atlas.tools.rank import BM25, tokenize
from atlas_eval.dataset import DATA_DIR

logger = logging.getLogger(__name__)

CORPUS_DIR = DATA_DIR / "corpus"

# Page text kept per document. Enough for a reader to find the relevant passages
# of a typical article; a hard cap because the corpus is committed, and fifty
# items of uncapped pages would be tens of megabytes of mostly navigation chrome.
CONTENT_CHARS = 12_000


class CorpusDocument(BaseModel):
    """One recorded search result, plus how it was found."""

    url: str
    title: str
    snippet: str
    # The page's text, capped at CONTENT_CHARS. Recorded so a reader that uses
    # more than the snippet can be measured without a second (paid) recording.
    content: str | None = None
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
        return corpus_path(self.item_id, root)


class MissingCorpusError(RuntimeError):
    """Raised in replay when an item has no recorded corpus.

    Loud on purpose. The alternative — falling through to live search, or
    returning an empty list — either breaks the offline guarantee or fabricates a
    "found nothing" result that the agent will faithfully report as fact.
    """


class FrozenSearch:
    """A deterministic stand-in for :func:`atlas.tools.search.search`.

    Ranking is BM25-style over the item's pool. The scoring function matters less
    than two properties it must have:

    * **Total order.** Ties are broken by URL, because a sort that leaves equal
      scores in arbitrary order makes two runs of identical code differ — which
      is exactly the variance this module exists to remove.
    * **No network, ever.** There is no fallback path to Tavily. If the corpus is
      missing, it raises.

    Page text is ranked on never and returned only when asked for, exactly as
    live search behaves: today's agent sees the snippets production gives it.
    """

    def __init__(self, corpus: QueryCorpus) -> None:
        self.corpus = corpus
        self._docs = corpus.documents
        self._index = BM25([tokenize(f"{d.title} {d.snippet}") for d in self._docs])
        self._urls = [d.url for d in self._docs]
        self.calls: list[str] = []

    async def search(
        self, query: str, max_results: int | None = None, include_content: bool = False
    ) -> list[Source]:
        """Rank the frozen pool. Signature matches ``atlas.tools.search.search``."""
        self.calls.append(query)
        limit = max_results if max_results is not None else 5
        # The URL is the tiebreak that makes the order total.
        ranked = self._index.rank(tokenize(query), keys=self._urls)
        chosen = [self._docs[i] for i in ranked[:limit]]
        return [
            Source(
                url=d.url,
                title=d.title,
                snippet=d.snippet,
                content=d.content if include_content else None,
            )
            for d in chosen
        ]


def corpus_path(item_id: str, root: Path = CORPUS_DIR) -> Path:
    # Gzipped: page text makes each pool several hundred kilobytes of JSON.
    return root / f"{item_id}.json.gz"


def load_corpus(item_id: str, root: Path = CORPUS_DIR) -> QueryCorpus:
    path = corpus_path(item_id, root)
    if not path.exists():
        raise MissingCorpusError(
            f"no recorded corpus for item {item_id!r} at {path}. "
            "Run `atlas_eval record` — replay never falls through to live search."
        )
    return QueryCorpus.model_validate_json(gzip.decompress(path.read_bytes()))


def save_corpus(corpus: QueryCorpus, root: Path = CORPUS_DIR) -> Path:
    path = corpus_path(corpus.item_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(corpus.model_dump_json())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    # mtime=0 so the same corpus always compresses to the same bytes, and an
    # unchanged pool does not show up as a modified file.
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
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
            content = source.content[:CONTENT_CHARS] if source.content else None
            existing = self.documents.get(source.url)
            if existing is None:
                self.documents[source.url] = CorpusDocument(
                    url=source.url,
                    title=source.title,
                    snippet=source.snippet,
                    content=content,
                    probe_queries=[query],
                )
                continue
            if existing.content is None:
                existing.content = content
            if query not in existing.probe_queries:
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
                    continue
                current = merged[document.url]
                if current.content is None:
                    current.content = document.content
                for probe in document.probe_queries:
                    if probe not in current.probe_queries:
                        current.probe_queries.append(probe)
        # Sorted by URL so the committed file is stable across recordings.
        documents = [merged[url] for url in sorted(merged)]
        return QueryCorpus(
            item_id=self.item_id,
            recorded_at=datetime.now(UTC).date().isoformat(),
            tavily_calls=calls,
            documents=documents,
        )
