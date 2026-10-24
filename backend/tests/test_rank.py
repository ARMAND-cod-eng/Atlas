"""BM25, shared by the eval's frozen search and the regulation library.

The ranking moved out of ``atlas_eval.corpus`` so production retrieval and the
eval's replay are one implementation. Moving it must not move a single result:
the committed baseline was scored against these rankings, and a refactor that
reordered them would change every later run's evidence while looking like a
pure code move. So the eval's rankings are pinned by digest.
"""

from __future__ import annotations

import hashlib

from atlas.tools.rank import BM25, tokenize
from atlas_eval.corpus import FrozenSearch, load_corpus
from atlas_eval.dataset import load_dataset

# sha256 over every item's top 10 for three query forms (the query itself, the
# researcher's " overview" retry, its first six words), recorded from the code
# the baseline was scored with.
EVAL_RANKINGS_DIGEST = "1fccecac2eb5a9ffd2ebb1d845f97c63b0646b4e79a6dc783be3fab0ec07c1dc"


async def test_the_eval_rankings_are_exactly_those_the_baseline_was_scored_on() -> None:
    digest = hashlib.sha256()
    for item in load_dataset().items:
        frozen = FrozenSearch(load_corpus(item.id))
        for query in (item.query, f"{item.query} overview", " ".join(item.query.split()[:6])):
            for source in await frozen.search(query, 10):
                digest.update(f"{item.id}|{query}|{source.url}\n".encode())

    assert digest.hexdigest() == EVAL_RANKINGS_DIGEST


def test_the_tokenizer_lowercases_and_splits_on_anything_not_alphanumeric() -> None:
    assert tokenize("GPT-4o, BM25: Okapi!") == ["gpt", "4o", "bm25", "okapi"]


def test_a_document_matching_more_rare_terms_ranks_first() -> None:
    bm25 = BM25([["cat", "sat"], ["dog", "sat"], ["cat", "dog", "cat"]])

    assert bm25.rank(["cat"], keys=["a", "b", "c"]) == [2, 0, 1]


def test_ties_are_broken_by_key_so_the_order_is_total() -> None:
    """Equal scores in arbitrary order would make identical runs differ."""
    bm25 = BM25([["x"], ["x"], ["x"]])

    assert bm25.rank(["x"], keys=["c", "a", "b"]) == [1, 2, 0]


def test_a_term_in_no_document_scores_nothing() -> None:
    bm25 = BM25([["a"], ["b"]])

    assert [bm25.score(i, ["zzz"]) for i in range(2)] == [0.0, 0.0]
