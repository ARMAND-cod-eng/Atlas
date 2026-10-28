"""Retrieval recall over the compliance dataset.

For each item, the share of its must-cite provisions the library search returns
in its top k. A provision counts as returned when a hit is it, contains it (a
paragraph holding the cited point), or sits inside it (a paragraph of the cited
article): in each case the words a citation needs are in what the researcher
read.

The query is the item's question as asked. The researcher searches with its
subtasks' wording instead, which the planner writes and which drifts from run to
run; the question is the fixed proxy, and the number it gives is the search's,
not the agent's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel

from atlas.library import Library
from atlas.library.search import Hit
from atlas_eval.dataset import EvalItem

Split = Literal["dev", "test"]


def split_of(item_id: str) -> Split:
    """Which half of the dataset an item is in, fixed by its id: odd ids develop,
    even ids test.

    Search changes are tried and kept on the dev half only, and the test half
    is measured once they are done. A change shaped on the questions it is then
    scored on reports how well it fits them, not how well it generalises.
    """
    return "dev" if int(item_id.lstrip("c")) % 2 == 1 else "test"


class Searcher(Protocol):
    def search(self, query: str, limit: int = 8) -> list[Hit]: ...


class Rewritten:
    """A search run on each question with its rewrite added, as the researcher
    runs it when QUERY_REWRITE is on."""

    def __init__(self, search: Searcher, rewrites: Mapping[str, str]) -> None:
        self._search = search
        self._rewrites = rewrites

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        return self._search.search(f"{query} {self._rewrites.get(query, '')}".strip(), limit)


class ItemRecall(BaseModel):
    item_id: str
    found: list[str]
    missed: list[str]
    hits: list[str]

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 1.0


def recall_at_k(
    items: Sequence[EvalItem], library: Library, searcher: Searcher, k: int
) -> list[ItemRecall]:
    results: list[ItemRecall] = []
    for item in items:
        if item.citations is None:
            continue
        hits = searcher.search(item.query, limit=k)
        found: list[str] = []
        missed: list[str] = []
        for label in item.citations.must:
            target = library.resolve(label)
            covered = target is not None and any(
                library.within(target.id, hit.unit.id) or library.within(hit.unit.id, target.id)
                for hit in hits
            )
            (found if covered else missed).append(label)
        results.append(
            ItemRecall(
                item_id=item.id,
                found=found,
                missed=missed,
                hits=[hit.unit.label for hit in hits],
            )
        )
    return results
