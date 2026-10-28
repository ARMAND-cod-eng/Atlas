"""Retrieval recall: does the library search surface the law an answer must cite?

Measured before the researcher uses the search, because a citation check can
only verify what was retrieved: a provision the search never returns is one no
answer can cite correctly, however good the model.
"""

from __future__ import annotations

from atlas.library import Library, Unit
from atlas.library.search import Hit
from atlas_eval.dataset import (
    DATASETS,
    EvalItem,
    ExpectedCitations,
    ExpectedSubtasks,
    RubricEntry,
    load_dataset,
)
from atlas_eval.retrieval import recall_at_k, split_of


def _unit(unit_id: str, label: str, parent: str | None = None) -> Unit:
    return Unit(
        id=unit_id, act="aia", kind="paragraph", label=label, title="", text="", parent=parent
    )


LIBRARY = Library(
    [
        _unit("a/1", "Art. 1"),
        _unit("a/1/1", "Art. 1(1)", parent="a/1"),
        _unit("a/2", "Art. 2"),
        _unit("a/2/1", "Art. 2(1)", parent="a/2"),
        _unit("a/3", "Art. 3"),
    ]
)


def _item(item_id: str, must: list[str]) -> EvalItem:
    return EvalItem(
        id=item_id,
        query=item_id,
        expected_subtasks=ExpectedSubtasks(min=1, max=1),
        rubric=[RubricEntry(id=f"{item_id}-r1", claim="c", check="judge")],
        wrong_but_plausible="w",
        recency="evergreen",
        citations=ExpectedCitations(must=must),
    )


def _hit(label: str) -> Hit:
    unit = LIBRARY.resolve(label)
    assert unit is not None
    return Hit(unit=unit, reason="match")


class FixedSearch:
    """Returns the same labelled units for every query, best first."""

    def __init__(self, labels: list[str]) -> None:
        self.hits = [_hit(label) for label in labels]

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        return self.hits[:limit]


def test_a_must_cite_counts_as_found_when_returned_inside_or_around_a_hit() -> None:
    """Art. 1 is found through its paragraph; Art. 2(1) through its article."""
    [result] = recall_at_k(
        [_item("q", ["Art. 1", "Art. 2(1)"])], LIBRARY, FixedSearch(["Art. 1(1)", "Art. 2"]), k=10
    )

    assert (result.found, result.missed, result.recall) == (["Art. 1", "Art. 2(1)"], [], 1.0)


def test_only_the_top_k_count() -> None:
    [result] = recall_at_k(
        [_item("q", ["Art. 3"])], LIBRARY, FixedSearch(["Art. 1", "Art. 2", "Art. 3"]), k=2
    )

    assert (result.found, result.missed, result.recall) == ([], ["Art. 3"], 0.0)


def test_recall_is_the_share_of_must_cites_found() -> None:
    [result] = recall_at_k(
        [_item("q", ["Art. 1", "Art. 3"])], LIBRARY, FixedSearch(["Art. 1"]), k=10
    )

    assert result.recall == 0.5


def test_the_split_is_fixed_by_id_and_halves_the_dataset() -> None:
    items = load_dataset(*DATASETS["compliance"]).items

    dev = [i.id for i in items if split_of(i.id) == "dev"]
    test = [i.id for i in items if split_of(i.id) == "test"]

    assert (len(dev), len(test)) == (20, 20)
    assert (dev[:3], test[:3]) == (["c01", "c03", "c05"], ["c02", "c04", "c06"])
