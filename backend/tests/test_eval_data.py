"""The dataset and the frozen corpus.

Two properties matter most here, and neither would announce itself if broken:
replay must be **deterministic** (or run-to-run variance is measuring the
retriever rather than the agent), and the dataset's **provenance claim** must not
outrun the review actually done.
"""

from __future__ import annotations

import pytest

from atlas_eval.corpus import (
    CorpusDocument,
    FrozenSearch,
    MissingCorpusError,
    QueryCorpus,
    Recorder,
    load_corpus,
)
from atlas_eval.dataset import (
    Dataset,
    DatasetMeta,
    EvalItem,
    ExpectedSubtasks,
    Provenance,
    RubricEntry,
    load_dataset,
    provenance_summary,
)

# --- the shipped dataset -------------------------------------------------------


def test_the_dataset_loads_and_is_fifty_items() -> None:
    """The README says fifty; the file must agree."""
    dataset = load_dataset()
    assert len(dataset.items) == 50


def test_the_distribution_covers_every_axis_the_suite_is_designed_to_test() -> None:
    dataset = load_dataset()
    tags = [tag for item in dataset.items for tag in item.tags]

    # Decomposability: narrow and wide both well represented, so a planner that
    # always emits five is detectable and so is one that always emits one.
    narrow = [i for i in dataset.items if i.expected_subtasks.max <= 2]
    wide = [i for i in dataset.items if i.expected_subtasks.min >= 4]
    assert len(narrow) >= 10
    assert len(wide) >= 10

    assert tags.count("adversarial") >= 4
    assert tags.count("failure-expected") == 5
    assert sum(1 for i in dataset.items if i.recency == "current") >= 8


def test_failure_expected_items_survived_into_the_dataset() -> None:
    """They were silently trimmed off the end once. Never again.

    Building the list as DRAFTS + FAILURE_EXPECTED and slicing to [:50] dropped
    exactly the category that tests whether Atlas admits what it cannot find, with
    no error and no warning.
    """
    dataset = load_dataset()
    failure_items = [i for i in dataset.items if "failure-expected" in i.tags]

    assert [i.id for i in failure_items] == ["q51", "q52", "q53", "q54", "q55"]
    for item in failure_items:
        assert any(
            word in item.wrong_but_plausible.lower()
            for word in ("invent", "fabricat", "producing a", "summarising")
        )


def test_the_smoke_subset_is_committed_and_stratified() -> None:
    """Fixed by id, never resampled — otherwise consecutive runs are incomparable."""
    dataset = load_dataset()
    smoke = dataset.smoke()

    assert len(smoke) == 12
    assert {i.id for i in smoke} == set(dataset.meta.smoke_ids)
    # Spans the width range and includes the awkward categories, so a smoke run
    # exercises the planner across its whole decision space.
    assert any(i.expected_subtasks.max <= 2 for i in smoke)
    assert any(i.expected_subtasks.min >= 4 for i in smoke)
    assert any("adversarial" in i.tags for i in smoke)
    assert any("failure-expected" in i.tags for i in smoke)


def test_every_deterministic_entry_can_actually_be_satisfied() -> None:
    """A deterministic entry with no patterns and no number scores zero forever."""
    for item in load_dataset().items:
        for entry in item.rubric:
            if entry.check == "deterministic":
                assert entry.patterns or entry.number is not None, f"{entry.id}"


# --- provenance: the claim cannot outrun the work ------------------------------


def _dataset(reviewed: int, total: int = 4) -> Dataset:
    items = [
        EvalItem(
            id=f"x{i}",
            query="q",
            expected_subtasks=ExpectedSubtasks(min=1, max=2),
            rubric=[RubricEntry(id=f"x{i}-r1", claim="c", check="judge")],
            wrong_but_plausible="w",
            recency="evergreen",
            provenance=Provenance(
                drafted_by="model",
                reviewed_by="human" if i < reviewed else None,
                reviewed_at="2026-08-10" if i < reviewed else None,
            ),
        )
        for i in range(total)
    ]
    meta = DatasetMeta(schema_version=1, dataset_version="v1", created="2026-08-10")
    return Dataset(meta=meta, items=items)


def test_a_fully_reviewed_dataset_may_be_called_hand_labeled() -> None:
    summary = provenance_summary(_dataset(reviewed=4, total=4))

    assert summary.fully_reviewed is True
    assert summary.claim == "4 hand-labeled query traces"


def test_a_partly_reviewed_dataset_may_not() -> None:
    """The specific thing being guarded against.

    "Hand-labeled" is a claim about whose judgements these are. With three of four
    reviewed, the honest phrasing is the weaker one, and it is produced
    automatically rather than left to whoever writes the README that week.
    """
    summary = provenance_summary(_dataset(reviewed=3, total=4))

    assert summary.fully_reviewed is False
    assert summary.claim == "4 model-drafted query traces, 3 of 4 human-reviewed"
    assert "hand-labeled" not in summary.claim


def test_the_shipped_dataset_claim_matches_its_review_state() -> None:
    """Whatever is true right now, the generated claim says it.

    This is the test that fails if someone writes "hand-labeled" into the README
    while items are still unreviewed.
    """
    summary = provenance_summary(load_dataset())

    if summary.fully_reviewed:
        assert summary.claim == "50 hand-labeled query traces"
    else:
        assert summary.claim.startswith("50 model-drafted query traces")
        assert "hand-labeled" not in summary.claim


# --- schema validation ---------------------------------------------------------


def test_an_inverted_subtask_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds max"):
        ExpectedSubtasks(min=5, max=2)


def test_an_empty_rubric_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvalItem(
            id="x",
            query="q",
            expected_subtasks=ExpectedSubtasks(min=1, max=2),
            rubric=[],
            wrong_but_plausible="w",
            recency="evergreen",
        )


def test_a_deterministic_entry_with_nothing_to_match_is_rejected() -> None:
    with pytest.raises(ValueError, match="never be satisfied"):
        RubricEntry(id="x-r1", claim="c", check="deterministic")


def test_smoke_ids_must_resolve() -> None:
    meta = DatasetMeta(
        schema_version=1, dataset_version="v1", created="2026-08-10", smoke_ids=["nope"]
    )
    with pytest.raises(ValueError, match="unknown items"):
        Dataset(meta=meta, items=_dataset(0, 1).items)


# --- the frozen corpus ---------------------------------------------------------


def _corpus() -> QueryCorpus:
    return QueryCorpus(
        item_id="t1",
        recorded_at="2026-08-10",
        tavily_calls=3,
        documents=[
            CorpusDocument(url="https://a.example", title="BM25 ranking", snippet="term frequency"),
            CorpusDocument(url="https://b.example", title="dense retrieval", snippet="embeddings"),
            CorpusDocument(
                url="https://c.example", title="hybrid search", snippet="bm25 and dense"
            ),
        ],
    )


async def test_replay_is_deterministic() -> None:
    """The property the whole variance measurement depends on.

    If replay reordered between calls, run-to-run variance would be measuring the
    retriever, and the paired SD that decides the gate threshold would be noise
    about noise.
    """
    first = await FrozenSearch(_corpus()).search("bm25 ranking", 3)
    second = await FrozenSearch(_corpus()).search("bm25 ranking", 3)

    assert [s.url for s in first] == [s.url for s in second]


async def test_ties_are_broken_by_url_not_left_to_chance() -> None:
    """A query matching nothing scores every document zero.

    Without an explicit tiebreak the order of equal scores is whatever sort
    stability happens to give, which differs across corpora and Python versions.
    """
    result = await FrozenSearch(_corpus()).search("zzzz nothing matches", 3)

    assert [s.url for s in result] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


async def test_ranking_actually_ranks() -> None:
    """Not just deterministic — the relevant document has to come first."""
    result = await FrozenSearch(_corpus()).search("embeddings dense", 1)

    assert [s.url for s in result] == ["https://b.example"]


async def test_max_results_is_honoured() -> None:
    result = await FrozenSearch(_corpus()).search("search", 2)
    assert len(result) == 2


async def test_the_search_calls_are_recorded_for_tool_scoring() -> None:
    frozen = FrozenSearch(_corpus())
    await frozen.search("first")
    await frozen.search("second")

    assert frozen.calls == ["first", "second"]


def test_a_missing_corpus_raises_rather_than_returning_empty(tmp_path: object) -> None:
    """The loud-failure requirement.

    Returning an empty list would be indistinguishable from "the web has nothing
    on this", and the agent would faithfully report that as a finding.
    """
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    with pytest.raises(MissingCorpusError, match="never falls through to live"):
        load_corpus("does-not-exist", tmp_path)


def test_recording_merges_passes_without_duplicating_documents() -> None:
    """Three recording passes over one item produce one pool, not three."""
    from atlas.graph.state import Source

    first = Recorder("t1")
    first.absorb("query one", [Source(url="https://a.example", title="A", snippet="s")])
    pass_one = first.finish()

    second = Recorder("t1")
    second.absorb("query two", [Source(url="https://a.example", title="A", snippet="s")])
    second.absorb("query two", [Source(url="https://b.example", title="B", snippet="s")])
    merged = second.finish(pass_one)

    assert [d.url for d in merged.documents] == ["https://a.example", "https://b.example"]
    # Both probes that found the shared document are remembered.
    shared = next(d for d in merged.documents if d.url == "https://a.example")
    assert sorted(shared.probe_queries) == ["query one", "query two"]
    assert merged.tavily_calls == 3
