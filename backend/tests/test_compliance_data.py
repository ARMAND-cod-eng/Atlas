"""The compliance dataset: questions about the EU AI Act and GDPR, with the law
each answer has to cite.

Its answer key is written in article numbers, and they are the part most worth
distrusting: a model drafted them, and a model misnumbering an article is the
very failure this dataset exists to measure. So every expected citation has to
resolve in the library — the Official Journal text, parsed — before the dataset
loads in CI. A typo'd or invented article fails here, not in a scored run.
"""

from __future__ import annotations

from collections import Counter

import pytest

from atlas.library import load_library
from atlas_eval.dataset import (
    COMPLIANCE_META_PATH,
    COMPLIANCE_PATH,
    EvalItem,
    ExpectedCitations,
    load_dataset,
    provenance_summary,
)


def _compliance() -> list[EvalItem]:
    return load_dataset(COMPLIANCE_PATH, COMPLIANCE_META_PATH).items


def test_it_loads_and_is_forty_items() -> None:
    assert len(_compliance()) == 40


def test_it_covers_both_acts_and_their_interplay() -> None:
    """Eighteen questions on each act, four that need both."""
    scope = Counter(
        next(tag for tag in item.tags if tag in {"ai-act", "gdpr", "cross-act"})
        for item in _compliance()
    )

    assert scope == {"ai-act": 18, "gdpr": 18, "cross-act": 4}


def test_some_questions_rest_on_a_false_premise() -> None:
    """An answer that accepts the premise is wrong however well it is cited."""
    premises = [item.id for item in _compliance() if "false-premise" in item.tags]

    assert len(premises) >= 4


def test_every_item_says_which_law_answers_it() -> None:
    assert [item.id for item in _compliance() if item.citations is None] == []


def test_every_expected_citation_names_a_real_provision() -> None:
    library = load_library()
    unresolved = [
        (item.id, label)
        for item in _compliance()
        if item.citations is not None
        for label in [*item.citations.must, *item.citations.acceptable]
        if library.resolve(label) is None
    ]

    assert unresolved == []


def test_the_claim_matches_the_review_state() -> None:
    summary = provenance_summary(load_dataset(COMPLIANCE_PATH, COMPLIANCE_META_PATH))

    if summary.fully_reviewed:
        assert summary.claim == "40 hand-labeled query traces"
    else:
        assert summary.claim.startswith("40 model-drafted query traces")


# --- the schema --------------------------------------------------------------------


def test_a_citation_cannot_be_both_required_and_merely_acceptable() -> None:
    with pytest.raises(ValueError, match="both must and acceptable"):
        ExpectedCitations(must=["GDPR Art. 33(1)"], acceptable=["GDPR Art. 33(1)"])


def test_an_item_needs_at_least_one_required_citation() -> None:
    with pytest.raises(ValueError):
        ExpectedCitations(must=[], acceptable=["GDPR Art. 33"])


# --- the review screen ---------------------------------------------------------------


def test_review_shows_each_expected_citation_with_the_laws_words(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reviewer checks the answer key against the law, not against the draft."""
    from atlas_eval import cli

    [item] = [i for i in _compliance() if i.id == "c10"]

    cli._print_citations(item)
    out = capsys.readouterr().out

    expected = "MUST CITE\n  AI Act Art. 99(3) [Penalties]\n    Non-compliance with the prohibition"
    assert expected in out


def test_review_flags_a_citation_that_names_no_provision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from atlas_eval import cli

    [item] = [i for i in _compliance() if i.id == "c10"]
    bogus = item.model_copy(update={"citations": ExpectedCitations(must=["AI Act Art. 114"])})

    cli._print_citations(bogus)

    assert "AI Act Art. 114: NOT IN THE LIBRARY" in capsys.readouterr().out
