"""The domain eval's deterministic measures: is each citation real, is it the right
law, and does the answer cite the law it rests on.

Run against the real library, so "exists" means what the Official Journal says.
"""

from __future__ import annotations

import pytest

from atlas.library import load_library
from atlas_eval.citation_metrics import CitationScore, score_answer, summarize_citations
from atlas_eval.dataset import ExpectedCitations

EXPECTED = ExpectedCitations(
    must=["AI Act Art. 6(2)", "AI Act Annex III(4)(a)"],
    acceptable=["AI Act Art. 6(3)"],
)


def _score(answer: str) -> CitationScore:
    return score_answer("c01", answer, EXPECTED, load_library())


def test_an_answer_citing_exactly_the_law_it_rests_on() -> None:
    score = _score("High-risk [AI Act Art. 6(2)] as recruitment AI [AI Act Annex III(4)(a)].")

    assert score.model_dump() == {
        "item_id": "c01",
        "cited": ["AI Act Art. 6(2)", "AI Act Annex III(4)(a)"],
        "nonexistent": [],
        "unsupported": [],
        "must_covered": ["AI Act Art. 6(2)", "AI Act Annex III(4)(a)"],
        "must_missed": [],
    }
    assert (score.validity, score.precision, score.recall, score.fabricated) == (
        1.0,
        1.0,
        1.0,
        False,
    )


def test_an_invented_article_is_fabricated() -> None:
    # The AI Act has 113 articles; the 2021 proposal's numbering is not the act's.
    score = _score("See [AI Act Art. 6(2)] and the registration duty [AI Act Art. 114].")

    assert (score.nonexistent, score.validity, score.fabricated) == (["AI Act Art. 114"], 0.5, True)


def test_citing_the_whole_article_covers_the_paragraph_it_contains() -> None:
    score = _score("[AI Act Art. 6] and [AI Act Annex III(4)]")

    assert (score.must_covered, score.recall) == (
        ["AI Act Art. 6(2)", "AI Act Annex III(4)(a)"],
        1.0,
    )


def test_citing_a_point_covers_the_paragraph_it_sits_in() -> None:
    expected = ExpectedCitations(must=["AI Act Art. 6(3)"])

    score = score_answer("c01", "[AI Act Art. 6(3)(a)]", expected, load_library())

    assert (score.must_covered, score.unsupported) == (["AI Act Art. 6(3)"], [])


def test_real_but_unrelated_law_counts_against_precision_not_validity() -> None:
    score = _score("[AI Act Art. 6(2)] and, for good measure, [GDPR Art. 5].")

    assert (score.unsupported, score.validity, score.precision, score.recall) == (
        ["GDPR Art. 5"],
        1.0,
        0.5,
        0.5,
    )


def test_an_answer_that_cites_nothing_has_no_validity_to_score_and_no_recall() -> None:
    score = _score("Yes, it is high-risk.")

    assert (score.cited, score.validity, score.precision, score.recall, score.fabricated) == (
        [],
        None,
        None,
        0.0,
        False,
    )


def test_a_repeated_citation_is_one_citation() -> None:
    assert _score("[GDPR Art. 5] ... [GDPR Art. 5]").cited == ["GDPR Art. 5"]


def test_the_suite_reports_shares_over_every_answer_and_means_over_those_that_cite() -> None:
    scores = [
        _score("[AI Act Art. 6(2)] [AI Act Annex III(4)(a)]"),  # perfect
        _score("[AI Act Art. 114]"),  # only an invented article
        _score("No citations at all."),
        _score("[AI Act Art. 6(2)] [GDPR Art. 5]"),
    ]

    summary = summarize_citations(scores)

    assert summary.answers == 4
    assert (summary.fabricated.value_pp, summary.uncited.value_pp) == (25.0, 25.0)
    # Validity over the three answers that cite: 1, 0, 1.
    assert summary.validity.value_pp == pytest.approx(200 / 3)
    assert summary.validity.items == 3
    # Recall over all four, an answer that cites nothing scoring zero.
    assert summary.recall.value_pp == pytest.approx((1 + 0 + 0 + 0.5) / 4 * 100)
    # Precision over answers with a real citation: 1 and 0.5.
    assert (summary.precision.value_pp, summary.precision.items) == (75.0, 2)


def test_acceptable_law_is_on_point_without_being_required() -> None:
    # 6(3) is the derogation: worth citing, not required, and not inside 6(2).
    score = _score("[AI Act Art. 6(2)] unless [AI Act Art. 6(3)] applies.")

    assert (score.unsupported, score.precision, score.recall) == ([], 1.0, 0.5)
