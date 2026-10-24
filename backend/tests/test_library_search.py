"""Searching the regulation library.

A researcher's question names the law in two ways, and the search honours both.
When it cites a provision outright ("Article 22 GDPR"), that provision comes
first, found by lookup and never left to ranking. Otherwise its words are ranked
with BM25 over the library's passages: paragraphs, definitions, annex points
and recitals, each read with its article's heading.
"""

from __future__ import annotations

import pytest

from atlas.library import load_library
from atlas.library.search import Hit, LibrarySearch, legal_terms


@pytest.fixture(scope="module")
def search() -> LibrarySearch:
    return LibrarySearch(load_library())


def _labels(hits: list[Hit]) -> list[str]:
    return [hit.unit.label for hit in hits]


# --- the tokenizer ---------------------------------------------------------------------


def test_british_and_american_spellings_meet() -> None:
    """The Official Journal writes "authorised"; a question may say "authorized"."""
    assert legal_terms("authorised organisation") == legal_terms("authorized organization")


def test_inflections_meet() -> None:
    assert legal_terms("processing processed") == legal_terms("processes process")


def test_function_words_carry_no_weight() -> None:
    assert legal_terms("What must the provider do under the Act?") == ["provid", "act"]


# --- references ------------------------------------------------------------------------


def test_a_cited_article_comes_first_by_lookup(search: LibrarySearch) -> None:
    hits = search.search("What does Article 22 GDPR require?")

    assert (hits[0].unit.label, hits[0].reason) == ("GDPR Art. 22", "reference")


def test_the_named_act_decides_which_article_6(search: LibrarySearch) -> None:
    references = [h for h in search.search("Article 6 of the AI Act") if h.reason == "reference"]

    assert _labels(references) == ["AI Act Art. 6"]


def test_with_no_act_named_both_are_looked_up(search: LibrarySearch) -> None:
    references = [h for h in search.search("What does Article 50 say?") if h.reason == "reference"]

    assert _labels(references) == ["AI Act Art. 50", "GDPR Art. 50"]


@pytest.mark.parametrize(
    ("query", "label"),
    [
        ("GDPR Article 35(3)(a) profiling", "GDPR Art. 35(3)(a)"),
        ("Art. 26(6) AI Act logs", "AI Act Art. 26(6)"),
        ("Annex III of the AI Act", "AI Act Annex III"),
        ("AI Act Annex III(4)(a) recruitment", "AI Act Annex III(4)(a)"),
        ("GDPR recital 26 anonymous data", "GDPR Recital 26"),
    ],
)
def test_paragraphs_points_annexes_and_recitals_are_looked_up(
    search: LibrarySearch, query: str, label: str
) -> None:
    assert search.search(query)[0].unit.label == label


def test_a_paragraph_that_does_not_exist_falls_back_to_its_article(
    search: LibrarySearch,
) -> None:
    assert search.search("Article 6(9) of the AI Act")[0].unit.label == "AI Act Art. 6"


def test_an_article_that_does_not_exist_is_not_invented(search: LibrarySearch) -> None:
    hits = search.search("Article 114 of the AI Act")

    assert [h for h in hits if h.reason == "reference"] == []


# --- definitions -------------------------------------------------------------------------


def test_a_question_about_meaning_gets_the_acts_definition(search: LibrarySearch) -> None:
    hits = search.search("How does the AI Act define an 'AI system'?")

    assert (hits[0].unit.label, hits[0].reason) == ("AI Act Art. 3(1)", "definition")


def test_quoted_terms_get_their_definitions_in_the_order_asked(search: LibrarySearch) -> None:
    hits = search.search("Under the AI Act, how is a 'provider' different from a 'deployer'?")

    assert [h.unit.label for h in hits if h.reason == "definition"] == [
        "AI Act Art. 3(3)",
        "AI Act Art. 3(4)",
    ]


def test_the_longest_defined_term_wins(search: LibrarySearch) -> None:
    """GDPR defines "personal data" (4(1)) and "personal data breach" (4(12)). A
    question about the second is not a question about the first."""
    hits = search.search("What counts as a 'personal data breach' under the GDPR?")

    assert [h.unit.label for h in hits if h.reason == "definition"] == ["GDPR Art. 4(12)"]


def test_a_term_merely_used_does_not_pull_in_its_definition(search: LibrarySearch) -> None:
    """Every question about providers mentions providers; few ask what one is."""
    hits = search.search("What must a provider of a high-risk AI system do under the AI Act?")

    assert [h for h in hits if h.reason == "definition"] == []


# --- ranking ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "label"),
    [
        (
            "deadline to notify the supervisory authority of a personal data breach",
            "GDPR Art. 33(1)",
        ),
        (
            "AI systems used to filter job applications and evaluate candidates",
            "AI Act Annex III(4)(a)",
        ),
        ("right to obtain human intervention and contest an automated decision", "GDPR Art. 22(3)"),
        ("inferring emotions of employees in the workplace", "AI Act Art. 5(1)(f)"),
    ],
)
def test_a_plain_language_question_finds_its_provision_in_the_top_five(
    search: LibrarySearch, query: str, label: str
) -> None:
    """Found means returned, or inside a passage returned: Annex III(4) holds 4(a)."""
    target = search.library.resolve(label)
    assert target is not None

    hits = search.search(query, limit=5)

    assert any(search.library.within(target.id, hit.unit.id) for hit in hits), _labels(hits)


def test_a_paragraph_is_found_by_its_own_lead_in(search: LibrarySearch) -> None:
    """Article 35(3): "A data protection impact assessment ... shall in particular be
    required in the case of:" then three long points. Ranked on all of that text,
    the lead-in a question matches was diluted below the article's other paragraphs."""
    hits = search.search("When is a data protection impact assessment required under the GDPR?")

    assert hits[0].unit.label == "GDPR Art. 35(3)"


def test_a_question_that_names_one_act_is_answered_from_that_act(search: LibrarySearch) -> None:
    hits = search.search("Does the GDPR apply to a company outside the EU that monitors people?")

    assert {hit.unit.act for hit in hits} == {"gdpr"}


def test_the_eu_is_the_union_in_the_acts_own_words() -> None:
    assert legal_terms("data subjects in the EU") == legal_terms("data subjects in the Union")


def test_a_hit_already_inside_a_looked_up_provision_is_not_repeated(
    search: LibrarySearch,
) -> None:
    """Article 22's text includes 22(1); listing both spends a slot on nothing."""
    hits = search.search("Article 22 GDPR automated decision profiling", limit=8)

    assert [h.unit.label for h in hits if h.unit.label.startswith("GDPR Art. 22(")] == []


def test_the_same_query_gives_the_same_hits(search: LibrarySearch) -> None:
    query = "obligations of deployers of high-risk AI systems"

    assert _labels(search.search(query)) == _labels(search.search(query))


def test_the_limit_is_respected(search: LibrarySearch) -> None:
    assert len(search.search("personal data", limit=3)) == 3
