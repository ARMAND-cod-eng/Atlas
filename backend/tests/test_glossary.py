"""Everyday words, searched in the acts' own.

The queries here are new phrasings, none of them a dataset question: the
glossary has to carry to questions it was not shaped on.
"""

from __future__ import annotations

import pytest

from atlas.library import load_library
from atlas.library.glossary import expand
from atlas.library.search import LibrarySearch


@pytest.fixture(scope="module")
def search() -> LibrarySearch:
    return LibrarySearch(load_library())


def _labels(search: LibrarySearch, query: str, k: int) -> list[str]:
    return [hit.unit.label for hit in search.search(query, limit=k)]


def test_an_everyday_phrase_adds_the_acts_words_and_keeps_its_own() -> None:
    assert expand("Must our chatbot say it is a bot?") == (
        "Must our chatbot say it is a bot? "
        "AI systems intended to interact directly with natural persons"
    )
    assert expand("What does Article 5 prohibit?") == "What does Article 5 prohibit?"


@pytest.mark.parametrize(
    ("query", "provision"),
    [
        ("Do customer-service chatbots have to say they are not human?", "AI Act Art. 50(1)"),
        ("Is a résumé-ranking tool regulated by the AI Act?", "AI Act Annex III(4)(a)"),
        ("Which lawful basis covers sending newsletters under the GDPR?", "GDPR Art. 6(1)"),
        ("Can we store customer data on servers in a country outside the EU?", "GDPR Art. 45(1)"),
        ("From when do the AI Act's deadlines bite?", "AI Act Art. 113"),
        ("Do we have to label deepfakes made with our video tool?", "AI Act Art. 50(4)"),
    ],
)
def test_a_question_in_everyday_words_finds_the_provision(
    search: LibrarySearch, query: str, provision: str
) -> None:
    library = search.library
    target = library.resolve(provision)
    assert target is not None
    hits = search.search(query, limit=10)

    assert any(
        library.within(target.id, hit.unit.id) or library.within(hit.unit.id, target.id)
        for hit in hits
    ), _labels(search, query, 10)


def test_a_question_about_whether_something_counts_gets_the_definition(
    search: LibrarySearch,
) -> None:
    # "counts as" asks whether something meets a definition, as "what is" does.
    hits = search.search("Does a hashed email address count as personal data under the GDPR?")

    assert ("GDPR Art. 4(1)", "definition") in [(h.unit.label, h.reason) for h in hits]
