"""Checking an answer's citations of the law, without a model.

The failure this exists for is the one lawyers have been sanctioned over: a
citation that looks verifiable and is not. Three ways a citation of a provision
can be wrong, each decidable from the library alone:

* **unresolved**: the provision does not exist ("AI Act Art. 114");
* **unretrieved**: it exists, but the run never read it, so whatever the answer
  says it says came from somewhere other than the text;
* **misquoted**: the answer puts words in quotation marks and attributes them to
  a provision that does not contain them.
"""

from __future__ import annotations

import pytest

from atlas.graph.state import ResearchResult, Source
from atlas.library import load_library
from atlas.library.citations import CitationReport, check_citations, cited_labels
from atlas.library.sources import as_source


def _read(*labels: str) -> list[ResearchResult]:
    """One research result that retrieved these provisions."""
    library = load_library()
    sources: list[Source] = []
    for label in labels:
        unit = library.resolve(label)
        assert unit is not None, label
        sources.append(as_source(library, unit))
    return [ResearchResult(subtask_id="s0", findings="f", sources=sources)]


READ_BREACH = _read("GDPR Art. 33(1)", "GDPR Art. 34")


def _check(answer: str, results: list[ResearchResult] = READ_BREACH) -> CitationReport:
    return check_citations(answer, results, load_library())


# --- finding the citations -------------------------------------------------------------


def test_citations_are_read_in_the_libraries_own_spelling() -> None:
    answer = (
        "Notify within 72 hours [GDPR Article 33(1)], tell people too [GDPR Art 34(1)]; "
        "both [GDPR Art. 33(1); GDPR Art. 34(1)]."
    )

    assert cited_labels(answer) == [
        "GDPR Art. 33(1)",
        "GDPR Art. 34(1)",
        "GDPR Art. 33(1)",
        "GDPR Art. 34(1)",
    ]


def test_a_bracket_that_is_not_a_citation_is_left_alone() -> None:
    assert cited_labels("A footnote [1] and a URL [https://a.example] are not law.") == []


# --- the three failures ---------------------------------------------------------------


def test_a_citation_of_a_retrieved_provision_passes() -> None:
    report = _check("Notify the authority within 72 hours [GDPR Art. 33(1)].")

    assert report == CitationReport(cited=["GDPR Art. 33(1)"])
    assert report.ok is True


def test_a_provision_that_does_not_exist_is_unresolved() -> None:
    report = _check("High-risk systems must register [AI Act Art. 114].")

    assert (report.unresolved, report.ok) == (["AI Act Art. 114"], False)


def test_a_provision_the_run_never_read_is_unretrieved() -> None:
    """Art. 83(5) exists, but nothing in this run retrieved it."""
    report = _check("Fines reach EUR 20 million [GDPR Art. 83(5)].")

    assert (report.unresolved, report.unretrieved) == ([], ["GDPR Art. 83(5)"])


def test_citing_inside_or_around_what_was_read_is_not_unretrieved() -> None:
    """34(1) sits inside the retrieved Art. 34; Art. 33 holds the retrieved 33(1)."""
    report = _check("Tell people [GDPR Art. 34(1)]; notify the authority [GDPR Art. 33].")

    assert report.unretrieved == []


def test_a_quotation_must_be_the_provisions_words() -> None:
    report = _check('The controller has "thirty days to notify the authority" [GDPR Art. 33(1)].')

    assert report.misquoted == [("GDPR Art. 33(1)", "thirty days to notify the authority")]


@pytest.mark.parametrize(
    "quote",
    [
        "“not later than 72 hours after having become aware of it”",
        '"Not later than 72 hours after having become aware of it"',
        '"without undue delay and, where feasible, ... after having become aware of it"',
    ],
)
def test_a_faithful_quotation_passes_however_it_is_typeset(quote: str) -> None:
    """Curly or straight quotes, sentence-initial capitals and elided words are
    how quotations are written; none of them is a misquotation."""
    report = _check(f"The deadline: {quote} [GDPR Art. 33(1)].")

    assert report.misquoted == []


def test_a_quotation_is_checked_against_a_citation_in_its_own_sentence() -> None:
    """The question's own words in quotes are not attributed to anything."""
    report = _check('You asked about "thirty days". The rule differs [GDPR Art. 33(1)].')

    assert report.misquoted == []


def test_an_ungrounded_url_is_still_reported() -> None:
    report = _check("See https://invented.example/guide [GDPR Art. 33(1)].")

    assert report.ungrounded_urls == ["https://invented.example/guide"]


def test_the_feedback_names_each_problem() -> None:
    report = _check(
        'Register [AI Act Art. 114]. Fines [GDPR Art. 83(5)]. "thirty days" [GDPR Art. 33(1)].'
    )

    assert report.feedback() == (
        "Citation problems in the answer, found by checking it against the law:\n"
        "- [AI Act Art. 114] is not a provision of that act. Remove it or cite the "
        "provision that actually says this.\n"
        "- [GDPR Art. 83(5)] was not among the provisions retrieved for this question, "
        "so nothing it says was read. Cite only retrieved provisions.\n"
        '- "thirty days" does not appear in [GDPR Art. 33(1)]. Quote the provision '
        "exactly, or paraphrase without quotation marks."
    )
