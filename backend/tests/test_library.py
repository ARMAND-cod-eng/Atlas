"""The regulation library: the official text, parsed into units a citation can name.

Everything downstream trusts this parse. A citation check that resolves
"AI Act Art. 6(2)" against the wrong paragraph does not fail; it vouches for an
answer that cites the wrong law. So the parse is pinned to facts about the acts
that do not come from the parser: how many articles, recitals and annexes each
one has, and the exact words of the provisions the domain eval leans on.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

import pytest

from atlas.library import ACTS, SOURCES_DIR, Library, load_library


@pytest.fixture(scope="module")
def library() -> Library:
    return load_library()


# --- the sources ------------------------------------------------------------------


def test_each_committed_source_is_the_file_its_manifest_describes() -> None:
    """The raw text is committed with its checksum, so an edit to it is visible."""
    manifest = json.loads((SOURCES_DIR / "manifest.json").read_text(encoding="utf-8"))

    for act in ACTS.values():
        raw = gzip.decompress((SOURCES_DIR / f"{act.id}.xhtml.gz").read_bytes())
        entry = manifest[act.id]
        assert (entry["celex"], entry["sha256"]) == (act.celex, hashlib.sha256(raw).hexdigest())


# --- golden counts, from the acts themselves -----------------------------------------


@pytest.mark.parametrize(
    ("act", "articles", "recitals", "annexes"),
    [("aia", 113, 180, 13), ("gdpr", 99, 173, 0)],
)
def test_every_article_recital_and_annex_is_parsed(
    library: Library, act: str, articles: int, recitals: int, annexes: int
) -> None:
    kinds = Counter(unit.kind for unit in library.units.values() if unit.act == act)

    assert (kinds["article"], kinds["recital"], kinds["annex"]) == (articles, recitals, annexes)


def test_every_numbered_paragraph_of_the_ai_act_is_parsed(library: Library) -> None:
    """500: the act's paragraph anchors (``id="006.003"``) that sit in the article
    they number. The text has 509 anchors; the other nine are paragraphs of *other*
    regulations, quoted inside the AI Act's amending articles (105 to 109)."""
    paragraphs = [u for u in library.units.values() if u.act == "aia" and u.kind == "paragraph"]

    assert len(paragraphs) == 500


@pytest.mark.parametrize("label", ["AI Act Art. 8(5)", "AI Act Art. 5(12)", "AI Act Art. 19(4)"])
def test_a_quoted_amendment_is_not_filed_as_an_ai_act_paragraph(
    library: Library, label: str
) -> None:
    """Article 105 inserts a paragraph 5 into Article 8 of *another* regulation.
    Filed under the AI Act, it would make "AI Act Art. 8(5)" — which does not
    exist — a citation the check accepts."""
    assert library.resolve(label) is None


def test_every_label_names_exactly_one_unit(library: Library) -> None:
    labels = Counter(unit.label for unit in library.units.values())

    assert [label for label, count in labels.items() if count > 1] == []


# --- the provisions the domain leans on, word for word --------------------------------
#
# The Official Journal quotes defined terms in typographic quotes, U+2018 and
# U+2019, and the parse keeps them: a quotation check must match the act's
# characters, not a keyboard's.


@pytest.mark.parametrize(
    ("label", "starts_with"),
    [
        (
            "AI Act Art. 6(1)",
            "Irrespective of whether an AI system is placed on the market or put into service",
        ),
        (
            "AI Act Art. 6(3)",
            "By derogation from paragraph 2, an AI system referred to in Annex III shall not be "
            "considered to be high-risk",
        ),
        ("AI Act Art. 3(1)", "\u2018AI system\u2019 means a machine-based system"),
        ("AI Act Art. 3(3)", "\u2018provider\u2019 means a natural or legal person"),
        ("AI Act Art. 50(1)", "Providers shall ensure that AI systems intended to interact"),
        ("AI Act Annex III(4)", "Employment, workers\u2019 management and access to self-"),
        (
            "GDPR Art. 35(3)(a)",
            "a systematic and extensive evaluation of personal aspects relating to natural persons",
        ),
        ("GDPR Art. 4(1)", "\u2018personal data\u2019 means any information relating to an"),
        ("GDPR Art. 6(1)(a)", "the data subject has given consent to the processing"),
        ("GDPR Recital 1", "The protection of natural persons in relation to the processing"),
    ],
)
def test_a_label_resolves_to_the_provision_it_names(
    library: Library, label: str, starts_with: str
) -> None:
    unit = library.resolve(label)

    assert unit is not None, label
    assert unit.text.startswith(starts_with), unit.text[:120]


def test_an_article_carries_its_heading(library: Library) -> None:
    unit = library.resolve("AI Act Art. 6")

    assert unit is not None
    assert (unit.kind, unit.title) == ("article", "Classification rules for high-risk AI systems")


def test_a_point_sits_inside_its_paragraph_and_article(library: Library) -> None:
    point = library.resolve("GDPR Art. 35(3)(a)")
    paragraph = library.resolve("GDPR Art. 35(3)")
    article = library.resolve("GDPR Art. 35")

    assert point is not None and paragraph is not None and article is not None
    assert (point.parent, paragraph.parent) == (paragraph.id, article.id)
    assert library.within(point.id, article.id) is True
    assert library.within(article.id, point.id) is False
    # A parent's text holds its children's.
    assert point.text in paragraph.text and paragraph.text in article.text


def test_an_unknown_label_resolves_to_nothing(library: Library) -> None:
    """The AI Act has 113 articles. A citation of the 114th is invented."""
    assert library.resolve("AI Act Art. 114") is None
    assert library.resolve("GDPR Art. 35(12)") is None
