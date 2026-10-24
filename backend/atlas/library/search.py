"""Searching the regulation library.

Two ways in, in this order:

1. **Lookup.** A question that cites a provision — "Article 22 GDPR", "Art.
   26(6) AI Act", "Annex III", "Recital 26" — gets that provision first, by
   resolving the reference, never by ranking. Ranking a citation is how a
   search returns Article 21 for a question about Article 22.
2. **Ranking.** BM25 (atlas.tools.rank) over the library's *passages*: numbered
   paragraphs, the points of definitions articles and annexes, articles too
   short to have either, and recitals. Each passage is indexed with its
   article's heading, which is often the most precise words it has.

Results are deduplicated by containment: a passage inside a provision already
returned adds nothing a reader does not have, so it does not take a slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import snowballstemmer

from atlas.library import ACTS, Library, Unit
from atlas.tools.rank import BM25, tokenize

# Words that say nothing about which provision a question is about. "Shall",
# "may" and "must" are the law's own grammar and appear in nearly every passage.
_STOPWORDS = frozenset(
    """
    a about all also an and any are as at be been being by can could did do does
    for from had has have he her his how i if in into is it its may me might must
    my no not of on or our shall she should such than that the their them then
    there these they this those to under us we what when where which who whom will
    with would you your
    """.split()
)

_STEMMER = snowballstemmer.stemmer("english")

# The Official Journal spells British (-ise, -isation); questions may not. Folded
# before stemming, because the stemmer alone splits them: "authorised" stems to
# "authoris", "authorized" to "author".
_BRITISH = re.compile(r"(?<=[a-z]{2})is(e|ed|es|ing|ation|ations)$")


# The acts' own word for what a question calls the EU. The Official Journal does
# not write "EU" in running text: it is "the Union", "Union law", "data subjects
# who are in the Union".
_EU = {"eu": "union"}


def legal_terms(text: str) -> list[str]:
    """Index terms: lowercased, spelling-folded, stemmed, function words dropped."""
    words = [
        _BRITISH.sub(r"iz\1", _EU.get(word, word))
        for word in tokenize(text)
        if word not in _STOPWORDS
    ]
    stems: list[str] = _STEMMER.stemWords(words)
    return stems


# --- references --------------------------------------------------------------------

_ACT_NAMES: dict[str, re.Pattern[str]] = {
    "aia": re.compile(r"\b(ai act|artificial intelligence act|2024/1689)\b", re.IGNORECASE),
    "gdpr": re.compile(r"\b(gdpr|general data protection regulation|2016/679)\b", re.IGNORECASE),
}
_SUBDIVISIONS = r"((?:\(\s*[0-9a-z]{1,4}\s*\))*)"
_ARTICLE_REF = re.compile(r"\bart(?:icle|\.)?\s*(\d{1,3})\s*" + _SUBDIVISIONS, re.IGNORECASE)
_ANNEX_REF = re.compile(r"\bannex\s+([IVXLC]+)\b\s*" + _SUBDIVISIONS, re.IGNORECASE)
_RECITAL_REF = re.compile(r"\brecitals?\s+(\d{1,3})\b", re.IGNORECASE)


def _acts_named(query: str) -> list[str]:
    named = [act for act, pattern in _ACT_NAMES.items() if pattern.search(query)]
    return named or list(ACTS)


def _subdivisions(raw: str) -> list[str]:
    return re.findall(r"\(\s*([0-9a-z]{1,4})\s*\)", raw.lower())


def _most_specific(library: Library, base: str, parts: list[str]) -> Unit | None:
    """The deepest unit the reference names that exists: Art. 6(9) falls back to Art. 6."""
    for depth in range(len(parts), -1, -1):
        unit = library.resolve(base + "".join(f"({part})" for part in parts[:depth]))
        if unit is not None:
            return unit
    return None


def references(library: Library, query: str) -> list[Unit]:
    """Every provision the query cites outright, in the order it cites them."""
    found: list[Unit] = []
    for act_id in _acts_named(query):
        short = ACTS[act_id].short
        for match in _ARTICLE_REF.finditer(query):
            base = f"{short} Art. {int(match.group(1))}"
            unit = _most_specific(library, base, _subdivisions(match.group(2)))
            if unit is not None:
                found.append(unit)
        for match in _ANNEX_REF.finditer(query):
            base = f"{short} Annex {match.group(1).upper()}"
            unit = _most_specific(library, base, _subdivisions(match.group(2)))
            if unit is not None:
                found.append(unit)
        for match in _RECITAL_REF.finditer(query):
            unit = library.resolve(f"{short} Recital {int(match.group(1))}")
            if unit is not None:
                found.append(unit)
    return found


# --- definitions -------------------------------------------------------------------

# A definition opens with its term in the OJ's typographic quotes (U+2018, U+2019).
_DEFINED_TERM = re.compile("^\u2018([^\u2019]+)\u2019")
# The question asks what something is, or quotes a term. Without a cue like this,
# every question that mentions a provider would spend a slot on what one is.
_ASKS_MEANING = re.compile(
    r"\b(define[sd]?|definitions?|meaning|mean|means|what is an?|differen\w*)\b"
    "|['\"\u2018\u201c][^'\"\u2019\u201d]{2,60}['\"\u2019\u201d]",
    re.IGNORECASE,
)


def definitions(library: Library) -> list[tuple[tuple[str, ...], Unit]]:
    """Each defined term, as index terms, with the unit that defines it."""
    found: list[tuple[tuple[str, ...], Unit]] = []
    for unit in library.units.values():
        parent = library.units[unit.parent] if unit.parent is not None else None
        if unit.kind != "point" or parent is None or parent.title != "Definitions":
            continue
        match = _DEFINED_TERM.match(unit.text)
        if match is not None:
            found.append((tuple(legal_terms(match.group(1))), unit))
    return found


def _defined_in(
    query: str, terms: list[tuple[tuple[str, ...], Unit]], acts: set[str]
) -> list[Unit]:
    """Definitions of the terms a meaning-seeking query uses, longest match first."""
    if _ASKS_MEANING.search(query) is None:
        return []
    words = legal_terms(query)
    spans: list[tuple[int, int, Unit]] = []
    for term, unit in terms:
        if unit.act not in acts or not term:
            continue
        for start in range(len(words) - len(term) + 1):
            if tuple(words[start : start + len(term)]) == term:
                spans.append((start, -len(term), unit))
    chosen: list[Unit] = []
    taken: set[int] = set()
    for start, negative_length, unit in sorted(spans, key=lambda s: (s[0], s[1], s[2].id)):
        covered = set(range(start, start - negative_length))
        if covered & taken:
            continue
        taken |= covered
        chosen.append(unit)
    return chosen


# --- ranking -----------------------------------------------------------------------


_TRAILING_MARKER = re.compile(r"\(?[0-9a-z]{1,4}\)?\.?\s*$")


def _own_words(unit: Unit, points: list[Unit]) -> str:
    """A unit's text without its points': the lead-in before them and any text after.

    "Processing shall be lawful only if ... at least one of the following
    applies:" is what a question about lawful bases matches. Indexed together
    with all six bases, it is one long passage BM25's length normalisation
    pushes down; the points are indexed on their own anyway.
    """
    if not points:
        return unit.text
    first = unit.text.find(points[0].text)
    last = unit.text.rfind(points[-1].text)
    if first < 0 or last < 0:
        return unit.text
    lead = _TRAILING_MARKER.sub("", unit.text[:first])
    return f"{lead} {unit.text[last + len(points[-1].text) :]}".strip()


@dataclass(frozen=True, slots=True)
class Passage:
    """A unit ranking runs over, and the words it is ranked on."""

    unit: Unit
    indexed: str


def passages(library: Library) -> list[Passage]:
    """The units ranking runs over: each a self-contained piece a reader would quote.

    * Paragraphs, ranked on their own words (see ``_own_words``).
    * Every point and sub-point, ranked on its own words where it has points of
      its own: Article 5(1) holds all eight prohibitions, and a question about one
      of them should find that one; Annex III(4)(a) is what a question about
      recruitment tools has to find, not the heading of point 4.
    * Articles whose only children are points (definitions, Article 113's
      dates), ranked on their own words; articles and annexes with no children.
    * Recitals.

    Not whole articles that have paragraphs: an article competing with its own
    paragraphs wins on length and says less. Each passage is ranked with its
    article's or annex's heading, often the most precise words it has.
    """
    children: dict[str, list[Unit]] = {}
    for unit in library.units.values():
        if unit.parent is not None:
            children.setdefault(unit.parent, []).append(unit)
    chosen: list[Passage] = []
    for unit in library.units.values():
        kids = children.get(unit.id, [])
        points = [kid for kid in kids if kid.kind == "point"]
        if unit.kind in {"paragraph", "recital", "point"} or (
            unit.kind in {"article", "annex"} and len(points) == len(kids)
        ):
            chosen.append(Passage(unit=unit, indexed=f"{unit.title} {_own_words(unit, points)}"))
    return chosen


@dataclass(frozen=True, slots=True)
class Hit:
    unit: Unit
    reason: Literal["reference", "definition", "match"]
    score: float = 0.0


class LibrarySearch:
    """Lookup, then definitions asked about, then BM25 over passages.

    Deterministic: ties break by id.
    """

    def __init__(self, library: Library) -> None:
        self.library = library
        found = passages(library)
        self._passages = [passage.unit for passage in found]
        self._index = BM25([legal_terms(passage.indexed) for passage in found])
        self._ids = [unit.id for unit in self._passages]
        self._definitions = definitions(library)

    def _covered(self, unit: Unit, chosen: list[Hit]) -> bool:
        return any(self.library.within(unit.id, hit.unit.id) for hit in chosen)

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        chosen: list[Hit] = []
        for unit in references(self.library, query):
            if not self._covered(unit, chosen):
                chosen.append(Hit(unit=unit, reason="reference"))

        # A question that names one act is about that act. Ranked across both, a
        # GDPR question about territorial scope came back with AI Act recitals.
        acts = set(_acts_named(query))
        for unit in _defined_in(query, self._definitions, acts):
            if not self._covered(unit, chosen):
                chosen.append(Hit(unit=unit, reason="definition"))

        terms = legal_terms(query)
        scored = sorted(
            (
                (-self._index.score(i, terms), self._ids[i], i)
                for i in range(len(self._passages))
                if self._passages[i].act in acts
            ),
        )
        for negative, _, index in scored:
            if len(chosen) >= limit or negative >= 0:
                break
            unit = self._passages[index]
            if not self._covered(unit, chosen):
                chosen.append(Hit(unit=unit, reason="match", score=-negative))
        return chosen[:limit]
