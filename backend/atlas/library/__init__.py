"""The regulation library: EU acts, parsed into units a citation can name.

Each act's Official Journal text is committed as XHTML (``sources/``, fetched by
``scripts/ingest_regulation.py``) and parsed here into units — articles,
numbered paragraphs, points and sub-points, annexes and their points, recitals —
each with a stable id and the label a citation uses for it::

    aia/art/6/par/3          "AI Act Art. 6(3)"
    gdpr/art/35/par/3/pt/a   "GDPR Art. 35(3)(a)"
    aia/art/3/pt/1           "AI Act Art. 3(1)"      the definitions are points
    aia/anx/III/pt/4         "AI Act Annex III(4)"
    aia/rct/53               "AI Act Recital 53"

A unit's text includes its children's, so a point is findable on its own and
still sits, word for word, inside the paragraph a reader would quote.

The parse is pinned by ``tests/test_library.py`` to facts that do not come from
the parser: each act's count of articles, recitals and annexes, and the exact
words of the provisions the domain eval relies on.
"""

from __future__ import annotations

import functools
import gzip
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

SOURCES_DIR = Path(__file__).parent / "sources"


@dataclass(frozen=True, slots=True)
class Act:
    """One act in the library, and how its citations name it."""

    id: str
    # The name in a citation label: "AI Act Art. 6(2)".
    short: str
    title: str
    celex: str
    eli: str


ACTS: dict[str, Act] = {
    act.id: act
    for act in (
        Act(
            id="aia",
            short="AI Act",
            title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
            celex="32024R1689",
            eli="http://data.europa.eu/eli/reg/2024/1689/oj",
        ),
        Act(
            id="gdpr",
            short="GDPR",
            title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
            celex="32016R0679",
            eli="http://data.europa.eu/eli/reg/2016/679/oj",
        ),
    )
}

UnitKind = Literal["article", "paragraph", "point", "annex", "recital"]


class Unit(BaseModel):
    """One citable piece of an act."""

    id: str
    act: str
    kind: UnitKind
    label: str
    # The heading of the article or annex the unit belongs to.
    title: str
    text: str
    parent: str | None = None


class Library:
    """Every unit of every act, by id and by citation label."""

    def __init__(self, units: list[Unit]) -> None:
        self.units: dict[str, Unit] = {unit.id: unit for unit in units}
        self._by_label: dict[str, Unit] = {unit.label: unit for unit in units}

    def resolve(self, label: str) -> Unit | None:
        """The unit a citation label names, or None for a provision that does not exist."""
        return self._by_label.get(label)

    def within(self, unit_id: str, ancestor_id: str) -> bool:
        """Whether ``unit_id`` is ``ancestor_id`` or sits somewhere inside it."""
        current: str | None = unit_id
        while current is not None:
            if current == ancestor_id:
                return True
            current = self.units[current].parent
        return False


# --- parsing --------------------------------------------------------------------------

_XHTML = "{http://www.w3.org/1999/xhtml}"
_ARTICLE_ID = re.compile(r"^art_(\d+)$")
_PARAGRAPH_ID = re.compile(r"^\d{3}\.(\d{3})$")
_ANNEX_ID = re.compile(r"^anx_([IVXLC]+)$")
_RECITAL_ID = re.compile(r"^rct_(\d+)$")
# A point's marker cell: "(a)", "(1)", "(iv)", "1.", "(aa)".
_MARKER = re.compile(r"^\(?([0-9]+|[a-z]{1,3})\)?\.?$")
# The number a paragraph or recital opens with, which its label already carries.
_LEADING_NUMBER = re.compile(r"^(?:\d+\.|\(\d+\))\s*")


def _tag(element: ET.Element) -> str:
    return element.tag.removeprefix(_XHTML)


def _classes(element: ET.Element) -> set[str]:
    return set((element.get("class") or "").split())


def _text(element: ET.Element) -> str:
    """The element's words, with the OJ's non-breaking spaces and layout flattened."""
    return " ".join("".join(element.itertext()).replace("\xa0", " ").split())


def _join(elements: list[ET.Element]) -> str:
    return " ".join(text for text in (_text(e) for e in elements) if text)


class _Parser:
    def __init__(self, act: Act) -> None:
        self.act = act
        self.units: list[Unit] = []

    def add(
        self, kind: UnitKind, unit_id: str, label: str, title: str, text: str, parent: str | None
    ) -> None:
        self.units.append(
            Unit(
                id=unit_id,
                act=self.act.id,
                kind=kind,
                label=label,
                title=title,
                text=text,
                parent=parent,
            )
        )

    def parse(self, root: ET.Element) -> list[Unit]:
        for element in root.iter(f"{_XHTML}div"):
            element_id = element.get("id") or ""
            if match := _RECITAL_ID.match(element_id):
                self.recital(element, match.group(1))
            elif match := _ARTICLE_ID.match(element_id):
                self.article(element, match.group(1))
            elif match := _ANNEX_ID.match(element_id):
                self.annex(element, match.group(1))
        return self.units

    def recital(self, element: ET.Element, number: str) -> None:
        text = _LEADING_NUMBER.sub("", _text(element), count=1)
        self.add(
            "recital",
            f"{self.act.id}/rct/{number}",
            f"{self.act.short} Recital {number}",
            "",
            text,
            None,
        )

    def article(self, element: ET.Element, number: str) -> None:
        heading = next(
            (child for child in element if "eli-title" in _classes(child)),
            None,
        )
        title = _text(heading) if heading is not None else ""
        body = [
            child
            for child in element
            if "oj-ti-art" not in _classes(child) and "eli-title" not in _classes(child)
        ]
        unit_id = f"{self.act.id}/art/{number}"
        label = f"{self.act.short} Art. {number}"
        self.add("article", unit_id, label, title, _join(body), None)
        for child in body:
            if match := _PARAGRAPH_ID.match(child.get("id") or ""):
                self.paragraph(child, int(match.group(1)), unit_id, label, title)
            elif _tag(child) == "table":
                self.point(child, unit_id, label, title)

    def paragraph(
        self, element: ET.Element, number: int, article_id: str, article_label: str, title: str
    ) -> None:
        unit_id = f"{article_id}/par/{number}"
        label = f"{article_label}({number})"
        text = _LEADING_NUMBER.sub("", _text(element), count=1)
        self.add("paragraph", unit_id, label, title, text, article_id)
        for child in element:
            if _tag(child) == "table":
                self.point(child, unit_id, label, title)

    def point(self, table: ET.Element, parent_id: str, parent_label: str, title: str) -> None:
        """A point is a two-cell row: its marker, then its text (and any sub-points)."""
        row = next(table.iter(f"{_XHTML}tr"), None)
        cells = row.findall(f"{_XHTML}td") if row is not None else []
        if len(cells) != 2:
            return
        match = _MARKER.match(_text(cells[0]))
        if match is None:
            return
        key = match.group(1)
        unit_id = f"{parent_id}/pt/{key}"
        label = f"{parent_label}({key})"
        self.add("point", unit_id, label, title, _text(cells[1]), parent_id)
        for child in cells[1]:
            if _tag(child) == "table":
                self.point(child, unit_id, label, title)

    def annex(self, element: ET.Element, number: str) -> None:
        headings = [child for child in element if "oj-doc-ti" in _classes(child)]
        # The first heading is "ANNEX III" itself; the rest is its title.
        title = _join(headings[1:])
        body = [child for child in element if "oj-doc-ti" not in _classes(child)]
        unit_id = f"{self.act.id}/anx/{number}"
        label = f"{self.act.short} Annex {number}"
        self.add("annex", unit_id, label, title, _join(body), None)
        for child in body:
            if _tag(child) == "table":
                self.point(child, unit_id, label, title)


def parse_act(act: Act, xhtml: bytes) -> list[Unit]:
    """Every unit of one act, from its Official Journal XHTML."""
    return _Parser(act).parse(ET.fromstring(xhtml))


@functools.lru_cache(maxsize=1)
def load_library() -> Library:
    """Parse every committed act. Cached: the sources only change by commit."""
    units: list[Unit] = []
    for act in ACTS.values():
        raw = gzip.decompress((SOURCES_DIR / f"{act.id}.xhtml.gz").read_bytes())
        units.extend(parse_act(act, raw))
    return Library(units)
