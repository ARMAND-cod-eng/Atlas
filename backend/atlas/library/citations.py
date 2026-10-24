"""Checking an answer's citations of the law against the library, without a model.

Prompting a model not to invent citations is an instruction, not a guarantee.
This is the check that knows. Each bracketed citation in an answer
("[GDPR Art. 33(1)]") is held to three tests:

* it names a provision that **exists** in the act;
* the provision was **retrieved** in this run: the cited unit is one a
  researcher read, sits inside one, or contains one (citing Article 33 having
  read 33(1) is ordinary legal practice);
* any **quotation** in the same sentence appears in the cited provision, after
  normalising what typesetting changes (curly quotes, case, spacing) and
  allowing elisions ("...").

Web citations keep the URL check they always had.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from atlas.graph.state import ResearchResult
from atlas.library import Library, Unit

_BRACKET = re.compile(r"\[([^\[\]]{3,200})\]")
_LABEL = re.compile(
    r"^(AI Act|GDPR)\s+(Art(?:icle)?\.?|Annex|Recital)\s*(\d{1,3}|[IVXLC]+)\s*"
    r"((?:\(\s*[0-9a-z]{1,4}\s*\))*)$",
    re.IGNORECASE,
)
_ACT_START = re.compile(r"^\s*(AI Act|GDPR)\b", re.IGNORECASE)
_SENTENCE_END = re.compile('(?<=[.!?])\\s+(?=[A-Z\\["\u201c])')
_QUOTE = re.compile('["\u201c]([^"\u201d]{3,500})["\u201d]')
_ELISION = re.compile(r"\s*(?:\[\s*(?:\.\.\.|\u2026)\s*\]|\.\.\.|\u2026)\s*")


def _label(text: str) -> str | None:
    """A citation in the library's own spelling, or None if it is not one."""
    match = _LABEL.match(" ".join(text.split()))
    if match is None:
        return None
    act = "AI Act" if match.group(1).lower() == "ai act" else "GDPR"
    kind = match.group(2).lower()
    subdivisions = re.sub(r"\s+", "", match.group(4).lower())
    if kind.startswith("art"):
        return f"{act} Art. {int(match.group(3))}{subdivisions}"
    if kind == "annex":
        return f"{act} Annex {match.group(3).upper()}{subdivisions}"
    return f"{act} Recital {int(match.group(3))}"


def cited_labels(text: str) -> list[str]:
    """Every provision the text cites in brackets, in order, repeats included.

    One bracket may hold several citations: "[GDPR Art. 33(1); GDPR Art. 34(1)]".
    A bracket that holds none (a footnote number, a URL) is not a citation.
    """
    labels: list[str] = []
    for bracket in _BRACKET.finditer(text):
        parts: list[str] = []
        for chunk in bracket.group(1).split(";"):
            # A comma separates two citations only when the act is named again.
            pieces = chunk.split(",")
            current = pieces[0]
            for piece in pieces[1:]:
                if _ACT_START.match(piece):
                    parts.append(current)
                    current = piece
                else:
                    current = f"{current},{piece}"
            parts.append(current)
        labels.extend(label for part in parts if (label := _label(part)) is not None)
    return labels


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for curly, straight in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
        text = text.replace(curly, straight)
    return " ".join(text.split()).casefold()


def quotes(provision: str, quotation: str) -> bool:
    """Whether a quotation, elisions allowed, is the provision's own words."""
    text = _normalise(provision)
    position = 0
    for fragment in _ELISION.split(quotation):
        words = _normalise(fragment).strip(" ,.;:")
        if not words:
            continue
        found = text.find(words, position)
        if found < 0:
            return False
        position = found + len(words)
    return True


class CitationReport(BaseModel):
    """What checking an answer's citations found. ``ok`` means nothing."""

    cited: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    unretrieved: list[str] = Field(default_factory=list)
    # (the provision cited, the quotation it does not contain)
    misquoted: list[tuple[str, str]] = Field(default_factory=list)
    ungrounded_urls: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unresolved or self.unretrieved or self.misquoted or self.ungrounded_urls)

    def feedback(self) -> str:
        """The problems, written for the synthesizer that has to fix them."""
        lines = ["Citation problems in the answer, found by checking it against the law:"]
        lines += [
            f"- [{label}] is not a provision of that act. Remove it or cite the provision "
            "that actually says this."
            for label in self.unresolved
        ]
        lines += [
            f"- [{label}] was not among the provisions retrieved for this question, so "
            "nothing it says was read. Cite only retrieved provisions."
            for label in self.unretrieved
        ]
        lines += [
            f'- "{quotation}" does not appear in [{label}]. Quote the provision exactly, '
            "or paraphrase without quotation marks."
            for label, quotation in self.misquoted
        ]
        lines += [
            f"- {url} is not among the sources retrieved for this question. Cite only "
            "retrieved sources."
            for url in self.ungrounded_urls
        ]
        return "\n".join(lines)


def _retrieved(results: list[ResearchResult], library: Library) -> list[Unit]:
    units = [
        library.resolve(source.label)
        for result in results
        for source in result.sources
        if source.label is not None
    ]
    return [unit for unit in units if unit is not None]


def check_citations(answer: str, results: list[ResearchResult], library: Library) -> CitationReport:
    """Check every citation in ``answer`` against the law and what was retrieved."""
    # Imported here: the graph's nodes import this module.
    from atlas.graph.nodes import ungrounded_urls

    read = _retrieved(results, library)
    report = CitationReport(ungrounded_urls=sorted(ungrounded_urls(answer, results)))
    for label in dict.fromkeys(cited_labels(answer)):
        report.cited.append(label)
        unit = library.resolve(label)
        if unit is None:
            report.unresolved.append(label)
        elif not any(library.within(unit.id, r.id) or library.within(r.id, unit.id) for r in read):
            report.unretrieved.append(label)

    for sentence in _SENTENCE_END.split(answer):
        provisions = [
            (label, unit)
            for label in cited_labels(sentence)
            if (unit := library.resolve(label)) is not None
        ]
        if not provisions:
            continue
        for match in _QUOTE.finditer(sentence):
            quotation = match.group(1)
            if not any(quotes(unit.text, quotation) for _, unit in provisions):
                report.misquoted.append((provisions[0][0], quotation))
    return report
