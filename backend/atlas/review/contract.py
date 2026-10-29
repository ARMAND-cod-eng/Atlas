"""A contract as Atlas reads it: its text, from the file it came in, then its clauses.

A clause is what a finding cites and quotes, so it gets a label a reviewer can
find in the document ("Clause 4.2", "Article 3") and an id that stays put
("c7"). Clauses are cut at the document's own numbered headings; a document
with none is read paragraph by paragraph ("¶ 3").

Text extraction is deliberately plain. A .docx is a zip of XML and is read with
the standard library; a PDF is read with pypdf, and one with no text in it (a
scan) is refused rather than reviewed as empty: an empty contract would be
found to lack every term, which is a finding about the file, not the contract.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_BYTES = 5_000_000
MAX_CHARS = 200_000

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class UnreadableContract(ValueError):
    """A file that cannot be reviewed, and why, in words for the person who sent it."""


class Clause(BaseModel):
    id: str
    label: str
    heading: str | None = None
    text: str

    @property
    def words(self) -> str:
        """Everything a quote may come from: the heading and the text."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def text_of(filename: str, data: bytes) -> str:
    """The text of an uploaded contract, or ``UnreadableContract`` saying why not."""
    if len(data) > MAX_BYTES:
        raise UnreadableContract(f"{filename} is larger than 5 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix in ("", ".txt", ".md"):
        text = data.decode("utf-8", errors="replace")
    elif suffix == ".docx":
        text = _docx_text(filename, data)
    elif suffix == ".pdf":
        text = _pdf_text(filename, data)
    else:
        raise UnreadableContract(f"Atlas can read .txt, .md, .docx and .pdf, not {suffix}.")

    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()
    if not text:
        if suffix == ".pdf":
            raise UnreadableContract(
                f"{filename} has no text in it: it looks scanned. "
                "Atlas needs a PDF with selectable text."
            )
        raise UnreadableContract(f"{filename} has no text in it.")
    if len(text) > MAX_CHARS:
        raise UnreadableContract(f"{filename} is longer than 200,000 characters.")
    return text


def _docx_text(filename: str, data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise UnreadableContract(f"{filename} could not be read as a .docx.") from exc
    paragraphs = []
    for paragraph in root.iter(f"{_W}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag == f"{_W}tab":
                parts.append("\t")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _pdf_text(filename: str, data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise UnreadableContract(f"{filename} could not be read as a PDF.") from exc


# A worded heading: "Clause 3", "Article 4.1:", "Section 2." and what follows it.
_WORDED = re.compile(
    r"^(Clause|Article|Section)\s+(\d{1,3}(?:\.\d{1,3})*)\b[.:)]?\s*(.*)$", re.IGNORECASE
)
# A bare number is a heading only with its punctuation ("4.", "4)") or as a
# sub-clause ("4.2"): "30 days after termination" starts no clause.
_NUMBERED = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)[.)]\s+(.*)$|^(\d{1,3}(?:\.\d{1,3})+)\s+(.*)$")


def _heading_of(line: str) -> tuple[str, str] | None:
    """(label, the rest of the line) when the line starts a clause."""
    stripped = line.strip()
    worded = _WORDED.match(stripped)
    if worded is not None:
        return f"{worded.group(1).capitalize()} {worded.group(2)}", worded.group(3).strip()
    numbered = _NUMBERED.match(stripped)
    if numbered is not None:
        number = numbered.group(1) or numbered.group(3)
        rest = numbered.group(2) if numbered.group(1) else numbered.group(4)
        return f"Clause {number}", (rest or "").strip()
    return None


# How a sentence ends. A line that does not end one is continued on the next.
_SENTENCE_ENDS = (".", ";", ":", "!", "?", ")")


def _headings(lines: list[str]) -> list[tuple[int, tuple[str, str]]]:
    """Where clauses start: (line index, (label, rest of the line)).

    A line that only looks like a heading is running text when both say so: the
    line before it does not finish a sentence, and what follows its number is
    lowercase. Found in the first sample contract, whose Clause 4 wraps onto
    "Article 32 of the GDPR"; "2.1 million records" is the same mistake.
    """
    found: list[tuple[int, tuple[str, str]]] = []
    unfinished = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            unfinished = False
            continue
        heading = _heading_of(stripped)
        running_text = heading is not None and unfinished and heading[1][:1].islower()
        if heading is not None and not running_text:
            found.append((index, heading))
            unfinished = False
        else:
            unfinished = not stripped.endswith(_SENTENCE_ENDS)
    return found


# A title names a clause in a few words; a sentence that wraps is longer.
_TITLE_WORDS = 8


def _is_title(rest: str, following: str) -> bool:
    """A heading's title, not the clause's first sentence: a few words, not a sentence, and
    not continued in lowercase on the next line, as a wrapped sentence is."""
    return (
        0 < len(rest) <= 80
        and len(rest.split()) <= _TITLE_WORDS
        and not rest.endswith((".", ";", ":", ","))
        and not following.strip()[:1].islower()
    )


def clauses_of(text: str) -> list[Clause]:
    """The contract's clauses, in order, each with a label and a stable id."""
    lines = text.split("\n")
    headings = _headings(lines)
    if not headings:
        return _paragraphs(text)

    pieces: list[tuple[str, str | None, str]] = []
    preamble = "\n".join(lines[: headings[0][0]]).strip()
    if preamble:
        pieces.append(("Preamble", None, preamble))
    for position, (index, (label, rest)) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = [line for line in lines[index + 1 : end]]
        if _is_title(rest, body[0] if body else ""):
            heading: str | None = rest
        else:
            heading = None
            body = [rest, *body]
        pieces.append((label, heading, "\n".join(body).strip()))
    return _numbered(pieces)


def _numbered(pieces: list[tuple[str, str | None, str]]) -> list[Clause]:
    """Ids in order, and labels told apart where the document numbers twice."""
    seen: dict[str, int] = {}
    clauses = []
    for position, (label, heading, body) in enumerate(pieces, start=1):
        seen[label] = seen.get(label, 0) + 1
        shown = label if seen[label] == 1 else f"{label} ({seen[label]})"
        clauses.append(Clause(id=f"c{position}", label=shown, heading=heading, text=body))
    return clauses


def _paragraphs(text: str) -> list[Clause]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    return [
        Clause(id=f"c{position}", label=f"¶ {position}", text=block)
        for position, block in enumerate(blocks, start=1)
    ]
