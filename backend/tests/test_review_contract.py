"""Reading a contract: its text from the file it came in, then its clauses.

A clause is what a reviewer cites and a finding quotes, so it needs a label the
reviewer can find in the document ("Clause 4.2") and an id that stays put. The
files are built here, a few lines each, so what each test reads is in front of it.
"""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

import pytest

from atlas.library import load_library
from atlas.review.contract import MAX_BYTES, MAX_CHARS, UnreadableContract, clauses_of, text_of
from atlas.review.requirements import GDPR_ART_28, provision_text


def tiny_docx(paragraphs: list[str]) -> bytes:
    """A .docx with one paragraph each, the second of each split across two runs."""
    body = "".join(
        "<w:p>"
        + (
            f'<w:r><w:t xml:space="preserve">{escape(p[: len(p) // 2])}</w:t></w:r>'
            f'<w:r><w:t xml:space="preserve">{escape(p[len(p) // 2 :])}</w:t></w:r>'
        )
        + "</w:p>"
        for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def tiny_pdf(lines: list[str]) -> bytes:
    """A one-page PDF, a line of text each; with no lines, a page with no text, as a scan has."""
    shown = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    content = f"BT /F1 11 Tf 72 720 Td {shown} ET" if lines else ""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n".encode()
    table = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n".encode()
    )
    return out


LINES = [
    "1. Subject matter",
    "The Processor processes personal data on behalf of the Controller.",
    "2. Instructions",
    "The Processor shall act only on documented instructions.",
]


# --- the text ----------------------------------------------------------------------


def test_plain_text_is_read_as_it_is() -> None:
    assert (
        text_of("dpa.txt", b"1. Subject matter\nThe Processor.")
        == "1. Subject matter\nThe Processor."
    )


def test_a_docx_is_read_paragraph_by_paragraph() -> None:
    assert text_of("dpa.docx", tiny_docx(LINES)).splitlines() == LINES


def test_a_pdf_is_read_line_by_line() -> None:
    assert [line.strip() for line in text_of("dpa.pdf", tiny_pdf(LINES)).splitlines()] == LINES


@pytest.mark.parametrize(
    ("filename", "data", "why"),
    [
        ("scan.pdf", tiny_pdf([]), "no text in it: it looks scanned"),
        ("dpa.pages", b"anything", "can read .txt, .md, .docx and .pdf"),
        ("broken.docx", b"not a zip", "could not be read as a .docx"),
        ("broken.pdf", b"not a pdf", "could not be read as a PDF"),
        ("empty.txt", b"   \n ", "no text in it"),
        ("huge.txt", b"a" * (MAX_BYTES + 1), "larger than 5 MB"),
        ("long.txt", b"a " * (MAX_CHARS // 2 + 1), "longer than 200,000 characters"),
    ],
    # Named, not derived from the data: the large ones would make an id megabytes long.
    ids=["scanned", "unsupported", "broken-docx", "broken-pdf", "empty", "too-big", "too-long"],
)
def test_a_file_that_cannot_be_read_says_why(filename: str, data: bytes, why: str) -> None:
    with pytest.raises(UnreadableContract) as raised:
        text_of(filename, data)

    assert why in str(raised.value)


# --- the clauses -----------------------------------------------------------------------

NUMBERED = """DATA PROCESSING AGREEMENT
between Acme Ltd (the Controller) and Cloudco SA (the Processor)

1. Subject matter
The Processor processes personal data on behalf of the Controller for
30 days after the order.

2. Sub-processors
2.1 The Processor shall not engage another processor without prior written authorisation.
2.2 The Processor shall impose the same obligations on any sub-processor.

Clause 3: Audits
The Processor shall allow audits by the Controller.
"""


def test_numbered_clauses_are_labelled_as_the_document_numbers_them() -> None:
    clauses = clauses_of(NUMBERED)

    assert [(c.id, c.label, c.heading) for c in clauses] == [
        ("c1", "Preamble", None),
        ("c2", "Clause 1", "Subject matter"),
        ("c3", "Clause 2", "Sub-processors"),
        ("c4", "Clause 2.1", None),
        ("c5", "Clause 2.2", None),
        ("c6", "Clause 3", "Audits"),
    ]
    # A line that starts with a number in running text ("30 days") starts no clause.
    assert clauses[1].text == (
        "The Processor processes personal data on behalf of the Controller for\n"
        "30 days after the order."
    )
    assert (
        clauses[3].text
        == "The Processor shall not engage another processor without prior written authorisation."
    )


def test_articles_and_sections_keep_their_own_word() -> None:
    labels = [c.label for c in clauses_of("Article 1 Scope\nText.\nSection 2. Term\nText.")]

    assert labels == ["Article 1", "Section 2"]


def test_a_number_used_twice_is_told_apart() -> None:
    labels = [c.label for c in clauses_of("1. Parties\nA.\n1. Annex\nB.")]

    assert labels == ["Clause 1", "Clause 1 (2)"]


def test_an_unnumbered_contract_is_read_paragraph_by_paragraph() -> None:
    clauses = clauses_of("The Processor acts on instructions.\n\nIt deletes the data at the end.\n")

    assert [(c.label, c.text) for c in clauses] == [
        ("¶ 1", "The Processor acts on instructions."),
        ("¶ 2", "It deletes the data at the end."),
    ]


# --- what a processor contract must contain ---------------------------------------------


def test_the_requirements_are_the_law_s_own_points() -> None:
    labels = [r.label for r in GDPR_ART_28.requirements]

    assert labels == ["GDPR Art. 28(3)", *(f"GDPR Art. 28(3)({p})" for p in "abcdefgh")]
    library = load_library()
    assert all(library.resolve(label) is not None for label in labels)


def test_a_requirement_reads_its_text_from_the_law() -> None:
    library = load_library()
    [details, instructions] = GDPR_ART_28.requirements[:2]

    # The paragraph's first sentence only: its points are requirements of their own.
    assert provision_text(details, library).endswith(
        "the obligations and rights of the controller."
    )
    assert provision_text(instructions, library).startswith(
        "processes the personal data only on documented instructions from the controller"
    )
