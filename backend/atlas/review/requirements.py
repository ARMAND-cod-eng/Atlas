"""What the law requires a contract to contain, as a checklist of its own provisions.

Each requirement names a provision by its citation label, and its text is read
from the library, never written into a prompt: what a contract is held to is the
law as published. A framework is one such checklist; GDPR Article 28(3), the
terms a processor contract must stipulate, is the first. DORA Article 30 would
be a second, once that act is in the library.
"""

from __future__ import annotations

from pydantic import BaseModel

from atlas.library import Library


class Requirement(BaseModel):
    id: str
    label: str
    title: str
    # The words contracts use for the term where the law's are not: a supplier's
    # "subcontractors", "safeguards", "hand back or erase". Searched with the
    # law's words, as the library glossary adds everyday words to a question's.
    # Chosen as a drafter writes them, then measured (atlas_eval contracts).
    words: str = ""


class Framework(BaseModel):
    id: str
    title: str
    requirements: tuple[Requirement, ...]


GDPR_ART_28 = Framework(
    id="gdpr-art-28",
    title="GDPR Art. 28: processor contracts",
    requirements=(
        Requirement(
            id="details",
            label="GDPR Art. 28(3)",
            title="What is processed: subject-matter, duration, nature and purpose, types of data, "
            "categories of data subjects, the controller's obligations and rights",
            words="subject matter duration term nature purpose categories types data subjects "
            "description schedule annex",
        ),
        Requirement(
            id="instructions",
            label="GDPR Art. 28(3)(a)",
            title="Only on documented instructions",
            words="instructions directions documented written instructed directed",
        ),
        Requirement(
            id="confidentiality",
            label="GDPR Art. 28(3)(b)",
            title="Staff bound to confidentiality",
            words="confidentiality confidential secrecy staff personnel employees authorised",
        ),
        Requirement(
            id="security",
            label="GDPR Art. 28(3)(c)",
            title="Security measures (Article 32)",
            words="security secure safeguards protect protection technical organisational safe",
        ),
        Requirement(
            id="sub-processors",
            label="GDPR Art. 28(3)(d)",
            title="Conditions for sub-processors",
            words="sub-processor subprocessor subcontractor subcontract another processor engage "
            "appoint authorisation consent",
        ),
        Requirement(
            id="data-subject-rights",
            label="GDPR Art. 28(3)(e)",
            title="Help with data subjects' requests",
            words="data subject requests rights individuals access rectification erasure respond",
        ),
        Requirement(
            id="assistance",
            label="GDPR Art. 28(3)(f)",
            title="Help with security, breaches and impact assessments (Articles 32 to 36)",
            words="assist breach notification impact assessment prior consultation supervisory "
            "authority",
        ),
        Requirement(
            id="end",
            label="GDPR Art. 28(3)(g)",
            title="Delete or return the data at the end",
            words="termination expiry end return delete deletion erase destroy hand back copies",
        ),
        Requirement(
            id="audits",
            label="GDPR Art. 28(3)(h)",
            title="Information and audits",
            words="audit audits inspection inspections information demonstrate compliance verify "
            "verification",
        ),
    ),
)

FRAMEWORKS: dict[str, Framework] = {GDPR_ART_28.id: GDPR_ART_28}

# Art. 28(3) holds its points (a) to (h); as a requirement of its own it is the
# sentence before them, which says what the contract must set out.
_POINTS_BEGIN = "That contract or other legal act shall stipulate"


def provision_text(requirement: Requirement, library: Library) -> str:
    """The law's words for a requirement."""
    unit = library.resolve(requirement.label)
    if unit is None:
        raise KeyError(f"{requirement.label} is not in the library")
    text = unit.text
    if requirement.label == "GDPR Art. 28(3)" and _POINTS_BEGIN in text:
        text = text[: text.index(_POINTS_BEGIN)].strip()
    return text
