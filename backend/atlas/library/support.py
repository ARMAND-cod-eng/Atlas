"""Does a cited provision say what the sentence citing it says?

The citation check (``atlas.library.citations``) knows whether a provision
exists and was read. It cannot know whether the provision says what the answer
attributes to it. A live run cited AI Act Art. 2(3), the national-security
exclusion, for the act reaching providers outside the Union, which is Art.
2(1)(c): a real provision, retrieved, and wrong. Telling the two apart takes
reading, so a model judges it.

The judgement is narrow and held to evidence:

* The judge sees one sentence and the cited provision's own text at a time,
  never the whole answer or its sources, and only for citations that passed
  the check against the law.
* "Supported" counts only with the provision's words quoted, and the quote is
  checked against the text. A support claimed without them is *unverified*:
  not evidence the citation is wrong, so nothing is sent back for it, and not
  evidence it is right, so it is not counted as supported.
* Only an explicit "does not support" sends an answer back, with the judge's
  reason, for the synthesizer to fix.

The library stays free of models: the judge is passed in.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from atlas.library import Library
from atlas.library.citations import CitationReport, cited_labels, quotes
from atlas.prompts import COMPLIANCE_SUPPORT_CHECK

# Citations judged per check. An answer citing more is checked up to here and
# the rest are counted unchecked: one call stays bounded however long the
# answer, and the count says what was left out.
MAX_CITATIONS = 12
# The provision text the judge reads, as much as a researcher was shown.
PROVISION_CHARS = 4000

_SENTENCE_END = re.compile('(?<=[.!?])\\s+(?=[A-Z\\["“])')


class CitedSentence(BaseModel):
    """One citation, and the sentence it is attached to."""

    label: str
    sentence: str


class SupportJudgement(BaseModel):
    """The judge's answer about one numbered citation."""

    item: int = Field(description="The number of the citation judged, as given.")
    supported: bool = Field(
        description="Whether the provision's text itself establishes what the sentence "
        "attributes to it."
    )
    quote: str = Field(
        default="",
        description="When supported: the shortest passage of the provision's text that "
        "establishes it, copied word for word.",
    )
    reason: str = Field(
        default="",
        description="When not supported: one short line on what the provision actually covers.",
    )


class SupportJudgements(BaseModel):
    judgements: list[SupportJudgement] = Field(default_factory=list)


class Support(BaseModel):
    """One judged citation."""

    label: str
    sentence: str
    quote: str = ""
    reason: str = ""


class SupportReport(BaseModel):
    """What judging an answer's citations found. ``ok`` means nothing to fix."""

    supported: list[Support] = Field(default_factory=list)
    unsupported: list[Support] = Field(default_factory=list)
    # Judged supported, but the quoted support is not in the provision's text.
    unverified: list[Support] = Field(default_factory=list)
    # Over the cap, or left without a judgement.
    unchecked: int = 0

    @property
    def ok(self) -> bool:
        return not self.unsupported

    def feedback(self) -> str:
        """The problems, written for the synthesizer that has to fix them."""
        lines = [
            "Citations that do not say what they are cited for, judged against each "
            "provision's text:"
        ]
        lines += [
            f'- [{s.label}] is cited for "{s.sentence}" but does not say that: '
            f"{s.reason.rstrip('.')}. Cite the provision that does, or drop the claim."
            for s in self.unsupported
        ]
        return "\n".join(lines)


def _sentences(answer: str) -> list[str]:
    """Sentences, and list items or lines without a full stop, as written."""
    return [
        " ".join(sentence.split())
        for line in answer.splitlines()
        for sentence in _SENTENCE_END.split(line)
        if sentence.strip()
    ]


def cited_sentences(answer: str, report: CitationReport) -> list[CitedSentence]:
    """Each citation that passed the check against the law, with its sentence.

    A provision that does not exist, or was never read, has already failed; it
    is not judged again. A sentence citing two provisions is two pairs.
    """
    failed = set(report.unresolved) | set(report.unretrieved)
    pairs = [
        (label, sentence)
        for sentence in _sentences(answer)
        for label in cited_labels(sentence)
        if label not in failed
    ]
    # A label cited twice in one sentence is one pair.
    return [
        CitedSentence(label=label, sentence=sentence) for label, sentence in dict.fromkeys(pairs)
    ]


def _provision_text(library: Library, label: str) -> str:
    unit = library.resolve(label)
    text = unit.text if unit is not None else ""
    if len(text) > PROVISION_CHARS:
        text = text[:PROVISION_CHARS].rsplit(" ", 1)[0] + " [...]"
    return text


def support_messages(pairs: list[CitedSentence], library: Library) -> list[BaseMessage]:
    """What the judge is sent: the rules, then each numbered citation."""
    items = "\n\n".join(
        f"[{n}] Sentence: {pair.sentence}\n"
        f"Cited: {pair.label}\n"
        f"Provision text:\n{_provision_text(library, pair.label)}"
        for n, pair in enumerate(pairs, start=1)
    )
    return [SystemMessage(COMPLIANCE_SUPPORT_CHECK), HumanMessage(items)]


Judge = Callable[[list[BaseMessage]], Awaitable[SupportJudgements]]


async def check_support(
    answer: str, report: CitationReport, library: Library, judge: Judge
) -> SupportReport:
    """Judge whether each checked citation says what its sentence says."""
    pairs = cited_sentences(answer, report)
    if not pairs:
        return SupportReport()
    asked = pairs[:MAX_CITATIONS]
    answered = await judge(support_messages(asked, library))

    by_item = {j.item: j for j in answered.judgements}
    result = SupportReport(unchecked=len(pairs) - len(asked))
    for n, pair in enumerate(asked, start=1):
        judgement = by_item.get(n)
        if judgement is None:
            result.unchecked += 1
            continue
        unit = library.resolve(pair.label)
        if not judgement.supported:
            result.unsupported.append(
                Support(label=pair.label, sentence=pair.sentence, reason=judgement.reason)
            )
        elif unit is not None and judgement.quote.strip() and quotes(unit.text, judgement.quote):
            result.supported.append(
                Support(label=pair.label, sentence=pair.sentence, quote=judgement.quote)
            )
        else:
            result.unverified.append(
                Support(label=pair.label, sentence=pair.sentence, quote=judgement.quote)
            )
    return result
