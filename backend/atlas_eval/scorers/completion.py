"""Task completion: did the answer satisfy the rubric?

Hybrid by design. Rubric entries that can be checked mechanically are checked
mechanically — a date, a number, a named entity — and only the rest reach the
judge. The realised split is computed per run and printed, because "mostly
deterministic" is the sort of claim that decays quietly as rubrics get written.

Score is the fraction of rubric entries satisfied, so it is a proportion over a
denominator the dataset fixes, not a holistic impression of quality.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from atlas_eval.dataset import EvalItem, RubricEntry
from atlas_eval.scorers.judge import judge

JUDGE_SYSTEM = """You are checking whether a specific required fact appears in an answer.

You are given one REQUIRED FACT and an ANSWER. Decide whether the answer actually \
asserts that fact.

Rules:
- Wording will differ. Judge the claim, not the phrasing.
- The answer may say more than the fact requires. That is fine.
- If the answer states the fact but hedges it ("some sources suggest"), it still counts, \
as long as the substance is there.
- If the answer contradicts the fact, or omits it, it does not count.
- If the answer only alludes to the fact without asserting it, it does not count.

Answer with `present` true or false, and one short sentence of evidence quoting the \
relevant part of the answer.
"""


class FactVerdict(BaseModel):
    """The judge's decision about one rubric entry."""

    present: bool
    evidence: str = ""


class EntryResult(BaseModel):
    """Per-entry outcome, kept so a score can be traced back to a decision."""

    entry_id: str
    claim: str
    check: str
    satisfied: bool
    evidence: str = ""


class CompletionScore(BaseModel):
    score: float
    entries: list[EntryResult] = Field(default_factory=list)

    @property
    def deterministic_count(self) -> int:
        return sum(1 for e in self.entries if e.check == "deterministic")

    @property
    def judge_count(self) -> int:
        return sum(1 for e in self.entries if e.check == "judge")


def normalize(text: str) -> str:
    """Casefold, strip accents, collapse whitespace and most punctuation.

    Deterministic matching has to survive "GPT-4o" vs "GPT 4o" without becoming
    so loose that it matches anything.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.casefold()
    return re.sub(r"[^a-z0-9%.\s]+", " ", lowered).strip()


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def check_deterministic(entry: RubricEntry, answer: str) -> bool:
    """True when a pattern matches, or a number appears within tolerance."""
    haystack = _collapse(normalize(answer))

    for pattern in entry.patterns:
        if _collapse(normalize(pattern)) in haystack:
            return True

    if entry.number is not None:
        for found in _NUMBER.findall(haystack):
            try:
                value = float(found)
            except ValueError:
                continue
            if abs(value - entry.number) <= entry.tolerance:
                return True

    return False


async def score_completion(item: EvalItem, answer: str) -> CompletionScore:
    """Fraction of the rubric the answer satisfies."""
    results: list[EntryResult] = []

    for entry in item.rubric:
        if entry.check == "deterministic":
            satisfied = check_deterministic(entry, answer)
            results.append(
                EntryResult(
                    entry_id=entry.id,
                    claim=entry.claim,
                    check=entry.check,
                    satisfied=satisfied,
                    evidence="pattern match" if satisfied else "no pattern matched",
                )
            )
            continue

        if not answer.strip():
            # No answer cannot satisfy anything, and asking the judge to confirm
            # that would spend a call to learn nothing.
            results.append(
                EntryResult(
                    entry_id=entry.id,
                    claim=entry.claim,
                    check=entry.check,
                    satisfied=False,
                    evidence="the run produced no answer",
                )
            )
            continue

        verdict = await judge(
            FactVerdict,
            JUDGE_SYSTEM,
            f"REQUIRED FACT:\n{entry.claim}\n\nANSWER:\n{answer}",
        )
        results.append(
            EntryResult(
                entry_id=entry.id,
                claim=entry.claim,
                check=entry.check,
                satisfied=verdict.present,
                evidence=verdict.evidence,
            )
        )

    satisfied_count = sum(1 for r in results if r.satisfied)
    score = satisfied_count / len(results) if results else 0.0
    return CompletionScore(score=score, entries=results)
